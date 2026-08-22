from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

from .adaptive_cognition import QueryCandidate, TaskState
from .body_checkpoint import checkpoint_dict, restore_runtime
from .causal_identification import GenerationScopedIdentifier, VersionSpaceSnapshot
from .causal_model_genesis import CausalModelGenesisEngine, GeneratedCausalModel, InterventionDescriptor
from .causal_predicate_genesis import BooleanCausalPredicateGenesisEngine, GeneratedPredicateModel
from .causal_program_genesis import CompositionalCausalProgramGenesisEngine, GeneratedCausalProgram
from .cognitive_runtime import CognitiveCycle, PersistentCognitiveRuntime
from .possibility_space import Fact, OperatorSpec
from .representation_genesis import MeasurementObservation, RepresentationGenesisEngine
from .semantic_genesis import ResidualObservation
from .world_coupling import WorldExecutor, WorldOutcomePair, WorldReceiptVerifier
from .world_model_ecology import (
    CausalWorldModel,
    EpistemicDepthPlan,
    EpistemicInterventionScore,
    WorldModelEcology,
)


EPISTEMIC_DEPTH_SCHEMA = "arte.epistemic_depth_same_body/v1"


@dataclass(frozen=True)
class CausalExpansionDecision:
    status: str
    generation: int
    origin: str
    active_model_ids: Tuple[str, ...]
    shadow_model_ids: Tuple[str, ...]
    reason: str


class EpistemicallyDeepPersistentCognitiveRuntime(PersistentCognitiveRuntime):
    """Same BODY with staged causal representation expansion.

    Candidate *presence* in persistent phenotype is evidence-independent: each
    generation first builds its unfiltered bounded shadow universe, persists only
    that universe, and then marks an evidence-compatible subset for immediate
    reasoning. This prevents evidence-conditioned candidate membership from
    becoming a covert authority channel across checkpoint/restore.

    Structural depth is selected by the BODY itself. Independently authenticated
    failure opens exactly the next generation supported by ancestry: named causal
    families (G1), compositional causal programs (G2), then synthesized Boolean
    activation predicates (G3). Exact identification is generation-scoped so older
    lineage cannot dilute a current generation's experiment-information gain.
    """

    def __init__(
        self,
        *args,
        world_models: Optional[WorldModelEcology] = None,
        model_genesis: Optional[CausalModelGenesisEngine] = None,
        program_genesis: Optional[CompositionalCausalProgramGenesisEngine] = None,
        predicate_genesis: Optional[BooleanCausalPredicateGenesisEngine] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.world_models = world_models or WorldModelEcology()
        self.model_genesis = model_genesis or CausalModelGenesisEngine()
        self.program_genesis = program_genesis or CompositionalCausalProgramGenesisEngine()
        self.predicate_genesis = predicate_genesis or BooleanCausalPredicateGenesisEngine()
        self.identifier = GenerationScopedIdentifier()
        self.last_epistemic_depth = self.world_models.depth_plan()

    def register_causal_world_models(self, models: Sequence[CausalWorldModel]) -> None:
        self.world_models.register(models)
        self.last_epistemic_depth = self.world_models.depth_plan()

    @staticmethod
    def _models(items) -> List[CausalWorldModel]:
        return [item.model if hasattr(item, "model") else item for item in items]

    @staticmethod
    def _restrict_active_to_shadow(active, shadow):
        shadow_ids = {item.model.model_id for item in shadow}
        return [item for item in active if item.model.model_id in shadow_ids]

    def structural_models(self, generation: int) -> List[CausalWorldModel]:
        return [
            model for model in self.world_models.models.values()
            if int(model.generation) == int(generation)
        ]

    def latest_structural_generation(self) -> int:
        generated = [
            int(model.generation) for model in self.world_models.models.values()
            if model.origin.startswith("GENERATED") and int(model.generation) > 0
        ]
        return max(generated) if generated else 0

    def generation_version_space(self, generation: int) -> VersionSpaceSnapshot:
        return self.identifier.snapshot(
            generation,
            list(self.world_models.models.values()),
            self.world_models.authoritative_evidence(),
        )

    def rank_generation_interventions(
        self,
        generation: int,
        candidates: Sequence[QueryCandidate],
    ) -> Sequence[EpistemicInterventionScore]:
        snapshot = self.generation_version_space(generation)
        return self.identifier.rank_interventions(
            candidates,
            snapshot.compatible_model_ids,
            cost_exponent=0.15,
        )

    def select_generation_intervention(
        self,
        generation: int,
        candidates: Sequence[QueryCandidate],
    ) -> Optional[EpistemicInterventionScore]:
        snapshot = self.generation_version_space(generation)
        return self.identifier.select_next(
            candidates,
            snapshot.compatible_model_ids,
            cost_exponent=0.15,
        )

    def generate_replacement_causal_models(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
    ) -> List[GeneratedCausalModel]:
        if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
            return []
        shadow = self.model_genesis.generate(variables, descriptors, ())
        self.world_models.register(self._models(shadow))
        active = self.model_genesis.generate(
            variables, descriptors, self.world_models.authoritative_evidence()
        )
        active = self._restrict_active_to_shadow(active, shadow)
        self.last_epistemic_depth = self.world_models.depth_plan()
        return active

    def generate_compositional_causal_models(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
    ) -> List[GeneratedCausalProgram]:
        if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
            return []
        if not any(model.origin == "GENERATED" for model in self.world_models.models.values()):
            return []
        existing = list(self.world_models.models.values())
        shadow = self.program_genesis.generate_novel(variables, descriptors, (), existing)
        self.world_models.register(self._models(shadow))
        active = self.program_genesis.generate_novel(
            variables, descriptors, self.world_models.authoritative_evidence(), existing
        )
        active = self._restrict_active_to_shadow(active, shadow)
        self.last_epistemic_depth = self.world_models.depth_plan()
        return active

    def generate_predicate_causal_models(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
    ) -> List[GeneratedPredicateModel]:
        """Open generation 3 only after compositional lineage failure.

        The full unfiltered predicate-equivalence universe is built first. If its
        configured budget truncates that universe, structural promotion fails
        closed rather than letting an evidence-filtered candidate outside the
        shadow universe encode authority via checkpoint membership.
        """
        if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
            return []
        if not any(
            model.origin == "GENERATED_COMPOSITIONAL"
            for model in self.world_models.models.values()
        ):
            return []
        existing = list(self.world_models.models.values())
        shadow = self.predicate_genesis.generate_novel(variables, descriptors, (), existing)
        shadow_truncated = bool(self.predicate_genesis.last_truncated)
        self.world_models.register(self._models(shadow))
        if shadow_truncated:
            self.last_epistemic_depth = self.world_models.depth_plan()
            return []
        active = self.predicate_genesis.generate_novel(
            variables, descriptors, self.world_models.authoritative_evidence(), existing
        )
        active = self._restrict_active_to_shadow(active, shadow)
        self.last_epistemic_depth = self.world_models.depth_plan()
        return active

    def expand_causal_model_class(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
    ) -> CausalExpansionDecision:
        """Autonomously open exactly the next structural generation.

        The caller supplies observable variables/intervention semantics, but does
        not choose the generator. The BODY uses authenticated class failure plus
        inherited structural ancestry to decide how much deeper to search.
        """
        if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
            return CausalExpansionDecision(
                "NO_EXPANSION_REQUIRED", self.latest_structural_generation(), "NONE", (), (),
                "current live model ecology retains at least one jointly compatible model",
            )

        current = self.latest_structural_generation()
        if current <= 0:
            generation = 1
            origin = "GENERATED"
            active = self.generate_replacement_causal_models(variables, descriptors)
        elif current == 1:
            generation = 2
            origin = "GENERATED_COMPOSITIONAL"
            active = self.generate_compositional_causal_models(variables, descriptors)
        elif current == 2:
            generation = 3
            origin = "GENERATED_PREDICATE"
            active = self.generate_predicate_causal_models(variables, descriptors)
        else:
            return CausalExpansionDecision(
                "MAX_GENERATION_REACHED", current, "NONE", (), (),
                "current bounded structural metalanguage has no generation beyond G3",
            )

        shadow = tuple(sorted(
            model.model_id for model in self.world_models.models.values()
            if int(model.generation) == generation and model.origin == origin
        ))
        active_ids = tuple(sorted(item.model.model_id for item in active))
        if generation == 3 and self.predicate_genesis.last_truncated:
            status = "FAIL_CLOSED_TRUNCATED_SHADOW_UNIVERSE"
            reason = "predicate-equivalence universe exceeded the bounded search budget"
        elif not shadow:
            status = "NO_STRUCTURAL_CANDIDATES"
            reason = "next structural generator produced no prediction-novel candidates"
        elif not active_ids:
            status = "NO_EVIDENCE_COMPATIBLE_CANDIDATES"
            reason = "shadow hypotheses persist but none satisfy current authoritative evidence"
        else:
            status = "EXPANDED"
            reason = "authenticated class failure opened the next ancestry-supported structural generation"
        return CausalExpansionDecision(status, generation, origin, active_ids, shadow, reason)

    def generated_model_queries(
        self,
        descriptors: Sequence[InterventionDescriptor],
        generated: Optional[Sequence[GeneratedCausalModel]] = None,
    ) -> List[QueryCandidate]:
        models = (
            [model for model in self.world_models.models.values() if model.origin == "GENERATED"]
            if generated is None else [item.model for item in generated]
        )
        return self.model_genesis.query_candidates(descriptors, models)

    def compositional_model_queries(
        self,
        descriptors: Sequence[InterventionDescriptor],
        generated: Optional[Sequence[GeneratedCausalProgram]] = None,
    ) -> List[QueryCandidate]:
        models = (
            [model for model in self.world_models.models.values() if model.origin == "GENERATED_COMPOSITIONAL"]
            if generated is None else [item.model for item in generated]
        )
        return self.model_genesis.query_candidates(descriptors, models)

    def predicate_model_queries(
        self,
        descriptors: Sequence[InterventionDescriptor],
        generated: Optional[Sequence[GeneratedPredicateModel]] = None,
    ) -> List[QueryCandidate]:
        models = (
            [model for model in self.world_models.models.values() if model.origin == "GENERATED_PREDICATE"]
            if generated is None else [item.model for item in generated]
        )
        return self.model_genesis.query_candidates(descriptors, models)

    def epistemic_depth_plan(self) -> EpistemicDepthPlan:
        self.last_epistemic_depth = self.world_models.depth_plan()
        return self.last_epistemic_depth

    def rank_epistemic_interventions(
        self,
        candidates: Sequence[QueryCandidate],
    ) -> Sequence[EpistemicInterventionScore]:
        return self.world_models.select_interventions(candidates)

    def cycle(
        self,
        task: TaskState,
        facts: Sequence[Fact] = (),
        residuals: Sequence[ResidualObservation] = (),
        measurements: Sequence[MeasurementObservation] = (),
        operator_spec: Optional[OperatorSpec] = None,
        possibility_budget: int = 32,
        experiment_reference_values: Optional[Mapping[str, float]] = None,
        world_context_id: Optional[str] = None,
    ) -> CognitiveCycle:
        depth = self.epistemic_depth_plan()
        effective_possibility_budget = int(possibility_budget)
        original_representation = self.representation
        if depth.mode != "COMPACT":
            effective_possibility_budget = max(effective_possibility_budget, depth.possibility_budget)
            if depth.representation_axis_budget > self.representation.axis_budget:
                self.representation = RepresentationGenesisEngine(
                    min_information_gain=self.representation.min_information_gain,
                    min_partition_support=self.representation.min_partition_support,
                    axis_budget=depth.representation_axis_budget,
                    enable_projection=self.representation.enable_projection,
                )
        try:
            return super().cycle(
                task=task,
                facts=facts,
                residuals=residuals,
                measurements=measurements,
                operator_spec=operator_spec,
                possibility_budget=effective_possibility_budget,
                experiment_reference_values=experiment_reference_values,
                world_context_id=world_context_id,
            )
        finally:
            self.representation = original_representation

    def execute_world_intervention(
        self,
        proposal,
        executor: WorldExecutor,
        verifier: Optional[WorldReceiptVerifier] = None,
    ) -> WorldOutcomePair:
        pair = super().execute_world_intervention(proposal, executor, verifier=verifier)
        self.world_models.observe_world_pair(pair)
        self.last_epistemic_depth = self.world_models.depth_plan()
        return pair


def epistemic_checkpoint_dict(runtime: EpistemicallyDeepPersistentCognitiveRuntime):
    payload = checkpoint_dict(runtime)
    payload["epistemic_depth_schema"] = EPISTEMIC_DEPTH_SCHEMA
    payload["world_model_ecology"] = {
        "models": [asdict(model) for _, model in sorted(runtime.world_models.models.items())],
    }
    return payload


def restore_epistemic_runtime(payload, world_verifier: Optional[WorldReceiptVerifier] = None):
    if payload.get("epistemic_depth_schema") not in (None, EPISTEMIC_DEPTH_SCHEMA):
        raise ValueError("unsupported epistemic depth schema")
    base = restore_runtime(payload, world_verifier=world_verifier)
    runtime = EpistemicallyDeepPersistentCognitiveRuntime(
        compiler=base.compiler,
        router=base.router,
        topology=base.topology,
        possibility=base.possibility,
        representation=base.representation,
        representation_value=base.representation_value,
        experiment=base.experiment,
        semantic=base.semantic,
        memory=base.memory,
        credit_engine=base.credit_engine,
        causal_law=base.causal_law,
        subgraph_finder=base.subgraph_finder,
        promotion_gate=base.promotion_gate,
        world_coupling=base.world_coupling,
        adaptive_projection_search=base.adaptive_projection_search,
    )
    models = []
    for item in payload.get("world_model_ecology", {}).get("models", []):
        models.append(CausalWorldModel(
            model_id=str(item["model_id"]),
            prior=float(item.get("prior", 1.0)),
            predictions=tuple((str(key), str(value)) for key, value in item.get("predictions", ())),
            origin=str(item.get("origin", "AUTHORED")),
            family=str(item.get("family", "UNSPECIFIED")),
            structure=tuple(str(value) for value in item.get("structure", ())),
            generation=int(item.get("generation", 0)),
            parent_model_ids=tuple(str(value) for value in item.get("parent_model_ids", ())),
            equivalent_structures=tuple(str(value) for value in item.get("equivalent_structures", ())),
        ))
    runtime.register_causal_world_models(models)
    for pair in runtime.world_coupling.pairs:
        runtime.world_models.observe_world_pair(pair)
    runtime.last_epistemic_depth = runtime.world_models.depth_plan()
    return runtime
