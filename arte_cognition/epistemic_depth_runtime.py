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
from .intervention_surface_genesis import InterventionSurfaceGenesisEngine
from .model_falsification import ModelFalsificationPolicy
from .possibility_space import Fact, OperatorSpec
from .representation_genesis import MeasurementObservation, RepresentationGenesisEngine
from .semantic_genesis import ResidualObservation
from .sparse_minterm_genesis import GeneratedSparseMintermModel, SparseMintermCausalGenesisEngine
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


@dataclass(frozen=True)
class SynthesizedInterventionDecision:
    status: str
    generation: int
    descriptor: Optional[InterventionDescriptor]
    expected_information_gain: float
    version_space_size: int
    generated_surface_size: int
    reason: str


@dataclass(frozen=True)
class FalsificationDecision:
    status: str
    generation: int
    descriptor: Optional[InterventionDescriptor]
    semantic_novelty: float
    structural_stress: float
    generated_surface_size: int
    reason: str


class EpistemicallyDeepPersistentCognitiveRuntime(PersistentCognitiveRuntime):
    """Same BODY with causal representation, intervention and falsification growth.

    Structural depth is selected by the BODY from authenticated class failure plus
    inherited ancestry: named families (G1), primitive composition (G2), bounded
    Boolean predicates (G3), then falsification-driven sparse exact-minterm gates
    (G4). G4 is not a new logical metalanguage: it expands Boolean complexity only
    after G3 has failed in the world.

    Concrete intervention semantics are synthesized from observable variables.
    Exact identification is not treated as truth; a separate falsification policy
    continues challenging a sole model in untested semantic regimes.
    """

    def __init__(
        self,
        *args,
        world_models: Optional[WorldModelEcology] = None,
        model_genesis: Optional[CausalModelGenesisEngine] = None,
        program_genesis: Optional[CompositionalCausalProgramGenesisEngine] = None,
        predicate_genesis: Optional[BooleanCausalPredicateGenesisEngine] = None,
        minterm_genesis: Optional[SparseMintermCausalGenesisEngine] = None,
        intervention_surface: Optional[InterventionSurfaceGenesisEngine] = None,
        falsification_policy: Optional[ModelFalsificationPolicy] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.world_models = world_models or WorldModelEcology()
        self.model_genesis = model_genesis or CausalModelGenesisEngine()
        self.program_genesis = program_genesis or CompositionalCausalProgramGenesisEngine()
        self.predicate_genesis = predicate_genesis or BooleanCausalPredicateGenesisEngine()
        self.minterm_genesis = minterm_genesis or SparseMintermCausalGenesisEngine()
        self.intervention_surface = intervention_surface or InterventionSurfaceGenesisEngine()
        self.falsification_policy = falsification_policy or ModelFalsificationPolicy()
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
        return [model for model in self.world_models.models.values() if int(model.generation) == int(generation)]

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

    def generation_falsified(self, generation: int) -> bool:
        snapshot = self.generation_version_space(generation)
        return bool(snapshot.model_ids) and not bool(snapshot.compatible_model_ids)

    def rank_generation_interventions(
        self,
        generation: int,
        candidates: Sequence[QueryCandidate],
    ) -> Sequence[EpistemicInterventionScore]:
        snapshot = self.generation_version_space(generation)
        return self.identifier.rank_interventions(candidates, snapshot.compatible_model_ids, cost_exponent=0.15)

    def select_generation_intervention(
        self,
        generation: int,
        candidates: Sequence[QueryCandidate],
    ) -> Optional[EpistemicInterventionScore]:
        snapshot = self.generation_version_space(generation)
        return self.identifier.select_next(candidates, snapshot.compatible_model_ids, cost_exponent=0.15)

    def synthesize_intervention_surface(
        self,
        variables: Sequence[str],
        observed_intervention_ids: Sequence[str] = (),
    ) -> List[InterventionDescriptor]:
        return self.intervention_surface.novel(variables, observed_intervention_ids)

    def _observed_generated_descriptors(self, variables: Sequence[str]) -> List[InterventionDescriptor]:
        full = self.intervention_surface.generate(variables)
        by_id = {row.intervention_id: row for row in full}
        observed_ids = {item.intervention_id for item in self.world_models.authoritative_evidence()}
        return [by_id[item] for item in sorted(observed_ids) if item in by_id]

    def select_synthesized_generation_intervention(
        self,
        generation: int,
        variables: Sequence[str],
        observed_intervention_ids: Sequence[str] = (),
    ) -> SynthesizedInterventionDecision:
        snapshot = self.generation_version_space(generation)
        if snapshot.identified:
            return SynthesizedInterventionDecision(
                "ALREADY_IDENTIFIED", generation, None, 0.0,
                len(snapshot.compatible_model_ids), 0,
                "exact version space already contains one model",
            )
        surface = self.synthesize_intervention_surface(variables, observed_intervention_ids)
        if self.intervention_surface.last_truncated:
            return SynthesizedInterventionDecision(
                "FAIL_CLOSED_TRUNCATED_INTERVENTION_SURFACE", generation, None, 0.0,
                len(snapshot.compatible_model_ids), len(surface),
                "generated intervention universe exceeded bounded surface budget",
            )
        queries = self.model_genesis.query_candidates(surface, self.structural_models(generation))
        selected = self.select_generation_intervention(generation, queries)
        if selected is None or selected.expected_information_gain <= 0.0:
            return SynthesizedInterventionDecision(
                "NO_DISCRIMINATING_INTERVENTION", generation, None, 0.0,
                len(snapshot.compatible_model_ids), len(surface),
                "generated interventions cannot split the current exact version space",
            )
        descriptor = next(row for row in surface if row.intervention_id == selected.intervention_id)
        return SynthesizedInterventionDecision(
            "SELECTED", generation, descriptor, selected.expected_information_gain,
            len(snapshot.compatible_model_ids), len(surface),
            "BODY-generated intervention maximizes generation-scoped information utility",
        )

    def select_falsification_intervention(
        self,
        generation: int,
        variables: Sequence[str],
        observed_intervention_ids: Sequence[str] = (),
    ) -> FalsificationDecision:
        snapshot = self.generation_version_space(generation)
        if not snapshot.identified:
            status = "GENERATION_ALREADY_FALSIFIED" if not snapshot.compatible_model_ids else "NOT_YET_IDENTIFIED"
            return FalsificationDecision(
                status, generation, None, 0.0, 0.0, 0,
                "falsification requires exactly one currently compatible model",
            )
        observed_ids = set(observed_intervention_ids)
        observed_ids.update(item.intervention_id for item in self.world_models.authoritative_evidence())
        surface = self.synthesize_intervention_surface(variables, tuple(sorted(observed_ids)))
        if self.intervention_surface.last_truncated:
            return FalsificationDecision(
                "FAIL_CLOSED_TRUNCATED_INTERVENTION_SURFACE", generation, None, 0.0, 0.0,
                len(surface), "generated intervention universe exceeded bounded surface budget",
            )
        if not surface:
            return FalsificationDecision(
                "BOUNDED_SURFACE_EXHAUSTED", generation, None, 0.0, 0.0, 0,
                "all generated intervention semantics have been externally tested",
            )
        observed = self._observed_generated_descriptors(variables)
        selected = self.falsification_policy.select(surface, observed=observed, cost_exponent=0.10)
        if selected is None:
            return FalsificationDecision(
                "NO_FALSIFICATION_CANDIDATE", generation, None, 0.0, 0.0,
                len(surface), "no unobserved generated intervention remains",
            )
        descriptor = next(row for row in surface if row.intervention_id == selected.intervention_id)
        return FalsificationDecision(
            "SELECTED", generation, descriptor, selected.semantic_novelty,
            selected.structural_stress, len(surface),
            "identified model is challenged in an untested high-coverage semantic regime",
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
        active = self.model_genesis.generate(variables, descriptors, self.world_models.authoritative_evidence())
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
        if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
            return []
        if not any(model.origin == "GENERATED_COMPOSITIONAL" for model in self.world_models.models.values()):
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

    def generate_sparse_minterm_causal_models(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
    ) -> List[GeneratedSparseMintermModel]:
        """Open G4 only after externally grounded failure of the G3 model class."""
        if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
            return []
        if not self.generation_falsified(3):
            return []
        existing = list(self.world_models.models.values())
        shadow = self.minterm_genesis.generate_novel(variables, descriptors, (), existing)
        shadow_truncated = bool(self.minterm_genesis.last_truncated)
        self.world_models.register(self._models(shadow))
        if shadow_truncated:
            self.last_epistemic_depth = self.world_models.depth_plan()
            return []
        active = self.minterm_genesis.generate_novel(
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
        if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
            return CausalExpansionDecision(
                "NO_EXPANSION_REQUIRED", self.latest_structural_generation(), "NONE", (), (),
                "current live model ecology retains at least one jointly compatible model",
            )
        current = self.latest_structural_generation()
        if current <= 0:
            generation, origin = 1, "GENERATED"
            active = self.generate_replacement_causal_models(variables, descriptors)
        elif current == 1:
            generation, origin = 2, "GENERATED_COMPOSITIONAL"
            active = self.generate_compositional_causal_models(variables, descriptors)
        elif current == 2:
            generation, origin = 3, "GENERATED_PREDICATE"
            active = self.generate_predicate_causal_models(variables, descriptors)
        elif current == 3:
            generation, origin = 4, "GENERATED_SPARSE_MINTERM"
            active = self.generate_sparse_minterm_causal_models(variables, descriptors)
        else:
            return CausalExpansionDecision(
                "MAX_GENERATION_REACHED", current, "NONE", (), (),
                "current bounded structural metalanguage has no generation beyond G4",
            )
        shadow = tuple(sorted(
            model.model_id for model in self.world_models.models.values()
            if int(model.generation) == generation and model.origin == origin
        ))
        active_ids = tuple(sorted(item.model.model_id for item in active))
        truncated = (
            generation == 3 and self.predicate_genesis.last_truncated
        ) or (
            generation == 4 and self.minterm_genesis.last_truncated
        )
        if truncated:
            status, reason = (
                "FAIL_CLOSED_TRUNCATED_SHADOW_UNIVERSE",
                "next structural candidate universe exceeded the bounded search budget",
            )
        elif not shadow:
            status, reason = "NO_STRUCTURAL_CANDIDATES", "next structural generator produced no prediction-novel candidates"
        elif not active_ids:
            status, reason = "NO_EVIDENCE_COMPATIBLE_CANDIDATES", "shadow hypotheses persist but none satisfy current authoritative evidence"
        else:
            status, reason = "EXPANDED", "authenticated class failure opened the next ancestry-supported structural generation"
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
