from __future__ import annotations

from dataclasses import asdict
from typing import List, Mapping, Optional, Sequence

from .adaptive_cognition import QueryCandidate, TaskState
from .body_checkpoint import checkpoint_dict, restore_runtime
from .causal_model_genesis import CausalModelGenesisEngine, GeneratedCausalModel, InterventionDescriptor
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


class EpistemicallyDeepPersistentCognitiveRuntime(PersistentCognitiveRuntime):
    """Same BODY with depth control and staged causal structure generation.

    Evidence-conditioned generation returns compatible candidates for immediate
    reasoning, but the BODY also preserves an evidence-unfiltered bounded shadow
    candidate pool. Otherwise the *absence* of rejected candidates in a checkpoint
    would leak external authority into a verifierless descendant.
    """

    def __init__(
        self,
        *args,
        world_models: Optional[WorldModelEcology] = None,
        model_genesis: Optional[CausalModelGenesisEngine] = None,
        program_genesis: Optional[CompositionalCausalProgramGenesisEngine] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.world_models = world_models or WorldModelEcology()
        self.model_genesis = model_genesis or CausalModelGenesisEngine()
        self.program_genesis = program_genesis or CompositionalCausalProgramGenesisEngine()
        self.last_epistemic_depth = self.world_models.depth_plan()

    def register_causal_world_models(self, models: Sequence[CausalWorldModel]) -> None:
        self.world_models.register(models)
        self.last_epistemic_depth = self.world_models.depth_plan()

    @staticmethod
    def _model_union(*groups) -> List[CausalWorldModel]:
        by_id = {}
        for group in groups:
            for item in group:
                model = item.model if hasattr(item, "model") else item
                by_id[model.model_id] = model
        return list(by_id.values())

    def generate_replacement_causal_models(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
    ) -> List[GeneratedCausalModel]:
        """Generate first-generation active candidates plus bounded shadow alternatives."""
        if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
            return []
        evidence = self.world_models.authoritative_evidence()
        active = self.model_genesis.generate(
            variables=variables,
            descriptors=descriptors,
            residual_evidence=evidence,
        )
        shadow = self.model_genesis.generate(
            variables=variables,
            descriptors=descriptors,
            residual_evidence=(),
        )
        self.world_models.register(self._model_union(shadow, active))
        self.last_epistemic_depth = self.world_models.depth_plan()
        return active

    def generate_compositional_causal_models(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
    ) -> List[GeneratedCausalProgram]:
        """Open second-generation composition after first-generation ancestry fails.

        Active candidates are evidence-compatible, while a bounded unfiltered
        compositional shadow pool is persisted too. Thus a restart without an
        external verifier retains hypotheses but cannot infer which ones survived
        earlier external evidence merely from candidate-set membership.
        """
        if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
            return []
        if not any(model.origin == "GENERATED" for model in self.world_models.models.values()):
            return []
        existing = list(self.world_models.models.values())
        active = self.program_genesis.generate_novel(
            variables=variables,
            descriptors=descriptors,
            residual_evidence=self.world_models.authoritative_evidence(),
            existing_models=existing,
        )
        shadow = self.program_genesis.generate_novel(
            variables=variables,
            descriptors=descriptors,
            residual_evidence=(),
            existing_models=existing,
        )
        self.world_models.register(self._model_union(shadow, active))
        self.last_epistemic_depth = self.world_models.depth_plan()
        return active

    def generated_model_queries(
        self,
        descriptors: Sequence[InterventionDescriptor],
        generated: Optional[Sequence[GeneratedCausalModel]] = None,
    ) -> List[QueryCandidate]:
        if generated is None:
            models = [
                model for model in self.world_models.models.values()
                if model.origin == "GENERATED"
            ]
        else:
            models = [item.model for item in generated]
        return self.model_genesis.query_candidates(descriptors, models)

    def compositional_model_queries(
        self,
        descriptors: Sequence[InterventionDescriptor],
        generated: Optional[Sequence[GeneratedCausalProgram]] = None,
    ) -> List[QueryCandidate]:
        if generated is None:
            models = [
                model for model in self.world_models.models.values()
                if model.origin == "GENERATED_COMPOSITIONAL"
            ]
        else:
            models = [item.model for item in generated]
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
    # Model definitions, including shadow alternatives and ancestry, persist. Their
    # posterior authority does not: signed receipts must be externally reverified.
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
            predictions=tuple(
                (str(key), str(value))
                for key, value in item.get("predictions", ())
            ),
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
