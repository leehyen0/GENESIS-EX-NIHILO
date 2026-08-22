from __future__ import annotations

from dataclasses import asdict
from typing import List, Mapping, Optional, Sequence

from .adaptive_cognition import QueryCandidate, TaskState
from .body_checkpoint import checkpoint_dict, restore_runtime
from .causal_model_genesis import CausalModelGenesisEngine, GeneratedCausalModel, InterventionDescriptor
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
    """Same PersistentCognitiveRuntime with causal-model depth and model genesis.

    This is not a second BODY. The runtime inherits router/topology/memory/world
    state directly, expands resources under model-class failure, and may generate
    replacement causal structures only when externally grounded evidence leaves no
    jointly compatible live model. Generated models are hypotheses, never evidence.
    """

    def __init__(
        self,
        *args,
        world_models: Optional[WorldModelEcology] = None,
        model_genesis: Optional[CausalModelGenesisEngine] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.world_models = world_models or WorldModelEcology()
        self.model_genesis = model_genesis or CausalModelGenesisEngine()
        self.last_epistemic_depth = self.world_models.depth_plan()

    def register_causal_world_models(self, models: Sequence[CausalWorldModel]) -> None:
        self.world_models.register(models)
        self.last_epistemic_depth = self.world_models.depth_plan()

    def generate_replacement_causal_models(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
    ) -> List[GeneratedCausalModel]:
        """Generate a new causal-model class only after authenticated class failure."""
        if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
            return []
        generated = self.model_genesis.generate(
            variables=variables,
            descriptors=descriptors,
            residual_evidence=self.world_models.authoritative_evidence(),
        )
        self.world_models.register([item.model for item in generated])
        self.last_epistemic_depth = self.world_models.depth_plan()
        return generated

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
    # Posterior, inadequacy and authority are never trusted serialized state. Model
    # definitions (including generated structure lineage) are persistent phenotype;
    # evidence authority is reconstructed from signed receipts after restore.
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
