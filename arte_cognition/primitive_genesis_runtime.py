from __future__ import annotations

from typing import Mapping, Optional, Sequence

from .causal_linear_primitive_genesis import (
    GeneratedLinearPrimitiveModel,
    LinearFormPrimitiveGenesisEngine,
)
from .causal_model_genesis import InterventionDescriptor
from .causal_primitive_genesis import GeneratedPrimitiveModel, RawThresholdPrimitiveGenesisEngine
from .epistemic_depth_runtime import (
    CausalExpansionDecision,
    EpistemicallyDeepPersistentCognitiveRuntime,
    restore_epistemic_runtime,
)
from .world_coupling import WorldReceiptVerifier


class WorldDrivenPrimitiveRuntime(EpistemicallyDeepPersistentCognitiveRuntime):
    """Same persistent BODY with falsification-driven primitive growth.

    The parent runtime remains the state owner. This extension does not create a
    parallel memory, authority system or world-model store. After complete G4
    failure it can synthesize single-channel threshold primitives (G5). If that
    complete primitive class is itself externally falsified, the same BODY can
    synthesize bounded multi-channel integer linear forms and threshold them (G6).
    """

    def __init__(
        self,
        *args,
        primitive_genesis: Optional[RawThresholdPrimitiveGenesisEngine] = None,
        linear_primitive_genesis: Optional[LinearFormPrimitiveGenesisEngine] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.primitive_genesis = primitive_genesis or RawThresholdPrimitiveGenesisEngine()
        self.linear_primitive_genesis = linear_primitive_genesis or LinearFormPrimitiveGenesisEngine()

    def generate_world_driven_primitive_models(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
        raw_observations: Mapping[str, Mapping[str, float]],
    ) -> list[GeneratedPrimitiveModel]:
        if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
            return []
        if not self.generation_falsified(4):
            return []
        existing = list(self.world_models.models.values())
        shadow = self.primitive_genesis.generate_novel(
            variables,
            descriptors,
            raw_observations,
            (),
            existing,
        )
        shadow_truncated = bool(self.primitive_genesis.last_truncated)
        self.world_models.register(self._models(shadow))
        if shadow_truncated:
            self.last_epistemic_depth = self.world_models.depth_plan()
            return []
        active = self.primitive_genesis.generate_novel(
            variables,
            descriptors,
            raw_observations,
            self.world_models.authoritative_evidence(),
            existing,
        )
        active = self._restrict_active_to_shadow(active, shadow)
        self.last_epistemic_depth = self.world_models.depth_plan()
        return active

    def generate_world_driven_linear_primitive_models(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
        raw_observations: Mapping[str, Mapping[str, float]],
    ) -> list[GeneratedLinearPrimitiveModel]:
        if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
            return []
        if not self.generation_falsified(5):
            return []
        existing = list(self.world_models.models.values())
        shadow = self.linear_primitive_genesis.generate_novel(
            variables,
            descriptors,
            raw_observations,
            (),
            existing,
        )
        shadow_truncated = bool(self.linear_primitive_genesis.last_truncated)
        self.world_models.register(self._models(shadow))
        if shadow_truncated:
            self.last_epistemic_depth = self.world_models.depth_plan()
            return []
        active = self.linear_primitive_genesis.generate_novel(
            variables,
            descriptors,
            raw_observations,
            self.world_models.authoritative_evidence(),
            existing,
        )
        active = self._restrict_active_to_shadow(active, shadow)
        self.last_epistemic_depth = self.world_models.depth_plan()
        return active

    def expand_causal_model_class_with_raw_observations(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
        raw_observations: Mapping[str, Mapping[str, float]],
    ) -> CausalExpansionDecision:
        current = self.latest_structural_generation()
        if current <= 3:
            return super().expand_causal_model_class(variables, descriptors)
        if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
            return CausalExpansionDecision(
                "NO_EXPANSION_REQUIRED",
                current,
                "NONE",
                (),
                (),
                "current live model ecology retains at least one jointly compatible model",
            )
        if current == 4:
            generation = 5
            origin = "GENERATED_PRIMITIVE_THRESHOLD"
            active = self.generate_world_driven_primitive_models(
                variables,
                descriptors,
                raw_observations,
            )
            shadow = tuple(sorted(
                model.model_id
                for model in self.world_models.models.values()
                if int(model.generation) == generation and model.origin == origin
            ))
            active_ids = tuple(sorted(item.model.model_id for item in active))
            if self.primitive_genesis.last_truncated:
                status, reason = (
                    "FAIL_CLOSED_TRUNCATED_SHADOW_UNIVERSE",
                    "raw-observation primitive candidate universe exceeded bounded search budget",
                )
            elif not shadow:
                status, reason = (
                    "NO_STRUCTURAL_CANDIDATES",
                    "no prediction-novel threshold primitive was generated from raw observations",
                )
            elif not active_ids:
                status, reason = (
                    "NO_EVIDENCE_COMPATIBLE_CANDIDATES",
                    "primitive shadow hypotheses persist but none satisfy current authoritative evidence",
                )
            else:
                status, reason = (
                    "EXPANDED",
                    "externally falsified G4 opened raw-observation primitive synthesis",
                )
            return CausalExpansionDecision(
                status,
                generation,
                origin,
                active_ids,
                shadow,
                reason,
            )
        if current == 5:
            generation = 6
            origin = "GENERATED_LINEAR_PRIMITIVE"
            active = self.generate_world_driven_linear_primitive_models(
                variables,
                descriptors,
                raw_observations,
            )
            shadow = tuple(sorted(
                model.model_id
                for model in self.world_models.models.values()
                if int(model.generation) == generation and model.origin == origin
            ))
            active_ids = tuple(sorted(item.model.model_id for item in active))
            if self.linear_primitive_genesis.last_truncated:
                status, reason = (
                    "FAIL_CLOSED_TRUNCATED_SHADOW_UNIVERSE",
                    "linear primitive candidate universe exceeded bounded search budget",
                )
            elif not shadow:
                status, reason = (
                    "NO_STRUCTURAL_CANDIDATES",
                    "no prediction-novel multi-channel linear primitive was generated",
                )
            elif not active_ids:
                status, reason = (
                    "NO_EVIDENCE_COMPATIBLE_CANDIDATES",
                    "linear primitive shadow hypotheses persist but none satisfy current authoritative evidence",
                )
            else:
                status, reason = (
                    "EXPANDED",
                    "externally falsified single-channel primitive class opened multi-channel relation synthesis",
                )
            return CausalExpansionDecision(
                status,
                generation,
                origin,
                active_ids,
                shadow,
                reason,
            )
        return CausalExpansionDecision(
            "MAX_GENERATION_REACHED",
            current,
            "NONE",
            (),
            (),
            "current bounded primitive metalanguage has no generation beyond linear-form G6",
        )


def restore_world_driven_primitive_runtime(
    payload,
    world_verifier: Optional[WorldReceiptVerifier] = None,
) -> WorldDrivenPrimitiveRuntime:
    base = restore_epistemic_runtime(payload, world_verifier=world_verifier)
    return WorldDrivenPrimitiveRuntime(
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
        world_models=base.world_models,
    )
