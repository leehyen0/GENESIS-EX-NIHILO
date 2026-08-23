from __future__ import annotations

from typing import Dict, Optional, Sequence

from .causal_model_genesis import InterventionDescriptor
from .composition_law_genesis import (
    CompositionLawGenesisEngine,
    GeneratedCompositionLaw,
    GeneratedCompositionLawModel,
)
from .primitive_genesis_runtime import (
    WorldDrivenPrimitiveRuntime,
    primitive_checkpoint_dict,
    restore_world_driven_primitive_runtime,
)
from .raw_observation_authority import RawObservationVerifier
from .world_coupling import WorldReceiptVerifier


REPRESENTATION_ALGEBRA_SCHEMA = "arte.representation_algebra_same_body/v1"


class WorldDrivenRepresentationAlgebraRuntime(WorldDrivenPrimitiveRuntime):
    """Persistent BODY whose next structural organ is a generated operation law.

    The inherited G7 BODY owns a fixed symbolic operator alphabet. This runtime
    opens a deeper structural generation only after that exact generation has been
    externally falsified. Generated finite composition laws are stored as lineage,
    not as authority. World and raw-observation authority are still rebuilt from
    external receipts after checkpoint restore.
    """

    def __init__(
        self,
        *args,
        composition_law_genesis: Optional[CompositionLawGenesisEngine] = None,
        composition_law_lineage: Optional[Sequence[GeneratedCompositionLawModel]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.composition_law_genesis = composition_law_genesis or CompositionLawGenesisEngine()
        self.composition_law_lineage: Dict[str, GeneratedCompositionLawModel] = {}
        for item in composition_law_lineage or ():
            self.composition_law_lineage[item.model.model_id] = item

    def _remember_composition_laws(self, items: Sequence[GeneratedCompositionLawModel]) -> None:
        for item in items:
            self.composition_law_lineage[item.model.model_id] = item

    def generate_world_driven_composition_law_models(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
    ) -> list[GeneratedCompositionLawModel]:
        if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
            return []
        if not self.generation_falsified(7):
            return []
        owned_raw = self.raw_observations_for(descriptors)
        if len(owned_raw) != len(descriptors):
            return []
        existing = list(self.world_models.models.values())
        shadow = self.composition_law_genesis.generate_novel(
            variables, descriptors, owned_raw, (), existing
        )
        shadow_truncated = bool(self.composition_law_genesis.last_truncated)
        self._remember_composition_laws(shadow)
        self.world_models.register(self._models(shadow))
        if shadow_truncated:
            self.last_epistemic_depth = self.world_models.depth_plan()
            return []
        active = self.composition_law_genesis.generate_novel(
            variables,
            descriptors,
            owned_raw,
            self.world_models.authoritative_evidence(),
            existing,
        )
        active = self._restrict_active_to_shadow(active, shadow)
        self._remember_composition_laws(active)
        self.last_epistemic_depth = self.world_models.depth_plan()
        return active

    def authorized_composition_law_model_ids(self) -> tuple[str, ...]:
        # Candidate persistence must not turn into authority by itself. With no
        # externally reverified world evidence after restart this set is empty.
        if not self.world_models.authoritative_evidence():
            return ()
        snapshot = self.generation_version_space(8)
        known = set(self.composition_law_lineage)
        return tuple(sorted(model_id for model_id in snapshot.compatible_model_ids if model_id in known))

    def expand_causal_model_class_with_raw_observations(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
        raw_observations=None,
    ):
        self._reject_unverified_raw_argument(raw_observations)
        current = self.latest_structural_generation()
        if current <= 6:
            return super().expand_causal_model_class_with_raw_observations(variables, descriptors)
        if current == 7:
            if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
                return self._decision(
                    8, "GENERATED_COMPOSITION_LAW", [], self.world_models, False,
                    "fixed symbolic generation is not yet externally falsified",
                    "no externally authorized composition law is required",
                    "no expansion required",
                )
            if len(self.raw_observations_for(descriptors)) != len(descriptors):
                from .epistemic_depth_runtime import CausalExpansionDecision
                return CausalExpansionDecision(
                    "RAW_OBSERVATION_AUTHORITY_INCOMPLETE",
                    current,
                    "NONE",
                    (),
                    (),
                    "composition-law genesis requires independently corroborated raw observations bound to authoritative world pairs",
                )
            active = self.generate_world_driven_composition_law_models(variables, descriptors)
            return self._decision(
                8,
                "GENERATED_COMPOSITION_LAW",
                active,
                self.world_models,
                self.composition_law_genesis.last_truncated,
                "no prediction-novel finite operation law was generated from authenticated raw observations",
                "composition-law shadow hypotheses persist but none satisfy current authoritative evidence",
                "externally falsified fixed symbolic alphabet opened outcome-free finite operation-law genesis",
            )
        from .epistemic_depth_runtime import CausalExpansionDecision
        return CausalExpansionDecision(
            "MAX_GENERATION_REACHED",
            current,
            "NONE",
            (),
            (),
            "current finite operation-table metalanguage has no deeper validated self-generated interpreter",
        )


def _lineage_dict(item: GeneratedCompositionLawModel):
    return {
        "model_id": item.model.model_id,
        "cause": item.cause,
        "sign": item.sign,
        "left_channel": item.left_channel,
        "right_channel": item.right_channel,
        "law": {
            "state_count": item.law.state_count,
            "identity_state": item.law.identity_state,
            "table": list(item.law.table),
            "active_states": list(item.law.active_states),
        },
        "equivalent_laws": list(item.equivalent_laws),
    }


def representation_algebra_checkpoint_dict(runtime: WorldDrivenRepresentationAlgebraRuntime):
    payload = primitive_checkpoint_dict(runtime)
    payload["representation_algebra_schema"] = REPRESENTATION_ALGEBRA_SCHEMA
    payload["composition_law_policy"] = {
        "model_budget": runtime.composition_law_genesis.model_budget,
        "state_count": runtime.composition_law_genesis.state_count,
        "identity_state": runtime.composition_law_genesis.identity_state,
        "tolerance": runtime.composition_law_genesis.tolerance,
        "min_active_channels": runtime.composition_law_genesis.min_active_channels,
    }
    payload["composition_law_lineage"] = [
        _lineage_dict(item)
        for _, item in sorted(runtime.composition_law_lineage.items())
    ]
    return payload


def restore_world_driven_representation_algebra_runtime(
    payload,
    world_verifier: Optional[WorldReceiptVerifier] = None,
    raw_observation_verifier: Optional[RawObservationVerifier] = None,
) -> WorldDrivenRepresentationAlgebraRuntime:
    schema = payload.get("representation_algebra_schema")
    if schema not in (None, REPRESENTATION_ALGEBRA_SCHEMA):
        raise ValueError("unsupported representation algebra schema")
    base = restore_world_driven_primitive_runtime(
        payload,
        world_verifier=world_verifier,
        raw_observation_verifier=raw_observation_verifier,
    )
    policy = payload.get("composition_law_policy", {})
    engine = CompositionLawGenesisEngine(
        model_budget=int(policy.get("model_budget", 4096)),
        state_count=int(policy.get("state_count", 3)),
        identity_state=int(policy.get("identity_state", 1)),
        tolerance=float(policy.get("tolerance", 1e-9)),
        min_active_channels=int(policy.get("min_active_channels", 2)),
    )

    lineage = []
    for raw in payload.get("composition_law_lineage", []):
        model_id = str(raw.get("model_id", ""))
        model = base.world_models.models.get(model_id)
        if model is None:
            raise ValueError("composition-law lineage references missing world-model state")
        law_raw = raw.get("law", {})
        law = GeneratedCompositionLaw(
            state_count=int(law_raw.get("state_count", 3)),
            identity_state=int(law_raw.get("identity_state", 1)),
            table=tuple(int(value) for value in law_raw.get("table", ())),
            active_states=tuple(int(value) for value in law_raw.get("active_states", ())),
        )
        if len(law.table) != law.state_count * law.state_count:
            raise ValueError("invalid checkpointed composition-law table")
        lineage.append(GeneratedCompositionLawModel(
            cause=str(raw.get("cause", "")),
            sign=str(raw.get("sign", "")),
            left_channel=str(raw.get("left_channel", "")),
            right_channel=str(raw.get("right_channel", "")),
            law=law,
            model=model,
            equivalent_laws=tuple(str(value) for value in raw.get("equivalent_laws", ())),
        ))

    return WorldDrivenRepresentationAlgebraRuntime(
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
        model_genesis=base.model_genesis,
        program_genesis=base.program_genesis,
        predicate_genesis=base.predicate_genesis,
        minterm_genesis=base.minterm_genesis,
        intervention_surface=base.intervention_surface,
        falsification_policy=base.falsification_policy,
        primitive_genesis=base.primitive_genesis,
        linear_primitive_genesis=base.linear_primitive_genesis,
        symbolic_primitive_genesis=base.symbolic_primitive_genesis,
        raw_observation_receipts=base.raw_observation_receipts,
        raw_observation_verifier=raw_observation_verifier,
        composition_law_genesis=engine,
        composition_law_lineage=lineage,
    )
