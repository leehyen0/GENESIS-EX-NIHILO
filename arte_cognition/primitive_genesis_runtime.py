from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

from .causal_linear_primitive_genesis import (
    GeneratedLinearPrimitiveModel,
    LinearFormPrimitiveGenesisEngine,
)
from .causal_model_genesis import InterventionDescriptor
from .causal_primitive_genesis import GeneratedPrimitiveModel, RawThresholdPrimitiveGenesisEngine
from .causal_symbolic_primitive_genesis import (
    GeneratedSymbolicPrimitiveModel,
    SymbolicPrimitiveGenesisEngine,
)
from .epistemic_depth_runtime import (
    CausalExpansionDecision,
    EpistemicallyDeepPersistentCognitiveRuntime,
    epistemic_checkpoint_dict,
    restore_epistemic_runtime,
)
from .raw_observation_authority import (
    RawObservationReceipt,
    RawObservationVerifier,
    corroborated_raw_observations,
)
from .world_coupling import WorldReceiptVerifier


PRIMITIVE_DEVELOPMENT_SCHEMA = "arte.primitive_development_same_body/v2"
LEGACY_PRIMITIVE_DEVELOPMENT_SCHEMAS = {"arte.primitive_development_same_body/v1"}


class WorldDrivenPrimitiveRuntime(EpistemicallyDeepPersistentCognitiveRuntime):
    """Same persistent BODY with falsification-driven primitive growth.

    G5-G7 raw observations are representation evidence, so their values cannot be
    accepted as transient evaluator arguments. The BODY stores signed raw receipts
    and derives its usable raw-observation memory only when two conditions hold:
    the exact raw payload is externally authenticated, and that receipt is bound to
    an already authoritative world outcome pair from the same execution identity.
    Independent authority classes must corroborate the exact raw channel map.

    Search-policy parameters and signed raw receipts are checkpointed. Cached raw
    values are never restored as authority: a descendant must reverify both world
    outcome receipts and raw observation receipts before primitive genesis resumes.
    """

    def __init__(
        self,
        *args,
        primitive_genesis: Optional[RawThresholdPrimitiveGenesisEngine] = None,
        linear_primitive_genesis: Optional[LinearFormPrimitiveGenesisEngine] = None,
        symbolic_primitive_genesis: Optional[SymbolicPrimitiveGenesisEngine] = None,
        raw_observation_receipts: Optional[Sequence[RawObservationReceipt]] = None,
        raw_observation_verifier: Optional[RawObservationVerifier] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.primitive_genesis = primitive_genesis or RawThresholdPrimitiveGenesisEngine()
        self.linear_primitive_genesis = linear_primitive_genesis or LinearFormPrimitiveGenesisEngine()
        self.symbolic_primitive_genesis = symbolic_primitive_genesis or SymbolicPrimitiveGenesisEngine()
        self.raw_observation_receipts: list[RawObservationReceipt] = []
        self.raw_observation_memory: Dict[str, Dict[str, float]] = {}
        if raw_observation_receipts:
            self.ingest_raw_observation_receipts(raw_observation_receipts, raw_observation_verifier)

    def _rebuild_raw_observation_memory(
        self,
        verifier: Optional[RawObservationVerifier],
    ) -> None:
        self.raw_observation_memory = corroborated_raw_observations(
            self.raw_observation_receipts,
            self.world_coupling.pairs,
            verifier,
            min_independent_classes=self.world_coupling.min_independent_classes,
        )

    def ingest_raw_observation_receipt(
        self,
        receipt: RawObservationReceipt,
        verifier: Optional[RawObservationVerifier],
    ) -> bool:
        if receipt.observation_id in {item.observation_id for item in self.raw_observation_receipts}:
            return False
        self.raw_observation_receipts.append(receipt)
        self._rebuild_raw_observation_memory(verifier)
        return receipt.intervention_id in self.raw_observation_memory

    def ingest_raw_observation_receipts(
        self,
        receipts: Sequence[RawObservationReceipt],
        verifier: Optional[RawObservationVerifier],
    ) -> None:
        seen = {item.observation_id for item in self.raw_observation_receipts}
        for receipt in receipts:
            if receipt.observation_id in seen:
                continue
            seen.add(receipt.observation_id)
            self.raw_observation_receipts.append(receipt)
        self._rebuild_raw_observation_memory(verifier)

    def reverify_raw_observation_authority(
        self,
        verifier: Optional[RawObservationVerifier],
    ) -> None:
        self._rebuild_raw_observation_memory(verifier)

    def raw_observations_for(
        self,
        descriptors: Sequence[InterventionDescriptor],
    ) -> Dict[str, Dict[str, float]]:
        return {
            descriptor.intervention_id: dict(self.raw_observation_memory.get(descriptor.intervention_id, {}))
            for descriptor in descriptors
            if descriptor.intervention_id in self.raw_observation_memory
        }

    @staticmethod
    def _reject_unverified_raw_argument(
        raw_observations: Optional[Mapping[str, Mapping[str, float]]],
    ) -> None:
        if raw_observations is not None:
            raise ValueError(
                "direct raw observations are non-authoritative; ingest signed RawObservationReceipt objects instead"
            )

    def generate_world_driven_primitive_models(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
        raw_observations: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> list[GeneratedPrimitiveModel]:
        self._reject_unverified_raw_argument(raw_observations)
        if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
            return []
        if not self.generation_falsified(4):
            return []
        owned_raw = self.raw_observations_for(descriptors)
        if len(owned_raw) != len(descriptors):
            return []
        existing = list(self.world_models.models.values())
        shadow = self.primitive_genesis.generate_novel(
            variables, descriptors, owned_raw, (), existing
        )
        shadow_truncated = bool(self.primitive_genesis.last_truncated)
        self.world_models.register(self._models(shadow))
        if shadow_truncated:
            self.last_epistemic_depth = self.world_models.depth_plan()
            return []
        active = self.primitive_genesis.generate_novel(
            variables,
            descriptors,
            owned_raw,
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
        raw_observations: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> list[GeneratedLinearPrimitiveModel]:
        self._reject_unverified_raw_argument(raw_observations)
        if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
            return []
        if not self.generation_falsified(5):
            return []
        owned_raw = self.raw_observations_for(descriptors)
        if len(owned_raw) != len(descriptors):
            return []
        existing = list(self.world_models.models.values())
        shadow = self.linear_primitive_genesis.generate_novel(
            variables, descriptors, owned_raw, (), existing
        )
        shadow_truncated = bool(self.linear_primitive_genesis.last_truncated)
        self.world_models.register(self._models(shadow))
        if shadow_truncated:
            self.last_epistemic_depth = self.world_models.depth_plan()
            return []
        active = self.linear_primitive_genesis.generate_novel(
            variables,
            descriptors,
            owned_raw,
            self.world_models.authoritative_evidence(),
            existing,
        )
        active = self._restrict_active_to_shadow(active, shadow)
        self.last_epistemic_depth = self.world_models.depth_plan()
        return active

    def generate_world_driven_symbolic_primitive_models(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
        raw_observations: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> list[GeneratedSymbolicPrimitiveModel]:
        self._reject_unverified_raw_argument(raw_observations)
        if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
            return []
        if not self.generation_falsified(6):
            return []
        owned_raw = self.raw_observations_for(descriptors)
        if len(owned_raw) != len(descriptors):
            return []
        existing = list(self.world_models.models.values())
        shadow = self.symbolic_primitive_genesis.generate_novel(
            variables, descriptors, owned_raw, (), existing
        )
        shadow_truncated = bool(self.symbolic_primitive_genesis.last_truncated)
        self.world_models.register(self._models(shadow))
        if shadow_truncated:
            self.last_epistemic_depth = self.world_models.depth_plan()
            return []
        active = self.symbolic_primitive_genesis.generate_novel(
            variables,
            descriptors,
            owned_raw,
            self.world_models.authoritative_evidence(),
            existing,
        )
        active = self._restrict_active_to_shadow(active, shadow)
        self.last_epistemic_depth = self.world_models.depth_plan()
        return active

    @staticmethod
    def _decision(
        generation: int,
        origin: str,
        active,
        world_models,
        truncated: bool,
        no_shadow_reason: str,
        no_active_reason: str,
        expanded_reason: str,
    ) -> CausalExpansionDecision:
        shadow = tuple(sorted(
            model.model_id
            for model in world_models.models.values()
            if int(model.generation) == generation and model.origin == origin
        ))
        active_ids = tuple(sorted(item.model.model_id for item in active))
        if truncated:
            status, reason = (
                "FAIL_CLOSED_TRUNCATED_SHADOW_UNIVERSE",
                "next primitive candidate universe exceeded bounded search budget",
            )
        elif not shadow:
            status, reason = "NO_STRUCTURAL_CANDIDATES", no_shadow_reason
        elif not active_ids:
            status, reason = "NO_EVIDENCE_COMPATIBLE_CANDIDATES", no_active_reason
        else:
            status, reason = "EXPANDED", expanded_reason
        return CausalExpansionDecision(status, generation, origin, active_ids, shadow, reason)

    def expand_causal_model_class_with_raw_observations(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
        raw_observations: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> CausalExpansionDecision:
        self._reject_unverified_raw_argument(raw_observations)
        current = self.latest_structural_generation()
        if current <= 3:
            return super().expand_causal_model_class(variables, descriptors)
        if self.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS":
            return CausalExpansionDecision(
                "NO_EXPANSION_REQUIRED", current, "NONE", (), (),
                "current live model ecology retains at least one jointly compatible model",
            )
        if len(self.raw_observations_for(descriptors)) != len(descriptors):
            return CausalExpansionDecision(
                "RAW_OBSERVATION_AUTHORITY_INCOMPLETE", current, "NONE", (), (),
                "primitive expansion requires independently corroborated raw observations bound to authoritative world pairs",
            )
        if current == 4:
            active = self.generate_world_driven_primitive_models(variables, descriptors)
            return self._decision(
                5,
                "GENERATED_PRIMITIVE_THRESHOLD",
                active,
                self.world_models,
                self.primitive_genesis.last_truncated,
                "no prediction-novel threshold primitive was generated from authenticated raw observations",
                "primitive shadow hypotheses persist but none satisfy current authoritative evidence",
                "externally falsified G4 plus authenticated raw observations opened primitive synthesis",
            )
        if current == 5:
            active = self.generate_world_driven_linear_primitive_models(variables, descriptors)
            return self._decision(
                6,
                "GENERATED_LINEAR_PRIMITIVE",
                active,
                self.world_models,
                self.linear_primitive_genesis.last_truncated,
                "no prediction-novel multi-channel linear primitive was generated",
                "linear primitive shadow hypotheses persist but none satisfy current authoritative evidence",
                "externally falsified single-channel class plus authenticated raw observations opened relation synthesis",
            )
        if current == 6:
            active = self.generate_world_driven_symbolic_primitive_models(variables, descriptors)
            return self._decision(
                7,
                "GENERATED_SYMBOLIC_PRIMITIVE",
                active,
                self.world_models,
                self.symbolic_primitive_genesis.last_truncated,
                "no prediction-novel symbolic expression primitive was generated",
                "symbolic shadow hypotheses persist but none satisfy current authoritative evidence",
                "externally falsified linear class plus authenticated raw observations opened symbolic search",
            )
        return CausalExpansionDecision(
            "MAX_GENERATION_REACHED",
            current,
            "NONE",
            (),
            (),
            "current bounded symbolic operation alphabet has no deeper validated expansion",
        )


def _raw_receipt_dict(receipt: RawObservationReceipt):
    return {
        "observation_id": receipt.observation_id,
        "intervention_id": receipt.intervention_id,
        "channel_values": [[name, float(value)] for name, value in receipt.normalized_values],
        "source_id": receipt.source_id,
        "context_id": receipt.context_id,
        "challenge_id": receipt.challenge_id,
        "epoch": int(receipt.epoch),
        "externally_generated": bool(receipt.externally_generated),
        "issuer_id": receipt.issuer_id,
        "signature": receipt.signature,
    }


def _restore_raw_receipt(item) -> RawObservationReceipt:
    return RawObservationReceipt(
        observation_id=str(item["observation_id"]),
        intervention_id=str(item["intervention_id"]),
        channel_values=tuple((str(name), float(value)) for name, value in item.get("channel_values", ())),
        source_id=str(item["source_id"]),
        context_id=str(item["context_id"]),
        challenge_id=str(item["challenge_id"]),
        epoch=int(item["epoch"]),
        externally_generated=bool(item.get("externally_generated", False)),
        issuer_id=str(item.get("issuer_id", "UNSIGNED")),
        signature=str(item.get("signature", "")),
    )


def primitive_checkpoint_dict(runtime: WorldDrivenPrimitiveRuntime):
    payload = epistemic_checkpoint_dict(runtime)
    payload["primitive_development_schema"] = PRIMITIVE_DEVELOPMENT_SCHEMA
    payload["raw_observation_receipts"] = [
        _raw_receipt_dict(receipt)
        for receipt in runtime.raw_observation_receipts
    ]
    # Audit-only cache. Restore never trusts this field; authority is re-derived
    # exclusively from signed raw receipts plus reverified world pairs.
    payload["raw_observation_memory_cache"] = {
        intervention_id: {
            channel: float(value)
            for channel, value in sorted(row.items())
        }
        for intervention_id, row in sorted(runtime.raw_observation_memory.items())
    }
    payload["primitive_genesis_policy"] = {
        "threshold": {
            "model_budget": runtime.primitive_genesis.model_budget,
            "min_distinct_values": runtime.primitive_genesis.min_distinct_values,
        },
        "linear": {
            "model_budget": runtime.linear_primitive_genesis.model_budget,
            "max_coefficient_abs": runtime.linear_primitive_genesis.max_coefficient_abs,
            "min_active_channels": runtime.linear_primitive_genesis.min_active_channels,
        },
        "symbolic": {
            "model_budget": runtime.symbolic_primitive_genesis.model_budget,
            "expression_budget": runtime.symbolic_primitive_genesis.expression_budget,
            "max_depth": runtime.symbolic_primitive_genesis.max_depth,
            "operators": list(runtime.symbolic_primitive_genesis.operators),
            "min_active_channels": runtime.symbolic_primitive_genesis.min_active_channels,
        },
    }
    return payload


def restore_world_driven_primitive_runtime(
    payload,
    world_verifier: Optional[WorldReceiptVerifier] = None,
    raw_observation_verifier: Optional[RawObservationVerifier] = None,
) -> WorldDrivenPrimitiveRuntime:
    schema = payload.get("primitive_development_schema")
    if schema not in (None, PRIMITIVE_DEVELOPMENT_SCHEMA) and schema not in LEGACY_PRIMITIVE_DEVELOPMENT_SCHEMAS:
        raise ValueError("unsupported primitive development schema")
    base = restore_epistemic_runtime(payload, world_verifier=world_verifier)
    policy = payload.get("primitive_genesis_policy", {})
    threshold_policy = policy.get("threshold", {})
    linear_policy = policy.get("linear", {})
    symbolic_policy = policy.get("symbolic", {})
    raw_receipts = (
        [_restore_raw_receipt(item) for item in payload.get("raw_observation_receipts", [])]
        if schema == PRIMITIVE_DEVELOPMENT_SCHEMA
        else []
    )
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
        primitive_genesis=RawThresholdPrimitiveGenesisEngine(
            model_budget=int(threshold_policy.get("model_budget", 4096)),
            min_distinct_values=int(threshold_policy.get("min_distinct_values", 3)),
        ),
        linear_primitive_genesis=LinearFormPrimitiveGenesisEngine(
            model_budget=int(linear_policy.get("model_budget", 8192)),
            max_coefficient_abs=int(linear_policy.get("max_coefficient_abs", 2)),
            min_active_channels=int(linear_policy.get("min_active_channels", 2)),
        ),
        symbolic_primitive_genesis=SymbolicPrimitiveGenesisEngine(
            model_budget=int(symbolic_policy.get("model_budget", 16384)),
            expression_budget=int(symbolic_policy.get("expression_budget", 2048)),
            max_depth=int(symbolic_policy.get("max_depth", 2)),
            operators=tuple(symbolic_policy.get("operators", ("ADD", "SUB", "MUL", "ABS"))),
            min_active_channels=int(symbolic_policy.get("min_active_channels", 2)),
        ),
        raw_observation_receipts=raw_receipts,
        raw_observation_verifier=raw_observation_verifier,
    )
