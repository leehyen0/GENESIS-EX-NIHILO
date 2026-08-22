from __future__ import annotations

import unittest

from arte_cognition.canonical_body_checkpoint import (
    KIND_BASE,
    KIND_EPISTEMIC,
    KIND_PRIMITIVE,
    checkpoint_dict,
    integrity_sha256,
    restore_runtime,
)
from arte_cognition.causal_linear_primitive_genesis import LinearFormPrimitiveGenesisEngine
from arte_cognition.causal_primitive_genesis import RawThresholdPrimitiveGenesisEngine
from arte_cognition.causal_symbolic_primitive_genesis import SymbolicPrimitiveGenesisEngine
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.epistemic_depth_runtime import EpistemicallyDeepPersistentCognitiveRuntime
from arte_cognition.primitive_genesis_runtime import WorldDrivenPrimitiveRuntime
from arte_cognition.raw_observation_authority import RawObservationReceipt
from arte_cognition.world_model_ecology import CausalWorldModel


class CanonicalBodyCheckpointTests(unittest.TestCase):
    def test_base_runtime_roundtrip_uses_base_kind(self):
        runtime = PersistentCognitiveRuntime()
        payload = checkpoint_dict(runtime)
        self.assertEqual(payload["canonical_body"]["runtime_kind"], KIND_BASE)
        restored = restore_runtime(payload)
        self.assertIs(type(restored), PersistentCognitiveRuntime)

    def test_epistemic_runtime_roundtrip_preserves_world_model_ecology(self):
        runtime = EpistemicallyDeepPersistentCognitiveRuntime()
        runtime.register_causal_world_models([
            CausalWorldModel(
                model_id="epistemic-model",
                prior=1.0,
                predictions=(("probe", "POSITIVE_EFFECT"),),
                origin="GENERATED",
                family="DIRECT",
                structure=("x->y",),
                generation=1,
            )
        ])
        payload = checkpoint_dict(runtime)
        self.assertEqual(payload["canonical_body"]["runtime_kind"], KIND_EPISTEMIC)
        restored = restore_runtime(payload)
        self.assertIsInstance(restored, EpistemicallyDeepPersistentCognitiveRuntime)
        self.assertIn("epistemic-model", restored.world_models.models)

    def test_primitive_runtime_cannot_silently_downcast(self):
        runtime = WorldDrivenPrimitiveRuntime(
            primitive_genesis=RawThresholdPrimitiveGenesisEngine(
                model_budget=321,
                min_distinct_values=4,
            ),
            linear_primitive_genesis=LinearFormPrimitiveGenesisEngine(
                model_budget=654,
                max_coefficient_abs=3,
                min_active_channels=2,
            ),
            symbolic_primitive_genesis=SymbolicPrimitiveGenesisEngine(
                model_budget=987,
                expression_budget=111,
                max_depth=1,
                operators=("MUL", "ABS"),
                min_active_channels=2,
            ),
        )
        receipt = RawObservationReceipt(
            observation_id="raw-a",
            intervention_id="probe-a",
            channel_values=(("opaque-1", 1.25), ("opaque-2", -3.5)),
            source_id="source-a",
            context_id="context-a",
            challenge_id="challenge-a",
            epoch=1,
            externally_generated=True,
            issuer_id="issuer-a",
            signature="audit-signature",
        )
        runtime.ingest_raw_observation_receipts([receipt], verifier=None)

        payload = checkpoint_dict(runtime)
        self.assertEqual(payload["canonical_body"]["runtime_kind"], KIND_PRIMITIVE)
        self.assertEqual(len(payload["raw_observation_receipts"]), 1)
        restored = restore_runtime(payload)
        self.assertIsInstance(restored, WorldDrivenPrimitiveRuntime)
        self.assertEqual(len(restored.raw_observation_receipts), 1)
        # Serialized signatures survive for audit/reverification, but without a
        # verifier the raw values regain no learning authority.
        self.assertEqual(restored.raw_observation_memory, {})
        self.assertEqual(restored.primitive_genesis.model_budget, 321)
        self.assertEqual(restored.primitive_genesis.min_distinct_values, 4)
        self.assertEqual(restored.linear_primitive_genesis.model_budget, 654)
        self.assertEqual(restored.linear_primitive_genesis.max_coefficient_abs, 3)
        self.assertEqual(restored.symbolic_primitive_genesis.model_budget, 987)
        self.assertEqual(restored.symbolic_primitive_genesis.expression_budget, 111)
        self.assertEqual(restored.symbolic_primitive_genesis.max_depth, 1)
        self.assertEqual(restored.symbolic_primitive_genesis.operators, ("MUL", "ABS"))

    def test_integrity_hash_detects_accidental_state_truncation(self):
        payload = checkpoint_dict(WorldDrivenPrimitiveRuntime())
        payload.pop("primitive_genesis_policy")
        with self.assertRaisesRegex(ValueError, "required state namespaces|integrity"):
            restore_runtime(payload)

    def test_rehashed_downcast_still_fails_runtime_kind_contract(self):
        payload = checkpoint_dict(WorldDrivenPrimitiveRuntime())
        payload.pop("primitive_development_schema")
        payload.pop("raw_observation_receipts")
        payload.pop("primitive_genesis_policy")
        payload.pop("raw_observation_memory_cache", None)
        # Even if an accidental migration recomputes the unkeyed hash, the
        # declared BODY kind cannot be reconciled with the shallower schema.
        payload["canonical_body"]["integrity_sha256"] = integrity_sha256(payload)
        with self.assertRaisesRegex(ValueError, "downcast/schema mismatch"):
            restore_runtime(payload)

    def test_required_namespace_contract_cannot_be_weakened_by_rehash(self):
        payload = checkpoint_dict(WorldDrivenPrimitiveRuntime())
        payload["canonical_body"]["required_namespaces"] = ["policy"]
        payload["canonical_body"]["integrity_sha256"] = integrity_sha256(payload)
        with self.assertRaisesRegex(ValueError, "required-namespace contract mismatch"):
            restore_runtime(payload)


if __name__ == "__main__":
    unittest.main()
