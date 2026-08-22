from __future__ import annotations

import unittest

from arte_cognition.causal_linear_primitive_genesis import LinearFormPrimitiveGenesisEngine
from arte_cognition.causal_primitive_genesis import RawThresholdPrimitiveGenesisEngine
from arte_cognition.causal_symbolic_primitive_genesis import SymbolicPrimitiveGenesisEngine
from arte_cognition.primitive_genesis_runtime import (
    WorldDrivenPrimitiveRuntime,
    primitive_checkpoint_dict,
    restore_world_driven_primitive_runtime,
)


class PrimitiveDevelopmentalCheckpointTests(unittest.TestCase):
    def test_search_policy_roundtrip_without_restoring_raw_authority(self):
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
        payload = primitive_checkpoint_dict(runtime)
        # The cache is audit-only and must never self-authorize after restore.
        payload["raw_observation_memory_cache"] = {
            "probe-a": {"opaque-1": 1.25, "opaque-2": -3.5},
        }
        restored = restore_world_driven_primitive_runtime(payload)

        self.assertEqual(restored.raw_observation_memory, {})
        self.assertEqual(restored.raw_observation_receipts, [])
        self.assertEqual(restored.primitive_genesis.model_budget, 321)
        self.assertEqual(restored.primitive_genesis.min_distinct_values, 4)
        self.assertEqual(restored.linear_primitive_genesis.model_budget, 654)
        self.assertEqual(restored.linear_primitive_genesis.max_coefficient_abs, 3)
        self.assertEqual(restored.symbolic_primitive_genesis.model_budget, 987)
        self.assertEqual(restored.symbolic_primitive_genesis.expression_budget, 111)
        self.assertEqual(restored.symbolic_primitive_genesis.max_depth, 1)
        self.assertEqual(restored.symbolic_primitive_genesis.operators, ("MUL", "ABS"))

    def test_legacy_v1_plain_raw_memory_is_deauthorized(self):
        runtime = WorldDrivenPrimitiveRuntime()
        payload = primitive_checkpoint_dict(runtime)
        payload["primitive_development_schema"] = "arte.primitive_development_same_body/v1"
        payload.pop("raw_observation_receipts", None)
        payload.pop("raw_observation_memory_cache", None)
        payload["raw_observation_memory"] = {
            "legacy-probe": {"opaque": 9.0},
        }
        restored = restore_world_driven_primitive_runtime(payload)
        self.assertEqual(restored.raw_observation_memory, {})
        self.assertEqual(restored.raw_observation_receipts, [])
        self.assertEqual(restored.symbolic_primitive_genesis.max_depth, 2)

    def test_legacy_epistemic_checkpoint_restores_with_empty_raw_memory(self):
        runtime = WorldDrivenPrimitiveRuntime()
        payload = primitive_checkpoint_dict(runtime)
        payload.pop("primitive_development_schema", None)
        payload.pop("raw_observation_receipts", None)
        payload.pop("raw_observation_memory_cache", None)
        payload.pop("primitive_genesis_policy", None)
        restored = restore_world_driven_primitive_runtime(payload)
        self.assertEqual(restored.raw_observation_memory, {})
        self.assertEqual(restored.raw_observation_receipts, [])
        self.assertEqual(restored.symbolic_primitive_genesis.max_depth, 2)


if __name__ == "__main__":
    unittest.main()
