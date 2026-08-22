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
    def test_raw_observation_memory_and_search_policy_roundtrip(self):
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
        runtime.ingest_raw_observations({
            "probe-a": {"opaque-1": 1.25, "opaque-2": -3.5},
            "probe-b": {"opaque-1": 2.25, "opaque-2": 7.5},
        })
        payload = primitive_checkpoint_dict(runtime)
        restored = restore_world_driven_primitive_runtime(payload)

        self.assertEqual(restored.raw_observation_memory, runtime.raw_observation_memory)
        self.assertEqual(restored.primitive_genesis.model_budget, 321)
        self.assertEqual(restored.primitive_genesis.min_distinct_values, 4)
        self.assertEqual(restored.linear_primitive_genesis.model_budget, 654)
        self.assertEqual(restored.linear_primitive_genesis.max_coefficient_abs, 3)
        self.assertEqual(restored.symbolic_primitive_genesis.model_budget, 987)
        self.assertEqual(restored.symbolic_primitive_genesis.expression_budget, 111)
        self.assertEqual(restored.symbolic_primitive_genesis.max_depth, 1)
        self.assertEqual(restored.symbolic_primitive_genesis.operators, ("MUL", "ABS"))

    def test_legacy_epistemic_checkpoint_restores_with_empty_raw_memory(self):
        runtime = WorldDrivenPrimitiveRuntime()
        payload = primitive_checkpoint_dict(runtime)
        payload.pop("primitive_development_schema")
        payload.pop("raw_observation_memory")
        payload.pop("primitive_genesis_policy")
        restored = restore_world_driven_primitive_runtime(payload)
        self.assertEqual(restored.raw_observation_memory, {})
        self.assertEqual(restored.symbolic_primitive_genesis.max_depth, 2)


if __name__ == "__main__":
    unittest.main()
