from __future__ import annotations

import unittest

from arte_cognition.causal_linear_primitive_genesis import LinearFormPrimitiveGenesisEngine
from arte_cognition.causal_model_genesis import InterventionDescriptor
from arte_cognition.causal_primitive_genesis import RawThresholdPrimitiveGenesisEngine
from arte_cognition.primitive_genesis_runtime import WorldDrivenPrimitiveRuntime
from arte_cognition.world_model_ecology import ModelEvidence


class LinearFormPrimitiveGenesisTests(unittest.TestCase):
    def setUp(self):
        self.x = "sensor_x"
        self.a = "raw_a"
        self.b = "raw_b"
        self.descriptors = [
            InterventionDescriptor(f"probe-{index}", (self.x,), cost=1.0)
            for index in range(6)
        ]
        # Individually each channel is non-monotonic with respect to the target
        # partition below; a multi-channel linear relation can separate it.
        a_values = [0, 1, 2, 3, 4, 5]
        b_values = [0, 3, 1, 5, 2, 4]
        self.raw = {
            descriptor.intervention_id: {
                self.a: float(a_values[index]),
                self.b: float(b_values[index]),
            }
            for index, descriptor in enumerate(self.descriptors)
        }

    def test_generates_two_channel_linear_forms_without_named_sum_or_difference(self):
        g5_engine = RawThresholdPrimitiveGenesisEngine(model_budget=2048)
        g5 = g5_engine.generate_novel([self.x], self.descriptors, self.raw, (), ())
        engine = LinearFormPrimitiveGenesisEngine(model_budget=4096, max_coefficient_abs=2)
        g6 = engine.generate_novel(
            [self.x], self.descriptors, self.raw, (), [item.model for item in g5]
        )
        self.assertFalse(engine.last_truncated)
        self.assertTrue(g6)
        self.assertTrue(all(len(item.primitive.coefficients) >= 2 for item in g6))
        self.assertTrue(any(
            {weight for _channel, weight in item.primitive.coefficients} & {-2, -1, 1, 2}
            for item in g6
        ))
        self.assertTrue(all(item.model.family == "RAW_LINEAR_FORM_THRESHOLD" for item in g6))

    def test_authoritative_evidence_filters_preexisting_linear_shadow(self):
        g5_engine = RawThresholdPrimitiveGenesisEngine(model_budget=2048)
        g5 = g5_engine.generate_novel([self.x], self.descriptors, self.raw, (), ())
        existing = [item.model for item in g5]
        engine = LinearFormPrimitiveGenesisEngine(model_budget=4096, max_coefficient_abs=2)
        shadow = engine.generate_novel([self.x], self.descriptors, self.raw, (), existing)
        target = shadow[len(shadow) // 2]
        evidence = tuple(
            ModelEvidence(
                evidence_id=f"e-{index}",
                intervention_id=descriptor.intervention_id,
                observed_outcome=target.model.prediction_for(descriptor.intervention_id) or "NO_EFFECT",
                source_class="independent-A",
                context_id="unit",
                authoritative=True,
            )
            for index, descriptor in enumerate(self.descriptors)
        )
        active = engine.generate_novel([self.x], self.descriptors, self.raw, evidence, existing)
        self.assertEqual({item.model.model_id for item in active}, {target.model.model_id})

    def test_runtime_cannot_open_g6_before_g5_is_falsified(self):
        g5_engine = RawThresholdPrimitiveGenesisEngine(model_budget=2048)
        g5 = g5_engine.generate_novel([self.x], self.descriptors, self.raw, (), ())
        runtime = WorldDrivenPrimitiveRuntime()
        runtime.register_causal_world_models([item.model for item in g5])
        generated = runtime.generate_world_driven_linear_primitive_models(
            [self.x], self.descriptors, self.raw
        )
        self.assertEqual(generated, [])
        self.assertFalse(runtime.generation_falsified(5))


if __name__ == "__main__":
    unittest.main()
