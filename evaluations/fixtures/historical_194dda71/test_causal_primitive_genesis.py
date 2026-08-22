from __future__ import annotations

import unittest

from arte_cognition.causal_model_genesis import InterventionDescriptor
from arte_cognition.causal_primitive_genesis import RawThresholdPrimitiveGenesisEngine
from arte_cognition.primitive_genesis_runtime import WorldDrivenPrimitiveRuntime
from arte_cognition.world_model_ecology import CausalWorldModel, ModelEvidence


class RawThresholdPrimitiveGenesisTests(unittest.TestCase):
    def setUp(self):
        self.x = "sensor_x"
        self.channel = "raw_channel_7f3a"
        self.descriptors = [
            InterventionDescriptor(f"probe-{index}", (self.x,), cost=1.0)
            for index in range(4)
        ]
        self.raw = {
            descriptor.intervention_id: {self.channel: float(index)}
            for index, descriptor in enumerate(self.descriptors)
        }

    def test_shadow_thresholds_are_generated_without_outcome_semantics(self):
        engine = RawThresholdPrimitiveGenesisEngine(model_budget=128)
        shadow = engine.generate_novel([self.x], self.descriptors, self.raw, (), ())
        self.assertFalse(engine.last_truncated)
        self.assertTrue(shadow)
        self.assertTrue(any(
            item.primitive.channel == self.channel
            and item.primitive.direction == ">="
            and abs(item.primitive.threshold - 1.5) < 1e-12
            for item in shadow
        ))

    def test_authoritative_evidence_filters_preexisting_shadow_universe(self):
        engine = RawThresholdPrimitiveGenesisEngine(model_budget=128)
        shadow = engine.generate_novel([self.x], self.descriptors, self.raw, (), ())
        target = next(
            item for item in shadow
            if item.sign == "POS"
            and item.primitive.direction == ">="
            and abs(item.primitive.threshold - 1.5) < 1e-12
        )
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
        active = engine.generate_novel([self.x], self.descriptors, self.raw, evidence, ())
        active_ids = {item.model.model_id for item in active}
        self.assertEqual(active_ids, {target.model.model_id})

    def test_runtime_cannot_open_g5_before_g4_is_falsified(self):
        runtime = WorldDrivenPrimitiveRuntime()
        g4 = CausalWorldModel(
            model_id="g4-survivor",
            prior=1.0,
            predictions=tuple((d.intervention_id, "NO_EFFECT") for d in self.descriptors),
            origin="GENERATED_SPARSE_MINTERM",
            family="SPARSE_EXACT_MINTERM_GATE",
            structure=("G4",),
            generation=4,
        )
        runtime.register_causal_world_models([g4])
        generated = runtime.generate_world_driven_primitive_models(
            [self.x], self.descriptors, self.raw
        )
        self.assertEqual(generated, [])
        self.assertFalse(runtime.generation_falsified(4))


if __name__ == "__main__":
    unittest.main()
