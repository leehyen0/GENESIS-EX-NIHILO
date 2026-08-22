from __future__ import annotations

import unittest

from arte_cognition.causal_linear_primitive_genesis import LinearFormPrimitiveGenesisEngine
from arte_cognition.causal_model_genesis import InterventionDescriptor
from arte_cognition.causal_symbolic_primitive_genesis import SymbolicPrimitiveGenesisEngine
from arte_cognition.primitive_genesis_runtime import WorldDrivenPrimitiveRuntime
from arte_cognition.world_model_ecology import ModelEvidence


class SymbolicPrimitiveGenesisTests(unittest.TestCase):
    def setUp(self):
        self.x = "sensor_x"
        self.a = "raw_a"
        self.b = "raw_b"
        self.descriptors = [
            InterventionDescriptor(f"probe-{index}", (self.x,), cost=1.0)
            for index in range(8)
        ]
        a_values = [0, 1, 2, 3, 4, 5, 6, 7]
        b_values = [7, 2, 5, 0, 6, 1, 4, 3]
        self.raw = {
            descriptor.intervention_id: {
                self.a: float(a_values[index]),
                self.b: float(b_values[index]),
            }
            for index, descriptor in enumerate(self.descriptors)
        }

    def _g6(self):
        engine = LinearFormPrimitiveGenesisEngine(model_budget=8192, max_coefficient_abs=2)
        return engine.generate_novel([self.x], self.descriptors, self.raw, (), ())

    def test_symbolic_search_generates_nonlinear_two_channel_expressions(self):
        g6 = self._g6()
        engine = SymbolicPrimitiveGenesisEngine(
            model_budget=8192,
            expression_budget=512,
            max_depth=1,
            operators=("ADD", "SUB", "MUL", "ABS"),
        )
        g7 = engine.generate_novel(
            [self.x], self.descriptors, self.raw, (), [item.model for item in g6]
        )
        self.assertFalse(engine.last_truncated)
        self.assertTrue(g7)
        self.assertTrue(any(" * " in item.primitive.expression.render() for item in g7))
        g6_signatures = {tuple(sorted(item.model.predictions)) for item in g6}
        self.assertTrue(all(tuple(sorted(item.model.predictions)) not in g6_signatures for item in g7))

    def test_authoritative_evidence_filters_outcome_independent_symbolic_shadow(self):
        g6 = self._g6()
        existing = [item.model for item in g6]
        engine = SymbolicPrimitiveGenesisEngine(
            model_budget=8192,
            expression_budget=512,
            max_depth=1,
        )
        shadow = engine.generate_novel([self.x], self.descriptors, self.raw, (), existing)
        self.assertTrue(shadow)
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

    def test_runtime_cannot_open_symbolic_search_before_g6_falsification(self):
        g6 = self._g6()
        runtime = WorldDrivenPrimitiveRuntime()
        runtime.register_causal_world_models([item.model for item in g6])
        # A surviving G6 class blocks symbolic expansion independently of whether
        # authenticated raw representation evidence has already been collected.
        generated = runtime.generate_world_driven_symbolic_primitive_models(
            [self.x], self.descriptors
        )
        self.assertEqual(generated, [])
        self.assertFalse(runtime.generation_falsified(6))


if __name__ == "__main__":
    unittest.main()
