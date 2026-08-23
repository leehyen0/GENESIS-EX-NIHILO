from __future__ import annotations

import unittest

from arte_cognition.causal_model_genesis import InterventionDescriptor
from arte_cognition.causal_symbolic_primitive_genesis import SymbolicPrimitiveGenesisEngine
from arte_cognition.composition_law_genesis import (
    CompositionLawGenesisEngine,
    GeneratedCompositionLaw,
)
from arte_cognition.world_model_ecology import ModelEvidence


class CompositionLawGenesisTests(unittest.TestCase):
    def _fixture(self, scale: float = 1.0, prefix: str = "train"):
        values = (-1.0, 0.0, 1.0)
        descriptors = []
        raw = {}
        for i, left in enumerate(values):
            for j, right in enumerate(values):
                intervention_id = f"{prefix}-{i}-{j}"
                descriptors.append(InterventionDescriptor(intervention_id, ("cause",)))
                raw[intervention_id] = {
                    "opaque_a": float(scale) * left,
                    "opaque_b": float(scale) * right,
                }
        return tuple(descriptors), raw

    @staticmethod
    def _evaluator_owned_target_law():
        # This table is not supplied to candidate generation. It is evaluator-owned
        # only so the test can produce the external consequence pattern.
        return GeneratedCompositionLaw(
            state_count=3,
            identity_state=1,
            table=(
                0, 0, 0,
                0, 1, 2,
                0, 2, 0,
            ),
            active_states=(0,),
        )

    def _evidence(self, engine, descriptors, raw):
        hidden = self._evaluator_owned_target_law()
        out = []
        for index, descriptor in enumerate(descriptors):
            outcome = engine.predict(
                "cause", "POS", "opaque_a", "opaque_b", hidden, descriptor, raw
            )
            out.append(ModelEvidence(
                evidence_id=f"e-{index}",
                intervention_id=descriptor.intervention_id,
                observed_outcome=outcome,
                source_class="external-a" if index % 2 == 0 else "external-b",
                context_id="law-train",
                authoritative=True,
            ))
        return tuple(out)

    def test_complete_fixed_symbolic_alphabet_cannot_fit_relation(self):
        descriptors, raw = self._fixture()
        law_engine = CompositionLawGenesisEngine()
        evidence = self._evidence(law_engine, descriptors, raw)
        predecessor = SymbolicPrimitiveGenesisEngine(
            model_budget=16384,
            expression_budget=2048,
            max_depth=2,
            operators=("ADD", "SUB", "MUL", "ABS"),
            min_active_channels=2,
        )
        active = predecessor.generate_novel(
            ("cause",), descriptors, raw, evidence, ()
        )
        self.assertFalse(predecessor.last_truncated)
        self.assertEqual(active, [])

    def test_operation_tables_are_generated_without_named_arithmetic_operator(self):
        engine = CompositionLawGenesisEngine()
        laws = engine.shadow_laws()
        self.assertEqual(engine.last_table_count, 27)
        self.assertEqual(len(laws), 162)
        self.assertEqual({law.identity_state for law in laws}, {1})
        self.assertTrue(all(len(law.table) == 9 for law in laws))

    def test_world_evidence_filters_frozen_shadow_to_new_prediction_class(self):
        descriptors, raw = self._fixture()
        engine = CompositionLawGenesisEngine(model_budget=4096)
        shadow = engine.generate_novel(("cause",), descriptors, raw, (), ())
        shadow_ids = {item.model.model_id for item in shadow}
        evidence = self._evidence(engine, descriptors, raw)
        active = engine.generate_novel(("cause",), descriptors, raw, evidence, ())

        self.assertFalse(engine.last_truncated)
        self.assertGreater(len(shadow), 0)
        self.assertEqual(len(active), 1)
        self.assertIn(active[0].model.model_id, shadow_ids)
        self.assertEqual(active[0].model.origin, "GENERATED_COMPOSITION_LAW")
        self.assertEqual(active[0].model.generation, 8)

    def test_frozen_generated_law_transfers_to_fresh_scaled_context_and_wrong_law_fails(self):
        train_descriptors, train_raw = self._fixture()
        engine = CompositionLawGenesisEngine(model_budget=4096)
        evidence = self._evidence(engine, train_descriptors, train_raw)
        active = engine.generate_novel(("cause",), train_descriptors, train_raw, evidence, ())
        self.assertEqual(len(active), 1)
        selected = active[0]

        heldout_descriptors, heldout_raw = self._fixture(scale=7.0, prefix="heldout")
        hidden = self._evaluator_owned_target_law()
        expected = tuple(
            engine.predict("cause", "POS", "opaque_a", "opaque_b", hidden, d, heldout_raw)
            for d in heldout_descriptors
        )
        treatment = tuple(
            engine.predict(
                selected.cause,
                selected.sign,
                selected.left_channel,
                selected.right_channel,
                selected.law,
                d,
                heldout_raw,
            )
            for d in heldout_descriptors
        )
        self.assertEqual(treatment, expected)

        shadow = engine.generate_novel(("cause",), train_descriptors, train_raw, (), ())
        wrong = next(
            item for item in shadow
            if tuple(item.model.predictions) != tuple(selected.model.predictions)
        )
        wrong_predictions = tuple(
            engine.predict(
                wrong.cause,
                wrong.sign,
                wrong.left_channel,
                wrong.right_channel,
                wrong.law,
                d,
                heldout_raw,
            )
            for d in heldout_descriptors
        )
        self.assertNotEqual(wrong_predictions, expected)


if __name__ == "__main__":
    unittest.main()
