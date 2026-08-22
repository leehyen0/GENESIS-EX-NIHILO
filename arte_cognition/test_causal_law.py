import unittest

from arte_cognition.causal_law import CausalLawEvaluator, InterventionObservation
from arte_cognition.semantic_genesis import LawCandidate


class CausalLawTests(unittest.TestCase):
    def _law(self, status="BOUNDED_LAW"):
        return LawCandidate(
            law_id="LAW::x",
            concept_id="CONCEPT::x",
            predicted_outcome="FAIL",
            train_support=6,
            train_accuracy=1.0,
            heldout_support=2,
            heldout_accuracy=1.0,
            counterexamples=(),
            status=status,
        )

    def test_predictive_law_does_not_become_causal_without_intervention(self):
        result = CausalLawEvaluator().assess(self._law(), [])
        self.assertEqual(result.status, "PREDICTIVE_LAW")

    def test_nonrandomized_intervention_stops_at_supported_relation(self):
        rows = [
            InterventionObservation("t1", "TREATMENT", 1.0, "NATURAL_EXPERIMENT", "A"),
            InterventionObservation("t2", "TREATMENT", 0.9, "NATURAL_EXPERIMENT", "B"),
            InterventionObservation("c1", "CONTROL", 0.2, "NATURAL_EXPERIMENT", "A"),
            InterventionObservation("c2", "CONTROL", 0.1, "NATURAL_EXPERIMENT", "B"),
        ]
        result = CausalLawEvaluator().assess(self._law(), rows)
        self.assertEqual(result.status, "INTERVENTION_SUPPORTED_RELATION")
        self.assertGreater(result.estimated_effect, 0.5)

    def test_randomized_multisource_with_negative_control_closes_bounded_causal_gate(self):
        rows = [
            InterventionObservation("t1", "TREATMENT", 1.0, "RANDOMIZED", "A"),
            InterventionObservation("t2", "TREATMENT", 0.9, "RANDOMIZED", "B"),
            InterventionObservation("c1", "CONTROL", 0.2, "RANDOMIZED", "A"),
            InterventionObservation("c2", "CONTROL", 0.1, "RANDOMIZED", "B"),
            InterventionObservation("nt1", "TREATMENT", 0.50, "RANDOMIZED", "A", negative_control=True),
            InterventionObservation("nt2", "TREATMENT", 0.52, "RANDOMIZED", "B", negative_control=True),
            InterventionObservation("nc1", "CONTROL", 0.49, "RANDOMIZED", "A", negative_control=True),
            InterventionObservation("nc2", "CONTROL", 0.51, "RANDOMIZED", "B", negative_control=True),
        ]
        result = CausalLawEvaluator().assess(self._law(), rows)
        self.assertEqual(result.status, "CAUSAL_LAW_BOUNDED")
        self.assertTrue(result.randomized)
        self.assertTrue(result.negative_control_pass)

    def test_failed_negative_control_blocks_causal_promotion(self):
        rows = [
            InterventionObservation("t1", "TREATMENT", 1.0, "RANDOMIZED", "A"),
            InterventionObservation("t2", "TREATMENT", 0.9, "RANDOMIZED", "B"),
            InterventionObservation("c1", "CONTROL", 0.2, "RANDOMIZED", "A"),
            InterventionObservation("c2", "CONTROL", 0.1, "RANDOMIZED", "B"),
            InterventionObservation("nt1", "TREATMENT", 0.9, "RANDOMIZED", "A", negative_control=True),
            InterventionObservation("nt2", "TREATMENT", 0.8, "RANDOMIZED", "B", negative_control=True),
            InterventionObservation("nc1", "CONTROL", 0.1, "RANDOMIZED", "A", negative_control=True),
            InterventionObservation("nc2", "CONTROL", 0.2, "RANDOMIZED", "B", negative_control=True),
        ]
        result = CausalLawEvaluator().assess(self._law(), rows)
        self.assertEqual(result.status, "INTERVENTION_SUPPORTED_RELATION")
        self.assertFalse(result.negative_control_pass)

    def test_unverified_semantic_law_remains_associative(self):
        result = CausalLawEvaluator().assess(self._law("HELDOUT_REQUIRED"), [])
        self.assertEqual(result.status, "ASSOCIATIVE_PATTERN")


if __name__ == "__main__":
    unittest.main()
