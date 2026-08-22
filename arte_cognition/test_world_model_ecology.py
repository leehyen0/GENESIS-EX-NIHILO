import unittest

from arte_cognition.adaptive_cognition import QueryCandidate
from arte_cognition.world_model_ecology import CausalWorldModel, ModelEvidence, WorldModelEcology


class WorldModelEcologyTests(unittest.TestCase):
    def _ecology(self):
        ecology = WorldModelEcology()
        ecology.register([
            CausalWorldModel(
                "MODEL_A",
                1.0,
                (("EXPENSIVE", "POSITIVE_EFFECT"), ("CHEAP", "NO_EFFECT"), ("SURPRISE", "POSITIVE_EFFECT")),
            ),
            CausalWorldModel(
                "MODEL_B",
                1.0,
                (("EXPENSIVE", "NEGATIVE_EFFECT"), ("CHEAP", "NO_EFFECT"), ("SURPRISE", "POSITIVE_EFFECT")),
            ),
        ])
        return ecology

    def test_high_uncertainty_can_choose_expensive_discriminating_intervention(self):
        ecology = self._ecology()
        ranked = ecology.rank_interventions([
            QueryCandidate("CHEAP", {"MODEL_A": "NO_EFFECT", "MODEL_B": "NO_EFFECT"}, cost=1.0, intervention=True),
            QueryCandidate("EXPENSIVE", {"MODEL_A": "POSITIVE_EFFECT", "MODEL_B": "NEGATIVE_EFFECT"}, cost=20.0, intervention=True),
        ])
        self.assertEqual(ecology.depth_plan().mode, "DEEP_DISCRIMINATION")
        self.assertEqual(ranked[0].intervention_id, "EXPENSIVE")
        self.assertGreater(ranked[0].expected_information_gain, 0.9)
        self.assertEqual(ranked[0].cost, 20.0)

    def test_authoritative_evidence_can_concentrate_posterior(self):
        ecology = self._ecology()
        ecology.observe(ModelEvidence("e1", "EXPENSIVE", "POSITIVE_EFFECT", "A", "ctx", True))
        ecology.observe(ModelEvidence("e2", "EXPENSIVE", "POSITIVE_EFFECT", "B", "ctx", True))
        posterior = ecology.posterior()
        self.assertGreater(posterior["MODEL_A"], 0.99)
        self.assertEqual(ecology.depth_plan().mode, "COMPACT")

    def test_world_surprise_triggers_model_class_expansion_not_forced_fit(self):
        ecology = self._ecology()
        ecology.observe(ModelEvidence("e1", "EXPENSIVE", "POSITIVE_EFFECT", "A", "ctx", True))
        ecology.observe(ModelEvidence("e2", "EXPENSIVE", "POSITIVE_EFFECT", "B", "ctx", True))
        self.assertEqual(ecology.depth_plan().mode, "COMPACT")
        ecology.observe(ModelEvidence("s1", "SURPRISE", "NO_EFFECT", "A", "ctx2", True))
        plan = ecology.depth_plan()
        self.assertEqual(plan.mode, "EXPAND_MODEL_CLASS")
        self.assertTrue(plan.model_class_inadequate)
        self.assertGreater(plan.possibility_budget, 32)
        self.assertGreater(plan.representation_axis_budget, 16)
        self.assertLess(plan.cost_exponent, 0.5)

    def test_unverified_surprise_cannot_expand_model_class(self):
        ecology = self._ecology()
        ecology.observe(ModelEvidence("fake", "SURPRISE", "NO_EFFECT", "UNVERIFIED", "ctx", False))
        self.assertFalse(ecology.model_class_inadequate)
        self.assertEqual(ecology.depth_plan().mode, "DEEP_DISCRIMINATION")


if __name__ == "__main__":
    unittest.main()
