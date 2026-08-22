import unittest

from arte_cognition.causal_credit import OutcomeAblationCreditEngine
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.adaptive_cognition import TaskState


class OutcomeAblationCreditTests(unittest.TestCase):
    def test_modules_receive_distinct_outcome_credit(self):
        engine = OutcomeAblationCreditEngine()
        credits = engine.assign(
            full_outcome=0.90,
            ablation_outcomes={"MODAL_EXPANSION": 0.30, "QUESTION_FIELD": 0.95},
            active_modules=["MODAL_EXPANSION", "QUESTION_FIELD"],
        )
        by_module = {c.module: c for c in credits}
        self.assertAlmostEqual(by_module["MODAL_EXPANSION"].marginal_contribution, 0.60)
        self.assertAlmostEqual(by_module["MODAL_EXPANSION"].causal_credit, 0.60)
        self.assertAlmostEqual(by_module["QUESTION_FIELD"].marginal_contribution, -0.05)
        self.assertAlmostEqual(by_module["QUESTION_FIELD"].causal_harm, 0.05)

    def test_unmatched_compute_cannot_update_credit(self):
        engine = OutcomeAblationCreditEngine()
        credit = engine.assign(
            full_outcome=1.0,
            ablation_outcomes={"MODAL_EXPANSION": 0.0},
            active_modules=["MODAL_EXPANSION"],
            matched_compute={"MODAL_EXPANSION": False},
        )[0]
        self.assertEqual(credit.causal_credit, 0.0)
        self.assertEqual(credit.causal_harm, 0.0)

    def test_pair_synergy_is_not_double_counted_as_individual_credit(self):
        engine = OutcomeAblationCreditEngine()
        synergy = engine.pair_synergy(
            full_outcome=1.0,
            without_i=0.4,
            without_j=0.5,
            without_both=0.1,
            module_i="A",
            module_j="B",
        )
        self.assertAlmostEqual(synergy.synergy, 0.2)

    def test_outcome_ablation_learning_changes_future_routing(self):
        runtime = PersistentCognitiveRuntime()
        task = TaskState(goal="novel", novelty=0.62)
        self.assertNotIn("MODAL_EXPANSION", runtime.cycle(task).plan.active_subgraph)
        for _ in range(3):
            runtime.learn_from_ablation_outcomes(
                active_modules=["MODAL_EXPANSION"],
                full_outcome=1.0,
                ablation_outcomes={"MODAL_EXPANSION": 0.0},
                matched_compute={"MODAL_EXPANSION": True},
            )
        self.assertIn("MODAL_EXPANSION", runtime.cycle(task).plan.active_subgraph)


if __name__ == "__main__":
    unittest.main()
