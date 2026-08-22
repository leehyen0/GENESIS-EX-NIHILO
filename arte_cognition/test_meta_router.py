import unittest

from arte_cognition.adaptive_cognition import Hypothesis, TaskState
from arte_cognition.meta_router import OutcomeLearnedCognitionRouter


class MetaRouterTests(unittest.TestCase):
    def test_no_routing_change_before_minimum_evidence(self):
        router = OutcomeLearnedCognitionRouter()
        task = TaskState(goal="marginal contradiction", hypotheses=[Hypothesis("H1")], contradictions=["x"])
        self.assertNotIn("COUNTEREXAMPLE_SEARCH", router.compile(task).active_subgraph)

        for _ in range(2):
            router.learn_from_outcome(
                active_modules=["COUNTEREXAMPLE_SEARCH"],
                baseline_decision="A",
                treatment_decision="B",
                baseline_outcome=0.0,
                treatment_outcome=1.0,
                ablation_decisions={"COUNTEREXAMPLE_SEARCH": "A"},
            )
        self.assertNotIn("COUNTEREXAMPLE_SEARCH", router.compile(task).active_subgraph)

    def test_repeated_positive_causal_credit_changes_future_routing(self):
        router = OutcomeLearnedCognitionRouter()
        task = TaskState(goal="marginal contradiction", hypotheses=[Hypothesis("H1")], contradictions=["x"])
        self.assertNotIn("COUNTEREXAMPLE_SEARCH", router.compile(task).active_subgraph)

        for _ in range(3):
            router.learn_from_outcome(
                active_modules=["COUNTEREXAMPLE_SEARCH"],
                baseline_decision="A",
                treatment_decision="B",
                baseline_outcome=0.0,
                treatment_outcome=1.0,
                ablation_decisions={"COUNTEREXAMPLE_SEARCH": "A"},
            )
        plan = router.compile(task)
        self.assertIn("COUNTEREXAMPLE_SEARCH", plan.active_subgraph)
        self.assertTrue(any("outcome-learned router activated" in r for r in plan.reasons))

    def test_negative_outcomes_can_demote_marginal_modal_expansion(self):
        router = OutcomeLearnedCognitionRouter()
        task = TaskState(goal="single residual", hypotheses=[Hypothesis("H1")], residuals=["r:x"])
        self.assertIn("MODAL_EXPANSION", router.compile(task).active_subgraph)

        for _ in range(3):
            router.learn_from_outcome(
                active_modules=["MODAL_EXPANSION"],
                baseline_decision="A",
                treatment_decision="B",
                baseline_outcome=1.0,
                treatment_outcome=0.0,
                ablation_decisions={"MODAL_EXPANSION": "A"},
            )
        plan = router.compile(task)
        self.assertNotIn("MODAL_EXPANSION", plan.active_subgraph)
        self.assertIn("MODAL_EXPANSION", plan.shadow_subgraph)

    def test_hard_representation_escape_cannot_be_learned_away(self):
        router = OutcomeLearnedCognitionRouter()
        task = TaskState(
            goal="distinguish collision",
            hypotheses=[
                Hypothesis("H1", representation_signature=("x",), predicts={"o": "1"}),
                Hypothesis("H2", representation_signature=("x",), predicts={"o": "2"}),
            ],
        )
        plan = router.compile(task)
        self.assertIn("REPRESENTATION_ESCAPE", plan.active_subgraph)


if __name__ == "__main__":
    unittest.main()
