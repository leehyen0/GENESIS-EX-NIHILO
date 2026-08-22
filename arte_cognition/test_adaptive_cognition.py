import unittest

from arte_cognition.adaptive_cognition import (
    AdaptiveCognitionCompiler,
    Hypothesis,
    QueryCandidate,
    TaskState,
)


class AdaptiveCognitionTests(unittest.TestCase):
    def setUp(self):
        self.c = AdaptiveCognitionCompiler()

    def test_sparse_easy_task_does_not_activate_everything(self):
        task = TaskState(goal="simple lookup", hypotheses=[Hypothesis("H1")])
        plan = self.c.compile(task)
        self.assertEqual(plan.active_subgraph, ["STATE_IDENTITY", "CAUSAL_MODEL", "VERIFICATION"])
        self.assertLess(len(plan.active_subgraph), 5)

    def test_question_field_prefers_discriminating_query(self):
        task = TaskState(
            goal="identify cause",
            hypotheses=[Hypothesis("H1", 0.5), Hypothesis("H2", 0.5)],
            candidate_queries=[
                QueryCandidate("Q_bad", {"H1": "same", "H2": "same"}),
                QueryCandidate("Q_good", {"H1": "a", "H2": "b"}),
            ],
        )
        plan = self.c.compile(task)
        self.assertEqual(plan.question_field[0].query_id, "Q_good")
        self.assertIn("QUESTION_FIELD", plan.active_subgraph)

    def test_boundary_shadow_triggers_representation_escape(self):
        task = TaskState(
            goal="separate states",
            hypotheses=[
                Hypothesis("H1", representation_signature=("x",), predicts={"sensor": "up"}),
                Hypothesis("H2", representation_signature=("x",), predicts={"sensor": "down"}),
            ],
            residuals=["servo:start:jerk", "servo:start:jerk2"],
        )
        plan = self.c.compile(task)
        self.assertTrue(plan.boundary_shadow)
        self.assertTrue(plan.preobject_pressure)
        self.assertIn("REPRESENTATION_ESCAPE", plan.active_subgraph)
        self.assertIn("SPLIT", plan.representation_actions)
        self.assertIn("EXTEND", plan.representation_actions)
        self.assertIn("IMAGINARY", plan.modal_basis)

    def test_stakes_add_counterfactual_without_all_active(self):
        task = TaskState(
            goal="choose action",
            hypotheses=[Hypothesis("H1"), Hypothesis("H2")],
            stakes=0.9,
            action_required=True,
        )
        plan = self.c.compile(task)
        self.assertIn("COUNTERFACTUAL", plan.modal_basis)
        self.assertIn("WORLD_INTERACTION", plan.active_subgraph)
        self.assertLess(len(plan.active_subgraph), 10)

    def test_causal_credit_requires_behavior_and_outcome(self):
        credits = self.c.assign_causal_credit(
            active_modules=["QUESTION_FIELD", "MODAL_EXPANSION"],
            baseline_decision="A",
            treatment_decision="B",
            baseline_outcome=0.2,
            treatment_outcome=0.8,
            ablation_decisions={"QUESTION_FIELD": "A", "MODAL_EXPANSION": "B"},
        )
        by_module = {x.module: x for x in credits}
        self.assertGreater(by_module["QUESTION_FIELD"].causal_credit, 0)
        self.assertEqual(by_module["MODAL_EXPANSION"].causal_credit, 0)

    def test_shadow_reactivation_is_failure_specific(self):
        plan = self.c.compile(TaskState(goal="x", hypotheses=[Hypothesis("H1")]))
        self.assertIn("MODAL_EXPANSION", plan.shadow_subgraph)
        self.assertEqual(self.c.reactivate_shadow(plan, ["PREMATURE_CLOSURE"]), ["MODAL_EXPANSION"])

    def test_duplicate_representation_is_quotiented(self):
        task = TaskState(
            goal="compress basis",
            hypotheses=[Hypothesis("H1")],
            current_representation=["axis_a", "axis_a"],
        )
        self.assertIn("QUOTIENT", self.c.compile(task).representation_actions)


if __name__ == "__main__":
    unittest.main()
