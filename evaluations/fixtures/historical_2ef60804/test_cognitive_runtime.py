import unittest

from arte_cognition import Hypothesis, TaskState
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.possibility_space import Fact, OperatorSpec
from arte_cognition.semantic_genesis import ResidualObservation


class CognitiveRuntimeTests(unittest.TestCase):
    def test_cycle_integrates_routing_possibility_and_semantic_genesis(self):
        runtime = PersistentCognitiveRuntime()
        task = TaskState(
            goal="explain repeated failure",
            hypotheses=[Hypothesis("h1"), Hypothesis("h2")],
            residuals=["servo:startup:1", "servo:startup:2"],
            novelty=0.8,
        )
        residuals = [
            ResidualObservation("r1", ("startup", "loaded"), "JERK"),
            ResidualObservation("r2", ("startup", "loaded"), "JERK"),
            ResidualObservation("r3", ("steady",), "OK"),
            ResidualObservation("r4", ("steady",), "OK"),
            ResidualObservation("h1", ("startup", "loaded"), "JERK", heldout=True),
        ]
        cycle = runtime.cycle(
            task,
            facts=[Fact("servo", "state", "startup")],
            residuals=residuals,
            operator_spec=OperatorSpec(relation_opposites={"state": "not_state"}),
        )
        self.assertIn("QUESTION_FIELD", cycle.plan.active_subgraph)
        self.assertTrue(cycle.possibilities)
        self.assertTrue(cycle.concepts)
        self.assertTrue(any(law.status == "BOUNDED_LAW" for law in cycle.laws))
        self.assertTrue(cycle.active_generated_concepts)

    def test_unverified_concept_remains_shadow(self):
        runtime = PersistentCognitiveRuntime()
        rows = [
            ResidualObservation("r1", ("hot",), "FAIL"),
            ResidualObservation("r2", ("hot",), "FAIL"),
            ResidualObservation("r3", ("hot",), "FAIL"),
            ResidualObservation("r4", ("cold",), "PASS"),
        ]
        cycle = runtime.cycle(TaskState(goal="x"), residuals=rows)
        self.assertFalse(cycle.active_generated_concepts)
        self.assertTrue(cycle.shadow_generated_concepts)
        self.assertTrue(all(law.status == "HELDOUT_REQUIRED" for law in cycle.laws))

    def test_new_world_counterexample_demotes_active_generated_concept(self):
        runtime = PersistentCognitiveRuntime()
        rows = [
            ResidualObservation("r1", ("hot",), "FAIL"),
            ResidualObservation("r2", ("hot",), "FAIL"),
            ResidualObservation("r3", ("hot",), "FAIL"),
            ResidualObservation("r4", ("cold",), "PASS"),
            ResidualObservation("h1", ("hot",), "FAIL", heldout=True),
        ]
        cycle = runtime.cycle(TaskState(goal="x"), residuals=rows)
        self.assertTrue(cycle.active_generated_concepts)
        target = cycle.active_generated_concepts[0]
        mutations = runtime.observe_world([
            ResidualObservation("world-new", ("hot",), "PASS")
        ])
        self.assertTrue(any(m.action == "DEMOTE" for m in mutations))
        self.assertNotIn(target, runtime.memory.active_concepts())
        self.assertIn(target, runtime.memory.shadow_concepts())

    def test_demotion_preserves_refuted_lineage(self):
        runtime = PersistentCognitiveRuntime()
        rows = [
            ResidualObservation("r1", ("hot",), "FAIL"),
            ResidualObservation("r2", ("hot",), "FAIL"),
            ResidualObservation("r3", ("hot",), "FAIL"),
            ResidualObservation("r4", ("cold",), "PASS"),
            ResidualObservation("h1", ("hot",), "FAIL", heldout=True),
        ]
        runtime.cycle(TaskState(goal="x"), residuals=rows)
        runtime.observe_world([ResidualObservation("world-new", ("hot",), "PASS")])
        self.assertTrue(any(record.refutations == ["world-new"] for record in runtime.memory.laws.values()))
        self.assertTrue(any(m.action == "DEMOTE" for m in runtime.memory.mutation_log))

    def test_runtime_outcome_learning_changes_future_routing(self):
        runtime = PersistentCognitiveRuntime()
        task = TaskState(goal="novel", novelty=0.62)
        self.assertNotIn("MODAL_EXPANSION", runtime.cycle(task).plan.active_subgraph)
        for _ in range(3):
            runtime.learn_from_outcome(
                active_modules=["MODAL_EXPANSION"],
                baseline_decision="A",
                treatment_decision="B",
                baseline_outcome=0.0,
                treatment_outcome=1.0,
                ablation_decisions={"MODAL_EXPANSION": "A"},
            )
        self.assertIn("MODAL_EXPANSION", runtime.cycle(task).plan.active_subgraph)

    def test_generated_imaginary_possibility_is_never_world_fact(self):
        runtime = PersistentCognitiveRuntime()
        task = TaskState(goal="novel", novelty=0.9)
        cycle = runtime.cycle(task, facts=[Fact("x", "rel", "y")])
        imaginary = [p for p in cycle.possibilities if p.mode == "IMAGINARY"]
        self.assertTrue(imaginary)
        self.assertTrue(all(not p.asserted for p in imaginary))


if __name__ == "__main__":
    unittest.main()
