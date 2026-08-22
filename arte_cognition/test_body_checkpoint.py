import unittest

from arte_cognition.adaptive_cognition import TaskState
from arte_cognition.body_checkpoint import checkpoint_json, restore_json
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.representation_genesis import MeasurementObservation
from arte_cognition.semantic_genesis import ResidualObservation


class BodyCheckpointTests(unittest.TestCase):
    def _trained_runtime(self):
        runtime = PersistentCognitiveRuntime()
        for _ in range(3):
            runtime.learn_from_ablation_outcomes(
                active_modules=["MODAL_EXPANSION"],
                full_outcome=1.0,
                ablation_outcomes={"MODAL_EXPANSION": 0.0},
                matched_compute={"MODAL_EXPANSION": True},
            )

        sequence = ["QUESTION_FIELD", "MODAL_EXPANSION", "REPRESENTATION_ESCAPE"]
        synergy = {
            ("QUESTION_FIELD", "MODAL_EXPANSION"): 0.8,
            ("MODAL_EXPANSION", "REPRESENTATION_ESCAPE"): 0.8,
        }
        for _ in range(3):
            runtime.learn_topology(sequence, synergy)

        rows = [
            ResidualObservation("r1", ("hot",), "FAIL"),
            ResidualObservation("r2", ("hot",), "FAIL"),
            ResidualObservation("r3", ("hot",), "FAIL"),
            ResidualObservation("r4", ("cold",), "PASS"),
            ResidualObservation("h1", ("hot",), "FAIL", heldout=True),
        ]
        runtime.cycle(TaskState(goal="induce"), residuals=rows)
        runtime.observe_world([ResidualObservation("world-new", ("hot",), "PASS")])
        return runtime

    @staticmethod
    def _representation_runtime():
        measurements = [
            MeasurementObservation("a1", {"x": -2, "y": -3, "z": -1}, "A"),
            MeasurementObservation("a2", {"x": 1, "y": 3, "z": -3}, "A"),
            MeasurementObservation("a3", {"x": -5, "y": 5, "z": -2}, "A"),
            MeasurementObservation("b1", {"x": -1, "y": -4, "z": 5}, "B"),
            MeasurementObservation("b2", {"x": 2, "y": 1, "z": 3}, "B"),
            MeasurementObservation("a4", {"x": -1, "y": 3, "z": 2}, "A"),
            MeasurementObservation("b3", {"x": 3, "y": 2, "z": -5}, "B"),
            MeasurementObservation("b4", {"x": 1, "y": 0, "z": -3}, "B"),
            MeasurementObservation("ha", {"x": 0, "y": 2, "z": 0}, "A", heldout=True),
            MeasurementObservation("hb", {"x": 0, "y": 0, "z": 0}, "B", heldout=True),
        ]
        residuals = [
            ResidualObservation(
                residual_id=row.observation_id,
                features=("raw",),
                outcome=row.outcome,
                heldout=row.heldout,
            )
            for row in measurements
        ]
        runtime = PersistentCognitiveRuntime()
        runtime.cycle(
            TaskState(goal="discover latent representation", novelty=0.9),
            residuals=residuals,
            measurements=measurements,
            experiment_reference_values={"x": 0.0, "y": 1.0, "z": 0.0},
        )
        return runtime

    def test_checkpoint_restore_preserves_router_behavior(self):
        runtime = self._trained_runtime()
        task = TaskState(goal="novel", novelty=0.62)
        before = runtime.cycle(task).plan.active_subgraph
        restored = restore_json(checkpoint_json(runtime))
        after = restored.cycle(task).plan.active_subgraph
        self.assertEqual(before, after)
        self.assertAlmostEqual(
            runtime.router.policy.threshold_shift("MODAL_EXPANSION"),
            restored.router.policy.threshold_shift("MODAL_EXPANSION"),
        )

    def test_checkpoint_restore_preserves_refuted_lineage(self):
        runtime = self._trained_runtime()
        restored = restore_json(checkpoint_json(runtime))
        self.assertEqual(runtime.memory.active_concepts(), restored.memory.active_concepts())
        self.assertEqual(runtime.memory.shadow_concepts(), restored.memory.shadow_concepts())
        self.assertTrue(any(record.refutations == ["world-new"] for record in restored.memory.laws.values()))
        self.assertTrue(any(m.action == "DEMOTE" for m in restored.memory.mutation_log))

    def test_checkpoint_restore_preserves_learned_topology_and_macro_proposals(self):
        runtime = self._trained_runtime()
        restored = restore_json(checkpoint_json(runtime))
        reversed_modules = ["REPRESENTATION_ESCAPE", "MODAL_EXPANSION", "QUESTION_FIELD"]
        self.assertEqual(runtime.topology.reorder(reversed_modules), restored.topology.reorder(reversed_modules))
        self.assertEqual(runtime.topology.propose_macros(), restored.topology.propose_macros())
        self.assertTrue(restored.topology.propose_macros())

    def test_checkpoint_restore_preserves_exact_generated_representation_and_experiment(self):
        runtime = self._representation_runtime()
        before_axes = runtime.persisted_representation_axes()
        before_proposals = runtime.persisted_intervention_proposals()
        self.assertTrue(any(axis.family == "PROJECTION" for axis in before_axes))
        self.assertTrue(before_proposals)

        restored = restore_json(checkpoint_json(runtime))
        self.assertEqual(before_axes, restored.persisted_representation_axes())
        self.assertEqual(before_proposals, restored.persisted_intervention_proposals())
        projection = next(axis for axis in restored.persisted_representation_axes() if axis.family == "PROJECTION")
        self.assertTrue(projection.coefficients)
        self.assertTrue(any(p.axis_id == projection.axis_id for p in restored.persisted_intervention_proposals()))

    def test_checkpoint_roundtrip_is_semantically_stable(self):
        runtime = self._trained_runtime()
        first = checkpoint_json(runtime)
        second = checkpoint_json(restore_json(first))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
