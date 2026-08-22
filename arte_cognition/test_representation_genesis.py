import unittest

from arte_cognition.adaptive_cognition import TaskState
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.representation_genesis import MeasurementObservation, RepresentationGenesisEngine
from arte_cognition.semantic_genesis import ResidualObservation


class RepresentationGenesisTests(unittest.TestCase):
    def _measurements(self):
        return [
            MeasurementObservation("s1", {"load": 40, "capacity": 100}, "STABLE"),
            MeasurementObservation("s2", {"load": 60, "capacity": 150}, "STABLE"),
            MeasurementObservation("s3", {"load": 80, "capacity": 200}, "STABLE"),
            MeasurementObservation("j1", {"load": 90, "capacity": 100}, "JERK"),
            MeasurementObservation("j2", {"load": 135, "capacity": 150}, "JERK"),
            MeasurementObservation("j3", {"load": 180, "capacity": 200}, "JERK"),
            MeasurementObservation("hs", {"load": 50, "capacity": 125}, "STABLE", heldout=True),
            MeasurementObservation("hj", {"load": 112.5, "capacity": 125}, "JERK", heldout=True),
        ]

    def _residuals(self):
        rows = []
        for row in self._measurements():
            rows.append(ResidualObservation(
                residual_id=row.observation_id,
                features=("startup",),
                outcome=row.outcome,
                heldout=row.heldout,
            ))
        return rows

    def test_new_axis_generated_when_supplied_feature_vocabulary_is_inadequate(self):
        engine = RepresentationGenesisEngine()
        axes = engine.propose_axes(self._measurements())
        self.assertTrue(axes)
        self.assertTrue(any(axis.family in {"DIFFERENCE", "ABS_DIFFERENCE", "RATIO", "INTERACTION"} for axis in axes))
        self.assertGreaterEqual(max(axis.information_gain for axis in axes), 0.9)

    def test_generated_axis_enters_semantic_concept_and_heldout_law(self):
        runtime = PersistentCognitiveRuntime()
        cycle = runtime.cycle(
            TaskState(goal="explain startup residual", novelty=0.8),
            residuals=self._residuals(),
            measurements=self._measurements(),
        )
        self.assertTrue(cycle.representation_axes)
        derived = [
            concept for concept in cycle.concepts
            if any(feature.startswith("AXIS::") for feature in concept.defining_features)
        ]
        self.assertTrue(derived)
        derived_ids = {concept.concept_id for concept in derived}
        self.assertTrue(any(
            law.concept_id in derived_ids and law.status == "BOUNDED_LAW"
            for law in cycle.laws
        ))

    def test_axis_budget_and_partition_quotient_prevent_explosion(self):
        engine = RepresentationGenesisEngine(axis_budget=2)
        axes = engine.propose_axes(self._measurements())
        self.assertLessEqual(len(axes), 2)
        partitions = [axis.positive_partition for axis in axes]
        self.assertEqual(len(partitions), len(set(partitions)))


if __name__ == "__main__":
    unittest.main()
