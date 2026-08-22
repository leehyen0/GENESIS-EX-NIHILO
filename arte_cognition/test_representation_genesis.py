import unittest

from arte_cognition.adaptive_cognition import TaskState
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.experiment_genesis import ExperimentGenesisEngine
from arte_cognition.representation_genesis import MeasurementObservation, RepresentationAxis, RepresentationGenesisEngine
from arte_cognition.representation_value import RepresentationValueEvaluator
from arte_cognition.semantic_genesis import ResidualObservation


class RepresentationGenesisTests(unittest.TestCase):
    def _measurements(self):
        # Raw load overlaps classes and capacity is deliberately non-discriminating.
        # The ratio/load-capacity relationship is the useful new representation.
        return [
            MeasurementObservation("s1", {"load": 40, "capacity": 100}, "STABLE"),
            MeasurementObservation("s2", {"load": 80, "capacity": 200}, "STABLE"),
            MeasurementObservation("s3", {"load": 120, "capacity": 300}, "STABLE"),
            MeasurementObservation("j1", {"load": 90, "capacity": 100}, "JERK"),
            MeasurementObservation("j2", {"load": 180, "capacity": 200}, "JERK"),
            MeasurementObservation("j3", {"load": 270, "capacity": 300}, "JERK"),
            MeasurementObservation("hs", {"load": 60, "capacity": 150}, "STABLE", heldout=True),
            MeasurementObservation("hj", {"load": 135, "capacity": 150}, "JERK", heldout=True),
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

    def test_generated_axis_must_add_value_beyond_raw_parents(self):
        engine = RepresentationGenesisEngine()
        value = RepresentationValueEvaluator(min_incremental_gain=0.05, min_heldout_support=2)
        assessments = [value.assess(axis, self._measurements()) for axis in engine.propose_axes(self._measurements())]
        eligible = [a for a in assessments if a.status == "INCREMENTAL_REPRESENTATION_VALUE"]
        self.assertTrue(eligible)
        self.assertTrue(all(a.incremental_gain >= 0.05 for a in eligible))

    def test_redundant_derived_axis_is_blocked(self):
        rows = [
            MeasurementObservation("s1", {"load": 10, "capacity": 100}, "STABLE"),
            MeasurementObservation("s2", {"load": 20, "capacity": 100}, "STABLE"),
            MeasurementObservation("j1", {"load": 90, "capacity": 100}, "JERK"),
            MeasurementObservation("j2", {"load": 100, "capacity": 100}, "JERK"),
            MeasurementObservation("hs", {"load": 15, "capacity": 100}, "STABLE", heldout=True),
            MeasurementObservation("hj", {"load": 95, "capacity": 100}, "JERK", heldout=True),
        ]
        axis = RepresentationAxis(
            axis_id="AXIS::RATIO::load|capacity",
            family="RATIO",
            inputs=("load", "capacity"),
            threshold=0.55,
            direction="GT",
            information_gain=1.0,
            train_support=4,
            positive_partition=("j1", "j2"),
            formula="load/capacity",
        )
        assessment = RepresentationValueEvaluator().assess(axis, rows)
        self.assertEqual(assessment.status, "REDUNDANT_WITH_PARENT_REPRESENTATION")
        self.assertAlmostEqual(assessment.incremental_gain, 0.0)

    def test_incremental_axis_enters_semantic_concept_and_heldout_law(self):
        runtime = PersistentCognitiveRuntime()
        cycle = runtime.cycle(
            TaskState(goal="explain startup residual", novelty=0.8),
            residuals=self._residuals(),
            measurements=self._measurements(),
            experiment_reference_values={"load": 100.0, "capacity": 150.0},
        )
        eligible_ids = {
            item.axis_id for item in cycle.representation_value
            if item.status == "INCREMENTAL_REPRESENTATION_VALUE"
        }
        self.assertTrue(eligible_ids)
        derived = [
            concept for concept in cycle.concepts
            if any(feature in eligible_ids for feature in concept.defining_features)
        ]
        self.assertTrue(derived)
        derived_ids = {concept.concept_id for concept in derived}
        self.assertTrue(any(
            law.concept_id in derived_ids and law.status == "BOUNDED_LAW"
            for law in cycle.laws
        ))
        self.assertTrue(cycle.intervention_proposals)
        self.assertTrue(all(p.status == "PROPOSAL_ONLY" for p in cycle.intervention_proposals))

    def test_experiment_proposal_crosses_ratio_threshold_without_becoming_evidence(self):
        axis = RepresentationAxis(
            axis_id="AXIS::RATIO::load|capacity",
            family="RATIO",
            inputs=("load", "capacity"),
            threshold=0.65,
            direction="GT",
            information_gain=1.0,
            train_support=6,
            positive_partition=("j1", "j2", "j3"),
            formula="load/capacity",
        )
        proposals = ExperimentGenesisEngine().propose(axis, {"load": 100.0, "capacity": 150.0})
        self.assertTrue(proposals)
        self.assertTrue(all(p.status == "PROPOSAL_ONLY" for p in proposals))
        self.assertTrue(any(p.manipulated_variable == "load" for p in proposals))

    def test_axis_budget_and_partition_quotient_prevent_explosion(self):
        engine = RepresentationGenesisEngine(axis_budget=2)
        axes = engine.propose_axes(self._measurements())
        self.assertLessEqual(len(axes), 2)
        partitions = [axis.positive_partition for axis in axes]
        self.assertEqual(len(partitions), len(set(partitions)))


if __name__ == "__main__":
    unittest.main()
