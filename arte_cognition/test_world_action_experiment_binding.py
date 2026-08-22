import unittest

from arte_cognition.experiment_genesis import InterventionProposal
from arte_cognition.world_action_policy import EvidenceBoundWorldActionPolicy
from arte_cognition.world_coupling import WorldCouplingEngine, WorldOutcomePair


def proposal(axis_id, variable):
    return InterventionProposal(
        experiment_id=f"EXPERIMENT::{axis_id}::{variable}",
        axis_id=axis_id,
        manipulated_variable=variable,
        held_fixed=(),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="LE_THRESHOLD",
        predicted_high_side="GT_THRESHOLD",
        reason="exact-experiment binding regression",
    )


def evidence(axis_id, variable, source_id, effect):
    return WorldOutcomePair(
        pair_id=f"PAIR::{axis_id}::{variable}::{source_id}",
        experiment_id=f"EXPERIMENT::{axis_id}::{variable}",
        axis_id=axis_id,
        source_id=source_id,
        context_id="CTX",
        challenge_id=f"challenge::{source_id}",
        epoch=1,
        low_outcome=0.0,
        high_outcome=float(effect),
        low_value=0.0,
        high_value=1.0,
        matched_budget=True,
        externally_generated=True,
        issuer_id=f"issuer::{source_id}",
        independence_class_id=f"class::{source_id}",
        authority_verified=True,
    )


class ExactExperimentBindingTests(unittest.TestCase):
    def test_same_axis_evidence_does_not_promote_untested_experiment(self):
        world = WorldCouplingEngine()
        untested = proposal("AXIS::A", "x")
        tested = proposal("AXIS::A", "y")
        world.record_pair(evidence("AXIS::A", "y", "source-1", 1.0))
        world.record_pair(evidence("AXIS::A", "y", "source-2", 1.5))

        decision = EvidenceBoundWorldActionPolicy().select(
            [untested, tested],
            world,
            context_id="CTX",
        )
        self.assertEqual(decision.status, "WORLD_SUPPORTED_ACTION")
        self.assertEqual(decision.proposal.experiment_id, tested.experiment_id)
        self.assertNotEqual(decision.proposal.experiment_id, untested.experiment_id)
        self.assertEqual(decision.independent_evidence_classes, 2)


if __name__ == "__main__":
    unittest.main()
