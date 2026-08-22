import unittest

from arte_cognition.experiment_genesis import InterventionProposal
from arte_cognition.world_action_policy import EvidenceBoundWorldActionPolicy
from arte_cognition.world_coupling import WorldCouplingEngine, WorldOutcomePair


def proposal(axis_id):
    return InterventionProposal(
        experiment_id=f"EXPERIMENT::{axis_id}::x",
        axis_id=axis_id,
        manipulated_variable="x",
        held_fixed=(),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="LE_THRESHOLD",
        predicted_high_side="GT_THRESHOLD",
        reason="action-policy test",
    )


def pair(axis_id, source_id, challenge_id, context_id="CTX", effect=1.0):
    return WorldOutcomePair(
        pair_id=f"PAIR::{axis_id}::{context_id}::{source_id}::{challenge_id}",
        experiment_id=f"EXPERIMENT::{axis_id}::x",
        axis_id=axis_id,
        source_id=source_id,
        context_id=context_id,
        challenge_id=challenge_id,
        epoch=1,
        low_outcome=0.0,
        high_outcome=float(effect),
        low_value=0.0,
        high_value=1.0,
        matched_budget=True,
        externally_generated=True,
        issuer_id="issuer",
        authority_verified=True,
    )


class WorldActionPolicyTests(unittest.TestCase):
    def test_generated_proposal_is_exploration_only_before_world_evidence(self):
        world = WorldCouplingEngine()
        decision = EvidenceBoundWorldActionPolicy().select(
            [proposal("AXIS::A")],
            world,
            context_id="CTX",
        )
        self.assertEqual(decision.status, "EXPLORE_ONLY_NO_WORLD_SUPPORTED_ACTION")
        self.assertIsNone(decision.proposal)

    def test_authenticated_independent_outcomes_promote_future_action_choice(self):
        world = WorldCouplingEngine()
        proposals = [proposal("AXIS::A"), proposal("AXIS::B")]
        world.record_pair(pair("AXIS::B", "source-1", "challenge-1", effect=1.0))
        world.record_pair(pair("AXIS::B", "source-2", "challenge-2", effect=1.5))
        decision = EvidenceBoundWorldActionPolicy().select(proposals, world, context_id="CTX")
        self.assertEqual(decision.status, "WORLD_SUPPORTED_ACTION")
        self.assertEqual(decision.proposal.axis_id, "AXIS::B")
        self.assertEqual(decision.independent_evidence_classes, 2)
        self.assertGreater(decision.routing_score, 0.0)

    def test_conflicting_supported_regimes_force_contextless_action_abstention(self):
        world = WorldCouplingEngine()
        proposals = [proposal("AXIS::A"), proposal("AXIS::B")]
        for index in (1, 2):
            world.record_pair(pair("AXIS::A", f"calm-{index}", f"calm-c-{index}", context_id="CALM", effect=2.0))
            world.record_pair(pair("AXIS::B", f"turb-{index}", f"turb-c-{index}", context_id="TURBULENT", effect=2.0))
        policy = EvidenceBoundWorldActionPolicy()
        global_decision = policy.select(proposals, world)
        self.assertEqual(global_decision.status, "ABSTAIN_REGIME_CONFLICT")
        self.assertIsNone(global_decision.proposal)
        self.assertEqual(policy.select(proposals, world, context_id="CALM").proposal.axis_id, "AXIS::A")
        self.assertEqual(policy.select(proposals, world, context_id="TURBULENT").proposal.axis_id, "AXIS::B")


if __name__ == "__main__":
    unittest.main()
