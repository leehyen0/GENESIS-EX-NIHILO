import json
import unittest

from arte_cognition.body_checkpoint import checkpoint_json, restore_json
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.experiment_genesis import InterventionProposal
from arte_cognition.world_coupling import WorldOutcomeReceipt


class LinearExecutor:
    def __init__(self, effects, source_id="source-a", context_id="ctx", challenge_id="challenge-a", epoch=1, matched=True):
        self.effects = dict(effects)
        self.source_id = source_id
        self.context_id = context_id
        self.challenge_id = challenge_id
        self.epoch = epoch
        self.matched = matched

    def execute(self, proposal, arm, value):
        coefficient = float(self.effects.get(proposal.axis_id, 0.0))
        outcome = coefficient * float(value)
        token = "budget-shared" if self.matched else f"budget-{arm.lower()}"
        return WorldOutcomeReceipt(
            receipt_id=f"receipt::{self.source_id}::{self.challenge_id}::{proposal.axis_id}::{arm}",
            experiment_id=proposal.experiment_id,
            axis_id=proposal.axis_id,
            arm=arm,
            intervention_value=float(value),
            outcome=outcome,
            source_id=self.source_id,
            context_id=self.context_id,
            challenge_id=self.challenge_id,
            epoch=self.epoch,
            budget_token=token,
            externally_generated=True,
        )


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
        reason="test world-coupled ranking",
    )


class WorldCouplingTests(unittest.TestCase):
    def test_world_outcomes_change_future_experiment_order(self):
        runtime = PersistentCognitiveRuntime()
        proposals = [proposal("AXIS::A"), proposal("AXIS::B")]
        self.assertEqual(runtime.rank_intervention_proposals(proposals)[0].axis_id, "AXIS::A")

        for source_id, challenge_id in (("source-1", "challenge-1"), ("source-2", "challenge-2")):
            executor = LinearExecutor(
                {"AXIS::A": 0.05, "AXIS::B": 1.5},
                source_id=source_id,
                challenge_id=challenge_id,
            )
            for item in proposals:
                runtime.execute_world_intervention(item, executor)

        ranked = runtime.rank_intervention_proposals(proposals)
        self.assertEqual(ranked[0].axis_id, "AXIS::B")
        summary = runtime.world_axis_summary("AXIS::B")
        self.assertEqual(summary.independent_evidence_classes, 2)
        self.assertGreater(summary.routing_score, runtime.world_axis_summary("AXIS::A").routing_score)

    def test_repeated_correlated_challenge_does_not_inflate_independence(self):
        runtime = PersistentCognitiveRuntime()
        item = proposal("AXIS::B")
        executor = LinearExecutor({"AXIS::B": 2.0}, source_id="same-source", challenge_id="same-challenge")
        runtime.execute_world_intervention(item, executor)
        runtime.execute_world_intervention(item, executor)
        summary = runtime.world_axis_summary("AXIS::B")
        self.assertEqual(summary.independent_evidence_classes, 1)
        self.assertEqual(len(runtime.world_coupling.pairs), 1)

    def test_unmatched_budget_cannot_steer_world_routing(self):
        runtime = PersistentCognitiveRuntime()
        proposals = [proposal("AXIS::A"), proposal("AXIS::B")]
        executor = LinearExecutor({"AXIS::B": 100.0}, matched=False)
        runtime.execute_world_intervention(proposals[1], executor)
        self.assertEqual(runtime.world_axis_summary("AXIS::B").routing_score, 0.0)
        self.assertEqual(runtime.rank_intervention_proposals(proposals)[0].axis_id, "AXIS::A")

    def test_checkpoint_restore_preserves_world_caused_behavior_change(self):
        runtime = PersistentCognitiveRuntime()
        proposals = [proposal("AXIS::A"), proposal("AXIS::B")]
        for source_id, challenge_id in (("source-1", "challenge-1"), ("source-2", "challenge-2")):
            executor = LinearExecutor(
                {"AXIS::A": 0.0, "AXIS::B": 1.0},
                source_id=source_id,
                challenge_id=challenge_id,
            )
            for item in proposals:
                runtime.execute_world_intervention(item, executor)

        encoded = checkpoint_json(runtime)
        payload = json.loads(encoded)
        self.assertEqual(payload["schema"], "arte.cognition_body_checkpoint/v2")
        restored = restore_json(encoded)
        self.assertEqual(restored.rank_intervention_proposals(proposals)[0].axis_id, "AXIS::B")
        self.assertEqual(
            runtime.world_axis_summary("AXIS::B"),
            restored.world_axis_summary("AXIS::B"),
        )

    def test_v1_checkpoint_remains_readable(self):
        runtime = PersistentCognitiveRuntime()
        payload = json.loads(checkpoint_json(runtime))
        payload["schema"] = "arte.cognition_body_checkpoint/v1"
        payload.pop("world_coupling", None)
        restored = restore_json(json.dumps(payload))
        self.assertEqual(restored.world_coupling.pairs, [])
        self.assertEqual(checkpoint_json(restored), checkpoint_json(restore_json(checkpoint_json(restored))))


if __name__ == "__main__":
    unittest.main()
