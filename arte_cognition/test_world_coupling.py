from dataclasses import replace
import json
import unittest

from arte_cognition.body_checkpoint import checkpoint_json, restore_json
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.experiment_genesis import InterventionProposal
from arte_cognition.world_coupling import (
    HMACWorldReceiptSigner,
    HMACWorldReceiptVerifier,
    WorldOutcomeReceipt,
)


UNIT_SECRET = b"arte-unit-world-receipt-key-v1"
UNIT_ISSUER = "unit-world-lab"


def unit_verifier():
    return HMACWorldReceiptVerifier({UNIT_ISSUER: UNIT_SECRET})


class LinearExecutor:
    def __init__(
        self,
        effects,
        source_id="source-a",
        context_id="ctx",
        challenge_id="challenge-a",
        epoch=1,
        matched=True,
        signed=True,
        tamper_after_sign=False,
    ):
        self.effects = dict(effects)
        self.source_id = source_id
        self.context_id = context_id
        self.challenge_id = challenge_id
        self.epoch = epoch
        self.matched = matched
        self.signed = signed
        self.tamper_after_sign = tamper_after_sign
        self.signer = HMACWorldReceiptSigner(UNIT_ISSUER, UNIT_SECRET)
        self.verifier = unit_verifier()

    def execute(self, proposal, arm, value):
        coefficient = float(self.effects.get(proposal.axis_id, 0.0))
        outcome = coefficient * float(value)
        token = "budget-shared" if self.matched else f"budget-{arm.lower()}"
        receipt = WorldOutcomeReceipt(
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
        if self.signed:
            receipt = self.signer.sign(receipt)
        if self.tamper_after_sign:
            receipt = replace(receipt, outcome=receipt.outcome + 100.0)
        return receipt


def enact(runtime, item, executor, use_verifier=True):
    return runtime.execute_world_intervention(
        item,
        executor,
        verifier=executor.verifier if use_verifier else None,
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
                enact(runtime, item, executor)
        self.assertEqual(runtime.rank_intervention_proposals(proposals)[0].axis_id, "AXIS::B")
        self.assertEqual(runtime.world_axis_summary("AXIS::B").independent_evidence_classes, 2)
        self.assertTrue(all(pair.authority_verified for pair in runtime.world_coupling.pairs))

    def test_unsigned_external_claim_cannot_steer_world_routing(self):
        runtime = PersistentCognitiveRuntime()
        proposals = [proposal("AXIS::A"), proposal("AXIS::B")]
        executor = LinearExecutor({"AXIS::B": 100.0}, signed=False)
        pair = enact(runtime, proposals[1], executor)
        self.assertTrue(pair.externally_generated)
        self.assertFalse(pair.authority_verified)
        self.assertEqual(runtime.world_axis_summary("AXIS::B").routing_score, 0.0)
        self.assertEqual(runtime.rank_intervention_proposals(proposals), proposals)

    def test_tampered_signed_receipt_cannot_steer_world_routing(self):
        runtime = PersistentCognitiveRuntime()
        proposals = [proposal("AXIS::A"), proposal("AXIS::B")]
        executor = LinearExecutor({"AXIS::B": 100.0}, tamper_after_sign=True)
        pair = enact(runtime, proposals[1], executor)
        self.assertFalse(pair.authority_verified)
        self.assertEqual(runtime.world_axis_summary("AXIS::B").routing_score, 0.0)

    def test_missing_verifier_cannot_steer_even_valid_signature(self):
        runtime = PersistentCognitiveRuntime()
        proposals = [proposal("AXIS::A"), proposal("AXIS::B")]
        executor = LinearExecutor({"AXIS::B": 100.0})
        pair = enact(runtime, proposals[1], executor, use_verifier=False)
        self.assertFalse(pair.authority_verified)
        self.assertEqual(runtime.rank_intervention_proposals(proposals), proposals)

    def test_repeated_correlated_challenge_does_not_inflate_independence(self):
        runtime = PersistentCognitiveRuntime()
        item = proposal("AXIS::B")
        executor = LinearExecutor({"AXIS::B": 2.0}, source_id="same-source", challenge_id="same-challenge")
        enact(runtime, item, executor)
        enact(runtime, item, executor)
        self.assertEqual(runtime.world_axis_summary("AXIS::B").independent_evidence_classes, 1)
        self.assertEqual(len(runtime.world_coupling.pairs), 1)

    def test_unmatched_budget_cannot_steer_world_routing(self):
        runtime = PersistentCognitiveRuntime()
        proposals = [proposal("AXIS::A"), proposal("AXIS::B")]
        executor = LinearExecutor({"AXIS::B": 100.0}, matched=False)
        pair = enact(runtime, proposals[1], executor)
        self.assertTrue(pair.authority_verified)
        self.assertFalse(pair.matched_budget)
        self.assertEqual(runtime.world_axis_summary("AXIS::B").routing_score, 0.0)
        self.assertEqual(runtime.rank_intervention_proposals(proposals)[0].axis_id, "AXIS::A")

    def test_checkpoint_restore_requires_external_reverification(self):
        runtime = PersistentCognitiveRuntime()
        proposals = [proposal("AXIS::A"), proposal("AXIS::B")]
        for source_id, challenge_id in (("source-1", "challenge-1"), ("source-2", "challenge-2")):
            executor = LinearExecutor(
                {"AXIS::A": 0.0, "AXIS::B": 1.0},
                source_id=source_id,
                challenge_id=challenge_id,
            )
            for item in proposals:
                enact(runtime, item, executor)

        encoded = checkpoint_json(runtime)
        payload = json.loads(encoded)
        self.assertEqual(payload["schema"], "arte.cognition_body_checkpoint/v3")
        self.assertNotIn(UNIT_SECRET.hex(), encoded)
        self.assertNotIn("trusted_keys", encoded)

        unverified_descendant = restore_json(encoded)
        self.assertTrue(unverified_descendant.world_coupling.pairs)
        self.assertTrue(all(not pair.authority_verified for pair in unverified_descendant.world_coupling.pairs))
        self.assertEqual(unverified_descendant.rank_intervention_proposals(proposals), proposals)

        restored = restore_json(encoded, world_verifier=unit_verifier())
        self.assertEqual(restored.rank_intervention_proposals(proposals)[0].axis_id, "AXIS::B")
        self.assertEqual(runtime.world_axis_summary("AXIS::B"), restored.world_axis_summary("AXIS::B"))
        self.assertTrue(all(pair.authority_verified for pair in restored.world_coupling.pairs))

    def test_checkpoint_boolean_or_pair_tamper_cannot_self_authorize(self):
        runtime = PersistentCognitiveRuntime()
        item = proposal("AXIS::B")
        executor = LinearExecutor({"AXIS::B": 2.0})
        enact(runtime, item, executor)
        payload = json.loads(checkpoint_json(runtime))
        pair = payload["world_coupling"]["pairs"][0]
        pair["authority_verified"] = True
        pair["high_outcome"] = 999.0
        restored = restore_json(json.dumps(payload), world_verifier=unit_verifier())
        self.assertFalse(restored.world_coupling.pairs[0].authority_verified)
        self.assertEqual(restored.world_axis_summary("AXIS::B").routing_score, 0.0)

    def test_v1_checkpoint_remains_readable(self):
        runtime = PersistentCognitiveRuntime()
        payload = json.loads(checkpoint_json(runtime))
        payload["schema"] = "arte.cognition_body_checkpoint/v1"
        payload.pop("world_coupling", None)
        restored = restore_json(json.dumps(payload))
        self.assertEqual(restored.world_coupling.pairs, [])
        self.assertEqual(checkpoint_json(restored), checkpoint_json(restore_json(checkpoint_json(restored))))

    def test_v2_world_evidence_is_loaded_but_deauthorized(self):
        runtime = PersistentCognitiveRuntime()
        proposals = [proposal("AXIS::A"), proposal("AXIS::B")]
        for index in (1, 2):
            executor = LinearExecutor(
                {"AXIS::A": 0.0, "AXIS::B": 2.0},
                source_id=f"source-{index}",
                challenge_id=f"challenge-{index}",
            )
            for item in proposals:
                enact(runtime, item, executor)
        self.assertEqual(runtime.rank_intervention_proposals(proposals)[0].axis_id, "AXIS::B")
        payload = json.loads(checkpoint_json(runtime))
        payload["schema"] = "arte.cognition_body_checkpoint/v2"
        for pair in payload["world_coupling"]["pairs"]:
            pair.pop("issuer_id", None)
            pair.pop("authority_verified", None)
            pair.pop("low_receipt", None)
            pair.pop("high_receipt", None)
        restored = restore_json(json.dumps(payload), world_verifier=unit_verifier())
        self.assertTrue(restored.world_coupling.pairs)
        self.assertTrue(all(not pair.authority_verified for pair in restored.world_coupling.pairs))
        self.assertEqual(restored.rank_intervention_proposals(proposals), proposals)

    def test_opposite_regimes_learn_opposite_intervention_preferences(self):
        runtime = PersistentCognitiveRuntime()
        proposals = [proposal("AXIS::A"), proposal("AXIS::B")]
        for index in (1, 2):
            calm = LinearExecutor(
                {"AXIS::A": 2.0, "AXIS::B": 0.05},
                source_id=f"calm-source-{index}", context_id="CALM", challenge_id=f"calm-challenge-{index}",
            )
            turbulent = LinearExecutor(
                {"AXIS::A": 0.05, "AXIS::B": 2.0},
                source_id=f"turbulent-source-{index}", context_id="TURBULENT", challenge_id=f"turbulent-challenge-{index}",
            )
            for item in proposals:
                enact(runtime, item, calm)
                enact(runtime, item, turbulent)
        self.assertEqual(runtime.rank_intervention_proposals(proposals, context_id="CALM")[0].axis_id, "AXIS::A")
        self.assertEqual(runtime.rank_intervention_proposals(proposals, context_id="TURBULENT")[0].axis_id, "AXIS::B")

    def test_same_evaluator_challenge_across_regimes_is_not_globally_independent(self):
        runtime = PersistentCognitiveRuntime()
        item = proposal("AXIS::A")
        for context_id in ("CALM", "TURBULENT"):
            executor = LinearExecutor(
                {"AXIS::A": 1.0}, source_id="shared-source", context_id=context_id, challenge_id="shared-challenge"
            )
            enact(runtime, item, executor)
        self.assertEqual(len(runtime.world_coupling.pairs), 2)
        self.assertEqual(runtime.world_axis_summary("AXIS::A", context_id="CALM").independent_evidence_classes, 1)
        self.assertEqual(runtime.world_axis_summary("AXIS::A", context_id="TURBULENT").independent_evidence_classes, 1)
        self.assertEqual(runtime.world_axis_summary("AXIS::A").independent_evidence_classes, 1)

    def test_context_conditioning_survives_descendant_reverification(self):
        runtime = PersistentCognitiveRuntime()
        proposals = [proposal("AXIS::A"), proposal("AXIS::B")]
        for index in (1, 2):
            executor = LinearExecutor(
                {"AXIS::A": 0.0, "AXIS::B": 1.5},
                source_id=f"source-{index}", context_id="REGIME-B", challenge_id=f"challenge-{index}",
            )
            for item in proposals:
                enact(runtime, item, executor)
        restored = restore_json(checkpoint_json(runtime), world_verifier=unit_verifier())
        self.assertEqual(restored.rank_intervention_proposals(proposals, context_id="REGIME-B")[0].axis_id, "AXIS::B")
        self.assertEqual(
            restored.world_axis_summary("AXIS::B", context_id="REGIME-B"),
            runtime.world_axis_summary("AXIS::B", context_id="REGIME-B"),
        )

    def test_conflicting_supported_regimes_block_contextless_transport(self):
        runtime = PersistentCognitiveRuntime()
        proposals = [proposal("AXIS::A"), proposal("AXIS::B")]
        for index in (1, 2):
            calm = LinearExecutor(
                {"AXIS::A": 2.0, "AXIS::B": 0.05},
                source_id=f"calm-source-{index}", context_id="CALM", challenge_id=f"calm-challenge-{index}",
            )
            turbulent = LinearExecutor(
                {"AXIS::A": 0.05, "AXIS::B": 2.0},
                source_id=f"turb-source-{index}", context_id="TURBULENT", challenge_id=f"turb-challenge-{index}",
            )
            for item in proposals:
                enact(runtime, item, calm)
                enact(runtime, item, turbulent)
        assessment = runtime.assess_world_transport(proposals)
        self.assertEqual(assessment.status, "REGIME_CONFLICT_BLOCK_GLOBAL_TRANSPORT")
        self.assertFalse(assessment.safe_for_global_transport)
        self.assertEqual(runtime.rank_intervention_proposals(proposals), proposals)

    def test_agreeing_supported_regimes_allow_bounded_global_transport(self):
        runtime = PersistentCognitiveRuntime()
        proposals = [proposal("AXIS::A"), proposal("AXIS::B")]
        for context_id in ("CALM", "TURBULENT"):
            for index in (1, 2):
                executor = LinearExecutor(
                    {"AXIS::A": 0.05, "AXIS::B": 2.0},
                    source_id=f"{context_id}-source-{index}",
                    context_id=context_id,
                    challenge_id=f"{context_id}-challenge-{index}",
                )
                for item in proposals:
                    enact(runtime, item, executor)
        assessment = runtime.assess_world_transport(proposals)
        self.assertEqual(assessment.status, "GLOBAL_TRANSPORT_SUPPORTED_BOUNDED")
        self.assertTrue(assessment.safe_for_global_transport)
        self.assertEqual(runtime.rank_intervention_proposals(proposals)[0].axis_id, "AXIS::B")

    def test_transport_abstention_survives_descendant_reverification(self):
        runtime = PersistentCognitiveRuntime()
        proposals = [proposal("AXIS::A"), proposal("AXIS::B")]
        for index in (1, 2):
            calm = LinearExecutor(
                {"AXIS::A": 2.0, "AXIS::B": 0.0},
                source_id=f"calm-source-{index}", context_id="CALM", challenge_id=f"calm-challenge-{index}",
            )
            turbulent = LinearExecutor(
                {"AXIS::A": 0.0, "AXIS::B": 2.0},
                source_id=f"turb-source-{index}", context_id="TURBULENT", challenge_id=f"turb-challenge-{index}",
            )
            for item in proposals:
                enact(runtime, item, calm)
                enact(runtime, item, turbulent)
        before = runtime.assess_world_transport(proposals)
        restored = restore_json(checkpoint_json(runtime), world_verifier=unit_verifier())
        after = restored.assess_world_transport(proposals)
        self.assertEqual(before, after)
        self.assertFalse(after.safe_for_global_transport)
        self.assertEqual(restored.rank_intervention_proposals(proposals), proposals)


if __name__ == "__main__":
    unittest.main()
