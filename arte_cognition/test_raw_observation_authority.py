from __future__ import annotations

from dataclasses import replace
import unittest

from arte_cognition.raw_observation_authority import (
    HMACRawObservationSigner,
    HMACRawObservationVerifier,
    RawObservationReceipt,
    corroborated_raw_observations,
)
from arte_cognition.world_coupling import WorldOutcomePair


class RawObservationAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.keys = {"issuer-a": b"raw-a", "issuer-b": b"raw-b"}
        self.classes = {"issuer-a": "class-a", "issuer-b": "class-b"}
        self.signers = {
            issuer: HMACRawObservationSigner(issuer, secret)
            for issuer, secret in self.keys.items()
        }
        self.verifier = HMACRawObservationVerifier(self.keys, self.classes)

    @staticmethod
    def pair(issuer, independence, source, challenge):
        return WorldOutcomePair(
            pair_id=f"PAIR::probe::ctx::{issuer}::{source}::{challenge}",
            experiment_id="probe",
            axis_id="axis",
            source_id=source,
            context_id="ctx",
            challenge_id=challenge,
            epoch=1,
            low_outcome=0.0,
            high_outcome=1.0,
            low_value=0.0,
            high_value=1.0,
            matched_budget=True,
            externally_generated=True,
            issuer_id=issuer,
            independence_class_id=independence,
            authority_verified=True,
        )

    def receipt(self, issuer, source, challenge, values=(("opaque-a", 1.0), ("opaque-b", 2.0))):
        return self.signers[issuer].sign(RawObservationReceipt(
            observation_id=f"raw::{issuer}::{challenge}",
            intervention_id="probe",
            channel_values=values,
            source_id=source,
            context_id="ctx",
            challenge_id=challenge,
            epoch=1,
            externally_generated=True,
        ))

    def test_one_independence_class_cannot_authorize_raw_row(self):
        receipt = self.receipt("issuer-a", "source-a", "challenge-a")
        result = corroborated_raw_observations(
            [receipt],
            [self.pair("issuer-a", "class-a", "source-a", "challenge-a")],
            self.verifier,
            min_independent_classes=2,
        )
        self.assertEqual(result, {})

    def test_two_independent_classes_must_agree_exactly(self):
        receipts = [
            self.receipt("issuer-a", "source-a", "challenge-a"),
            self.receipt("issuer-b", "source-b", "challenge-b"),
        ]
        pairs = [
            self.pair("issuer-a", "class-a", "source-a", "challenge-a"),
            self.pair("issuer-b", "class-b", "source-b", "challenge-b"),
        ]
        result = corroborated_raw_observations(receipts, pairs, self.verifier, 2)
        self.assertEqual(result, {"probe": {"opaque-a": 1.0, "opaque-b": 2.0}})

    def test_post_signature_tamper_cannot_supply_quorum(self):
        honest_a = self.receipt("issuer-a", "source-a", "challenge-a")
        honest_b = self.receipt("issuer-b", "source-b", "challenge-b")
        tampered_b = replace(
            honest_b,
            observation_id=honest_b.observation_id + "::tampered",
            channel_values=(("opaque-a", 999.0), ("opaque-b", 2.0)),
        )
        pairs = [
            self.pair("issuer-a", "class-a", "source-a", "challenge-a"),
            self.pair("issuer-b", "class-b", "source-b", "challenge-b"),
        ]
        result = corroborated_raw_observations([honest_a, tampered_b], pairs, self.verifier, 2)
        self.assertEqual(result, {})

    def test_valid_signatures_with_conflicting_raw_values_fail_closed(self):
        receipt_a = self.receipt("issuer-a", "source-a", "challenge-a")
        receipt_b = self.receipt(
            "issuer-b",
            "source-b",
            "challenge-b",
            values=(("opaque-a", 7.0), ("opaque-b", 2.0)),
        )
        pairs = [
            self.pair("issuer-a", "class-a", "source-a", "challenge-a"),
            self.pair("issuer-b", "class-b", "source-b", "challenge-b"),
        ]
        result = corroborated_raw_observations([receipt_a, receipt_b], pairs, self.verifier, 2)
        self.assertEqual(result, {})

    def test_signed_raw_receipt_must_bind_to_authoritative_world_identity(self):
        receipts = [
            self.receipt("issuer-a", "source-a", "challenge-a"),
            self.receipt("issuer-b", "source-b", "challenge-b"),
        ]
        pairs = [
            self.pair("issuer-a", "class-a", "source-a", "wrong-challenge"),
            self.pair("issuer-b", "class-b", "source-b", "challenge-b"),
        ]
        result = corroborated_raw_observations(receipts, pairs, self.verifier, 2)
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
