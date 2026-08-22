from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple
import hashlib
import hmac
import json

from .experiment_genesis import InterventionProposal


@dataclass(frozen=True)
class WorldOutcomeReceipt:
    """Outcome returned by an executor that owns the world/evaluation surface.

    `externally_generated` is descriptive metadata only; it is never sufficient
    for learning authority. A receipt can steer the BODY only when a separately
    supplied verifier authenticates its issuer and exact payload.
    """

    receipt_id: str
    experiment_id: str
    axis_id: str
    arm: str
    intervention_value: float
    outcome: float
    source_id: str
    context_id: str
    challenge_id: str
    epoch: int
    budget_token: str
    externally_generated: bool = True
    issuer_id: str = "UNSIGNED"
    signature: str = ""


def receipt_payload(receipt: WorldOutcomeReceipt) -> bytes:
    data = {
        "receipt_id": receipt.receipt_id,
        "experiment_id": receipt.experiment_id,
        "axis_id": receipt.axis_id,
        "arm": receipt.arm,
        "intervention_value": float(receipt.intervention_value),
        "outcome": float(receipt.outcome),
        "source_id": receipt.source_id,
        "context_id": receipt.context_id,
        "challenge_id": receipt.challenge_id,
        "epoch": int(receipt.epoch),
        "budget_token": receipt.budget_token,
        "externally_generated": bool(receipt.externally_generated),
        "issuer_id": receipt.issuer_id,
    }
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


class WorldReceiptVerifier(Protocol):
    def verify(self, receipt: WorldOutcomeReceipt) -> bool:
        ...


class HMACWorldReceiptSigner:
    """Reference source-side signer; secret is never BODY checkpoint state."""

    def __init__(self, issuer_id: str, secret: bytes) -> None:
        if not issuer_id:
            raise ValueError("issuer_id is required")
        if not secret:
            raise ValueError("signing secret is required")
        self.issuer_id = issuer_id
        self._secret = bytes(secret)

    def sign(self, receipt: WorldOutcomeReceipt) -> WorldOutcomeReceipt:
        unsigned = replace(receipt, issuer_id=self.issuer_id, signature="")
        signature = hmac.new(self._secret, receipt_payload(unsigned), hashlib.sha256).hexdigest()
        return replace(unsigned, signature=signature)


class HMACWorldReceiptVerifier:
    """LAB-side verifier with a frozen issuer->key trust map."""

    def __init__(self, trusted_keys: Mapping[str, bytes]) -> None:
        self._trusted_keys = {
            str(issuer): bytes(secret)
            for issuer, secret in trusted_keys.items()
            if issuer and secret
        }

    def verify(self, receipt: WorldOutcomeReceipt) -> bool:
        secret = self._trusted_keys.get(receipt.issuer_id)
        if secret is None or not receipt.signature:
            return False
        expected = hmac.new(
            secret,
            receipt_payload(replace(receipt, signature="")),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, receipt.signature)


@dataclass(frozen=True)
class WorldOutcomePair:
    pair_id: str
    experiment_id: str
    axis_id: str
    source_id: str
    context_id: str
    challenge_id: str
    epoch: int
    low_outcome: float
    high_outcome: float
    low_value: float
    high_value: float
    matched_budget: bool
    externally_generated: bool
    issuer_id: str = "UNVERIFIED"
    authority_verified: bool = False
    low_receipt: Optional[WorldOutcomeReceipt] = None
    high_receipt: Optional[WorldOutcomeReceipt] = None

    @property
    def effect(self) -> float:
        return float(self.high_outcome) - float(self.low_outcome)

    @property
    def independence_key(self) -> Tuple[str, str, str]:
        return (self.issuer_id, self.source_id, self.challenge_id)

    @property
    def contextual_evidence_key(self) -> Tuple[str, str, str, str]:
        return (self.context_id, self.issuer_id, self.source_id, self.challenge_id)


@dataclass(frozen=True)
class AxisWorldSummary:
    axis_id: str
    context_id: Optional[str]
    independent_evidence_classes: int
    mean_effect: float
    mean_abs_effect: float
    routing_score: float


@dataclass(frozen=True)
class WorldTransportAssessment:
    status: str
    safe_for_global_transport: bool
    evidence_ready_contexts: Tuple[str, ...]
    top_axis_by_context: Tuple[Tuple[str, str], ...]
    reasons: Tuple[str, ...]


class WorldExecutor(Protocol):
    def execute(
        self,
        proposal: InterventionProposal,
        arm: str,
        value: float,
    ) -> WorldOutcomeReceipt:
        ...


class WorldCouplingEngine:
    """Consume authenticated consequences and preserve re-verifiable evidence.

    Signed raw receipts are retained in the BODY evidence lineage, but the secret
    trust material is not. On restart, cached `authority_verified` flags are not
    trusted: a LAB verifier must re-authenticate the stored receipts and their
    consistency with the derived pair before they regain learning authority.
    """

    def __init__(self, min_independent_classes: int = 2) -> None:
        self.min_independent_classes = max(1, int(min_independent_classes))
        self.pairs: List[WorldOutcomePair] = []

    @staticmethod
    def _authoritative(pair: WorldOutcomePair) -> bool:
        return bool(pair.matched_budget and pair.externally_generated and pair.authority_verified)

    @staticmethod
    def _pair_matches_receipts(pair: WorldOutcomePair) -> bool:
        low, high = pair.low_receipt, pair.high_receipt
        if low is None or high is None:
            return False
        if low.arm.upper() != "LOW" or high.arm.upper() != "HIGH":
            return False
        common_low = (
            low.experiment_id,
            low.axis_id,
            low.issuer_id,
            low.source_id,
            low.context_id,
            low.challenge_id,
            int(low.epoch),
        )
        common_high = (
            high.experiment_id,
            high.axis_id,
            high.issuer_id,
            high.source_id,
            high.context_id,
            high.challenge_id,
            int(high.epoch),
        )
        if common_low != common_high:
            return False
        expected_pair_id = (
            f"PAIR::{low.experiment_id}::{low.context_id}::"
            f"{low.issuer_id}::{low.source_id}::{low.challenge_id}"
        )
        return bool(
            pair.pair_id == expected_pair_id
            and pair.experiment_id == low.experiment_id
            and pair.axis_id == low.axis_id
            and pair.issuer_id == low.issuer_id
            and pair.source_id == low.source_id
            and pair.context_id == low.context_id
            and pair.challenge_id == low.challenge_id
            and int(pair.epoch) == int(low.epoch)
            and float(pair.low_outcome) == float(low.outcome)
            and float(pair.high_outcome) == float(high.outcome)
            and float(pair.low_value) == float(low.intervention_value)
            and float(pair.high_value) == float(high.intervention_value)
            and bool(pair.externally_generated) == bool(low.externally_generated and high.externally_generated)
            and bool(pair.matched_budget) == bool(low.budget_token and low.budget_token == high.budget_token)
        )

    def _contextual_evidence_keys(self, axis_id: str, context_id: str) -> set[Tuple[str, str, str, str]]:
        return {
            pair.contextual_evidence_key
            for pair in self.pairs
            if pair.axis_id == axis_id
            and pair.context_id == context_id
            and self._authoritative(pair)
        }

    def record_pair(self, pair: WorldOutcomePair) -> bool:
        if pair.pair_id in {item.pair_id for item in self.pairs}:
            return False
        if self._authoritative(pair):
            if pair.contextual_evidence_key in self._contextual_evidence_keys(pair.axis_id, pair.context_id):
                return False
        self.pairs.append(pair)
        return True

    def execute(
        self,
        proposal: InterventionProposal,
        executor: WorldExecutor,
        verifier: Optional[WorldReceiptVerifier] = None,
    ) -> WorldOutcomePair:
        low = executor.execute(proposal, "LOW", float(proposal.low_value))
        high = executor.execute(proposal, "HIGH", float(proposal.high_value))

        for receipt, arm in ((low, "LOW"), (high, "HIGH")):
            if receipt.arm.upper() != arm:
                raise ValueError(f"executor returned wrong arm for {arm}")
            if receipt.experiment_id != proposal.experiment_id:
                raise ValueError("executor receipt experiment_id mismatch")
            if receipt.axis_id != proposal.axis_id:
                raise ValueError("executor receipt axis_id mismatch")

        identity_low = (low.issuer_id, low.source_id, low.context_id, low.challenge_id, low.epoch)
        identity_high = (high.issuer_id, high.source_id, high.context_id, high.challenge_id, high.epoch)
        if identity_low != identity_high:
            raise ValueError("LOW/HIGH receipts must belong to one authenticated world challenge")

        provisional = WorldOutcomePair(
            pair_id=(
                f"PAIR::{proposal.experiment_id}::{low.context_id}::"
                f"{low.issuer_id}::{low.source_id}::{low.challenge_id}"
            ),
            experiment_id=proposal.experiment_id,
            axis_id=proposal.axis_id,
            source_id=low.source_id,
            context_id=low.context_id,
            challenge_id=low.challenge_id,
            epoch=int(low.epoch),
            low_outcome=float(low.outcome),
            high_outcome=float(high.outcome),
            low_value=float(low.intervention_value),
            high_value=float(high.intervention_value),
            matched_budget=bool(low.budget_token and low.budget_token == high.budget_token),
            externally_generated=bool(low.externally_generated and high.externally_generated),
            issuer_id=low.issuer_id,
            authority_verified=False,
            low_receipt=low,
            high_receipt=high,
        )
        authority_verified = bool(
            verifier is not None
            and verifier.verify(low)
            and verifier.verify(high)
            and self._pair_matches_receipts(provisional)
        )
        pair = replace(provisional, authority_verified=authority_verified)
        self.record_pair(pair)
        return pair

    def summary(self, axis_id: str, context_id: Optional[str] = None) -> AxisWorldSummary:
        valid = [
            pair for pair in self.pairs
            if pair.axis_id == axis_id
            and self._authoritative(pair)
            and (context_id is None or pair.context_id == context_id)
        ]
        by_key: Dict[Tuple[str, str, str], WorldOutcomePair] = {}
        for pair in valid:
            by_key.setdefault(pair.independence_key, pair)
        unique = list(by_key.values())
        if not unique:
            return AxisWorldSummary(axis_id, context_id, 0, 0.0, 0.0, 0.0)
        effects = [pair.effect for pair in unique]
        mean_effect = sum(effects) / len(effects)
        mean_abs_effect = sum(abs(value) for value in effects) / len(effects)
        independence_factor = min(1.0, len(unique) / self.min_independent_classes)
        return AxisWorldSummary(
            axis_id=axis_id,
            context_id=context_id,
            independent_evidence_classes=len(unique),
            mean_effect=mean_effect,
            mean_abs_effect=mean_abs_effect,
            routing_score=mean_abs_effect * independence_factor,
        )

    def _rank_without_transport_guard(
        self,
        proposals: Sequence[InterventionProposal],
        context_id: Optional[str],
    ) -> List[InterventionProposal]:
        indexed = list(enumerate(proposals))
        return [
            proposal
            for _, proposal in sorted(
                indexed,
                key=lambda item: (
                    -self.summary(item[1].axis_id, context_id=context_id).routing_score,
                    item[0],
                ),
            )
        ]

    def assess_transport(self, proposals: Sequence[InterventionProposal]) -> WorldTransportAssessment:
        contexts = sorted({pair.context_id for pair in self.pairs if self._authoritative(pair)})
        evidence_ready: List[str] = []
        top_by_context: List[Tuple[str, str]] = []
        for context in contexts:
            summaries = [self.summary(item.axis_id, context_id=context) for item in proposals]
            if not summaries or max(s.independent_evidence_classes for s in summaries) < self.min_independent_classes:
                continue
            ranked = self._rank_without_transport_guard(proposals, context)
            if ranked:
                evidence_ready.append(context)
                top_by_context.append((context, ranked[0].axis_id))

        if len(evidence_ready) < 2:
            return WorldTransportAssessment(
                "GLOBAL_TRANSPORT_NOT_CONTRADICTED",
                True,
                tuple(evidence_ready),
                tuple(top_by_context),
                ("fewer than two independently supported regimes available",),
            )
        if len({axis_id for _, axis_id in top_by_context}) > 1:
            return WorldTransportAssessment(
                "REGIME_CONFLICT_BLOCK_GLOBAL_TRANSPORT",
                False,
                tuple(evidence_ready),
                tuple(top_by_context),
                ("independently supported regimes prefer different interventions",),
            )
        return WorldTransportAssessment(
            "GLOBAL_TRANSPORT_SUPPORTED_BOUNDED",
            True,
            tuple(evidence_ready),
            tuple(top_by_context),
            ("supported regimes agree on the same preferred intervention",),
        )

    def rank_proposals(
        self,
        proposals: Sequence[InterventionProposal],
        context_id: Optional[str] = None,
    ) -> List[InterventionProposal]:
        if context_id is None and not self.assess_transport(proposals).safe_for_global_transport:
            return list(proposals)
        return self._rank_without_transport_guard(proposals, context_id)

    def restore_pairs(
        self,
        pairs: Iterable[WorldOutcomePair],
        verifier: Optional[WorldReceiptVerifier] = None,
    ) -> None:
        self.pairs = []
        seen_ids = set()
        for raw_pair in pairs:
            if raw_pair.pair_id in seen_ids:
                continue
            seen_ids.add(raw_pair.pair_id)
            reverified = bool(
                verifier is not None
                and raw_pair.low_receipt is not None
                and raw_pair.high_receipt is not None
                and verifier.verify(raw_pair.low_receipt)
                and verifier.verify(raw_pair.high_receipt)
                and self._pair_matches_receipts(raw_pair)
            )
            self.pairs.append(replace(raw_pair, authority_verified=reverified))
