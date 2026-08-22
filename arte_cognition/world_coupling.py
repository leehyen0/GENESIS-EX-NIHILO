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
    supplied verifier authenticates its issuer and payload.
    """

    receipt_id: str
    experiment_id: str
    axis_id: str
    arm: str  # LOW or HIGH
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
    """Canonical authenticated payload; signature itself is excluded."""
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
    """Reference source-side signer for authenticated evaluator receipts.

    The signing secret belongs to the evaluator/source side and is not part of the
    persistent BODY checkpoint. HMAC gives payload integrity/authentication under
    the configured trust key; it does not establish independent organization
    custody by itself.
    """

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
        expected = hmac.new(secret, receipt_payload(replace(receipt, signature="")), hashlib.sha256).hexdigest()
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

    @property
    def effect(self) -> float:
        return float(self.high_outcome) - float(self.low_outcome)

    @property
    def independence_key(self) -> Tuple[str, str, str]:
        # Context is deliberately excluded. Replaying one evaluator/challenge in
        # several regimes can teach regime-specific behavior, but cannot become
        # several globally independent sources. Issuer identity is included so two
        # separately trusted issuers do not collide merely on source labels.
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
    """Consume authenticated external consequences and change future behavior.

    A raw receipt is not evidence merely because it says it is external. Only
    matched-budget pairs whose LOW and HIGH receipts authenticate under a supplied
    LAB verifier can influence world summaries, transport decisions or future
    intervention ranking. Unverified pairs remain in the audit lineage but have
    zero learning authority.
    """

    def __init__(self, min_independent_classes: int = 2) -> None:
        self.min_independent_classes = max(1, int(min_independent_classes))
        self.pairs: List[WorldOutcomePair] = []

    @staticmethod
    def _authoritative(pair: WorldOutcomePair) -> bool:
        return bool(pair.matched_budget and pair.externally_generated and pair.authority_verified)

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

        # Invalid-for-routing receipts are retained once for audit/failure memory.
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

        identity_low = (
            low.issuer_id,
            low.source_id,
            low.context_id,
            low.challenge_id,
            low.epoch,
        )
        identity_high = (
            high.issuer_id,
            high.source_id,
            high.context_id,
            high.challenge_id,
            high.epoch,
        )
        if identity_low != identity_high:
            raise ValueError("LOW/HIGH receipts must belong to one authenticated world challenge")

        authority_verified = bool(
            verifier is not None
            and verifier.verify(low)
            and verifier.verify(high)
        )
        pair = WorldOutcomePair(
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
            authority_verified=authority_verified,
        )
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
        routing_score = mean_abs_effect * independence_factor
        return AxisWorldSummary(
            axis_id=axis_id,
            context_id=context_id,
            independent_evidence_classes=len(unique),
            mean_effect=mean_effect,
            mean_abs_effect=mean_abs_effect,
            routing_score=routing_score,
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

    def assess_transport(
        self,
        proposals: Sequence[InterventionProposal],
    ) -> WorldTransportAssessment:
        contexts = sorted({
            pair.context_id for pair in self.pairs
            if self._authoritative(pair)
        })
        evidence_ready: List[str] = []
        top_by_context: List[Tuple[str, str]] = []

        for context in contexts:
            summaries = [self.summary(item.axis_id, context_id=context) for item in proposals]
            if not summaries or max(s.independent_evidence_classes for s in summaries) < self.min_independent_classes:
                continue
            ranked = self._rank_without_transport_guard(proposals, context)
            if not ranked:
                continue
            evidence_ready.append(context)
            top_by_context.append((context, ranked[0].axis_id))

        if len(evidence_ready) < 2:
            return WorldTransportAssessment(
                status="GLOBAL_TRANSPORT_NOT_CONTRADICTED",
                safe_for_global_transport=True,
                evidence_ready_contexts=tuple(evidence_ready),
                top_axis_by_context=tuple(top_by_context),
                reasons=("fewer than two independently supported regimes available",),
            )

        distinct_tops = {axis_id for _, axis_id in top_by_context}
        if len(distinct_tops) > 1:
            return WorldTransportAssessment(
                status="REGIME_CONFLICT_BLOCK_GLOBAL_TRANSPORT",
                safe_for_global_transport=False,
                evidence_ready_contexts=tuple(evidence_ready),
                top_axis_by_context=tuple(top_by_context),
                reasons=("independently supported regimes prefer different interventions",),
            )

        return WorldTransportAssessment(
            status="GLOBAL_TRANSPORT_SUPPORTED_BOUNDED",
            safe_for_global_transport=True,
            evidence_ready_contexts=tuple(evidence_ready),
            top_axis_by_context=tuple(top_by_context),
            reasons=("supported regimes agree on the same preferred intervention",),
        )

    def rank_proposals(
        self,
        proposals: Sequence[InterventionProposal],
        context_id: Optional[str] = None,
    ) -> List[InterventionProposal]:
        if context_id is None:
            assessment = self.assess_transport(proposals)
            if not assessment.safe_for_global_transport:
                return list(proposals)
        return self._rank_without_transport_guard(proposals, context_id)

    def restore_pairs(self, pairs: Iterable[WorldOutcomePair]) -> None:
        self.pairs = []
        seen_ids = set()
        for pair in pairs:
            if pair.pair_id in seen_ids:
                continue
            seen_ids.add(pair.pair_id)
            self.pairs.append(pair)
