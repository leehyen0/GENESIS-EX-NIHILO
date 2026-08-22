from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from .experiment_genesis import InterventionProposal


@dataclass(frozen=True)
class WorldOutcomeReceipt:
    """Outcome returned by an executor that owns the world/evaluation surface.

    The BODY receives the enacted value and realized outcome, but not the hidden
    mechanism used by the executor. `budget_token` identifies matched execution
    conditions for the two intervention arms.
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

    @property
    def effect(self) -> float:
        return float(self.high_outcome) - float(self.low_outcome)

    @property
    def independence_key(self) -> Tuple[str, str]:
        # Context is deliberately excluded. Replaying one evaluator/challenge in
        # several regimes can teach regime-specific behavior, but must not be
        # miscounted as several independent evidence sources globally.
        return (self.source_id, self.challenge_id)

    @property
    def contextual_evidence_key(self) -> Tuple[str, str, str]:
        return (self.context_id, self.source_id, self.challenge_id)


@dataclass(frozen=True)
class AxisWorldSummary:
    axis_id: str
    context_id: Optional[str]
    independent_evidence_classes: int
    mean_effect: float
    mean_abs_effect: float
    routing_score: float


class WorldExecutor(Protocol):
    def execute(
        self,
        proposal: InterventionProposal,
        arm: str,
        value: float,
    ) -> WorldOutcomeReceipt:
        ...


class WorldCouplingEngine:
    """Consume external intervention consequences and change future behavior.

    World value is conditioned on context/regime when one is supplied. This avoids
    transporting a cognition/intervention preference that was useful in one world
    regime into another merely because the axis name is the same. Evidence is
    deduplicated by evaluator/challenge identity; unmatched-budget or non-external
    events remain auditable but cannot steer future experiment order.
    """

    def __init__(self, min_independent_classes: int = 2) -> None:
        self.min_independent_classes = max(1, int(min_independent_classes))
        self.pairs: List[WorldOutcomePair] = []

    def _contextual_evidence_keys(self, axis_id: str, context_id: str) -> set[Tuple[str, str, str]]:
        return {
            pair.contextual_evidence_key
            for pair in self.pairs
            if pair.axis_id == axis_id
            and pair.context_id == context_id
            and pair.matched_budget
            and pair.externally_generated
        }

    def record_pair(self, pair: WorldOutcomePair) -> bool:
        if pair.pair_id in {item.pair_id for item in self.pairs}:
            return False

        if pair.matched_budget and pair.externally_generated:
            if pair.contextual_evidence_key in self._contextual_evidence_keys(pair.axis_id, pair.context_id):
                return False

        # Invalid-for-routing receipts are still retained once for audit.
        self.pairs.append(pair)
        return True

    def execute(
        self,
        proposal: InterventionProposal,
        executor: WorldExecutor,
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

        identity_low = (low.source_id, low.context_id, low.challenge_id, low.epoch)
        identity_high = (high.source_id, high.context_id, high.challenge_id, high.epoch)
        if identity_low != identity_high:
            raise ValueError("LOW/HIGH receipts must belong to one world challenge")

        pair = WorldOutcomePair(
            pair_id=f"PAIR::{proposal.experiment_id}::{low.context_id}::{low.source_id}::{low.challenge_id}",
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
        )
        self.record_pair(pair)
        return pair

    def summary(self, axis_id: str, context_id: Optional[str] = None) -> AxisWorldSummary:
        valid = [
            pair for pair in self.pairs
            if pair.axis_id == axis_id
            and pair.matched_budget
            and pair.externally_generated
            and (context_id is None or pair.context_id == context_id)
        ]
        # Within a regime and globally, one evaluator/challenge contributes at most
        # one independent evidence class. Context-specific pairs are still retained
        # so the BODY can learn opposite values in distinct regimes.
        by_key: Dict[Tuple[str, str], WorldOutcomePair] = {}
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

    def rank_proposals(
        self,
        proposals: Sequence[InterventionProposal],
        context_id: Optional[str] = None,
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

    def restore_pairs(self, pairs: Iterable[WorldOutcomePair]) -> None:
        self.pairs = []
        seen_ids = set()
        for pair in pairs:
            if pair.pair_id in seen_ids:
                continue
            seen_ids.add(pair.pair_id)
            self.pairs.append(pair)
