from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class ValidationObservation:
    observation_id: str
    source_id: str
    context_id: str
    epoch: int
    split: str  # TRAIN, HELDOUT, DELAYED
    variant: str  # FULL, REMOVE, WRONG_SWAP, BASELINE
    score: float


@dataclass(frozen=True)
class ValidationGateResult:
    status: str
    full_heldout_mean: float
    remove_heldout_mean: float
    wrong_heldout_mean: float
    delayed_full_mean: float
    source_disjoint: bool
    delayed_present: bool
    negative_transfer_pass: bool
    reasons: Tuple[str, ...]


class RobustPromotionGate:
    """Promotion gate for generated cognition beyond a single held-out split.

    Requires source-disjoint held-out evidence, delayed reproduction, FULL beating
    REMOVE and WRONG_SWAP, and no material negative transfer on listed protected
    contexts. This is an authority gate; it does not generate cognition.
    """

    def __init__(
        self,
        min_advantage: float = 0.02,
        max_negative_transfer: float = 0.02,
    ) -> None:
        self.min_advantage = float(min_advantage)
        self.max_negative_transfer = max(0.0, float(max_negative_transfer))

    @staticmethod
    def _mean(rows: Sequence[ValidationObservation]) -> float:
        return sum(float(row.score) for row in rows) / len(rows) if rows else float("-inf")

    def assess(
        self,
        observations: Sequence[ValidationObservation],
        protected_contexts: Iterable[str] = (),
    ) -> ValidationGateResult:
        train = [row for row in observations if row.split.upper() == "TRAIN"]
        heldout = [row for row in observations if row.split.upper() == "HELDOUT"]
        delayed = [row for row in observations if row.split.upper() == "DELAYED"]
        train_sources = {row.source_id for row in train}
        heldout_sources = {row.source_id for row in heldout}
        source_disjoint = bool(heldout_sources) and train_sources.isdisjoint(heldout_sources)
        delayed_present = bool(delayed)

        full_h = [r for r in heldout if r.variant.upper() == "FULL"]
        remove_h = [r for r in heldout if r.variant.upper() == "REMOVE"]
        wrong_h = [r for r in heldout if r.variant.upper() == "WRONG_SWAP"]
        full_d = [r for r in delayed if r.variant.upper() == "FULL"]
        baseline_d = [r for r in delayed if r.variant.upper() == "BASELINE"]

        full_mean = self._mean(full_h)
        remove_mean = self._mean(remove_h)
        wrong_mean = self._mean(wrong_h)
        delayed_full = self._mean(full_d)

        reasons: List[str] = []
        if not source_disjoint:
            reasons.append("held-out source classes overlap training sources or are missing")
        if not delayed_present or not full_d:
            reasons.append("delayed reproduction missing")
        if not full_h or not remove_h or not wrong_h:
            reasons.append("FULL/REMOVE/WRONG_SWAP held-out matrix incomplete")
        else:
            if full_mean < remove_mean + self.min_advantage:
                reasons.append("FULL does not beat REMOVE by required margin")
            if full_mean < wrong_mean + self.min_advantage:
                reasons.append("FULL does not beat WRONG_SWAP by required margin")

        protected = set(protected_contexts)
        negative_transfer_pass = True
        if protected:
            for context in sorted(protected):
                f = [r.score for r in full_d if r.context_id == context]
                b = [r.score for r in baseline_d if r.context_id == context]
                if not f or not b:
                    negative_transfer_pass = False
                    reasons.append(f"protected context {context} lacks delayed FULL/BASELINE comparison")
                    continue
                if (sum(b) / len(b)) - (sum(f) / len(f)) > self.max_negative_transfer:
                    negative_transfer_pass = False
                    reasons.append(f"negative transfer exceeded tolerance in {context}")

        if reasons:
            status = "ROBUST_PROMOTION_BLOCKED"
        else:
            status = "ROBUST_PROMOTION_ELIGIBLE"

        return ValidationGateResult(
            status=status,
            full_heldout_mean=full_mean,
            remove_heldout_mean=remove_mean,
            wrong_heldout_mean=wrong_mean,
            delayed_full_mean=delayed_full,
            source_disjoint=source_disjoint,
            delayed_present=delayed_present,
            negative_transfer_pass=negative_transfer_pass,
            reasons=tuple(reasons),
        )
