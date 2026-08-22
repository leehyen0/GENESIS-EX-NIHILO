from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .causal_model_genesis import InterventionDescriptor


@dataclass(frozen=True)
class FalsificationScore:
    intervention_id: str
    semantic_novelty: float
    structural_stress: float
    cost: float
    utility: float


class ModelFalsificationPolicy:
    """Challenge a sole surviving model instead of treating identification as truth.

    Once an exact version space collapses to one model, information gain among
    hypotheses is zero. A separate objective is therefore required. This policy
    searches BODY-generated, unobserved interventions for semantic coverage gaps
    and structural stress. It is deliberately independent of hidden outcomes.
    Repeated calls with newly observed ids sweep previously untested causal regimes
    until either a counterexample appears or the bounded surface is exhausted.
    """

    @staticmethod
    def _features(d: InterventionDescriptor) -> Set[str]:
        features = {f"TARGET:{name}" for name in d.targets}
        features.update(f"BLOCK:{name}" for name in d.blocked)
        if int(d.delay_steps) > 0:
            features.add(f"DELAY:{int(d.delay_steps)}")
        if bool(d.context_shift):
            features.add("CONTEXT")
        if len(d.targets) > 1:
            features.add("MULTI_TARGET")
        if d.blocked:
            features.add("BLOCKING")
        return features

    @classmethod
    def _distance(cls, left: InterventionDescriptor, right: InterventionDescriptor) -> float:
        a, b = cls._features(left), cls._features(right)
        union = a | b
        if not union:
            return 0.0
        return 1.0 - (len(a & b) / len(union))

    @classmethod
    def _structural_stress(cls, d: InterventionDescriptor) -> float:
        return float(
            len(d.targets)
            + 1.5 * len(d.blocked)
            + 1.25 * max(0, int(d.delay_steps))
            + (1.5 if d.context_shift else 0.0)
        )

    @classmethod
    def rank(
        cls,
        candidates: Sequence[InterventionDescriptor],
        observed: Sequence[InterventionDescriptor] = (),
        cost_exponent: float = 0.10,
    ) -> List[FalsificationScore]:
        if not candidates:
            return []
        scored: List[FalsificationScore] = []
        for candidate in candidates:
            if observed:
                novelty = min(cls._distance(candidate, prior) for prior in observed)
            else:
                novelty = 1.0
            stress = cls._structural_stress(candidate)
            cost = max(1e-9, float(candidate.cost))
            utility = (2.0 * novelty + stress) / (cost ** max(0.0, float(cost_exponent)))
            scored.append(FalsificationScore(
                intervention_id=candidate.intervention_id,
                semantic_novelty=novelty,
                structural_stress=stress,
                cost=cost,
                utility=utility,
            ))
        return sorted(
            scored,
            key=lambda item: (
                -item.utility,
                -item.semantic_novelty,
                -item.structural_stress,
                item.cost,
                item.intervention_id,
            ),
        )

    @classmethod
    def select(
        cls,
        candidates: Sequence[InterventionDescriptor],
        observed: Sequence[InterventionDescriptor] = (),
        cost_exponent: float = 0.10,
    ) -> Optional[FalsificationScore]:
        ranked = cls.rank(candidates, observed=observed, cost_exponent=cost_exponent)
        return ranked[0] if ranked else None
