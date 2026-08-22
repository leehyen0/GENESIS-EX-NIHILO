from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math

from .adaptive_cognition import QueryCandidate
from .world_model_ecology import CausalWorldModel, EpistemicInterventionScore, ModelEvidence


@dataclass(frozen=True)
class VersionSpaceSnapshot:
    generation: int
    model_ids: Tuple[str, ...]
    compatible_model_ids: Tuple[str, ...]
    authoritative_evidence_count: int

    @property
    def identified(self) -> bool:
        return len(self.compatible_model_ids) == 1

    @property
    def identified_model_id(self) -> Optional[str]:
        return self.compatible_model_ids[0] if self.identified else None


class GenerationScopedIdentifier:
    """Exact deterministic identification inside one structural generation.

    The ordinary world-model ecology intentionally keeps a soft posterior across
    the whole lineage. Structural identification needs a different object: the
    version space of models in the *current* generation that are not contradicted
    by authoritative evidence. Experiments are ranked only by how much they split
    that surviving version space. Older generations therefore cannot dilute EIG by
    appearing as UNKNOWN predictions in a newer generation's query surface.
    """

    @staticmethod
    def authoritative(evidence: Sequence[ModelEvidence]) -> Tuple[ModelEvidence, ...]:
        return tuple(item for item in evidence if item.authoritative)

    @classmethod
    def compatible_ids(
        cls,
        models: Sequence[CausalWorldModel],
        evidence: Sequence[ModelEvidence],
    ) -> Tuple[str, ...]:
        authoritative = cls.authoritative(evidence)
        compatible: List[str] = []
        for model in models:
            contradicted = False
            for item in authoritative:
                prediction = model.prediction_for(item.intervention_id)
                if prediction is not None and prediction != item.observed_outcome:
                    contradicted = True
                    break
            if not contradicted:
                compatible.append(model.model_id)
        return tuple(sorted(compatible))

    @classmethod
    def snapshot(
        cls,
        generation: int,
        models: Sequence[CausalWorldModel],
        evidence: Sequence[ModelEvidence],
    ) -> VersionSpaceSnapshot:
        selected = tuple(model for model in models if int(model.generation) == int(generation))
        return VersionSpaceSnapshot(
            generation=int(generation),
            model_ids=tuple(sorted(model.model_id for model in selected)),
            compatible_model_ids=cls.compatible_ids(selected, evidence),
            authoritative_evidence_count=len(cls.authoritative(evidence)),
        )

    @staticmethod
    def _entropy_uniform(n: int) -> float:
        return math.log2(n) if n > 1 else 0.0

    @classmethod
    def rank_interventions(
        cls,
        candidates: Sequence[QueryCandidate],
        compatible_model_ids: Sequence[str],
        cost_exponent: float = 0.15,
    ) -> List[EpistemicInterventionScore]:
        ids = tuple(sorted({str(model_id) for model_id in compatible_model_ids if str(model_id)}))
        if len(ids) <= 1:
            return []
        prior_entropy = cls._entropy_uniform(len(ids))
        scored: List[EpistemicInterventionScore] = []
        for candidate in candidates:
            buckets: Dict[str, int] = {}
            for model_id in ids:
                outcome = str(candidate.distinguishes.get(model_id, "__UNKNOWN__"))
                buckets[outcome] = buckets.get(outcome, 0) + 1
            expected_entropy = 0.0
            for count in buckets.values():
                mass = count / len(ids)
                expected_entropy += mass * cls._entropy_uniform(count)
            eig = max(0.0, prior_entropy - expected_entropy)
            cost = max(1e-9, float(candidate.cost))
            utility = eig / (cost ** max(0.0, float(cost_exponent)))
            scored.append(EpistemicInterventionScore(
                intervention_id=candidate.query_id,
                expected_information_gain=eig,
                cost=cost,
                utility=utility,
            ))
        return sorted(
            scored,
            key=lambda item: (-item.utility, -item.expected_information_gain, item.cost, item.intervention_id),
        )

    @classmethod
    def select_next(
        cls,
        candidates: Sequence[QueryCandidate],
        compatible_model_ids: Sequence[str],
        cost_exponent: float = 0.15,
    ) -> Optional[EpistemicInterventionScore]:
        ranked = cls.rank_interventions(candidates, compatible_model_ids, cost_exponent=cost_exponent)
        if not ranked or ranked[0].expected_information_gain <= 0.0:
            return None
        return ranked[0]
