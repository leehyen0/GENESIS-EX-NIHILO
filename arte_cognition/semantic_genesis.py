from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import itertools
import math


@dataclass(frozen=True)
class ResidualObservation:
    residual_id: str
    features: Tuple[str, ...]
    outcome: str
    source_class: str = "OBSERVATION"
    heldout: bool = False


@dataclass(frozen=True)
class ConceptCandidate:
    concept_id: str
    defining_features: Tuple[str, ...]
    support: int
    information_gain: float
    covered_residuals: Tuple[str, ...]
    status: str = "PROPOSAL_ONLY"


@dataclass(frozen=True)
class LawCandidate:
    law_id: str
    concept_id: str
    predicted_outcome: str
    train_support: int
    train_accuracy: float
    heldout_support: int
    heldout_accuracy: float
    counterexamples: Tuple[str, ...]
    status: str


@dataclass(frozen=True)
class SemanticQuery:
    query_id: str
    target_feature: str
    reason: str
    expected_discrimination: float


def _entropy(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = len(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


class SemanticGenesisEngine:
    """Turn recurring residual structure into bounded concept and law candidates.

    Generated concepts are proposals, not facts. Generated laws remain candidates
    until they satisfy explicit train support plus held-out reproduction. This
    separates generative representation escape from epistemic promotion.
    """

    def __init__(
        self,
        min_support: int = 2,
        min_information_gain: float = 0.05,
        max_features_per_concept: int = 2,
        concept_budget: int = 16,
    ) -> None:
        self.min_support = max(1, int(min_support))
        self.min_information_gain = max(0.0, float(min_information_gain))
        self.max_features_per_concept = max(1, int(max_features_per_concept))
        self.concept_budget = max(1, int(concept_budget))

    @staticmethod
    def _present(row: ResidualObservation, feature_set: Iterable[str]) -> bool:
        return set(feature_set).issubset(set(row.features))

    def information_gain(
        self,
        rows: Sequence[ResidualObservation],
        feature_set: Sequence[str],
    ) -> float:
        if not rows:
            return 0.0
        present = [row for row in rows if self._present(row, feature_set)]
        absent = [row for row in rows if row not in present]
        if not present or not absent:
            return 0.0
        base = _entropy([row.outcome for row in rows])
        total = len(rows)
        conditional = (
            len(present) / total * _entropy([row.outcome for row in present])
            + len(absent) / total * _entropy([row.outcome for row in absent])
        )
        return max(0.0, base - conditional)

    def propose_concepts(
        self,
        observations: Sequence[ResidualObservation],
    ) -> List[ConceptCandidate]:
        train = [row for row in observations if not row.heldout]
        features = sorted({feature for row in train for feature in row.features})
        candidates: List[ConceptCandidate] = []

        max_k = min(self.max_features_per_concept, len(features))
        for width in range(1, max_k + 1):
            for combo in itertools.combinations(features, width):
                covered = [row for row in train if self._present(row, combo)]
                if len(covered) < self.min_support:
                    continue
                gain = self.information_gain(train, combo)
                if gain < self.min_information_gain:
                    continue
                candidates.append(ConceptCandidate(
                    concept_id="CONCEPT::" + "+".join(combo),
                    defining_features=tuple(combo),
                    support=len(covered),
                    information_gain=gain,
                    covered_residuals=tuple(sorted(row.residual_id for row in covered)),
                ))

        candidates.sort(key=lambda c: (
            -c.information_gain,
            -c.support,
            len(c.defining_features),
            c.concept_id,
        ))

        # Quotient concepts that distinguish exactly the same residual set.
        out: List[ConceptCandidate] = []
        seen_coverage = set()
        for candidate in candidates:
            if candidate.covered_residuals in seen_coverage:
                continue
            seen_coverage.add(candidate.covered_residuals)
            out.append(candidate)
            if len(out) >= self.concept_budget:
                break
        return out

    def induce_law(
        self,
        concept: ConceptCandidate,
        observations: Sequence[ResidualObservation],
        min_train_support: int = 3,
        min_train_accuracy: float = 0.80,
        min_heldout_support: int = 1,
        min_heldout_accuracy: float = 1.0,
    ) -> LawCandidate:
        train = [
            row for row in observations
            if not row.heldout and self._present(row, concept.defining_features)
        ]
        heldout = [
            row for row in observations
            if row.heldout and self._present(row, concept.defining_features)
        ]

        if not train:
            return LawCandidate(
                law_id="LAW::" + concept.concept_id,
                concept_id=concept.concept_id,
                predicted_outcome="",
                train_support=0,
                train_accuracy=0.0,
                heldout_support=len(heldout),
                heldout_accuracy=0.0,
                counterexamples=(),
                status="INSUFFICIENT_TRAIN_SUPPORT",
            )

        counts = {}
        for row in train:
            counts[row.outcome] = counts.get(row.outcome, 0) + 1
        predicted = sorted(counts, key=lambda label: (-counts[label], label))[0]
        train_accuracy = sum(row.outcome == predicted for row in train) / len(train)
        heldout_accuracy = (
            sum(row.outcome == predicted for row in heldout) / len(heldout)
            if heldout else 0.0
        )
        counterexamples = tuple(sorted(
            row.residual_id for row in observations
            if self._present(row, concept.defining_features) and row.outcome != predicted
        ))

        if len(train) < min_train_support:
            status = "INSUFFICIENT_TRAIN_SUPPORT"
        elif train_accuracy < min_train_accuracy:
            status = "COUNTEREXAMPLE_BOUND"
        elif len(heldout) < min_heldout_support:
            status = "HELDOUT_REQUIRED"
        elif heldout_accuracy < min_heldout_accuracy:
            status = "HELDOUT_REFUTED"
        else:
            status = "BOUNDED_LAW"

        return LawCandidate(
            law_id="LAW::" + concept.concept_id,
            concept_id=concept.concept_id,
            predicted_outcome=predicted,
            train_support=len(train),
            train_accuracy=train_accuracy,
            heldout_support=len(heldout),
            heldout_accuracy=heldout_accuracy,
            counterexamples=counterexamples,
            status=status,
        )

    def propose_queries(
        self,
        observations: Sequence[ResidualObservation],
        concepts: Sequence[ConceptCandidate],
        budget: int = 8,
    ) -> List[SemanticQuery]:
        train = [row for row in observations if not row.heldout]
        all_features = sorted({feature for row in train for feature in row.features})
        concept_features = {feature for concept in concepts for feature in concept.defining_features}
        scored: List[SemanticQuery] = []

        # Probe unmodeled features that still discriminate outcome regimes.
        for feature in all_features:
            if feature in concept_features:
                continue
            gain = self.information_gain(train, (feature,))
            if gain > 0:
                scored.append(SemanticQuery(
                    query_id="QUERY::FEATURE::" + feature,
                    target_feature=feature,
                    reason="unmodeled feature may separate residual outcome regimes",
                    expected_discrimination=gain,
                ))

        # If a concept still mixes outcomes, search for a feature that can split it.
        for concept in concepts:
            covered = [row for row in train if self._present(row, concept.defining_features)]
            if _entropy([row.outcome for row in covered]) <= 0:
                continue
            for feature in all_features:
                if feature in concept.defining_features:
                    continue
                present = [row for row in covered if feature in row.features]
                if not present or len(present) == len(covered):
                    continue
                gain = self.information_gain(covered, (feature,))
                if gain > 0:
                    scored.append(SemanticQuery(
                        query_id=f"QUERY::{concept.concept_id}::{feature}",
                        target_feature=feature,
                        reason="split a concept whose current representation still mixes outcomes",
                        expected_discrimination=gain,
                    ))

        dedup = {}
        for query in scored:
            old = dedup.get(query.query_id)
            if old is None or query.expected_discrimination > old.expected_discrimination:
                dedup[query.query_id] = query
        return sorted(
            dedup.values(),
            key=lambda q: (-q.expected_discrimination, q.query_id),
        )[:max(1, int(budget))]
