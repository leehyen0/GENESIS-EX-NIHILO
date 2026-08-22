from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import math

from .representation_genesis import MeasurementObservation, RepresentationAxis, RepresentationGenesisEngine


@dataclass(frozen=True)
class RepresentationValueAssessment:
    axis_id: str
    derived_information_gain: float
    best_parent_information_gain: float
    incremental_gain: float
    heldout_accuracy: float
    heldout_support: int
    status: str
    reason: str


class RepresentationValueEvaluator:
    """Require a generated axis to add value beyond its parent variables.

    A syntactically new axis is not a representation escape. It must distinguish
    outcomes better than the best threshold on any parent variable and reproduce
    on held-out observations. This prevents relabeling an already-sufficient raw
    feature as cognitive progress.
    """

    def __init__(self, min_incremental_gain: float = 0.05, min_heldout_support: int = 2) -> None:
        self.min_incremental_gain = max(0.0, float(min_incremental_gain))
        self.min_heldout_support = max(1, int(min_heldout_support))
        self._axis_engine = RepresentationGenesisEngine(min_partition_support=1)

    @staticmethod
    def _entropy(labels: Sequence[str]) -> float:
        if not labels:
            return 0.0
        counts: Dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        n = len(labels)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())

    def _best_scalar_gain(self, rows: Sequence[MeasurementObservation], variable: str) -> float:
        usable = [row for row in rows if variable in row.values]
        if len(usable) < 2:
            return 0.0
        values = sorted({float(row.values[variable]) for row in usable})
        if len(values) < 2:
            return 0.0
        base = self._entropy([row.outcome for row in usable])
        best = 0.0
        for threshold in ((a + b) / 2.0 for a, b in zip(values, values[1:])):
            low = [row for row in usable if float(row.values[variable]) <= threshold]
            high = [row for row in usable if float(row.values[variable]) > threshold]
            if not low or not high:
                continue
            conditional = (
                len(low) / len(usable) * self._entropy([r.outcome for r in low])
                + len(high) / len(usable) * self._entropy([r.outcome for r in high])
            )
            best = max(best, base - conditional)
        return max(0.0, best)

    @staticmethod
    def _predict_positive(axis: RepresentationAxis, value: float) -> bool:
        return value > axis.threshold if axis.direction == "GT" else value <= axis.threshold

    def assess(
        self,
        axis: RepresentationAxis,
        observations: Sequence[MeasurementObservation],
    ) -> RepresentationValueAssessment:
        train = [row for row in observations if not row.heldout]
        heldout = [row for row in observations if row.heldout]
        parent_gain = max((self._best_scalar_gain(train, name) for name in axis.inputs), default=0.0)
        incremental = float(axis.information_gain) - parent_gain

        # Learn the positive-side label from training rows covered by the axis partition.
        positive_ids = set(axis.positive_partition)
        positive_train = [row for row in train if row.observation_id in positive_ids]
        if not positive_train:
            positive_label = ""
        else:
            counts: Dict[str, int] = {}
            for row in positive_train:
                counts[row.outcome] = counts.get(row.outcome, 0) + 1
            positive_label = sorted(counts, key=lambda k: (-counts[k], k))[0]

        correct = 0
        support = 0
        previous_by_context: Dict[str, MeasurementObservation] = {}
        ordered = sorted(
            heldout,
            key=lambda r: (r.context_id, float(r.time_index) if r.time_index is not None else 0.0, r.observation_id),
        )
        for row in ordered:
            previous = previous_by_context.get(row.context_id)
            value = self._axis_engine.axis_value(axis, row, previous)
            if row.time_index is not None:
                previous_by_context[row.context_id] = row
            if value is None:
                continue
            support += 1
            predicted_positive = self._predict_positive(axis, value)
            # This gate only evaluates whether the learned discriminating side
            # reproduces its training label; the complement is evaluated by inequality.
            if (row.outcome == positive_label) == predicted_positive:
                correct += 1
        heldout_accuracy = correct / support if support else 0.0

        if incremental < self.min_incremental_gain:
            status = "REDUNDANT_WITH_PARENT_REPRESENTATION"
            reason = "derived axis does not add enough information beyond parent variables"
        elif support < self.min_heldout_support:
            status = "HELDOUT_REQUIRED"
            reason = "incremental representation value has insufficient held-out support"
        elif heldout_accuracy < 1.0:
            status = "HELDOUT_REFUTED"
            reason = "derived distinction failed held-out reproduction"
        else:
            status = "INCREMENTAL_REPRESENTATION_VALUE"
            reason = "derived axis adds parent-conditional information and reproduces held-out"

        return RepresentationValueAssessment(
            axis_id=axis.axis_id,
            derived_information_gain=float(axis.information_gain),
            best_parent_information_gain=parent_gain,
            incremental_gain=incremental,
            heldout_accuracy=heldout_accuracy,
            heldout_support=support,
            status=status,
            reason=reason,
        )
