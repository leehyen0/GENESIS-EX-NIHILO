from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import itertools
import math

from .semantic_genesis import ResidualObservation


@dataclass(frozen=True)
class MeasurementObservation:
    observation_id: str
    values: Mapping[str, float]
    outcome: str
    heldout: bool = False
    time_index: Optional[float] = None
    context_id: str = "default"


@dataclass(frozen=True)
class RepresentationAxis:
    axis_id: str
    family: str
    inputs: Tuple[str, ...]
    threshold: float
    direction: str
    information_gain: float
    train_support: int
    positive_partition: Tuple[str, ...]
    formula: str
    status: str = "PROPOSAL_ONLY"


class RepresentationGenesisEngine:
    """Generate measurable candidate axes from raw numeric observations.

    This moves beyond supplied categorical features by constructing new axes such
    as differences, ratios, interactions and temporal derivatives. Axes remain
    proposals until downstream held-out/causal validation promotes the concepts
    that use them.
    """

    def __init__(
        self,
        min_information_gain: float = 0.05,
        min_partition_support: int = 2,
        axis_budget: int = 16,
    ) -> None:
        self.min_information_gain = max(0.0, float(min_information_gain))
        self.min_partition_support = max(1, int(min_partition_support))
        self.axis_budget = max(1, int(axis_budget))

    @staticmethod
    def _entropy(labels: Sequence[str]) -> float:
        if not labels:
            return 0.0
        counts: Dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        n = len(labels)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())

    def _best_split(
        self,
        rows: Sequence[MeasurementObservation],
        values: Mapping[str, float],
    ) -> Optional[Tuple[float, str, float, Tuple[str, ...]]]:
        usable = [row for row in rows if row.observation_id in values]
        if len(usable) < 2 * self.min_partition_support:
            return None
        unique = sorted({float(values[row.observation_id]) for row in usable})
        if len(unique) < 2:
            return None
        thresholds = [(a + b) / 2.0 for a, b in zip(unique, unique[1:])]
        base_entropy = self._entropy([row.outcome for row in usable])
        best = None
        for threshold in thresholds:
            low = [row for row in usable if values[row.observation_id] <= threshold]
            high = [row for row in usable if values[row.observation_id] > threshold]
            if len(low) < self.min_partition_support or len(high) < self.min_partition_support:
                continue
            conditional = (
                len(low) / len(usable) * self._entropy([r.outcome for r in low])
                + len(high) / len(usable) * self._entropy([r.outcome for r in high])
            )
            gain = max(0.0, base_entropy - conditional)
            for direction, positive in (("GT", high), ("LE", low)):
                signature = tuple(sorted(r.observation_id for r in positive))
                candidate = (gain, -abs(len(high) - len(low)), -threshold, direction, signature, threshold)
                if best is None or candidate > best[0]:
                    best = (candidate, threshold, direction, gain, signature)
        if best is None:
            return None
        _, threshold, direction, gain, signature = best
        return threshold, direction, gain, signature

    @staticmethod
    def _pair_values(
        rows: Sequence[MeasurementObservation],
        a: str,
        b: str,
        family: str,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for row in rows:
            if a not in row.values or b not in row.values:
                continue
            x, y = float(row.values[a]), float(row.values[b])
            if family == "DIFFERENCE":
                out[row.observation_id] = x - y
            elif family == "ABS_DIFFERENCE":
                out[row.observation_id] = abs(x - y)
            elif family == "RATIO":
                if abs(y) > 1e-12:
                    out[row.observation_id] = x / y
            elif family == "INTERACTION":
                out[row.observation_id] = x * y
        return out

    @staticmethod
    def _derivative_values(
        rows: Sequence[MeasurementObservation],
        variable: str,
    ) -> Dict[str, float]:
        by_context: Dict[str, List[MeasurementObservation]] = {}
        for row in rows:
            if variable in row.values and row.time_index is not None:
                by_context.setdefault(row.context_id, []).append(row)
        out: Dict[str, float] = {}
        for context_rows in by_context.values():
            ordered = sorted(context_rows, key=lambda r: (float(r.time_index), r.observation_id))
            for prev, cur in zip(ordered, ordered[1:]):
                dt = float(cur.time_index) - float(prev.time_index)
                if abs(dt) <= 1e-12:
                    continue
                out[cur.observation_id] = (
                    float(cur.values[variable]) - float(prev.values[variable])
                ) / dt
        return out

    def propose_axes(
        self,
        observations: Sequence[MeasurementObservation],
    ) -> List[RepresentationAxis]:
        train = [row for row in observations if not row.heldout]
        variables = sorted({key for row in train for key in row.values})
        raw: List[RepresentationAxis] = []

        def consider(family: str, inputs: Tuple[str, ...], values: Mapping[str, float], formula: str) -> None:
            split = self._best_split(train, values)
            if split is None:
                return
            threshold, direction, gain, signature = split
            if gain < self.min_information_gain:
                return
            raw.append(RepresentationAxis(
                axis_id=f"AXIS::{family}::{'|'.join(inputs)}",
                family=family,
                inputs=inputs,
                threshold=threshold,
                direction=direction,
                information_gain=gain,
                train_support=len(values),
                positive_partition=signature,
                formula=formula,
            ))

        for a, b in itertools.combinations(variables, 2):
            consider("DIFFERENCE", (a, b), self._pair_values(train, a, b, "DIFFERENCE"), f"{a}-{b}")
            consider("ABS_DIFFERENCE", (a, b), self._pair_values(train, a, b, "ABS_DIFFERENCE"), f"abs({a}-{b})")
            consider("RATIO", (a, b), self._pair_values(train, a, b, "RATIO"), f"{a}/{b}")
            consider("RATIO", (b, a), self._pair_values(train, b, a, "RATIO"), f"{b}/{a}")
            consider("INTERACTION", (a, b), self._pair_values(train, a, b, "INTERACTION"), f"{a}*{b}")

        for variable in variables:
            consider("DERIVATIVE", (variable,), self._derivative_values(train, variable), f"d({variable})/dt")

        raw.sort(key=lambda axis: (-axis.information_gain, len(axis.inputs), axis.axis_id))
        # Quotient axes that induce the same discriminating partition.
        out: List[RepresentationAxis] = []
        seen_partitions = set()
        for axis in raw:
            if axis.positive_partition in seen_partitions:
                continue
            seen_partitions.add(axis.positive_partition)
            out.append(axis)
            if len(out) >= self.axis_budget:
                break
        return out

    @staticmethod
    def axis_value(axis: RepresentationAxis, row: MeasurementObservation, previous: Optional[MeasurementObservation] = None) -> Optional[float]:
        try:
            if axis.family == "DIFFERENCE":
                return float(row.values[axis.inputs[0]]) - float(row.values[axis.inputs[1]])
            if axis.family == "ABS_DIFFERENCE":
                return abs(float(row.values[axis.inputs[0]]) - float(row.values[axis.inputs[1]]))
            if axis.family == "RATIO":
                denom = float(row.values[axis.inputs[1]])
                return None if abs(denom) <= 1e-12 else float(row.values[axis.inputs[0]]) / denom
            if axis.family == "INTERACTION":
                return float(row.values[axis.inputs[0]]) * float(row.values[axis.inputs[1]])
            if axis.family == "DERIVATIVE" and previous is not None:
                if row.time_index is None or previous.time_index is None:
                    return None
                dt = float(row.time_index) - float(previous.time_index)
                if abs(dt) <= 1e-12:
                    return None
                return (float(row.values[axis.inputs[0]]) - float(previous.values[axis.inputs[0]])) / dt
        except (KeyError, TypeError, ValueError):
            return None
        return None

    def augment_residuals(
        self,
        residuals: Sequence[ResidualObservation],
        measurements: Sequence[MeasurementObservation],
        axes: Sequence[RepresentationAxis],
    ) -> List[ResidualObservation]:
        by_id = {row.observation_id: row for row in measurements}
        previous_by_id: Dict[str, MeasurementObservation] = {}
        for context in sorted({row.context_id for row in measurements}):
            ordered = sorted(
                [row for row in measurements if row.context_id == context and row.time_index is not None],
                key=lambda r: (float(r.time_index), r.observation_id),
            )
            for prev, cur in zip(ordered, ordered[1:]):
                previous_by_id[cur.observation_id] = prev

        out: List[ResidualObservation] = []
        for residual in residuals:
            row = by_id.get(residual.residual_id)
            derived = list(residual.features)
            if row is not None:
                for axis in axes:
                    value = self.axis_value(axis, row, previous_by_id.get(row.observation_id))
                    if value is None:
                        continue
                    positive = value > axis.threshold if axis.direction == "GT" else value <= axis.threshold
                    if positive:
                        derived.append(axis.axis_id)
            out.append(ResidualObservation(
                residual_id=residual.residual_id,
                features=tuple(sorted(set(derived))),
                outcome=residual.outcome,
                source_class=residual.source_class,
                heldout=residual.heldout,
            ))
        return out
