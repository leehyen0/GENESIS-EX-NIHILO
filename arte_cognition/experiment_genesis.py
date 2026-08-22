from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Tuple

from .representation_genesis import RepresentationAxis


@dataclass(frozen=True)
class InterventionProposal:
    experiment_id: str
    axis_id: str
    manipulated_variable: str
    held_fixed: Tuple[Tuple[str, float], ...]
    low_value: float
    high_value: float
    predicted_low_side: str
    predicted_high_side: str
    reason: str
    status: str = "PROPOSAL_ONLY"


class ExperimentGenesisEngine:
    """Translate measurable representation axes into discriminating interventions.

    Proposals cross the learned representation threshold while holding other parent
    variables fixed where possible. They are experiment proposals, not evidence or
    actions, until an external executor performs them and returns outcomes.
    """

    def __init__(self, relative_margin: float = 0.15, max_proposals: int = 8) -> None:
        self.relative_margin = max(0.01, float(relative_margin))
        self.max_proposals = max(1, int(max_proposals))

    def _around(self, value: float) -> Tuple[float, float]:
        margin = max(abs(value) * self.relative_margin, self.relative_margin)
        return value - margin, value + margin

    def propose(
        self,
        axis: RepresentationAxis,
        reference_values: Mapping[str, float],
    ) -> List[InterventionProposal]:
        proposals: List[InterventionProposal] = []
        threshold = float(axis.threshold)

        def add(
            variable: str,
            low: float,
            high: float,
            fixed: Mapping[str, float],
            reason: str,
            low_side: str = "LE_THRESHOLD",
            high_side: str = "GT_THRESHOLD",
        ) -> None:
            proposals.append(InterventionProposal(
                experiment_id=f"EXPERIMENT::{axis.axis_id}::{variable}",
                axis_id=axis.axis_id,
                manipulated_variable=variable,
                held_fixed=tuple(sorted((k, float(v)) for k, v in fixed.items())),
                low_value=float(low),
                high_value=float(high),
                predicted_low_side=low_side,
                predicted_high_side=high_side,
                reason=reason,
            ))

        if axis.family in {"DIFFERENCE", "ABS_DIFFERENCE"} and len(axis.inputs) == 2:
            a, b = axis.inputs
            if b in reference_values:
                b0 = float(reference_values[b])
                low_axis, high_axis = self._around(threshold)
                add(a, low_axis + b0, high_axis + b0, {b: b0}, "cross derived difference threshold while holding second parent fixed")
            if a in reference_values:
                a0 = float(reference_values[a])
                low_axis, high_axis = self._around(threshold)
                add(b, a0 - high_axis, a0 - low_axis, {a: a0}, "cross derived difference threshold while holding first parent fixed")

        elif axis.family == "RATIO" and len(axis.inputs) == 2:
            numerator, denominator = axis.inputs
            if denominator in reference_values and abs(float(reference_values[denominator])) > 1e-12:
                d0 = float(reference_values[denominator])
                low_ratio, high_ratio = self._around(threshold)
                add(numerator, low_ratio * d0, high_ratio * d0, {denominator: d0}, "cross ratio threshold by manipulating numerator at fixed denominator")
            if numerator in reference_values and abs(threshold) > 1e-12:
                n0 = float(reference_values[numerator])
                low_ratio, high_ratio = self._around(threshold)
                if abs(low_ratio) > 1e-12 and abs(high_ratio) > 1e-12:
                    candidates = sorted((n0 / high_ratio, n0 / low_ratio))
                    add(denominator, candidates[0], candidates[1], {numerator: n0}, "cross ratio threshold by manipulating denominator at fixed numerator")

        elif axis.family == "INTERACTION" and len(axis.inputs) == 2:
            a, b = axis.inputs
            if b in reference_values and abs(float(reference_values[b])) > 1e-12:
                b0 = float(reference_values[b])
                low_product, high_product = self._around(threshold)
                add(a, low_product / b0, high_product / b0, {b: b0}, "cross interaction threshold while holding second parent fixed")
            if a in reference_values and abs(float(reference_values[a])) > 1e-12:
                a0 = float(reference_values[a])
                low_product, high_product = self._around(threshold)
                add(b, low_product / a0, high_product / a0, {a: a0}, "cross interaction threshold while holding first parent fixed")

        elif axis.family == "DERIVATIVE" and len(axis.inputs) == 1:
            variable = axis.inputs[0]
            if variable in reference_values:
                base = float(reference_values[variable])
                low_rate, high_rate = self._around(threshold)
                add(variable, base + low_rate, base + high_rate, {}, "create two one-step trajectories that straddle the learned derivative threshold")

        elif axis.family == "PROJECTION" and axis.coefficients:
            low_score, high_score = self._around(threshold)
            coeffs = dict(axis.coefficients)
            # Prefer variables with larger absolute leverage, yielding smaller and
            # more local interventions around the reference state.
            for variable, coefficient in sorted(coeffs.items(), key=lambda kv: (-abs(kv[1]), kv[0])):
                if abs(coefficient) <= 1e-12:
                    continue
                others = {name: float(reference_values[name]) for name in coeffs if name != variable and name in reference_values}
                if len(others) != len(coeffs) - 1:
                    continue
                fixed_score = float(axis.bias) + sum(coeffs[name] * value for name, value in others.items())
                x_for_low_score = (low_score - fixed_score) / coefficient
                x_for_high_score = (high_score - fixed_score) / coefficient
                if x_for_low_score <= x_for_high_score:
                    add(
                        variable,
                        x_for_low_score,
                        x_for_high_score,
                        others,
                        "cross learned latent projection threshold while holding other projection parents fixed",
                    )
                else:
                    add(
                        variable,
                        x_for_high_score,
                        x_for_low_score,
                        others,
                        "cross learned latent projection threshold while holding other projection parents fixed",
                        low_side="GT_THRESHOLD",
                        high_side="LE_THRESHOLD",
                    )

        valid = [p for p in proposals if p.low_value == p.low_value and p.high_value == p.high_value and p.low_value < p.high_value]
        return valid[: self.max_proposals]
