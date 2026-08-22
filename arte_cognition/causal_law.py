from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .semantic_genesis import LawCandidate


@dataclass(frozen=True)
class InterventionObservation:
    observation_id: str
    arm: str  # TREATMENT or CONTROL
    outcome: float
    assignment: str = "OBSERVATIONAL"  # OBSERVATIONAL, NATURAL_EXPERIMENT, RANDOMIZED
    source_class: str = "DEFAULT"
    negative_control: bool = False


@dataclass(frozen=True)
class CausalLawAssessment:
    law_id: str
    status: str
    treatment_support: int
    control_support: int
    treatment_mean: float
    control_mean: float
    estimated_effect: float
    randomized: bool
    independent_source_classes: int
    negative_control_pass: bool
    reasons: Tuple[str, ...]


class CausalLawEvaluator:
    """Promote predictive laws through explicit causal-evidence stages.

    `BOUNDED_LAW` from semantic genesis is interpreted only as predictive.
    Intervention evidence can raise it to INTERVENTION_SUPPORTED_RELATION.
    CAUSAL_LAW_BOUNDED additionally requires randomized assignment, multiple
    source classes and a passing negative control. This is deliberately strict.
    """

    def __init__(
        self,
        min_arm_support: int = 2,
        min_abs_effect: float = 0.05,
        max_negative_control_effect: float = 0.05,
        min_source_classes: int = 2,
    ) -> None:
        self.min_arm_support = max(1, int(min_arm_support))
        self.min_abs_effect = max(0.0, float(min_abs_effect))
        self.max_negative_control_effect = max(0.0, float(max_negative_control_effect))
        self.min_source_classes = max(1, int(min_source_classes))

    @staticmethod
    def _mean(rows: Sequence[InterventionObservation]) -> float:
        return sum(float(row.outcome) for row in rows) / len(rows) if rows else 0.0

    def assess(
        self,
        law: LawCandidate,
        observations: Sequence[InterventionObservation],
    ) -> CausalLawAssessment:
        reasons: List[str] = []
        if law.status != "BOUNDED_LAW":
            return CausalLawAssessment(
                law_id=law.law_id,
                status="ASSOCIATIVE_PATTERN",
                treatment_support=0,
                control_support=0,
                treatment_mean=0.0,
                control_mean=0.0,
                estimated_effect=0.0,
                randomized=False,
                independent_source_classes=0,
                negative_control_pass=False,
                reasons=("predictive held-out gate not closed",),
            )

        primary = [row for row in observations if not row.negative_control]
        treatment = [row for row in primary if row.arm.upper() == "TREATMENT"]
        control = [row for row in primary if row.arm.upper() == "CONTROL"]
        t_mean, c_mean = self._mean(treatment), self._mean(control)
        effect = t_mean - c_mean if treatment and control else 0.0

        if len(treatment) < self.min_arm_support or len(control) < self.min_arm_support:
            return CausalLawAssessment(
                law_id=law.law_id,
                status="PREDICTIVE_LAW",
                treatment_support=len(treatment),
                control_support=len(control),
                treatment_mean=t_mean,
                control_mean=c_mean,
                estimated_effect=effect,
                randomized=False,
                independent_source_classes=len({r.source_class for r in primary}),
                negative_control_pass=False,
                reasons=("intervention arm support insufficient",),
            )

        if abs(effect) < self.min_abs_effect:
            return CausalLawAssessment(
                law_id=law.law_id,
                status="PREDICTIVE_LAW",
                treatment_support=len(treatment),
                control_support=len(control),
                treatment_mean=t_mean,
                control_mean=c_mean,
                estimated_effect=effect,
                randomized=False,
                independent_source_classes=len({r.source_class for r in primary}),
                negative_control_pass=False,
                reasons=("intervention effect below minimum",),
            )

        status = "INTERVENTION_SUPPORTED_RELATION"
        reasons.append("treatment/control intervention contrast reproduced")
        randomized = bool(primary) and all(r.assignment.upper() == "RANDOMIZED" for r in primary)
        source_classes = len({r.source_class for r in primary})

        neg = [row for row in observations if row.negative_control]
        neg_t = [row for row in neg if row.arm.upper() == "TREATMENT"]
        neg_c = [row for row in neg if row.arm.upper() == "CONTROL"]
        negative_control_pass = False
        if len(neg_t) >= self.min_arm_support and len(neg_c) >= self.min_arm_support:
            negative_effect = self._mean(neg_t) - self._mean(neg_c)
            negative_control_pass = abs(negative_effect) <= self.max_negative_control_effect
            reasons.append(
                "negative control passed" if negative_control_pass else "negative control failed"
            )
        else:
            reasons.append("negative control support insufficient")

        if randomized and source_classes >= self.min_source_classes and negative_control_pass:
            status = "CAUSAL_LAW_BOUNDED"
            reasons.append("randomized multi-source intervention evidence closed bounded causal gate")
        else:
            if not randomized:
                reasons.append("randomized assignment not established")
            if source_classes < self.min_source_classes:
                reasons.append("source-class diversity insufficient")

        return CausalLawAssessment(
            law_id=law.law_id,
            status=status,
            treatment_support=len(treatment),
            control_support=len(control),
            treatment_mean=t_mean,
            control_mean=c_mean,
            estimated_effect=effect,
            randomized=randomized,
            independent_source_classes=source_classes,
            negative_control_pass=negative_control_pass,
            reasons=tuple(reasons),
        )
