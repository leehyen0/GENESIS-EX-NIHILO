from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence
import math

from .adaptive_cognition import (
    AdaptiveCognitionCompiler,
    CognitionPlan,
    ModuleCredit,
    Pressure,
    TaskState,
)


LEARNABLE_MODULES = (
    "QUESTION_FIELD",
    "COUNTEREXAMPLE_SEARCH",
    "MODAL_EXPANSION",
    "REPRESENTATION_ESCAPE",
    "TEMPORAL_IDENTITY",
    "WORLD_INTERACTION",
)


@dataclass
class ModuleExperience:
    evidence_count: int = 0
    signed_value: float = 0.0
    positive_count: int = 0
    negative_count: int = 0


@dataclass
class CognitionPolicyState:
    modules: Dict[str, ModuleExperience] = field(default_factory=dict)
    learning_rate: float = 0.25
    min_evidence_before_routing_change: int = 3
    max_threshold_shift: float = 0.10

    def experience(self, module: str) -> ModuleExperience:
        if module not in self.modules:
            self.modules[module] = ModuleExperience()
        return self.modules[module]

    def observe(self, credits: Iterable[ModuleCredit]) -> None:
        """Update only from causally relevant modules.

        Mere activation is ignored. Positive treatment delta moves signed_value up;
        a behavior-changing negative delta moves it down. Values are bounded to
        avoid one regime permanently dominating the router.
        """
        for credit in credits:
            if credit.module not in LEARNABLE_MODULES or not credit.decision_changed:
                continue
            exp = self.experience(credit.module)
            exp.evidence_count += 1
            signal = max(-1.0, min(1.0, float(credit.outcome_delta)))
            exp.signed_value = max(
                -1.0,
                min(1.0, (1.0 - self.learning_rate) * exp.signed_value + self.learning_rate * signal),
            )
            if signal > 0:
                exp.positive_count += 1
            elif signal < 0:
                exp.negative_count += 1

    def threshold_shift(self, module: str) -> float:
        exp = self.experience(module)
        if exp.evidence_count < self.min_evidence_before_routing_change:
            return 0.0
        # Positive validated value lowers the activation threshold; negative raises it.
        return -self.max_threshold_shift * math.tanh(2.0 * exp.signed_value)


class OutcomeLearnedCognitionRouter:
    """Bounded meta-router over the sparse cognition compiler.

    The base compiler supplies structural necessities. Learned routing may alter
    only marginal specialist activation. It cannot suppress hard representation,
    temporal-identity, or world-interaction requirements.
    """

    BASE_THRESHOLDS = {
        "QUESTION_FIELD": 0.25,
        "COUNTEREXAMPLE_SEARCH": 0.34,
        "MODAL_EXPANSION": 0.25,
    }

    def __init__(
        self,
        compiler: Optional[AdaptiveCognitionCompiler] = None,
        policy: Optional[CognitionPolicyState] = None,
    ) -> None:
        self.compiler = compiler or AdaptiveCognitionCompiler()
        self.policy = policy or CognitionPolicyState()

    def _trigger_strength(self, module: str, task: TaskState, pressure: Pressure) -> float:
        if module == "QUESTION_FIELD":
            return pressure.ambiguity
        if module == "COUNTEREXAMPLE_SEARCH":
            return max(pressure.contradiction, pressure.ambiguity)
        if module == "MODAL_EXPANSION":
            return max(pressure.residual, pressure.novelty)
        if module == "REPRESENTATION_ESCAPE":
            return 1.0 if self.compiler.detect_boundary_shadow(task) else 0.0
        if module == "TEMPORAL_IDENTITY":
            return pressure.temporal_identity
        if module == "WORLD_INTERACTION":
            return 1.0 if (task.external_world or task.action_required) else 0.0
        return 0.0

    def _hard_required(self, module: str, task: TaskState) -> bool:
        if module == "REPRESENTATION_ESCAPE":
            return self.compiler.detect_boundary_shadow(task)
        if module == "TEMPORAL_IDENTITY":
            return task.temporal_identity_pressure
        if module == "WORLD_INTERACTION":
            return task.external_world or task.action_required
        return False

    def compile(self, task: TaskState) -> CognitionPlan:
        plan = self.compiler.compile(task)
        active = list(plan.active_subgraph)
        shadow = list(plan.shadow_subgraph)
        reasons = list(plan.reasons)

        for module, base_threshold in self.BASE_THRESHOLDS.items():
            threshold = max(0.05, min(0.95, base_threshold + self.policy.threshold_shift(module)))
            strength = self._trigger_strength(module, task, plan.pressure)
            exp = self.policy.experience(module)

            if strength >= threshold and module not in active:
                active.append(module)
                if module in shadow:
                    shadow.remove(module)
                reasons.append(
                    f"outcome-learned router activated {module}: trigger={strength:.3f} threshold={threshold:.3f} evidence={exp.evidence_count}"
                )
            elif (
                module in active
                and not self._hard_required(module, task)
                and strength < threshold
                and exp.evidence_count >= self.policy.min_evidence_before_routing_change
            ):
                active.remove(module)
                if module not in shadow:
                    shadow.append(module)
                reasons.append(
                    f"outcome-learned router demoted {module}: trigger={strength:.3f} threshold={threshold:.3f} evidence={exp.evidence_count}"
                )

        plan.active_subgraph = self.compiler._dedupe(active)
        plan.shadow_subgraph = self.compiler._dedupe(shadow)
        plan.reasons = reasons
        return plan

    def learn_from_outcome(
        self,
        active_modules: Iterable[str],
        baseline_decision,
        treatment_decision,
        baseline_outcome: float,
        treatment_outcome: float,
        ablation_decisions: Optional[Mapping[str, object]] = None,
    ) -> List[ModuleCredit]:
        credits = self.compiler.assign_causal_credit(
            active_modules=active_modules,
            baseline_decision=baseline_decision,
            treatment_decision=treatment_decision,
            baseline_outcome=baseline_outcome,
            treatment_outcome=treatment_outcome,
            ablation_decisions=ablation_decisions,
        )
        self.policy.observe(credits)
        return credits
