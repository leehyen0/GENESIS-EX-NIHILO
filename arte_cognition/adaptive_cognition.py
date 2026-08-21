from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import math

CORE_MODULES = ("STATE_IDENTITY", "CAUSAL_MODEL", "VERIFICATION")
SPECIALIST_MODULES = (
    "QUESTION_FIELD", "COUNTEREXAMPLE_SEARCH", "MODAL_EXPANSION",
    "REPRESENTATION_ESCAPE", "TEMPORAL_IDENTITY", "WORLD_INTERACTION",
    "CAUSAL_CREDIT",
)

ROLE_BASIS = ("EXPLICIT", "OPPOSITE", "COMPLEMENT", "ABSENCE", "COUNTERFACTUAL", "IMAGINARY")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _entropy(probs: Sequence[float]) -> float:
    vals = [p for p in probs if p > 0]
    total = sum(vals)
    if total <= 0:
        return 0.0
    vals = [p / total for p in vals]
    return -sum(p * math.log2(p) for p in vals)


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    prior: float = 1.0
    representation_signature: Tuple[str, ...] = ()
    predicts: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryCandidate:
    query_id: str
    distinguishes: Mapping[str, str] = field(default_factory=dict)
    cost: float = 1.0
    intervention: bool = False
    source_class: str = "OBSERVATION"


@dataclass
class TaskState:
    goal: str
    hypotheses: List[Hypothesis] = field(default_factory=list)
    observations: Dict[str, Any] = field(default_factory=dict)
    contradictions: List[str] = field(default_factory=list)
    residuals: List[str] = field(default_factory=list)
    candidate_queries: List[QueryCandidate] = field(default_factory=list)
    stakes: float = 0.0
    novelty: float = 0.0
    action_required: bool = False
    external_world: bool = False
    temporal_identity_pressure: bool = False
    current_representation: List[str] = field(default_factory=list)
    shadow_representations: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Pressure:
    ambiguity: float
    contradiction: float
    novelty: float
    stakes: float
    residual: float
    externality: float
    temporal_identity: float

    @property
    def total(self) -> float:
        vals = [self.ambiguity, self.contradiction, self.novelty, self.stakes,
                self.residual, self.externality, self.temporal_identity]
        peak = max(vals)
        return _clamp01(peak + 0.15 * (sum(vals) - peak) / max(1, len(vals) - 1))


@dataclass
class QuestionScore:
    query_id: str
    expected_information_gain: float
    cost: float
    score: float
    intervention: bool
    source_class: str


@dataclass
class CognitionPlan:
    pressure: Pressure
    active_subgraph: List[str]
    shadow_subgraph: List[str]
    modal_basis: List[str]
    question_field: List[QuestionScore]
    representation_actions: List[str]
    boundary_shadow: bool
    preobject_pressure: bool
    recommended_mode: str
    reasons: List[str]


@dataclass
class ModuleCredit:
    module: str
    used: bool
    decision_changed: bool
    outcome_delta: float
    causal_credit: float


class AdaptiveCognitionCompiler:
    """Compile a task into a minimum-sufficient cognitive subgraph.

    The compiler keeps unused lineage in shadow, expands possibilities only under
    pressure, ranks discriminating questions by information gain, triggers
    representation change when the active basis collapses causally different
    states, and assigns credit only when behavior changed and downstream outcome
    improved.
    """

    def diagnose_pressure(self, task: TaskState) -> Pressure:
        n_h = len(task.hypotheses)
        return Pressure(
            ambiguity=0.0 if n_h <= 1 else _clamp01(math.log2(n_h) / 4.0),
            contradiction=_clamp01(len(task.contradictions) / 3.0),
            novelty=_clamp01(task.novelty),
            stakes=_clamp01(task.stakes),
            residual=_clamp01(len(task.residuals) / 4.0),
            externality=1.0 if task.external_world else 0.0,
            temporal_identity=1.0 if task.temporal_identity_pressure else 0.0,
        )

    def detect_boundary_shadow(self, task: TaskState) -> bool:
        if len(task.hypotheses) < 2:
            return False
        groups: Dict[Tuple[str, ...], List[Hypothesis]] = {}
        for h in task.hypotheses:
            groups.setdefault(tuple(h.representation_signature), []).append(h)
        for group in groups.values():
            if len(group) < 2:
                continue
            predictions = {
                tuple(sorted((str(k), str(v)) for k, v in h.predicts.items()))
                for h in group
            }
            if len(predictions) > 1:
                return True
        return False

    def detect_preobject_pressure(self, task: TaskState) -> bool:
        if len(task.residuals) < 2:
            return False
        families: Dict[str, int] = {}
        for residual in task.residuals:
            key = ":".join(str(residual).split(":", 2)[:2])
            families[key] = families.get(key, 0) + 1
        return max(families.values(), default=0) >= 2

    def route(self, task: TaskState, pressure: Pressure) -> Tuple[List[str], List[str], List[str]]:
        active: List[str] = list(CORE_MODULES)
        reasons: List[str] = []

        if pressure.ambiguity >= 0.25:
            active += ["QUESTION_FIELD", "COUNTEREXAMPLE_SEARCH"]
            reasons.append("multiple live hypotheses require discrimination")
        elif pressure.contradiction >= 0.34:
            active.append("COUNTEREXAMPLE_SEARCH")
            reasons.append("contradiction pressure requires adversarial checking")

        if pressure.residual >= 0.25 or pressure.novelty >= 0.65:
            active.append("MODAL_EXPANSION")
            reasons.append("residual or novelty justifies possibility-space expansion")

        if self.detect_boundary_shadow(task):
            active.append("REPRESENTATION_ESCAPE")
            reasons.append("current representation collapses causally different hypotheses")

        if pressure.temporal_identity >= 0.5:
            active.append("TEMPORAL_IDENTITY")
            reasons.append("referent identity must survive state change")

        if task.external_world or task.action_required:
            active.append("WORLD_INTERACTION")
            reasons.append("world-facing action requires explicit interaction and verification")

        if pressure.total >= 0.4 or task.action_required:
            active.append("CAUSAL_CREDIT")

        active = self._dedupe(active)
        shadow = [m for m in SPECIALIST_MODULES if m not in active]
        return active, shadow, reasons

    def expand_modal_basis(self, task: TaskState, pressure: Pressure) -> List[str]:
        basis = ["EXPLICIT"]
        if pressure.contradiction >= 0.25 or pressure.ambiguity >= 0.25:
            basis.append("OPPOSITE")
        if pressure.residual >= 0.25:
            basis += ["COMPLEMENT", "ABSENCE"]
        if pressure.stakes >= 0.5 or task.action_required:
            basis.append("COUNTERFACTUAL")
        if self.detect_boundary_shadow(task) or pressure.novelty >= 0.7:
            basis.append("IMAGINARY")
        return self._dedupe(basis)

    def rank_questions(self, task: TaskState) -> List[QuestionScore]:
        if len(task.hypotheses) <= 1 or not task.candidate_queries:
            return []

        priors = {h.hypothesis_id: max(0.0, float(h.prior)) for h in task.hypotheses}
        total = sum(priors.values()) or 1.0
        priors = {k: v / total for k, v in priors.items()}
        prior_entropy = _entropy(list(priors.values()))
        scored: List[QuestionScore] = []

        for q in task.candidate_queries:
            buckets: Dict[str, List[str]] = {}
            unresolved = 0
            for h in task.hypotheses:
                outcome = q.distinguishes.get(h.hypothesis_id)
                if outcome is None:
                    outcome = "__UNKNOWN__"
                    unresolved += 1
                buckets.setdefault(str(outcome), []).append(h.hypothesis_id)

            posterior_entropy = 0.0
            for members in buckets.values():
                mass = sum(priors[h] for h in members)
                local = [priors[h] / mass for h in members] if mass else []
                posterior_entropy += mass * _entropy(local)

            eig = max(0.0, prior_entropy - posterior_entropy)
            coverage = 1.0 - unresolved / max(1, len(task.hypotheses))
            eig *= 0.5 + 0.5 * coverage
            cost = max(1e-6, float(q.cost))
            bonus = 1.1 if q.intervention else 1.0
            scored.append(QuestionScore(
                query_id=q.query_id,
                expected_information_gain=eig,
                cost=cost,
                score=eig * bonus / cost,
                intervention=q.intervention,
                source_class=q.source_class,
            ))
        return sorted(scored, key=lambda x: (-x.score, -x.expected_information_gain, x.query_id))

    def representation_actions(self, task: TaskState) -> List[str]:
        actions: List[str] = []
        if self.detect_boundary_shadow(task):
            actions += ["SPLIT", "EXTEND"]
        if self.detect_preobject_pressure(task):
            actions.append("EXTEND")

        seen: Set[str] = set()
        duplicate = False
        for item in task.current_representation:
            canonical = str(item).strip().lower()
            duplicate = duplicate or canonical in seen
            seen.add(canonical)
        if duplicate:
            actions.append("QUOTIENT")
        if any(str(x).startswith("REDUNDANT:") for x in task.current_representation):
            actions.append("DELETE")
        return self._dedupe(actions)

    def compile(self, task: TaskState) -> CognitionPlan:
        pressure = self.diagnose_pressure(task)
        active, shadow, reasons = self.route(task, pressure)
        questions = self.rank_questions(task)
        boundary = self.detect_boundary_shadow(task)
        preobject = self.detect_preobject_pressure(task)

        if questions and questions[0].score > 0:
            mode = "ACQUIRE_INFORMATION"
        elif boundary or preobject:
            mode = "CHANGE_REPRESENTATION"
        elif task.action_required and pressure.stakes < 0.8:
            mode = "ACT_WITH_VERIFICATION"
        elif pressure.stakes >= 0.8 and len(task.hypotheses) > 1:
            mode = "DEFER_HIGH_STAKES_AND_VERIFY"
        else:
            mode = "REASON_AND_VERIFY"

        return CognitionPlan(
            pressure=pressure,
            active_subgraph=active,
            shadow_subgraph=shadow,
            modal_basis=self.expand_modal_basis(task, pressure),
            question_field=questions,
            representation_actions=self.representation_actions(task),
            boundary_shadow=boundary,
            preobject_pressure=preobject,
            recommended_mode=mode,
            reasons=reasons,
        )

    def assign_causal_credit(
        self,
        active_modules: Iterable[str],
        baseline_decision: Any,
        treatment_decision: Any,
        baseline_outcome: float,
        treatment_outcome: float,
        ablation_decisions: Optional[Mapping[str, Any]] = None,
    ) -> List[ModuleCredit]:
        ablation_decisions = dict(ablation_decisions or {})
        delta = float(treatment_outcome) - float(baseline_outcome)
        result: List[ModuleCredit] = []
        for module in self._dedupe(list(active_modules)):
            changed = treatment_decision != baseline_decision
            if module in ablation_decisions:
                changed = changed and ablation_decisions[module] != treatment_decision
            credit = max(0.0, delta) if changed and delta > 0 else 0.0
            result.append(ModuleCredit(module, True, bool(changed), delta, credit))
        return result

    def reactivate_shadow(self, plan: CognitionPlan, failure_tags: Sequence[str]) -> List[str]:
        tags = {str(x).upper() for x in failure_tags}
        requested: List[str] = []
        if {"MISSED_ALTERNATIVE", "PREMATURE_CLOSURE"} & tags:
            requested.append("MODAL_EXPANSION")
        if {"REPRESENTATION_COLLISION", "BOUNDARY_SHADOW"} & tags:
            requested.append("REPRESENTATION_ESCAPE")
        if {"WRONG_REFERENT", "TEMPORAL_IDENTITY"} & tags:
            requested.append("TEMPORAL_IDENTITY")
        if {"INSUFFICIENT_EVIDENCE", "AMBIGUOUS"} & tags:
            requested.append("QUESTION_FIELD")
        if {"WORLD_MISMATCH", "ACTION_FAILURE"} & tags:
            requested.append("WORLD_INTERACTION")
        return [m for m in self._dedupe(requested) if m in plan.shadow_subgraph]

    @staticmethod
    def _dedupe(items: Sequence[str]) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for item in items:
            if item not in seen:
                out.append(item)
                seen.add(item)
        return out


def plan_to_dict(plan: CognitionPlan) -> Dict[str, Any]:
    return {
        "pressure": asdict(plan.pressure),
        "active_subgraph": list(plan.active_subgraph),
        "shadow_subgraph": list(plan.shadow_subgraph),
        "modal_basis": list(plan.modal_basis),
        "question_field": [asdict(q) for q in plan.question_field],
        "representation_actions": list(plan.representation_actions),
        "boundary_shadow": plan.boundary_shadow,
        "preobject_pressure": plan.preobject_pressure,
        "recommended_mode": plan.recommended_mode,
        "reasons": list(plan.reasons),
    }
