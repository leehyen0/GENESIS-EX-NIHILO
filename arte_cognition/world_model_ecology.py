from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math

from .adaptive_cognition import QueryCandidate
from .world_coupling import WorldOutcomePair


def _entropy(probs: Sequence[float]) -> float:
    vals = [max(0.0, float(p)) for p in probs]
    total = sum(vals)
    if total <= 0:
        return 0.0
    vals = [p / total for p in vals if p > 0]
    return -sum(p * math.log2(p) for p in vals)


@dataclass(frozen=True)
class CausalWorldModel:
    model_id: str
    prior: float
    predictions: Tuple[Tuple[str, str], ...]

    def prediction_for(self, intervention_id: str) -> Optional[str]:
        return dict(self.predictions).get(intervention_id)


@dataclass(frozen=True)
class ModelEvidence:
    evidence_id: str
    intervention_id: str
    observed_outcome: str
    source_class: str
    context_id: str
    authoritative: bool


@dataclass(frozen=True)
class EpistemicDepthPlan:
    mode: str
    normalized_model_entropy: float
    model_class_inadequate: bool
    possibility_budget: int
    representation_axis_budget: int
    intervention_budget: int
    cost_exponent: float
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class EpistemicInterventionScore:
    intervention_id: str
    expected_information_gain: float
    cost: float
    utility: float


class WorldModelEcology:
    """Maintain competing causal models and spend more when epistemic depth demands it.

    Cost is a soft penalty, not the optimization target. When competing models
    remain unresolved or authenticated world evidence contradicts every live model,
    the BODY expands possibility/representation/intervention budgets and discounts
    cost more weakly so high-information interventions can dominate cheap but
    uninformative probes.
    """

    def __init__(self) -> None:
        self.models: Dict[str, CausalWorldModel] = {}
        self.evidence: List[ModelEvidence] = []
        self.inadequacy_events: List[str] = []

    def register(self, models: Iterable[CausalWorldModel]) -> None:
        for model in models:
            if not model.model_id:
                continue
            self.models[model.model_id] = model

    def posterior(self) -> Dict[str, float]:
        if not self.models:
            return {}
        weights = {mid: max(1e-9, float(model.prior)) for mid, model in self.models.items()}
        for ev in self.evidence:
            if not ev.authoritative:
                continue
            for mid, model in self.models.items():
                prediction = model.prediction_for(ev.intervention_id)
                if prediction is None:
                    likelihood = 0.5
                elif prediction == ev.observed_outcome:
                    likelihood = 0.95
                else:
                    likelihood = 0.05
                weights[mid] *= likelihood
        total = sum(weights.values()) or 1.0
        return {mid: value / total for mid, value in weights.items()}

    def normalized_entropy(self) -> float:
        posterior = self.posterior()
        if len(posterior) <= 1:
            return 0.0
        return _entropy(list(posterior.values())) / math.log2(len(posterior))

    def observe(self, evidence: ModelEvidence) -> bool:
        if evidence.evidence_id in {item.evidence_id for item in self.evidence}:
            return False
        self.evidence.append(evidence)
        if evidence.authoritative and self.models:
            predictions = [
                model.prediction_for(evidence.intervention_id)
                for model in self.models.values()
            ]
            known = [prediction for prediction in predictions if prediction is not None]
            if known and all(prediction != evidence.observed_outcome for prediction in known):
                self.inadequacy_events.append(evidence.evidence_id)
        return True

    @staticmethod
    def outcome_label(pair: WorldOutcomePair, epsilon: float = 1e-9) -> str:
        if pair.effect > epsilon:
            return "POSITIVE_EFFECT"
        if pair.effect < -epsilon:
            return "NEGATIVE_EFFECT"
        return "NO_EFFECT"

    def observe_world_pair(self, pair: WorldOutcomePair) -> bool:
        authoritative = bool(
            pair.authority_verified
            and pair.matched_budget
            and pair.externally_generated
            and pair.independence_class_id != "UNVERIFIED"
        )
        return self.observe(ModelEvidence(
            evidence_id=pair.pair_id,
            intervention_id=pair.experiment_id,
            observed_outcome=self.outcome_label(pair),
            source_class=pair.independence_class_id,
            context_id=pair.context_id,
            authoritative=authoritative,
        ))

    @property
    def model_class_inadequate(self) -> bool:
        return bool(self.inadequacy_events)

    def depth_plan(self) -> EpistemicDepthPlan:
        entropy = self.normalized_entropy()
        reasons: List[str] = []
        if self.model_class_inadequate:
            reasons.append("authenticated outcome contradicted every live causal model")
            return EpistemicDepthPlan(
                mode="EXPAND_MODEL_CLASS",
                normalized_model_entropy=entropy,
                model_class_inadequate=True,
                possibility_budget=128,
                representation_axis_budget=64,
                intervention_budget=8,
                cost_exponent=0.15,
                reasons=tuple(reasons),
            )
        if entropy >= 0.55 and len(self.models) >= 2:
            reasons.append("competing causal models remain unresolved")
            return EpistemicDepthPlan(
                mode="DEEP_DISCRIMINATION",
                normalized_model_entropy=entropy,
                model_class_inadequate=False,
                possibility_budget=64,
                representation_axis_budget=32,
                intervention_budget=4,
                cost_exponent=0.30,
                reasons=tuple(reasons),
            )
        reasons.append("current causal model posterior is sufficiently concentrated")
        return EpistemicDepthPlan(
            mode="COMPACT",
            normalized_model_entropy=entropy,
            model_class_inadequate=False,
            possibility_budget=32,
            representation_axis_budget=16,
            intervention_budget=2,
            cost_exponent=1.0,
            reasons=tuple(reasons),
        )

    def rank_interventions(self, candidates: Sequence[QueryCandidate]) -> List[EpistemicInterventionScore]:
        posterior = self.posterior()
        if not posterior or not candidates:
            return []
        prior_entropy = _entropy(list(posterior.values()))
        depth = self.depth_plan()
        scored: List[EpistemicInterventionScore] = []
        for candidate in candidates:
            buckets: Dict[str, List[str]] = {}
            for model_id in posterior:
                outcome = candidate.distinguishes.get(model_id, "__UNKNOWN__")
                buckets.setdefault(str(outcome), []).append(model_id)
            posterior_entropy = 0.0
            for members in buckets.values():
                mass = sum(posterior[mid] for mid in members)
                local = [posterior[mid] / mass for mid in members] if mass > 0 else []
                posterior_entropy += mass * _entropy(local)
            eig = max(0.0, prior_entropy - posterior_entropy)
            cost = max(1e-9, float(candidate.cost))
            utility = eig / (cost ** depth.cost_exponent)
            scored.append(EpistemicInterventionScore(
                intervention_id=candidate.query_id,
                expected_information_gain=eig,
                cost=cost,
                utility=utility,
            ))
        return sorted(scored, key=lambda item: (-item.utility, -item.expected_information_gain, item.intervention_id))

    def select_interventions(self, candidates: Sequence[QueryCandidate]) -> List[EpistemicInterventionScore]:
        ranked = self.rank_interventions(candidates)
        return ranked[: self.depth_plan().intervention_budget]
