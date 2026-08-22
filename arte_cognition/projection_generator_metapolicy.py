from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .world_coupling import WorldOutcomePair


DEFAULT_SHADOW_ALPHAS: Tuple[float, ...] = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class ProjectionGeneratorPolicy:
    status: str
    alpha: Optional[float]
    supporting_contexts: Tuple[str, ...]
    candidate_alpha_count: int
    strong_effect_threshold: float
    reason: str


@dataclass(frozen=True)
class ProjectionGeneratorFrontier:
    status: str
    candidate_scales: Tuple[float, ...]
    generator_alphas: Tuple[float, ...]
    learned_alpha: Optional[float]
    observed_authored_scales: Tuple[float, ...]
    best_verified_effect: float
    strong_effect_threshold: float
    reason: str


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def _proposal_scale_map(
    proposals: Iterable[InterventionProposal],
    probe_scale: Callable[[InterventionProposal], Optional[float]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for proposal in proposals:
        scale = probe_scale(proposal)
        if scale is not None and float(scale) > 0.0:
            out[proposal.experiment_id] = float(scale)
    return out


def _strong_scale_contexts(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    probe_scale: Callable[[InterventionProposal], Optional[float]],
    strong_effect_threshold: float,
) -> Dict[float, Dict[str, float]]:
    proposal_scale = _proposal_scale_map(proposals, probe_scale)
    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair) or pair.experiment_id not in proposal_scale:
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(
            pair.independence_class_id, pair
        )

    minimum = max(1, int(min_independent_classes))
    result: Dict[float, Dict[str, float]] = {}
    for (experiment_id, context_id), by_class in grouped.items():
        if len(by_class) < minimum:
            continue
        score = sum(abs(pair.effect) for pair in by_class.values()) / len(by_class)
        if score < float(strong_effect_threshold):
            continue
        scale = proposal_scale[experiment_id]
        current = result.setdefault(scale, {})
        current[context_id] = max(current.get(context_id, 0.0), float(score))
    return result


def _authored_bracket(authored_scales: Sequence[float], scale: float) -> Optional[Tuple[float, float]]:
    authored = tuple(sorted(set(float(value) for value in authored_scales if float(value) > 0.0)))
    for left, right in zip(authored, authored[1:]):
        if left < float(scale) < right:
            return left, right
    return None


def _alpha_for_scale(authored_scales: Sequence[float], scale: float) -> Optional[float]:
    bracket = _authored_bracket(authored_scales, scale)
    if bracket is None:
        return None
    left, right = bracket
    width = right - left
    if width <= 0.0:
        return None
    alpha = (float(scale) - left) / width
    if not (0.0 < alpha < 1.0):
        return None
    return round(alpha, 12)


def derive_projection_generator_policy(
    authored_scales: Sequence[float],
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    probe_scale: Callable[[InterventionProposal], Optional[float]],
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> ProjectionGeneratorPolicy:
    """Induce a reusable interpolation generator from externally validated atoms.

    A generated off-authored scale is mapped back to its normalized position alpha
    inside the surrounding authored interval. The BODY may reuse an alpha only when
    the corresponding generated atom has independently reproduced with strong effect
    across multiple world contexts. The policy is reconstructed from world evidence;
    alpha is not trusted as serialized authority.
    """
    authored = tuple(sorted(set(float(value) for value in authored_scales if float(value) > 0.0)))
    strong = _strong_scale_contexts(
        proposals,
        world_pairs,
        min_independent_classes,
        probe_scale,
        strong_effect_threshold,
    )
    by_alpha: Dict[float, Dict[str, float]] = {}
    for scale, contexts in strong.items():
        if any(abs(scale - base) <= 1e-12 for base in authored):
            continue
        alpha = _alpha_for_scale(authored, scale)
        if alpha is None:
            continue
        target = by_alpha.setdefault(alpha, {})
        for context_id, score in contexts.items():
            target[context_id] = max(target.get(context_id, 0.0), float(score))

    minimum_contexts = max(1, int(min_contexts))
    eligible = []
    for alpha, contexts in by_alpha.items():
        if len(contexts) < minimum_contexts:
            continue
        mean_score = sum(contexts.values()) / len(contexts)
        eligible.append((-len(contexts), -mean_score, abs(alpha - 0.5), alpha, tuple(sorted(contexts))))
    eligible.sort()
    if not eligible:
        return ProjectionGeneratorPolicy(
            status="NO_REPRODUCED_GENERATOR_POLICY",
            alpha=None,
            supporting_contexts=(),
            candidate_alpha_count=len(by_alpha),
            strong_effect_threshold=float(strong_effect_threshold),
            reason="no generated interpolation position has strong authenticated reproduction across enough contexts",
        )

    chosen = eligible[0]
    return ProjectionGeneratorPolicy(
        status="REPRODUCED_GENERATOR_POLICY",
        alpha=float(chosen[3]),
        supporting_contexts=tuple(chosen[4]),
        candidate_alpha_count=len(by_alpha),
        strong_effect_threshold=float(strong_effect_threshold),
        reason="minimum-description interpolation generator reconstructed from repeated authenticated generated-atom success",
    )


def derive_projection_generator_frontier(
    authored_scales: Sequence[float],
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    probe_scale: Callable[[InterventionProposal], Optional[float]],
    context_id: Optional[str],
    learned_policy: Optional[ProjectionGeneratorPolicy] = None,
    shadow_alphas: Sequence[float] = DEFAULT_SHADOW_ALPHAS,
    strong_effect_threshold: float = 0.9,
    max_candidates: int = 16,
) -> ProjectionGeneratorFrontier:
    """Generate refinement atoms with either a learned generator or shadow programs.

    Before a generator is learned, the BODY explores a bounded shadow program space
    of interpolation positions. After cross-context causal reproduction, only the
    learned generator is applied to every authored interval, enabling transfer of a
    refinement *rule* to a bracket not previously solved by that generated atom.
    """
    authored = tuple(sorted(set(float(value) for value in authored_scales if float(value) > 0.0)))
    proposal_scale = _proposal_scale_map(proposals, probe_scale)
    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair) or pair.experiment_id not in proposal_scale:
            continue
        if context_id is not None and pair.context_id != context_id:
            continue
        scale = proposal_scale[pair.experiment_id]
        if not any(abs(scale - base) <= 1e-12 for base in authored):
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(
            pair.independence_class_id, pair
        )

    minimum = max(1, int(min_independent_classes))
    scores: Dict[float, float] = {}
    for (experiment_id, _context), by_class in grouped.items():
        if len(by_class) < minimum:
            continue
        score = sum(abs(pair.effect) for pair in by_class.values()) / len(by_class)
        scale = proposal_scale[experiment_id]
        scores[scale] = max(scores.get(scale, 0.0), float(score))

    observed = tuple(sorted(scores))
    threshold = float(strong_effect_threshold)
    best = max(scores.values(), default=0.0)
    if len(observed) < 2:
        return ProjectionGeneratorFrontier(
            "INSUFFICIENT_AUTHENTICATED_BRACKET",
            (), (), None, observed, best, threshold,
            "at least two authored probe locations need authenticated outcomes in this context",
        )
    if best >= threshold:
        return ProjectionGeneratorFrontier(
            "STRONG_AUTHORED_SCALE_AVAILABLE",
            (), (), None, observed, best, threshold,
            "an authored probe scale already reaches the strong-effect criterion in this context",
        )

    if learned_policy is not None and learned_policy.alpha is not None and learned_policy.status == "REPRODUCED_GENERATOR_POLICY":
        alphas = (float(learned_policy.alpha),)
        learned_alpha: Optional[float] = float(learned_policy.alpha)
        status = "LEARNED_GENERATOR_TRANSFER"
        reason = "reproduced generator policy transferred to a new weak authored bracket"
    else:
        alphas = tuple(sorted(set(
            round(float(alpha), 12)
            for alpha in shadow_alphas
            if 0.0 < float(alpha) < 1.0
        )))
        learned_alpha = None
        status = "SHADOW_GENERATOR_PROGRAM_SEARCH"
        reason = "no reproduced generator policy; exploring bounded interpolation programs"

    candidates = []
    for left, right in zip(authored, authored[1:]):
        if left not in scores or right not in scores:
            continue
        width = right - left
        for alpha in alphas:
            scale = left + alpha * width
            if scale <= 0.0 or any(abs(scale - base) <= 1e-12 for base in authored):
                continue
            candidates.append(round(float(scale), 12))
    candidates = sorted(set(candidates))[: max(1, int(max_candidates))]
    return ProjectionGeneratorFrontier(
        status=status,
        candidate_scales=tuple(candidates),
        generator_alphas=alphas,
        learned_alpha=learned_alpha,
        observed_authored_scales=observed,
        best_verified_effect=best,
        strong_effect_threshold=threshold,
        reason=reason,
    )
