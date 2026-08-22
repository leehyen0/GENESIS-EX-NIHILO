from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .world_coupling import WorldOutcomePair


@dataclass(frozen=True)
class ProjectionScaleFrontier:
    status: str
    candidate_scales: Tuple[float, ...]
    observed_scales: Tuple[float, ...]
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


def projection_scale_scores(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    probe_scale: Callable[[InterventionProposal], Optional[float]],
    context_id: Optional[str] = None,
) -> Dict[float, float]:
    proposal_scale: Dict[str, float] = {}
    for proposal in proposals:
        scale = probe_scale(proposal)
        if scale is not None and float(scale) > 0.0:
            proposal_scale[proposal.experiment_id] = float(scale)

    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair) or pair.experiment_id not in proposal_scale:
            continue
        if context_id is not None and pair.context_id != context_id:
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(
            pair.independence_class_id, pair
        )

    minimum = max(1, int(min_independent_classes))
    scores: Dict[float, float] = {}
    for (experiment_id, _context), by_class in grouped.items():
        if len(by_class) < minimum:
            continue
        unique = list(by_class.values())
        score = sum(abs(pair.effect) for pair in unique) / len(unique)
        scale = proposal_scale[experiment_id]
        scores[scale] = max(scores.get(scale, 0.0), float(score))
    return scores


def validated_generated_projection_scales(
    authored_scales: Sequence[float],
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    probe_scale: Callable[[InterventionProposal], Optional[float]],
    strong_effect_threshold: float = 0.9,
) -> Tuple[float, ...]:
    authored = tuple(float(value) for value in authored_scales)
    scores = projection_scale_scores(
        proposals, world_pairs, min_independent_classes, probe_scale
    )
    generated = [
        scale
        for scale, score in scores.items()
        if score >= float(strong_effect_threshold)
        and all(abs(scale - base) > 1e-12 for base in authored)
    ]
    return tuple(sorted(set(generated)))


def derive_projection_scale_frontier(
    authored_scales: Sequence[float],
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    probe_scale: Callable[[InterventionProposal], Optional[float]],
    context_id: Optional[str] = None,
    strong_effect_threshold: float = 0.9,
    max_candidates: int = 8,
) -> ProjectionScaleFrontier:
    """Generate new numeric probe atoms from an externally observed search residual.

    If no already-tested scale has a strong authenticated world effect, the BODY
    refines the observed one-dimensional intervention vocabulary by generating
    arithmetic midpoints between adjacent tested scales. Outcome values prioritize
    which intervals are searched first but do not define the midpoint value itself.
    New scales remain proposal-only until separately executed and authenticated.

    This is bounded numeric parameter genesis. The midpoint refinement operator,
    strong-effect criterion and candidate budget remain authored; arbitrary code or
    unrestricted action-operator invention is not claimed.
    """
    scores = projection_scale_scores(
        proposals,
        world_pairs,
        min_independent_classes,
        probe_scale,
        context_id=context_id,
    )
    observed = tuple(sorted(scores))
    threshold = float(strong_effect_threshold)
    best = max(scores.values(), default=0.0)
    if not observed:
        return ProjectionScaleFrontier(
            "NO_AUTHENTICATED_SCALE_EVIDENCE", (), (), 0.0, threshold,
            "no exact probe scale has enough independent external authority",
        )
    if best >= threshold:
        return ProjectionScaleFrontier(
            "STRONG_SCALE_ALREADY_AVAILABLE", (), observed, best, threshold,
            "an already-tested scale reaches the bounded strong-effect criterion",
        )
    if len(observed) < 2:
        return ProjectionScaleFrontier(
            "INSUFFICIENT_BRACKET", (), observed, best, threshold,
            "at least two authenticated scale locations are required for refinement",
        )

    ranked = []
    for left, right in zip(observed, observed[1:]):
        midpoint = (left + right) / 2.0
        if midpoint <= 0.0 or any(abs(midpoint - value) <= 1e-12 for value in observed):
            continue
        endpoint_best = max(scores[left], scores[right])
        endpoint_mean = (scores[left] + scores[right]) / 2.0
        width = right - left
        ranked.append((
            -endpoint_best,
            -endpoint_mean,
            -width,
            midpoint,
            left,
            right,
        ))
    ranked.sort()
    budget = max(1, int(max_candidates))
    candidates = tuple(item[3] for item in ranked[:budget])
    if not candidates:
        return ProjectionScaleFrontier(
            "NO_NUMERIC_REFINEMENT_CANDIDATES", (), observed, best, threshold,
            "tested scale geometry produced no novel bounded midpoint",
        )
    return ProjectionScaleFrontier(
        "GENERATED_NUMERIC_REFINEMENT",
        candidates,
        observed,
        best,
        threshold,
        "world residual opened bounded midpoint refinement of the probe vocabulary",
    )
