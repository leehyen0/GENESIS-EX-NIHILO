from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .world_coupling import WorldOutcomePair


@dataclass(frozen=True)
class ProjectionSearchMetaPolicy:
    schedule: Tuple[float, ...]
    observed_contexts: Tuple[str, ...]
    covered_contexts: Tuple[str, ...]
    candidate_count: int
    material_effect_threshold: float
    reason: str


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def derive_projection_search_metapolicy(
    base_scales: Sequence[float],
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    probe_scale: Callable[[InterventionProposal], Optional[float]],
    material_effect_threshold: float = 0.5,
) -> ProjectionSearchMetaPolicy:
    """Search a bounded policy space from authenticated cross-context evidence.

    The policy grammar is the non-empty power set of the authored base scales.
    Selection is not a fixed top-k threshold rule: each candidate schedule must
    preserve a material world effect in every observed context for which the BODY
    has policy evidence. A singleton additionally requires material reproduction
    in at least two distinct contexts before the search vocabulary may collapse to
    one scale. This protects against one-context overfitting while allowing a
    heterogeneous minimal subset such as {1x, 4x} when different regimes require
    different probes.

    This is bounded metapolicy search, not unrestricted code/operator invention.
    """
    base = tuple(float(value) for value in base_scales)
    if len(base) <= 1:
        return ProjectionSearchMetaPolicy(
            base, (), (), 1, float(material_effect_threshold), "base vocabulary already minimal"
        )

    base_index = {scale: index for index, scale in enumerate(base)}
    proposal_scale: Dict[str, float] = {}
    for proposal in proposals:
        scale = probe_scale(proposal)
        if scale is None:
            continue
        matched = next((item for item in base if abs(item - float(scale)) <= 1e-12), None)
        if matched is not None:
            proposal_scale[proposal.experiment_id] = matched

    # context -> scale -> independently verified mean |effect|, taking the best
    # exact experiment at that scale inside the context.
    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair) or pair.experiment_id not in proposal_scale:
            continue
        key = (pair.experiment_id, pair.context_id)
        grouped.setdefault(key, {}).setdefault(pair.independence_class_id, pair)

    context_scores: Dict[str, Dict[float, float]] = {}
    minimum = max(1, int(min_independent_classes))
    for (experiment_id, context_id), by_class in grouped.items():
        if len(by_class) < minimum:
            continue
        scale = proposal_scale[experiment_id]
        unique = list(by_class.values())
        score = sum(abs(pair.effect) for pair in unique) / len(unique)
        current = context_scores.setdefault(context_id, {}).get(scale)
        if current is None or score > current:
            context_scores[context_id][scale] = float(score)

    contexts = tuple(sorted(context_scores))
    if not contexts:
        return ProjectionSearchMetaPolicy(
            base, (), (), (2 ** len(base)) - 1, float(material_effect_threshold),
            "no authenticated policy evidence",
        )

    # Every authored scale must have been tested somewhere before any scale can be
    # removed from the search vocabulary.
    observed_scales = {
        scale for scores in context_scores.values() for scale in scores
    }
    if any(scale not in observed_scales for scale in base):
        return ProjectionSearchMetaPolicy(
            base, contexts, contexts, (2 ** len(base)) - 1, float(material_effect_threshold),
            "at least one base scale remains globally untested",
        )

    threshold = float(material_effect_threshold)
    global_scores = {
        scale: max(
            (scores.get(scale, 0.0) for scores in context_scores.values()),
            default=0.0,
        )
        for scale in base
    }

    candidates = []
    for size in range(1, len(base) + 1):
        for subset in combinations(base, size):
            # A candidate must have at least one actually observed included scale
            # in every historical context, and one of those observed scales must
            # preserve a material effect there.
            context_best = []
            valid = True
            for context in contexts:
                observed = [
                    context_scores[context][scale]
                    for scale in subset
                    if scale in context_scores[context]
                ]
                if not observed or max(observed) < threshold:
                    valid = False
                    break
                context_best.append(max(observed))
            if not valid:
                continue

            # One-scale metapolicy promotion requires actual material reproduction
            # of that exact scale across two or more distinct world contexts.
            if size == 1:
                scale = subset[0]
                reproduced = sum(
                    1
                    for context in contexts
                    if context_scores[context].get(scale, 0.0) >= threshold
                )
                if reproduced < 2:
                    continue

            ordered = tuple(sorted(
                subset,
                key=lambda scale: (-global_scores[scale], base_index[scale]),
            ))
            worst = min(context_best)
            mean = sum(context_best) / len(context_best)
            index_signature = tuple(base_index[scale] for scale in subset)
            candidates.append((size, -worst, -mean, index_signature, ordered))

    if not candidates:
        return ProjectionSearchMetaPolicy(
            base, contexts, contexts, (2 ** len(base)) - 1, threshold,
            "no smaller policy preserves authenticated material capability across contexts",
        )

    candidates.sort()
    selected = candidates[0][4]
    reason = (
        "minimum authenticated cross-context capability-preserving search policy"
        if len(selected) < len(base)
        else "full authored vocabulary remains the minimum sufficient policy"
    )
    return ProjectionSearchMetaPolicy(
        selected,
        contexts,
        contexts,
        (2 ** len(base)) - 1,
        threshold,
        reason,
    )
