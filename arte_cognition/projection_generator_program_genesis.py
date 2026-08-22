from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .world_coupling import WorldOutcomePair


DEFAULT_PROGRAM_FAMILIES: Tuple[str, ...] = ("AFFINE", "GEOMETRIC", "HARMONIC")
DEFAULT_PROGRAM_ALPHAS: Tuple[float, ...] = (0.25, 0.5, 0.75)
PROGRAM_MARKER = "generator_programs="


@dataclass(frozen=True)
class ProjectionGeneratorProgram:
    program_id: str
    family: str
    alpha: float
    complexity: int

    def apply(self, left: float, right: float) -> Optional[float]:
        left = float(left)
        right = float(right)
        alpha = float(self.alpha)
        if not (0.0 < alpha < 1.0) or not (left < right):
            return None
        if self.family == "AFFINE":
            value = left + alpha * (right - left)
        elif self.family == "GEOMETRIC":
            if left <= 0.0 or right <= 0.0:
                return None
            value = math.exp((1.0 - alpha) * math.log(left) + alpha * math.log(right))
        elif self.family == "HARMONIC":
            if left <= 0.0 or right <= 0.0:
                return None
            denominator = (1.0 - alpha) / left + alpha / right
            if denominator <= 0.0:
                return None
            value = 1.0 / denominator
        else:
            return None
        if not math.isfinite(value) or not (left < value < right):
            return None
        return round(float(value), 12)


@dataclass(frozen=True)
class ProjectionGeneratorProgramCandidate:
    scale: float
    program_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ProjectionGeneratorProgramPolicy:
    status: str
    program_id: Optional[str]
    family: Optional[str]
    alpha: Optional[float]
    supporting_contexts: Tuple[str, ...]
    candidate_program_count: int
    reason: str


@dataclass(frozen=True)
class ProjectionGeneratorProgramFrontier:
    status: str
    candidates: Tuple[ProjectionGeneratorProgramCandidate, ...]
    bracket: Tuple[float, float]
    policy_program_id: Optional[str]
    shadow_program_count: int
    reason: str


def generate_projection_generator_programs(
    families: Sequence[str] = DEFAULT_PROGRAM_FAMILIES,
    alphas: Sequence[float] = DEFAULT_PROGRAM_ALPHAS,
) -> Tuple[ProjectionGeneratorProgram, ...]:
    """Generate a bounded, outcome-independent refinement program language."""
    complexity = {"AFFINE": 1, "GEOMETRIC": 2, "HARMONIC": 2}
    out = []
    for family in sorted(set(str(item).upper() for item in families)):
        if family not in complexity:
            continue
        for alpha in sorted(set(round(float(value), 12) for value in alphas if 0.0 < float(value) < 1.0)):
            out.append(ProjectionGeneratorProgram(
                program_id=f"GENERATOR::{family}::ALPHA::{alpha:g}",
                family=family,
                alpha=alpha,
                complexity=complexity[family],
            ))
    return tuple(sorted(out, key=lambda item: (item.complexity, item.family, item.alpha)))


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def parse_generator_program_ids(proposal: InterventionProposal) -> Tuple[str, ...]:
    reason = str(proposal.reason)
    if PROGRAM_MARKER not in reason:
        return ()
    tail = reason.split(PROGRAM_MARKER, 1)[1].strip().split()[0].rstrip(",;)")
    return tuple(sorted(set(item for item in tail.split("|") if item)))


def _strong_program_contexts(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float,
) -> Dict[str, Dict[str, float]]:
    programs_by_experiment = {
        proposal.experiment_id: parse_generator_program_ids(proposal)
        for proposal in proposals
        if parse_generator_program_ids(proposal)
    }
    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair) or pair.experiment_id not in programs_by_experiment:
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(
            pair.independence_class_id, pair
        )
    minimum = max(1, int(min_independent_classes))
    out: Dict[str, Dict[str, float]] = {}
    for (experiment_id, context_id), by_class in grouped.items():
        if len(by_class) < minimum:
            continue
        score = sum(abs(pair.effect) for pair in by_class.values()) / len(by_class)
        if score < float(strong_effect_threshold):
            continue
        for program_id in programs_by_experiment[experiment_id]:
            contexts = out.setdefault(program_id, {})
            contexts[context_id] = max(contexts.get(context_id, 0.0), float(score))
    return out


def derive_projection_generator_program_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    programs: Sequence[ProjectionGeneratorProgram] = (),
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> ProjectionGeneratorProgramPolicy:
    """Select a generator program only after repeated authenticated causal success."""
    program_space = tuple(programs) or generate_projection_generator_programs()
    by_id = {program.program_id: program for program in program_space}
    support = _strong_program_contexts(
        proposals,
        world_pairs,
        min_independent_classes,
        strong_effect_threshold,
    )
    minimum = max(1, int(min_contexts))
    eligible = []
    for program_id, contexts in support.items():
        program = by_id.get(program_id)
        if program is None or len(contexts) < minimum:
            continue
        mean_score = sum(contexts.values()) / len(contexts)
        eligible.append((
            -len(contexts),
            -mean_score,
            program.complexity,
            program.family,
            program.alpha,
            program_id,
            tuple(sorted(contexts)),
        ))
    eligible.sort()
    if not eligible:
        return ProjectionGeneratorProgramPolicy(
            status="NO_REPRODUCED_GENERATOR_PROGRAM",
            program_id=None,
            family=None,
            alpha=None,
            supporting_contexts=(),
            candidate_program_count=len(program_space),
            reason="no outcome-independent generator program reproduced strong authenticated effects across enough contexts",
        )
    chosen = eligible[0]
    program = by_id[chosen[5]]
    return ProjectionGeneratorProgramPolicy(
        status="REPRODUCED_GENERATOR_PROGRAM",
        program_id=program.program_id,
        family=program.family,
        alpha=program.alpha,
        supporting_contexts=tuple(chosen[6]),
        candidate_program_count=len(program_space),
        reason="minimum-complexity generator program retained after repeated authenticated cross-context success",
    )


def _scale_scores(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    probe_scale: Callable[[InterventionProposal], Optional[float]],
    context_id: str,
) -> Dict[float, float]:
    scale_by_experiment = {}
    for proposal in proposals:
        scale = probe_scale(proposal)
        if scale is not None:
            scale_by_experiment[proposal.experiment_id] = float(scale)
    grouped: Dict[str, Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if (
            pair.context_id == context_id
            and _authoritative(pair)
            and pair.experiment_id in scale_by_experiment
        ):
            grouped.setdefault(pair.experiment_id, {}).setdefault(pair.independence_class_id, pair)
    minimum = max(1, int(min_independent_classes))
    out: Dict[float, float] = {}
    for experiment_id, by_class in grouped.items():
        if len(by_class) < minimum:
            continue
        score = sum(abs(pair.effect) for pair in by_class.values()) / len(by_class)
        scale = scale_by_experiment[experiment_id]
        out[scale] = max(out.get(scale, 0.0), float(score))
    return out


def derive_projection_generator_program_frontier(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    probe_scale: Callable[[InterventionProposal], Optional[float]],
    context_id: str,
    left: float,
    right: float,
    policy: Optional[ProjectionGeneratorProgramPolicy] = None,
    programs: Sequence[ProjectionGeneratorProgram] = (),
    strong_effect_threshold: float = 0.9,
    max_candidates: int = 32,
) -> ProjectionGeneratorProgramFrontier:
    """Instantiate a weak-bracket generator frontier without using hidden outcomes.

    External outcomes gate whether refinement is needed, but candidate program
    semantics are generated independently of those outcomes. Once a program has
    cross-context authority, only that program is transferred into a fresh bracket.
    """
    left = float(left)
    right = float(right)
    if not (0.0 < left < right):
        return ProjectionGeneratorProgramFrontier(
            "INVALID_BRACKET", (), (left, right), None, 0, "positive ordered bracket required"
        )
    proposal_list = tuple(proposals)
    scores = _scale_scores(
        proposal_list,
        world_pairs,
        min_independent_classes,
        probe_scale,
        context_id,
    )
    endpoint_scores = [scores.get(left), scores.get(right)]
    if any(value is None for value in endpoint_scores):
        return ProjectionGeneratorProgramFrontier(
            "INSUFFICIENT_AUTHENTICATED_BRACKET",
            (),
            (left, right),
            None,
            0,
            "both bracket endpoints require authenticated independent outcomes",
        )
    if max(float(value) for value in endpoint_scores if value is not None) >= float(strong_effect_threshold):
        return ProjectionGeneratorProgramFrontier(
            "STRONG_ENDPOINT_AVAILABLE",
            (),
            (left, right),
            None,
            0,
            "an existing bracket endpoint already reaches the strong-effect criterion",
        )

    program_space = tuple(programs) or generate_projection_generator_programs()
    by_id = {program.program_id: program for program in program_space}
    if policy is not None and policy.status == "REPRODUCED_GENERATOR_PROGRAM" and policy.program_id in by_id:
        active_programs = (by_id[policy.program_id],)
        status = "LEARNED_GENERATOR_PROGRAM_TRANSFER"
        policy_program_id = policy.program_id
        reason = "reproduced generator program transferred to fresh weak bracket"
    else:
        active_programs = program_space
        status = "SHADOW_GENERATOR_PROGRAM_GENESIS"
        policy_program_id = None
        reason = "bounded outcome-independent generator program space opened by weak authenticated bracket"

    programs_by_scale: Dict[float, set[str]] = {}
    for program in active_programs:
        value = program.apply(left, right)
        if value is None:
            continue
        programs_by_scale.setdefault(value, set()).add(program.program_id)
    candidates = tuple(
        ProjectionGeneratorProgramCandidate(scale=scale, program_ids=tuple(sorted(program_ids)))
        for scale, program_ids in sorted(programs_by_scale.items())[: max(1, int(max_candidates))]
    )
    return ProjectionGeneratorProgramFrontier(
        status=status,
        candidates=candidates,
        bracket=(left, right),
        policy_program_id=policy_program_id,
        shadow_program_count=len(program_space),
        reason=reason,
    )
