from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .projection_transform_depth_genesis import ProjectionTransformDepthAssessment
from .world_coupling import WorldOutcomePair


TRANSFORM_PRIMITIVE_MARKER = "generated_transform_primitives="
DEFAULT_POWER_EXPONENTS: Tuple[float, ...] = (-2.0, -0.5, 0.5, 2.0)
DEFAULT_PRIMITIVE_ALPHAS: Tuple[float, ...] = (0.25, 0.5, 0.75)
PRIMITIVE_SIGNATURE_BRACKETS: Tuple[Tuple[float, float], ...] = (
    (2.0, 8.0),
    (3.0, 27.0),
    (4.0, 64.0),
    (5.0, 125.0),
)


@dataclass(frozen=True)
class ProjectionPowerPrimitiveProgram:
    primitive_id: str
    exponent: float
    alpha: float
    program_id: str
    complexity: int = 1

    def apply(self, left: float, right: float) -> Optional[float]:
        left = float(left)
        right = float(right)
        p = float(self.exponent)
        alpha = float(self.alpha)
        if not (0.0 < left < right and 0.0 < alpha < 1.0):
            return None
        if abs(p) <= 1e-12:
            return None
        try:
            mixed = (1.0 - alpha) * (left ** p) + alpha * (right ** p)
            if mixed <= 0.0:
                return None
            value = mixed ** (1.0 / p)
        except (ValueError, OverflowError, ZeroDivisionError):
            return None
        if not math.isfinite(value) or not (left < value < right):
            return None
        return round(float(value), 12)


@dataclass(frozen=True)
class ProjectionPrimitiveCandidate:
    scale: float
    program_ids: Tuple[str, ...]
    primitive_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ProjectionPrimitivePolicy:
    status: str
    primitive_id: Optional[str]
    exponent: Optional[float]
    alpha: Optional[float]
    program_id: Optional[str]
    supporting_contexts: Tuple[str, ...]
    candidate_program_count: int
    reason: str


@dataclass(frozen=True)
class ProjectionPrimitiveFrontier:
    status: str
    candidates: Tuple[ProjectionPrimitiveCandidate, ...]
    bracket: Tuple[float, float]
    shadow_program_count: int
    policy_program_id: Optional[str]
    reason: str


def _program_signature(program: ProjectionPowerPrimitiveProgram) -> Optional[Tuple[float, ...]]:
    values = []
    for left, right in PRIMITIVE_SIGNATURE_BRACKETS:
        value = program.apply(left, right)
        if value is None:
            return None
        values.append(round(value, 10))
    return tuple(values)


def generate_projection_power_primitive_programs(
    exponents: Sequence[float] = DEFAULT_POWER_EXPONENTS,
    alphas: Sequence[float] = DEFAULT_PRIMITIVE_ALPHAS,
) -> Tuple[ProjectionPowerPrimitiveProgram, ...]:
    """Generate a bounded transform-primitive shadow universe without outcomes.

    The schema is parameterized rather than naming SQRT/SQUARE/etc. Exponent and
    interpolation alpha are drawn from authored bounded lattices. Prediction-
    equivalent programs are quotiented before any world evidence is consumed.
    Exponents 1, -1 and 0 are excluded because they collapse to existing identity,
    inverse, or logarithmic-limit behavior.
    """
    p_values = tuple(sorted(set(
        round(float(value), 12)
        for value in exponents
        if abs(float(value)) > 1e-12
        and abs(float(value) - 1.0) > 1e-12
        and abs(float(value) + 1.0) > 1e-12
    )))
    alpha_values = tuple(sorted(set(
        round(float(value), 12)
        for value in alphas
        if 0.0 < float(value) < 1.0
    )))
    by_signature: Dict[Tuple[float, ...], ProjectionPowerPrimitiveProgram] = {}
    for exponent in p_values:
        primitive_id = f"GENPRIMITIVE::POWER::{exponent:g}"
        for alpha in alpha_values:
            program = ProjectionPowerPrimitiveProgram(
                primitive_id=primitive_id,
                exponent=exponent,
                alpha=alpha,
                program_id=f"{primitive_id}::ALPHA::{alpha:g}",
            )
            signature = _program_signature(program)
            if signature is None:
                continue
            previous = by_signature.get(signature)
            if previous is None or program.program_id < previous.program_id:
                by_signature[signature] = program
    return tuple(sorted(
        by_signature.values(),
        key=lambda item: (item.exponent, item.alpha, item.program_id),
    ))


def parse_primitive_program_ids(proposal: InterventionProposal) -> Tuple[str, ...]:
    reason = str(proposal.reason)
    if TRANSFORM_PRIMITIVE_MARKER not in reason:
        return ()
    tail = reason.split(TRANSFORM_PRIMITIVE_MARKER, 1)[1].strip().split()[0].rstrip(",;)")
    return tuple(sorted(set(item for item in tail.split("|") if item)))


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def derive_projection_primitive_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    programs: Sequence[ProjectionPowerPrimitiveProgram] = (),
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> ProjectionPrimitivePolicy:
    program_space = tuple(programs) or generate_projection_power_primitive_programs()
    by_id = {program.program_id: program for program in program_space}
    ids_by_experiment = {
        proposal.experiment_id: parse_primitive_program_ids(proposal)
        for proposal in proposals
        if parse_primitive_program_ids(proposal)
    }
    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair) or pair.experiment_id not in ids_by_experiment:
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(
            pair.independence_class_id, pair
        )

    minimum_classes = max(1, int(min_independent_classes))
    support: Dict[str, Dict[str, float]] = {}
    for (experiment_id, context_id), by_class in grouped.items():
        if len(by_class) < minimum_classes:
            continue
        score = sum(abs(pair.effect) for pair in by_class.values()) / len(by_class)
        if score < float(strong_effect_threshold):
            continue
        for program_id in ids_by_experiment[experiment_id]:
            contexts = support.setdefault(program_id, {})
            contexts[context_id] = max(contexts.get(context_id, 0.0), float(score))

    eligible = []
    required_contexts = max(1, int(min_contexts))
    for program_id, contexts in support.items():
        program = by_id.get(program_id)
        if program is None or len(contexts) < required_contexts:
            continue
        mean_score = sum(contexts.values()) / len(contexts)
        eligible.append((
            -len(contexts),
            -mean_score,
            abs(program.exponent),
            program.exponent,
            program.alpha,
            program.program_id,
            tuple(sorted(contexts)),
        ))
    eligible.sort()
    if not eligible:
        return ProjectionPrimitivePolicy(
            status="NO_REPRODUCED_TRANSFORM_PRIMITIVE",
            primitive_id=None,
            exponent=None,
            alpha=None,
            program_id=None,
            supporting_contexts=(),
            candidate_program_count=len(program_space),
            reason="no generated transform primitive has repeated authenticated strong effects",
        )
    chosen = eligible[0]
    program = by_id[chosen[5]]
    return ProjectionPrimitivePolicy(
        status="REPRODUCED_TRANSFORM_PRIMITIVE",
        primitive_id=program.primitive_id,
        exponent=program.exponent,
        alpha=program.alpha,
        program_id=program.program_id,
        supporting_contexts=tuple(chosen[6]),
        candidate_program_count=len(program_space),
        reason="bounded generated transform primitive retained by repeated authenticated cross-context success",
    )


def derive_projection_primitive_frontier(
    depth_assessment: ProjectionTransformDepthAssessment,
    left: float,
    right: float,
    policy: Optional[ProjectionPrimitivePolicy] = None,
    programs: Sequence[ProjectionPowerPrimitiveProgram] = (),
    max_candidates: int = 64,
) -> ProjectionPrimitiveFrontier:
    left = float(left)
    right = float(right)
    program_space = tuple(programs) or generate_projection_power_primitive_programs()
    if not (0.0 < left < right):
        return ProjectionPrimitiveFrontier(
            "INVALID_BRACKET", (), (left, right), len(program_space), None,
            "positive ordered bracket required",
        )
    if depth_assessment.status != "TRANSFORM_GRAMMAR_DEPTH_FALSIFIED_OPEN_NEXT":
        return ProjectionPrimitiveFrontier(
            "CURRENT_TRANSFORM_ALPHABET_NOT_EXHAUSTIVELY_FALSIFIED", (),
            (left, right), len(program_space), None,
            "primitive alphabet expansion requires repeated complete failure of the current transform grammar",
        )

    by_id = {program.program_id: program for program in program_space}
    if (
        policy is not None
        and policy.status == "REPRODUCED_TRANSFORM_PRIMITIVE"
        and policy.program_id in by_id
    ):
        active = (by_id[policy.program_id],)
        status = "LEARNED_TRANSFORM_PRIMITIVE_TRANSFER"
        policy_id = policy.program_id
        reason = "reproduced world-selected transform primitive transferred to a fresh bracket"
    else:
        active = program_space
        status = "SHADOW_TRANSFORM_PRIMITIVE_GENESIS"
        policy_id = None
        reason = "complete current-alphabet failure opened an outcome-independent parameterized primitive shadow"

    ids_by_scale: Dict[float, set[str]] = {}
    primitive_ids_by_scale: Dict[float, set[str]] = {}
    for program in active:
        value = program.apply(left, right)
        if value is None:
            continue
        ids_by_scale.setdefault(value, set()).add(program.program_id)
        primitive_ids_by_scale.setdefault(value, set()).add(program.primitive_id)
    candidates = tuple(
        ProjectionPrimitiveCandidate(
            scale=scale,
            program_ids=tuple(sorted(ids_by_scale[scale])),
            primitive_ids=tuple(sorted(primitive_ids_by_scale[scale])),
        )
        for scale in sorted(ids_by_scale)[: max(1, int(max_candidates))]
    )
    return ProjectionPrimitiveFrontier(
        status=status,
        candidates=candidates,
        bracket=(left, right),
        shadow_program_count=len(program_space),
        policy_program_id=policy_id,
        reason=reason,
    )
