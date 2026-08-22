from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .projection_generator_transform_grammar import (
    DEEP_TRANSFORM_SIGNATURE_ANCHORS,
    ProjectionTransformProgram,
    generate_projection_transform_programs,
    parse_transform_program_ids,
)
from .world_coupling import WorldOutcomePair


@dataclass(frozen=True)
class TransformDepthContextAssessment:
    context_id: str
    bracket: Tuple[float, float]
    expected_program_count: int
    evaluated_program_count: int
    strong_program_ids: Tuple[str, ...]
    missing_program_ids: Tuple[str, ...]
    status: str


@dataclass(frozen=True)
class ProjectionTransformDepthAssessment:
    status: str
    current_depth: int
    authorized_depth: int
    next_depth: int
    required_contexts: int
    falsified_contexts: Tuple[str, ...]
    supported_contexts: Tuple[str, ...]
    incomplete_contexts: Tuple[str, ...]
    current_program_count: int
    context_assessments: Tuple[TransformDepthContextAssessment, ...]
    reason: str


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def _program_effects_by_context(
    proposals: Sequence[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
) -> Dict[str, Dict[str, float]]:
    program_ids_by_experiment = {
        proposal.experiment_id: parse_transform_program_ids(proposal)
        for proposal in proposals
        if parse_transform_program_ids(proposal)
    }
    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair):
            continue
        if pair.experiment_id not in program_ids_by_experiment:
            continue
        grouped.setdefault((pair.context_id, pair.experiment_id), {}).setdefault(
            pair.independence_class_id, pair
        )

    minimum = max(1, int(min_independent_classes))
    effects: Dict[str, Dict[str, float]] = {}
    for (context_id, experiment_id), by_class in grouped.items():
        if len(by_class) < minimum:
            continue
        score = sum(abs(pair.effect) for pair in by_class.values()) / len(by_class)
        for program_id in program_ids_by_experiment[experiment_id]:
            context = effects.setdefault(context_id, {})
            context[program_id] = max(context.get(program_id, 0.0), float(score))
    return effects


def applicable_program_ids(
    programs: Sequence[ProjectionTransformProgram],
    left: float,
    right: float,
) -> Tuple[str, ...]:
    return tuple(sorted(
        program.program_id
        for program in programs
        if program.apply(float(left), float(right)) is not None
    ))


def derive_projection_transform_depth_assessment(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    context_brackets: Mapping[str, Tuple[float, float]],
    current_depth: int = 2,
    next_depth: Optional[int] = None,
    strong_effect_threshold: float = 0.9,
    min_falsified_contexts: int = 2,
) -> ProjectionTransformDepthAssessment:
    """Open one deeper transform grammar only after exhaustive authenticated failure.

    Absence is never treated as refutation. For a context to count as falsifying the
    current grammar, every transform program applicable to that bracket must have an
    externally authenticated outcome in the minimum number of independent evidence
    classes, and every such program must remain below the strong-effect threshold.
    Only repeated complete failures across distinct contexts authorize one additional
    transform-composition layer.
    """
    proposal_list = tuple(proposals)
    depth = max(0, int(current_depth))
    deeper = depth + 1 if next_depth is None else max(depth + 1, int(next_depth))
    programs = generate_projection_transform_programs(
        max_transform_depth=depth,
        signature_anchors=DEEP_TRANSFORM_SIGNATURE_ANCHORS,
    )
    effects_by_context = _program_effects_by_context(
        proposal_list,
        world_pairs,
        min_independent_classes,
    )

    assessments = []
    falsified = []
    supported = []
    incomplete = []
    threshold = float(strong_effect_threshold)

    for context_id, raw_bracket in sorted(context_brackets.items()):
        left, right = float(raw_bracket[0]), float(raw_bracket[1])
        if not (0.0 < left < right):
            assessments.append(TransformDepthContextAssessment(
                context_id=str(context_id),
                bracket=(left, right),
                expected_program_count=0,
                evaluated_program_count=0,
                strong_program_ids=(),
                missing_program_ids=(),
                status="INVALID_BRACKET",
            ))
            incomplete.append(str(context_id))
            continue

        expected = set(applicable_program_ids(programs, left, right))
        effects = effects_by_context.get(str(context_id), {})
        evaluated = expected.intersection(effects)
        missing = tuple(sorted(expected.difference(evaluated)))
        strong = tuple(sorted(
            program_id for program_id in evaluated
            if effects.get(program_id, 0.0) >= threshold
        ))

        if not expected or missing:
            status = "INCOMPLETE_CURRENT_DEPTH_EVALUATION"
            incomplete.append(str(context_id))
        elif strong:
            status = "CURRENT_DEPTH_HAS_STRONG_PROGRAM"
            supported.append(str(context_id))
        else:
            status = "CURRENT_DEPTH_EXHAUSTIVELY_FALSIFIED"
            falsified.append(str(context_id))

        assessments.append(TransformDepthContextAssessment(
            context_id=str(context_id),
            bracket=(left, right),
            expected_program_count=len(expected),
            evaluated_program_count=len(evaluated),
            strong_program_ids=strong,
            missing_program_ids=missing,
            status=status,
        ))

    required = max(1, int(min_falsified_contexts))
    if len(falsified) >= required:
        status = "TRANSFORM_GRAMMAR_DEPTH_FALSIFIED_OPEN_NEXT"
        authorized_depth = deeper
        reason = (
            "every applicable current-depth transform program was independently "
            "evaluated and remained weak in repeated contexts; one deeper AST layer is authorized"
        )
    elif supported:
        status = "CURRENT_TRANSFORM_DEPTH_RETAINS_SUPPORTED_PROGRAM"
        authorized_depth = depth
        reason = "at least one current-depth program has authenticated strong effect"
    else:
        status = "TRANSFORM_DEPTH_EVIDENCE_INCOMPLETE"
        authorized_depth = depth
        reason = (
            "current-depth grammar is not exhaustively falsified in enough contexts; "
            "missing or unverified candidates cannot authorize depth expansion"
        )

    return ProjectionTransformDepthAssessment(
        status=status,
        current_depth=depth,
        authorized_depth=authorized_depth,
        next_depth=deeper,
        required_contexts=required,
        falsified_contexts=tuple(sorted(falsified)),
        supported_contexts=tuple(sorted(supported)),
        incomplete_contexts=tuple(sorted(incomplete)),
        current_program_count=len(programs),
        context_assessments=tuple(assessments),
        reason=reason,
    )
