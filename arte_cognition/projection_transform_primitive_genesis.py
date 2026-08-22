from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .projection_generator_transform_grammar import (
    DEEP_TRANSFORM_SIGNATURE_ANCHORS,
    ProjectionTransformFrontier,
    ProjectionTransformPolicy,
    ProjectionTransformProgram,
    derive_projection_transform_frontier,
    derive_projection_transform_policy,
    generate_projection_transform_programs,
    normalize_transform_primitive,
)
from .projection_transform_depth_genesis import (
    ProjectionTransformDepthAssessment,
    derive_projection_transform_depth_assessment,
)
from .world_coupling import WorldOutcomePair


DEFAULT_POWER_PRIMITIVE_CANDIDATES: Tuple[str, ...] = (
    "POW:-2",
    "POW:-0.5",
    "POW:0.5",
    "POW:2",
)


@dataclass(frozen=True)
class ProjectionTransformPrimitiveGenesisAssessment:
    status: str
    depth_authority_status: str
    depth3_authorized: bool
    current_alphabet_status: str
    current_alphabet_program_count: int
    current_alphabet_falsified_contexts: Tuple[str, ...]
    current_alphabet_incomplete_contexts: Tuple[str, ...]
    generated_primitive_candidates: Tuple[str, ...]
    candidate_program_count: int
    reason: str


def generate_power_primitive_programs(
    primitive_candidates: Sequence[str] = DEFAULT_POWER_PRIMITIVE_CANDIDATES,
    alphas: Sequence[float] = (0.25, 0.5, 0.75),
) -> Tuple[ProjectionTransformProgram, ...]:
    """Generate bounded one-primitive transform programs without world outcomes."""
    normalized = tuple(sorted(set(
        primitive
        for primitive in (normalize_transform_primitive(item) for item in primitive_candidates)
        if primitive is not None and primitive.startswith("POW:")
    )))
    programs = generate_projection_transform_programs(
        primitives=normalized,
        alphas=alphas,
        max_transform_depth=1,
        signature_anchors=DEEP_TRANSFORM_SIGNATURE_ANCHORS,
    )
    return tuple(
        program for program in programs
        if len(program.operations) == 1 and program.operations[0] in normalized
    )


def derive_projection_transform_primitive_genesis_assessment(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    depth_authority_brackets: Mapping[str, Tuple[float, float]],
    alphabet_failure_brackets: Mapping[str, Tuple[float, float]],
    primitive_candidates: Sequence[str] = DEFAULT_POWER_PRIMITIVE_CANDIDATES,
    strong_effect_threshold: float = 0.9,
    min_falsified_contexts: int = 2,
) -> ProjectionTransformPrimitiveGenesisAssessment:
    """Open bounded primitive search only after authorized depth-3 class failure.

    The BODY must first earn depth 3 from exhaustive depth<=2 failure. Then every
    applicable program built from the current default alphabet {LOG, INV} through
    depth 3 must be independently evaluated and remain weak in repeated contexts.
    Missing candidates are never treated as refutation. Only then is the bounded
    outcome-independent POW:p candidate meta-language exposed for external testing.
    """
    proposal_list = tuple(proposals)
    depth_authority = derive_projection_transform_depth_assessment(
        proposals=proposal_list,
        world_pairs=world_pairs,
        min_independent_classes=min_independent_classes,
        context_brackets=depth_authority_brackets,
        current_depth=2,
        next_depth=3,
        strong_effect_threshold=strong_effect_threshold,
        min_falsified_contexts=min_falsified_contexts,
    )
    candidate_programs = generate_power_primitive_programs(primitive_candidates)
    candidate_primitives = tuple(sorted(set(
        program.operations[0] for program in candidate_programs if program.operations
    )))

    if depth_authority.authorized_depth < 3:
        return ProjectionTransformPrimitiveGenesisAssessment(
            status="TRANSFORM_PRIMITIVE_GENESIS_DEPTH3_NOT_AUTHORIZED",
            depth_authority_status=depth_authority.status,
            depth3_authorized=False,
            current_alphabet_status="NOT_EVALUATED_WITHOUT_DEPTH3_AUTHORITY",
            current_alphabet_program_count=0,
            current_alphabet_falsified_contexts=(),
            current_alphabet_incomplete_contexts=tuple(sorted(alphabet_failure_brackets)),
            generated_primitive_candidates=candidate_primitives,
            candidate_program_count=len(candidate_programs),
            reason="primitive genesis is closed until depth 3 is causally authorized by world evidence",
        )

    alphabet_assessment = derive_projection_transform_depth_assessment(
        proposals=proposal_list,
        world_pairs=world_pairs,
        min_independent_classes=min_independent_classes,
        context_brackets=alphabet_failure_brackets,
        current_depth=3,
        next_depth=4,
        strong_effect_threshold=strong_effect_threshold,
        min_falsified_contexts=min_falsified_contexts,
    )

    if alphabet_assessment.status == "TRANSFORM_GRAMMAR_DEPTH_FALSIFIED_OPEN_NEXT":
        status = "TRANSFORM_PRIMITIVE_GENESIS_OPEN"
        reason = (
            "authorized depth-3 {LOG, INV} program class was exhaustively evaluated "
            "and remained weak in repeated contexts; bounded POW:p primitive search is open"
        )
    elif alphabet_assessment.status == "CURRENT_TRANSFORM_DEPTH_RETAINS_SUPPORTED_PROGRAM":
        status = "CURRENT_TRANSFORM_ALPHABET_RETAINS_SUPPORTED_PROGRAM"
        reason = "current authorized transform alphabet still contains an authenticated strong program"
    else:
        status = "TRANSFORM_PRIMITIVE_GENESIS_EVIDENCE_INCOMPLETE"
        reason = (
            "authorized depth-3 current alphabet is not exhaustively falsified; "
            "missing or unverified programs cannot open primitive search"
        )

    return ProjectionTransformPrimitiveGenesisAssessment(
        status=status,
        depth_authority_status=depth_authority.status,
        depth3_authorized=True,
        current_alphabet_status=alphabet_assessment.status,
        current_alphabet_program_count=alphabet_assessment.current_program_count,
        current_alphabet_falsified_contexts=alphabet_assessment.falsified_contexts,
        current_alphabet_incomplete_contexts=alphabet_assessment.incomplete_contexts,
        generated_primitive_candidates=candidate_primitives,
        candidate_program_count=len(candidate_programs),
        reason=reason,
    )


def derive_projection_transform_primitive_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    genesis_assessment: ProjectionTransformPrimitiveGenesisAssessment,
    primitive_candidates: Sequence[str] = DEFAULT_POWER_PRIMITIVE_CANDIDATES,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> ProjectionTransformPolicy:
    programs = generate_power_primitive_programs(primitive_candidates)
    if genesis_assessment.status != "TRANSFORM_PRIMITIVE_GENESIS_OPEN":
        return ProjectionTransformPolicy(
            status="TRANSFORM_PRIMITIVE_POLICY_CLOSED",
            program_id=None,
            operations=(),
            alpha=None,
            supporting_contexts=(),
            candidate_program_count=len(programs),
            reason="bounded generated primitive policy cannot be learned before the primitive-genesis gate opens",
        )
    return derive_projection_transform_policy(
        proposals=tuple(proposals),
        world_pairs=world_pairs,
        min_independent_classes=min_independent_classes,
        programs=programs,
        strong_effect_threshold=strong_effect_threshold,
        min_contexts=min_contexts,
    )


def derive_projection_transform_primitive_frontier(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    probe_scale: Callable[[InterventionProposal], Optional[float]],
    context_id: str,
    left: float,
    right: float,
    genesis_assessment: ProjectionTransformPrimitiveGenesisAssessment,
    policy: Optional[ProjectionTransformPolicy] = None,
    primitive_candidates: Sequence[str] = DEFAULT_POWER_PRIMITIVE_CANDIDATES,
    strong_effect_threshold: float = 0.9,
    max_candidates: int = 32,
) -> ProjectionTransformFrontier:
    programs = generate_power_primitive_programs(primitive_candidates)
    if genesis_assessment.status != "TRANSFORM_PRIMITIVE_GENESIS_OPEN":
        return ProjectionTransformFrontier(
            status="TRANSFORM_PRIMITIVE_GENESIS_CLOSED",
            candidates=(),
            bracket=(float(left), float(right)),
            policy_program_id=None,
            shadow_program_count=len(programs),
            reason=genesis_assessment.reason,
        )
    return derive_projection_transform_frontier(
        proposals=tuple(proposals),
        world_pairs=world_pairs,
        min_independent_classes=min_independent_classes,
        probe_scale=probe_scale,
        context_id=context_id,
        left=left,
        right=right,
        policy=policy,
        programs=programs,
        strong_effect_threshold=strong_effect_threshold,
        max_candidates=max_candidates,
    )
