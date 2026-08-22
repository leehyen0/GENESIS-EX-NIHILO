from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .world_coupling import WorldOutcomePair


TRANSFORM_PROGRAM_MARKER = "generator_transform_programs="
DEFAULT_TRANSFORM_PRIMITIVES: Tuple[str, ...] = ("LOG", "INV")
DEFAULT_TRANSFORM_ALPHAS: Tuple[float, ...] = (0.25, 0.5, 0.75)
DEFAULT_TRANSFORM_SIGNATURE_ANCHORS: Tuple[float, ...] = (1.5, 2.0, 3.0, 4.0, 8.0, 16.0)
DEEP_TRANSFORM_SIGNATURE_ANCHORS: Tuple[float, ...] = (32.0, 64.0, 128.0, 256.0, 512.0, 1024.0)


def normalize_transform_primitive(spec: str) -> Optional[str]:
    """Normalize one bounded unary transform primitive specification.

    LOG and INV remain the default human-authored alphabet. Parameterized POW:p is
    a bounded meta-language extension: the parameter candidate can be generated
    without world outcomes while world evidence separately decides whether it earns
    authority. Identity-like p=1, inverse-duplicate p=-1, zero, non-finite, and
    extreme exponents are rejected so primitive search does not smuggle old atoms or
    unbounded numeric programs into the grammar.
    """
    text = str(spec).strip().upper().replace(" ", "")
    if text in {"LOG", "INV"}:
        return text
    if not text.startswith("POW:"):
        return None
    try:
        exponent = float(text.split(":", 1)[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(exponent) or abs(exponent) <= 1e-12:
        return None
    if abs(exponent) > 4.0:
        return None
    if abs(exponent - 1.0) <= 1e-12 or abs(exponent + 1.0) <= 1e-12:
        return None
    return f"POW:{exponent:g}"


@dataclass(frozen=True)
class ProjectionTransformProgram:
    program_id: str
    operations: Tuple[str, ...]
    alpha: float
    complexity: int

    @staticmethod
    def _forward_op(op: str, value: float) -> Optional[float]:
        try:
            if op == "LOG":
                if value <= 0.0:
                    return None
                result = math.log(value)
            elif op == "INV":
                if abs(value) <= 1e-12:
                    return None
                result = 1.0 / value
            elif op.startswith("POW:"):
                primitive = normalize_transform_primitive(op)
                if primitive is None or value <= 0.0:
                    return None
                exponent = float(primitive.split(":", 1)[1])
                result = math.pow(value, exponent)
            else:
                return None
        except (ValueError, OverflowError, ZeroDivisionError):
            return None
        return float(result) if math.isfinite(result) else None

    @staticmethod
    def _inverse_op(op: str, value: float) -> Optional[float]:
        try:
            if op == "LOG":
                if value >= 700.0:
                    return None
                result = math.exp(value)
            elif op == "INV":
                if abs(value) <= 1e-12:
                    return None
                result = 1.0 / value
            elif op.startswith("POW:"):
                primitive = normalize_transform_primitive(op)
                if primitive is None or value <= 0.0:
                    return None
                exponent = float(primitive.split(":", 1)[1])
                inverse_exponent = 1.0 / exponent
                result = math.pow(value, inverse_exponent)
            else:
                return None
        except (ValueError, OverflowError, ZeroDivisionError):
            return None
        return float(result) if math.isfinite(result) else None

    def transform(self, value: float) -> Optional[float]:
        current = float(value)
        for operation in self.operations:
            next_value = self._forward_op(operation, current)
            if next_value is None:
                return None
            current = next_value
        return current

    def inverse(self, value: float) -> Optional[float]:
        current = float(value)
        for operation in reversed(self.operations):
            next_value = self._inverse_op(operation, current)
            if next_value is None:
                return None
            current = next_value
        return current

    def apply(self, left: float, right: float) -> Optional[float]:
        left = float(left)
        right = float(right)
        if not (0.0 < left < right and 0.0 < self.alpha < 1.0):
            return None
        transformed_left = self.transform(left)
        transformed_right = self.transform(right)
        if transformed_left is None or transformed_right is None:
            return None
        mixed = (1.0 - self.alpha) * transformed_left + self.alpha * transformed_right
        value = self.inverse(mixed)
        if value is None or not (left < value < right):
            return None
        return round(float(value), 12)


@dataclass(frozen=True)
class ProjectionTransformCandidate:
    scale: float
    program_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ProjectionTransformPolicy:
    status: str
    program_id: Optional[str]
    operations: Tuple[str, ...]
    alpha: Optional[float]
    supporting_contexts: Tuple[str, ...]
    candidate_program_count: int
    reason: str


@dataclass(frozen=True)
class ProjectionTransformFrontier:
    status: str
    candidates: Tuple[ProjectionTransformCandidate, ...]
    bracket: Tuple[float, float]
    policy_program_id: Optional[str]
    shadow_program_count: int
    reason: str


def _transform_signature(
    operations: Tuple[str, ...],
    anchors: Sequence[float] = DEFAULT_TRANSFORM_SIGNATURE_ANCHORS,
) -> Optional[Tuple[float, ...]]:
    probe = ProjectionTransformProgram("probe", operations, 0.5, len(operations))
    values = []
    for anchor in anchors:
        value = probe.transform(float(anchor))
        if value is None:
            return None
        values.append(round(value, 10))
    return tuple(values)


def generate_projection_transform_programs(
    primitives: Sequence[str] = DEFAULT_TRANSFORM_PRIMITIVES,
    alphas: Sequence[float] = DEFAULT_TRANSFORM_ALPHAS,
    max_transform_depth: int = 2,
    signature_anchors: Sequence[float] = DEFAULT_TRANSFORM_SIGNATURE_ANCHORS,
) -> Tuple[ProjectionTransformProgram, ...]:
    """Generate and quotient a bounded transform grammar without world outcomes.

    The grammar contains only normalized unary transform primitives. Named
    interpolation families are not enumerated: each retained transform sequence is
    wrapped in the same inverse-transform/weighted-mix construction and identified
    by its primitive ancestry. Equivalent transform sequences are quotiented by
    anchor signatures.

    `signature_anchors` is a representation-domain parameter, not evidence. Deeper
    nested logarithms require a positive high-magnitude quotient domain; callers may
    therefore use `DEEP_TRANSFORM_SIGNATURE_ANCHORS` without exposing world outcomes.
    """
    normalized = []
    for item in primitives:
        primitive = normalize_transform_primitive(str(item))
        if primitive is not None:
            normalized.append(primitive)
    primitive_set = tuple(sorted(set(normalized)))
    anchors = tuple(float(value) for value in signature_anchors)
    if not anchors:
        anchors = DEFAULT_TRANSFORM_SIGNATURE_ANCHORS
    signatures: Dict[Tuple[float, ...], Tuple[str, ...]] = {}
    sequences = [()]
    for depth in range(1, max(0, int(max_transform_depth)) + 1):
        sequences.extend(itertools.product(primitive_set, repeat=depth))
    for raw in sequences:
        operations = tuple(raw)
        signature = _transform_signature(operations, anchors=anchors)
        if signature is None:
            continue
        previous = signatures.get(signature)
        if previous is None or (len(operations), operations) < (len(previous), previous):
            signatures[signature] = operations

    alpha_set = tuple(sorted(set(
        round(float(value), 12)
        for value in alphas
        if 0.0 < float(value) < 1.0
    )))
    programs = []
    for operations in sorted(signatures.values(), key=lambda item: (len(item), item)):
        transform_name = "IDENTITY" if not operations else ">".join(operations)
        for alpha in alpha_set:
            programs.append(ProjectionTransformProgram(
                program_id=f"GENERATOR_AST::{transform_name}::ALPHA::{alpha:g}",
                operations=operations,
                alpha=alpha,
                complexity=len(operations) + 1,
            ))
    return tuple(sorted(programs, key=lambda item: (item.complexity, item.operations, item.alpha)))


def parse_transform_program_ids(proposal: InterventionProposal) -> Tuple[str, ...]:
    reason = str(proposal.reason)
    if TRANSFORM_PROGRAM_MARKER not in reason:
        return ()
    tail = reason.split(TRANSFORM_PROGRAM_MARKER, 1)[1].strip().split()[0].rstrip(",;)")
    return tuple(sorted(set(item for item in tail.split("|") if item)))


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def derive_projection_transform_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    programs: Sequence[ProjectionTransformProgram] = (),
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> ProjectionTransformPolicy:
    program_space = tuple(programs) or generate_projection_transform_programs()
    by_id = {program.program_id: program for program in program_space}
    program_ids_by_experiment = {
        proposal.experiment_id: parse_transform_program_ids(proposal)
        for proposal in proposals
        if parse_transform_program_ids(proposal)
    }
    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair) or pair.experiment_id not in program_ids_by_experiment:
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
        for program_id in program_ids_by_experiment[experiment_id]:
            contexts = support.setdefault(program_id, {})
            contexts[context_id] = max(contexts.get(context_id, 0.0), float(score))

    eligible = []
    minimum_contexts = max(1, int(min_contexts))
    for program_id, contexts in support.items():
        program = by_id.get(program_id)
        if program is None or len(contexts) < minimum_contexts:
            continue
        mean_score = sum(contexts.values()) / len(contexts)
        eligible.append((
            -len(contexts),
            -mean_score,
            program.complexity,
            program.operations,
            program.alpha,
            program.program_id,
            tuple(sorted(contexts)),
        ))
    eligible.sort()
    if not eligible:
        return ProjectionTransformPolicy(
            status="NO_REPRODUCED_TRANSFORM_PROGRAM",
            program_id=None,
            operations=(),
            alpha=None,
            supporting_contexts=(),
            candidate_program_count=len(program_space),
            reason="no compositional transform program has repeated authenticated strong effects",
        )
    chosen = eligible[0]
    program = by_id[chosen[5]]
    return ProjectionTransformPolicy(
        status="REPRODUCED_TRANSFORM_PROGRAM",
        program_id=program.program_id,
        operations=program.operations,
        alpha=program.alpha,
        supporting_contexts=tuple(chosen[6]),
        candidate_program_count=len(program_space),
        reason="minimum-complexity compositional transform program retained by repeated authenticated cross-context success",
    )


def _endpoint_scores(
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
    out: Dict[float, float] = {}
    minimum = max(1, int(min_independent_classes))
    for experiment_id, by_class in grouped.items():
        if len(by_class) < minimum:
            continue
        score = sum(abs(pair.effect) for pair in by_class.values()) / len(by_class)
        scale = scale_by_experiment[experiment_id]
        out[scale] = max(out.get(scale, 0.0), float(score))
    return out


def derive_projection_transform_frontier(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    probe_scale: Callable[[InterventionProposal], Optional[float]],
    context_id: str,
    left: float,
    right: float,
    policy: Optional[ProjectionTransformPolicy] = None,
    programs: Sequence[ProjectionTransformProgram] = (),
    strong_effect_threshold: float = 0.9,
    max_candidates: int = 64,
) -> ProjectionTransformFrontier:
    left = float(left)
    right = float(right)
    if not (0.0 < left < right):
        return ProjectionTransformFrontier(
            "INVALID_BRACKET", (), (left, right), None, 0, "positive ordered bracket required"
        )
    proposal_list = tuple(proposals)
    scores = _endpoint_scores(
        proposal_list, world_pairs, min_independent_classes, probe_scale, context_id
    )
    if left not in scores or right not in scores:
        return ProjectionTransformFrontier(
            "INSUFFICIENT_AUTHENTICATED_BRACKET", (), (left, right), None, 0,
            "both bracket endpoints require authenticated independent outcomes",
        )
    if max(scores[left], scores[right]) >= float(strong_effect_threshold):
        return ProjectionTransformFrontier(
            "STRONG_ENDPOINT_AVAILABLE", (), (left, right), None, 0,
            "an existing bracket endpoint already reaches the strong-effect criterion",
        )

    program_space = tuple(programs) or generate_projection_transform_programs()
    by_id = {program.program_id: program for program in program_space}
    if (
        policy is not None
        and policy.status == "REPRODUCED_TRANSFORM_PROGRAM"
        and policy.program_id in by_id
    ):
        active = (by_id[policy.program_id],)
        status = "LEARNED_TRANSFORM_PROGRAM_TRANSFER"
        policy_id = policy.program_id
        reason = "reproduced compositional transform program transferred to a fresh weak bracket"
    else:
        active = program_space
        status = "SHADOW_TRANSFORM_PROGRAM_GENESIS"
        policy_id = None
        reason = "weak authenticated bracket opened bounded outcome-independent compositional transform grammar"

    ids_by_scale: Dict[float, set[str]] = {}
    for program in active:
        value = program.apply(left, right)
        if value is None:
            continue
        ids_by_scale.setdefault(value, set()).add(program.program_id)
    candidates = tuple(
        ProjectionTransformCandidate(scale, tuple(sorted(program_ids)))
        for scale, program_ids in sorted(ids_by_scale.items())[: max(1, int(max_candidates))]
    )
    return ProjectionTransformFrontier(
        status=status,
        candidates=candidates,
        bracket=(left, right),
        policy_program_id=policy_id,
        shadow_program_count=len(program_space),
        reason=reason,
    )
