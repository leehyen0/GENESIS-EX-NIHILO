from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import inspect
import json
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .software_upstream_failure_locus_genesis import (
    UpstreamPatchCandidate,
    locate_upstream_list_assignment,
)
from .software_upstream_patch_discrimination import UpstreamPatchSelector
from .world_coupling import WorldOutcomePair


REPRESENTATION_MARKER = "upstream_selector_representation="
REPRESENTATION_MODE_MARKER = "upstream_selector_representation_mode="

MODE_SIGNATURE_FALSE_DEFAULT_TRUE = "CALL_BINDING_FALSE_DEFAULT_EXPLICIT_TRUE_ANCHOR"
MODE_SHORTEST_STRUCTURAL_ARGUMENT = "SHORTEST_NONIDENTITY_ARGUMENT_SHAPE_ANCHOR"

RULE_TOWARD_ANCHOR = "REWRITE_NONANCHOR_TOWARD_UNIQUE_FLAGGED_ANCHOR_CORE"
RULE_AWAY_FROM_ANCHOR = "REWRITE_ANCHOR_CORE_AWAY_FROM_UNIQUE_FLAGGED_ANCHOR"


@dataclass(frozen=True)
class SelectorRepresentationContext:
    context_id: str
    inherited_frontier_count: int
    old_selected_candidate_count: int


@dataclass(frozen=True)
class SelectorRepresentationAssessment:
    status: str
    inexpressive_contexts: Tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class CallBindingSchema:
    function_name: str
    parameter_order: Tuple[str, ...]
    defaults: Tuple[Tuple[str, object], ...]
    false_default_parameters: Tuple[str, ...]

    @property
    def default_map(self) -> Dict[str, object]:
        return dict(self.defaults)


@dataclass(frozen=True)
class UpstreamSelectorRepresentation:
    mode: str

    @property
    def representation_id(self) -> str:
        digest = hashlib.sha256(self.mode.encode("utf-8")).hexdigest()[:20]
        return f"UPSTREAM_SELECTOR_REPRESENTATION::{digest}"


@dataclass(frozen=True)
class UpstreamSelectorRepresentationProposal:
    representation: UpstreamSelectorRepresentation
    proposal: InterventionProposal


@dataclass(frozen=True)
class UpstreamSelectorRepresentationPolicy:
    status: str
    representation_id: Optional[str]
    supporting_contexts: Tuple[str, ...]
    candidate_representation_count: int
    reason: str


def assess_selector_representation_inexpressivity(
    contexts: Sequence[SelectorRepresentationContext],
    min_contexts: int = 2,
) -> SelectorRepresentationAssessment:
    """Open representation escape only after repeated selector inapplicability.

    A nonempty inherited repair frontier is required in every counted context. A
    missing old selection is classified as representation inexpressivity, not as a
    refuted repair candidate or negative world outcome.
    """
    inexpressive = tuple(sorted(
        item.context_id
        for item in contexts
        if item.inherited_frontier_count > 0 and item.old_selected_candidate_count == 0
    ))
    if len(inexpressive) < max(1, int(min_contexts)):
        return SelectorRepresentationAssessment(
            status="INSUFFICIENT_SELECTOR_REPRESENTATION_INEXPRESSIVITY",
            inexpressive_contexts=inexpressive,
            reason="representation escape requires repeated nonempty frontiers that the inherited selector cannot denote",
        )
    return SelectorRepresentationAssessment(
        status="SELECTOR_REPRESENTATION_INEXPRESSIVE_OPEN_BINDING",
        inexpressive_contexts=inexpressive,
        reason="repeated nonempty repair frontiers are unreachable only because the inherited anchor representation is syntactically inexpressive",
    )


def derive_call_binding_schema(callable_obj) -> CallBindingSchema:
    signature = inspect.signature(callable_obj)
    order = []
    defaults = []
    false_defaults = []
    for name, parameter in signature.parameters.items():
        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            continue
        order.append(name)
        if parameter.default is not inspect.Parameter.empty:
            default = parameter.default
            if isinstance(default, (str, int, float, bool, type(None))):
                defaults.append((name, default))
            if default is False:
                false_defaults.append(name)
    name = getattr(callable_obj, "__name__", type(callable_obj).__name__)
    return CallBindingSchema(
        function_name=str(name),
        parameter_order=tuple(order),
        defaults=tuple(defaults),
        false_default_parameters=tuple(false_defaults),
    )


def generate_selector_representations(
    assessment: SelectorRepresentationAssessment,
) -> Tuple[UpstreamSelectorRepresentation, ...]:
    if assessment.status != "SELECTOR_REPRESENTATION_INEXPRESSIVE_OPEN_BINDING":
        return ()
    # Candidate representations exist before any execution outcome. The second is a
    # matched structural negative control; world evidence decides which receives authority.
    return (
        UpstreamSelectorRepresentation(MODE_SIGNATURE_FALSE_DEFAULT_TRUE),
        UpstreamSelectorRepresentation(MODE_SHORTEST_STRUCTURAL_ARGUMENT),
    )


def propose_selector_representation(
    representation: UpstreamSelectorRepresentation,
) -> UpstreamSelectorRepresentationProposal:
    digest = hashlib.sha256(representation.representation_id.encode("utf-8")).hexdigest()[:20]
    proposal = InterventionProposal(
        experiment_id=f"SOFTWARE_UPSTREAM_SELECTOR_REPRESENTATION::{digest}",
        axis_id=f"AXIS::SOFTWARE_SELECTOR_REPRESENTATION::{digest}",
        manipulated_variable=representation.representation_id,
        held_fixed=(),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="SURFACE_FORM_SELECTOR_INAPPLICABLE",
        predicted_high_side="BINDING_NORMALIZED_PRE_OUTCOME_SELECTION",
        reason=(
            "outcome-independent selector representation proposal; "
            f"{REPRESENTATION_MARKER}{representation.representation_id} "
            f"{REPRESENTATION_MODE_MARKER}{representation.mode}"
        ),
        status="PROPOSAL_ONLY",
    )
    return UpstreamSelectorRepresentationProposal(representation=representation, proposal=proposal)


def _assigned_list_by_name(source: str, name: str) -> Optional[ast.List]:
    try:
        tree = ast.parse(str(source))
    except SyntaxError:
        return None
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name and isinstance(node.value, ast.List):
            matches.append(node)
    if not matches:
        return None
    matches.sort(key=lambda node: int(node.lineno))
    return matches[-1].value


def _literal_erased_shape(value):
    if isinstance(value, ast.Constant):
        return ("Constant", type(value.value).__name__)
    if isinstance(value, ast.AST):
        return (
            type(value).__name__,
            tuple((field, _literal_erased_shape(getattr(value, field))) for field in value._fields),
        )
    if isinstance(value, (list, tuple)):
        return tuple(_literal_erased_shape(item) for item in value)
    return value


def _bind_call(call: ast.Call, schema: CallBindingSchema) -> Optional[Dict[str, ast.AST]]:
    if isinstance(call.func, ast.Name) and call.func.id != schema.function_name:
        return None
    bound: Dict[str, ast.AST] = {}
    positional = tuple(schema.parameter_order)
    if len(call.args) > len(positional):
        return None
    for index, value in enumerate(call.args):
        bound[positional[index]] = value
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg not in positional or keyword.arg in bound:
            return None
        bound[keyword.arg] = keyword.value
    return bound


def _constant_equals(node: ast.AST, expected: object) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is type(expected) and node.value == expected


def _binding_core(call: ast.Call, schema: CallBindingSchema) -> Optional[str]:
    bound = _bind_call(call, schema)
    if bound is None or not schema.parameter_order:
        return None
    defaults = schema.default_map
    identity = schema.parameter_order[0]
    payload = []
    for name in schema.parameter_order:
        if name == identity or name in schema.false_default_parameters:
            continue
        value = bound.get(name)
        if value is None:
            if name in defaults:
                payload.append((name, "DEFAULT", type(defaults[name]).__name__))
            else:
                payload.append((name, "MISSING"))
            continue
        if name in defaults and _constant_equals(value, defaults[name]):
            payload.append((name, "DEFAULT", type(defaults[name]).__name__))
        else:
            payload.append((name, _literal_erased_shape(value)))
    func_shape = _literal_erased_shape(call.func)
    return json.dumps({"func": func_shape, "bound": payload}, sort_keys=True, separators=(",", ":"))


def _binding_true_marker(call: ast.Call, schema: CallBindingSchema) -> bool:
    bound = _bind_call(call, schema)
    if bound is None:
        return False
    marked = [
        name for name in schema.false_default_parameters
        if name in bound and _constant_equals(bound[name], True)
    ]
    return len(marked) == 1


def _structural_width(call: ast.Call, schema: CallBindingSchema) -> Optional[int]:
    bound = _bind_call(call, schema)
    if bound is None or len(schema.parameter_order) < 2:
        return None
    value = bound.get(schema.parameter_order[1])
    if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
        return len(value.elts)
    return None


def _anchor_core(
    source: str,
    failure_line: int,
    representation: UpstreamSelectorRepresentation,
    schema: CallBindingSchema,
) -> Optional[str]:
    locus = locate_upstream_list_assignment(source, failure_line)
    if locus is None:
        return None
    value = _assigned_list_by_name(source, locus[2])
    if value is None:
        return None
    calls = [item for item in value.elts if isinstance(item, ast.Call)]
    if representation.mode == MODE_SIGNATURE_FALSE_DEFAULT_TRUE:
        anchors = [item for item in calls if _binding_true_marker(item, schema)]
    elif representation.mode == MODE_SHORTEST_STRUCTURAL_ARGUMENT:
        widths = [(item, _structural_width(item, schema)) for item in calls]
        widths = [(item, width) for item, width in widths if width is not None]
        if not widths:
            return None
        minimum = min(width for _, width in widths)
        anchors = [item for item, width in widths if width == minimum]
    else:
        return None
    cores = [core for item in anchors if (core := _binding_core(item, schema)) is not None]
    unique = tuple(sorted(set(cores)))
    return unique[0] if len(unique) == 1 else None


def _transition(
    original: str,
    candidate: str,
    failure_line: int,
    schema: CallBindingSchema,
) -> Optional[Tuple[int, str, str, bool, bool]]:
    locus = locate_upstream_list_assignment(original, failure_line)
    if locus is None:
        return None
    before = _assigned_list_by_name(original, locus[2])
    after = _assigned_list_by_name(candidate, locus[2])
    if before is None or after is None or len(before.elts) != len(after.elts):
        return None
    changed = [
        index for index, (left, right) in enumerate(zip(before.elts, after.elts))
        if ast.dump(left, include_attributes=False) != ast.dump(right, include_attributes=False)
    ]
    if len(changed) != 1:
        return None
    index = changed[0]
    left = before.elts[index]
    right = after.elts[index]
    if not isinstance(left, ast.Call) or not isinstance(right, ast.Call):
        return None
    left_core = _binding_core(left, schema)
    right_core = _binding_core(right, schema)
    if left_core is None or right_core is None:
        return None
    return (
        index,
        left_core,
        right_core,
        _binding_true_marker(left, schema),
        _binding_true_marker(right, schema),
    )


def select_upstream_patch_with_representation(
    selector: UpstreamPatchSelector,
    representation: UpstreamSelectorRepresentation,
    candidates: Sequence[UpstreamPatchCandidate],
    source: str,
    failure_line: int,
    schema: CallBindingSchema,
) -> Optional[UpstreamPatchCandidate]:
    """Apply a learned selector through a candidate call-binding representation."""
    anchor = _anchor_core(source, failure_line, representation, schema)
    if anchor is None:
        return None
    eligible = []
    for candidate in candidates:
        row = _transition(source, candidate.patched_source, failure_line, schema)
        if row is None:
            continue
        index, before_core, after_core, before_marker, after_marker = row
        if selector.selection_rule == RULE_TOWARD_ANCHOR:
            valid = before_core != anchor and after_core == anchor and not before_marker and not after_marker
        elif selector.selection_rule == RULE_AWAY_FROM_ANCHOR:
            valid = before_core == anchor and after_core != anchor and not before_marker and not after_marker
        else:
            valid = False
        if valid:
            eligible.append((index, candidate.candidate_id, candidate))
    eligible.sort(key=lambda row: (row[0], row[1]))
    return eligible[0][2] if eligible else None


def _parse_representation_id(proposal: InterventionProposal) -> Optional[str]:
    reason = str(proposal.reason)
    if REPRESENTATION_MARKER not in reason:
        return None
    return reason.split(REPRESENTATION_MARKER, 1)[1].strip().split()[0].rstrip(",;)") or None


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def derive_selector_representation_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> UpstreamSelectorRepresentationPolicy:
    representation_by_experiment: Dict[str, str] = {}
    all_representations = set()
    for proposal in proposals:
        representation_id = _parse_representation_id(proposal)
        if representation_id:
            representation_by_experiment[proposal.experiment_id] = representation_id
            all_representations.add(representation_id)
    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if pair.experiment_id not in representation_by_experiment or not _authoritative(pair):
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(
            pair.independence_class_id, pair
        )
    required = max(1, int(min_independent_classes))
    support: Dict[str, Dict[str, float]] = {}
    for (experiment_id, context_id), by_class in grouped.items():
        if len(by_class) < required:
            continue
        score = sum(max(0.0, float(pair.effect)) for pair in by_class.values()) / len(by_class)
        if score >= float(strong_effect_threshold):
            representation_id = representation_by_experiment[experiment_id]
            support.setdefault(representation_id, {})[context_id] = score
    eligible = []
    for representation_id, contexts in support.items():
        if len(contexts) < max(1, int(min_contexts)):
            continue
        eligible.append((-len(contexts), -sum(contexts.values()) / len(contexts), representation_id, tuple(sorted(contexts))))
    eligible.sort()
    if not eligible:
        return UpstreamSelectorRepresentationPolicy(
            status="NO_REPRODUCED_SELECTOR_REPRESENTATION",
            representation_id=None,
            supporting_contexts=(),
            candidate_representation_count=len(all_representations),
            reason="no selector representation has repeated verifier-derived single-patch support",
        )
    chosen = eligible[0]
    return UpstreamSelectorRepresentationPolicy(
        status="REPRODUCED_SELECTOR_REPRESENTATION",
        representation_id=chosen[2],
        supporting_contexts=chosen[3],
        candidate_representation_count=len(all_representations),
        reason="call-binding selector representation retained by repeated external executable outcomes",
    )


def select_authorized_selector_representation(
    representations: Sequence[UpstreamSelectorRepresentation],
    policy: UpstreamSelectorRepresentationPolicy,
) -> Optional[UpstreamSelectorRepresentation]:
    if policy.status != "REPRODUCED_SELECTOR_REPRESENTATION" or not policy.representation_id:
        return None
    return next((item for item in representations if item.representation_id == policy.representation_id), None)


class UpstreamSelectorRepresentationOrgan:
    def __init__(self, body) -> None:
        self.body = body

    def remember(self, proposals: Sequence[UpstreamSelectorRepresentationProposal]) -> None:
        for item in proposals:
            self.body.memory.remember_experiment(item.proposal)

    def policy(self) -> UpstreamSelectorRepresentationPolicy:
        return derive_selector_representation_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )
