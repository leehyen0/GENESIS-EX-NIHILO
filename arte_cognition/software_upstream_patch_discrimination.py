from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .software_upstream_failure_locus_genesis import (
    UpstreamPatchCandidate,
    locate_upstream_list_assignment,
)
from .world_coupling import WorldOutcomePair


UPSTREAM_SELECTOR_MARKER = "upstream_patch_selector="
UPSTREAM_SELECTOR_RULE_MARKER = "upstream_patch_selector_rule="

_RULE_TOWARD_ANCHOR = "REWRITE_NONANCHOR_TOWARD_UNIQUE_FLAGGED_ANCHOR_CORE"
_RULE_AWAY_FROM_ANCHOR = "REWRITE_ANCHOR_CORE_AWAY_FROM_UNIQUE_FLAGGED_ANCHOR"


@dataclass(frozen=True)
class UpstreamPatchSelector:
    selection_rule: str

    @property
    def selector_id(self) -> str:
        digest = hashlib.sha256(self.selection_rule.encode("utf-8")).hexdigest()[:20]
        return f"UPSTREAM_PATCH_SELECTOR::{digest}"


@dataclass(frozen=True)
class UpstreamPatchSelectorProposal:
    selector: UpstreamPatchSelector
    proposal: InterventionProposal


@dataclass(frozen=True)
class UpstreamPatchSelectorPolicy:
    status: str
    selector_id: Optional[str]
    supporting_contexts: Tuple[str, ...]
    candidate_selector_count: int
    reason: str


def generate_upstream_patch_selectors() -> Tuple[UpstreamPatchSelector, ...]:
    """Bounded outcome-independent selector shadow language.

    The rules inspect only AST relations among call elements. They do not inspect
    literal feature/outcome values, historical fixes, candidate execution results,
    file names, or concrete row identifiers.
    """
    return (
        UpstreamPatchSelector(_RULE_TOWARD_ANCHOR),
        UpstreamPatchSelector(_RULE_AWAY_FROM_ANCHOR),
    )


def propose_upstream_patch_selector(selector: UpstreamPatchSelector) -> UpstreamPatchSelectorProposal:
    digest = hashlib.sha256(selector.selector_id.encode("utf-8")).hexdigest()[:20]
    proposal = InterventionProposal(
        experiment_id=f"SOFTWARE_UPSTREAM_PATCH_SELECTOR::{digest}",
        axis_id=f"AXIS::SOFTWARE_UPSTREAM_PATCH_SELECTION::{digest}",
        manipulated_variable=selector.selector_id,
        held_fixed=(),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="UPSTREAM_PATCH_FRONTIER_UNSELECTED",
        predicted_high_side="STRUCTURALLY_SELECTED_SINGLE_UPSTREAM_PATCH",
        reason=(
            "outcome-independent upstream patch semantic selector proposal; "
            f"{UPSTREAM_SELECTOR_MARKER}{selector.selector_id} "
            f"{UPSTREAM_SELECTOR_RULE_MARKER}{selector.selection_rule}"
        ),
        status="PROPOSAL_ONLY",
    )
    return UpstreamPatchSelectorProposal(selector=selector, proposal=proposal)


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


def _core(call: ast.Call) -> str:
    payload = {
        "func": ast.dump(call.func, include_attributes=False),
        "args_after_identity": [
            ast.dump(item, include_attributes=False) for item in call.args[1:]
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _flagged(call: ast.Call) -> bool:
    return any(
        isinstance(item.value, ast.Constant) and item.value.value is True
        for item in call.keywords
    )


def _rewrite_transition(
    original: str,
    candidate: str,
    failure_line: int,
) -> Optional[Tuple[int, str, str, bool, bool]]:
    locus = locate_upstream_list_assignment(original, failure_line)
    if locus is None:
        return None
    name = locus[2]
    before = _assigned_list_by_name(original, name)
    after = _assigned_list_by_name(candidate, name)
    if before is None or after is None or len(before.elts) != len(after.elts):
        return None
    changed = []
    for index, (left, right) in enumerate(zip(before.elts, after.elts)):
        if ast.dump(left, include_attributes=False) != ast.dump(right, include_attributes=False):
            changed.append(index)
    if len(changed) != 1:
        return None
    index = changed[0]
    left = before.elts[index]
    right = after.elts[index]
    if not isinstance(left, ast.Call) or not isinstance(right, ast.Call):
        return None
    return index, _core(left), _core(right), _flagged(left), _flagged(right)


def _anchor_core(source: str, failure_line: int) -> Optional[str]:
    locus = locate_upstream_list_assignment(source, failure_line)
    if locus is None:
        return None
    value = _assigned_list_by_name(source, locus[2])
    if value is None:
        return None
    flagged_cores = [
        _core(item) for item in value.elts
        if isinstance(item, ast.Call) and _flagged(item)
    ]
    unique = tuple(sorted(set(flagged_cores)))
    return unique[0] if len(unique) == 1 else None


def select_upstream_patch(
    selector: UpstreamPatchSelector,
    candidates: Sequence[UpstreamPatchCandidate],
    source: str,
    failure_line: int,
) -> Optional[UpstreamPatchCandidate]:
    """Select exactly one upstream rewrite before any candidate execution outcome."""
    anchor = _anchor_core(source, failure_line)
    if anchor is None:
        return None
    eligible = []
    for candidate in candidates:
        transition = _rewrite_transition(source, candidate.patched_source, failure_line)
        if transition is None:
            continue
        index, before_core, after_core, before_flagged, after_flagged = transition
        if selector.selection_rule == _RULE_TOWARD_ANCHOR:
            valid = (
                before_core != anchor
                and after_core == anchor
                and not before_flagged
                and not after_flagged
            )
        elif selector.selection_rule == _RULE_AWAY_FROM_ANCHOR:
            valid = (
                before_core == anchor
                and after_core != anchor
                and not before_flagged
                and not after_flagged
            )
        else:
            valid = False
        if valid:
            eligible.append((index, candidate.candidate_id, candidate))
    eligible.sort(key=lambda row: (row[0], row[1]))
    return eligible[0][2] if eligible else None


def _parse_selector_id(proposal: InterventionProposal) -> Optional[str]:
    reason = str(proposal.reason)
    if UPSTREAM_SELECTOR_MARKER not in reason:
        return None
    return reason.split(UPSTREAM_SELECTOR_MARKER, 1)[1].strip().split()[0].rstrip(",;)") or None


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def derive_upstream_patch_selector_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> UpstreamPatchSelectorPolicy:
    selector_by_experiment: Dict[str, str] = {}
    all_selectors = set()
    for proposal in proposals:
        selector_id = _parse_selector_id(proposal)
        if selector_id:
            selector_by_experiment[proposal.experiment_id] = selector_id
            all_selectors.add(selector_id)
    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if pair.experiment_id not in selector_by_experiment or not _authoritative(pair):
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(
            pair.independence_class_id, pair
        )
    required_classes = max(1, int(min_independent_classes))
    support: Dict[str, Dict[str, float]] = {}
    for (experiment_id, context_id), by_class in grouped.items():
        if len(by_class) < required_classes:
            continue
        score = sum(max(0.0, float(pair.effect)) for pair in by_class.values()) / len(by_class)
        if score < float(strong_effect_threshold):
            continue
        selector_id = selector_by_experiment[experiment_id]
        support.setdefault(selector_id, {})[context_id] = score
    eligible = []
    for selector_id, contexts in support.items():
        if len(contexts) < max(1, int(min_contexts)):
            continue
        mean_score = sum(contexts.values()) / len(contexts)
        eligible.append((-len(contexts), -mean_score, selector_id, tuple(sorted(contexts))))
    eligible.sort()
    if not eligible:
        return UpstreamPatchSelectorPolicy(
            status="NO_REPRODUCED_UPSTREAM_PATCH_SELECTOR",
            selector_id=None,
            supporting_contexts=(),
            candidate_selector_count=len(all_selectors),
            reason="no selector has repeated verifier-derived single-patch success",
        )
    chosen = eligible[0]
    return UpstreamPatchSelectorPolicy(
        status="REPRODUCED_UPSTREAM_PATCH_SELECTOR",
        selector_id=chosen[2],
        supporting_contexts=chosen[3],
        candidate_selector_count=len(all_selectors),
        reason="structural upstream patch selector retained by repeated external single-patch success",
    )


def select_authorized_upstream_patch_selector(
    selectors: Sequence[UpstreamPatchSelector],
    policy: UpstreamPatchSelectorPolicy,
) -> Optional[UpstreamPatchSelector]:
    if policy.status != "REPRODUCED_UPSTREAM_PATCH_SELECTOR" or not policy.selector_id:
        return None
    return next((item for item in selectors if item.selector_id == policy.selector_id), None)


class UpstreamPatchSelectorOrgan:
    def __init__(self, body) -> None:
        self.body = body

    def remember(self, proposals: Sequence[UpstreamPatchSelectorProposal]) -> None:
        for item in proposals:
            self.body.memory.remember_experiment(item.proposal)

    def policy(self) -> UpstreamPatchSelectorPolicy:
        return derive_upstream_patch_selector_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )
