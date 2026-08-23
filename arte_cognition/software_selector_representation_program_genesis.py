from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import itertools
import json
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .software_call_binding_representation import (
    CallBindingSchema,
    UpstreamSelectorRepresentation,
    select_upstream_patch_with_representation,
)
from .software_upstream_failure_locus_genesis import UpstreamPatchCandidate
from .software_upstream_patch_discrimination import UpstreamPatchSelector
from .world_coupling import WorldOutcomePair


PROGRAM_MARKER = "selector_representation_program="
PROGRAM_OPS_MARKER = "selector_representation_program_ops="

OP_RESOLVE_LOCAL_CALL_ALIAS = "RESOLVE_LOCAL_CALL_ALIAS"
OP_EXPAND_LITERAL_KWARGS = "EXPAND_LITERAL_KWARGS"
OP_EXPAND_LITERAL_STARARGS = "EXPAND_LITERAL_STARARGS"

_PRIMITIVE_ORDER = (
    OP_RESOLVE_LOCAL_CALL_ALIAS,
    OP_EXPAND_LITERAL_KWARGS,
    OP_EXPAND_LITERAL_STARARGS,
)


@dataclass(frozen=True)
class SelectorRepresentationProgramAssessment:
    status: str
    inexpressive_contexts: Tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class SelectorRepresentationProgram:
    operations: Tuple[str, ...]

    @property
    def program_id(self) -> str:
        payload = json.dumps(self.operations, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        return f"SELECTOR_REPRESENTATION_PROGRAM::{digest}"


@dataclass(frozen=True)
class SelectorRepresentationProgramProposal:
    program: SelectorRepresentationProgram
    proposal: InterventionProposal


@dataclass(frozen=True)
class SelectorRepresentationProgramPolicy:
    status: str
    program_id: Optional[str]
    supporting_contexts: Tuple[str, ...]
    candidate_program_count: int
    reason: str


def assess_representation_program_inexpressivity(
    contexts: Sequence[Tuple[str, int, int]],
    min_contexts: int = 2,
) -> SelectorRepresentationProgramAssessment:
    """Open program genesis only after repeated nonempty-frontier representation failure."""
    inexpressive = tuple(sorted(
        str(context_id)
        for context_id, frontier_count, selected_count in contexts
        if int(frontier_count) > 0 and int(selected_count) == 0
    ))
    if len(inexpressive) < max(1, int(min_contexts)):
        return SelectorRepresentationProgramAssessment(
            status="INSUFFICIENT_REPRESENTATION_PROGRAM_INEXPRESSIVITY",
            inexpressive_contexts=inexpressive,
            reason="program genesis requires repeated nonempty repair frontiers unreachable by the inherited representation",
        )
    return SelectorRepresentationProgramAssessment(
        status="REPRESENTATION_PROGRAM_INEXPRESSIVE_OPEN_COMPOSITION",
        inexpressive_contexts=inexpressive,
        reason="repeated representation failure persists despite an unchanged inherited repair frontier",
    )


def generate_selector_representation_programs(
    assessment: SelectorRepresentationProgramAssessment,
    max_depth: int = 2,
) -> Tuple[SelectorRepresentationProgram, ...]:
    """Generate canonical primitive compositions before any candidate execution outcome."""
    if assessment.status != "REPRESENTATION_PROGRAM_INEXPRESSIVE_OPEN_COMPOSITION":
        return ()
    depth = max(1, min(int(max_depth), len(_PRIMITIVE_ORDER)))
    programs = []
    for size in range(1, depth + 1):
        for combo in itertools.combinations(_PRIMITIVE_ORDER, size):
            programs.append(SelectorRepresentationProgram(tuple(combo)))
    return tuple(programs)


def propose_selector_representation_program(
    program: SelectorRepresentationProgram,
) -> SelectorRepresentationProgramProposal:
    digest = hashlib.sha256(program.program_id.encode("utf-8")).hexdigest()[:20]
    proposal = InterventionProposal(
        experiment_id=f"SOFTWARE_SELECTOR_REPRESENTATION_PROGRAM::{digest}",
        axis_id=f"AXIS::SOFTWARE_SELECTOR_REPRESENTATION_PROGRAM::{digest}",
        manipulated_variable=program.program_id,
        held_fixed=(),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="INHERITED_REPRESENTATION_INEXPRESSIVE",
        predicted_high_side="COMPOSED_NORMALIZER_ENABLES_PRE_OUTCOME_SELECTION",
        reason=(
            "outcome-independent selector representation program proposal; "
            f"{PROGRAM_MARKER}{program.program_id} "
            f"{PROGRAM_OPS_MARKER}{'|'.join(program.operations)}"
        ),
        status="PROPOSAL_ONLY",
    )
    return SelectorRepresentationProgramProposal(program=program, proposal=proposal)


class _ResolveLocalAlias(ast.NodeTransformer):
    def __init__(self, target_name: str) -> None:
        self.target_name = str(target_name)
        self.aliases = set()

    def collect(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            value = node.value
            if (
                isinstance(target, ast.Name)
                and isinstance(value, ast.Name)
                and value.id == self.target_name
            ):
                self.aliases.add(target.id)

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.aliases:
            node.func = ast.copy_location(ast.Name(id=self.target_name, ctx=ast.Load()), node.func)
        return node


class _ExpandLiteralKwargs(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        expanded = []
        seen = {item.arg for item in node.keywords if item.arg is not None}
        for keyword in node.keywords:
            if keyword.arg is not None:
                expanded.append(keyword)
                continue
            if not isinstance(keyword.value, ast.Dict):
                expanded.append(keyword)
                continue
            pairs = []
            valid = True
            for key, value in zip(keyword.value.keys, keyword.value.values):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    valid = False
                    break
                name = key.value
                if name in seen or any(existing == name for existing, _ in pairs):
                    valid = False
                    break
                pairs.append((name, value))
            if not valid:
                expanded.append(keyword)
                continue
            for name, value in pairs:
                seen.add(name)
                expanded.append(ast.keyword(arg=name, value=value))
        node.keywords = expanded
        return node


class _ExpandLiteralStarargs(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        args = []
        for value in node.args:
            if isinstance(value, ast.Starred) and isinstance(value.value, (ast.Tuple, ast.List)):
                args.extend(value.value.elts)
            else:
                args.append(value)
        node.args = args
        return node


def normalize_source_with_representation_program(
    source: str,
    program: SelectorRepresentationProgram,
    target_function_name: str,
) -> str:
    """Apply a bounded AST normalizer program without consulting execution outcomes."""
    tree = ast.parse(str(source))
    for operation in program.operations:
        if operation == OP_RESOLVE_LOCAL_CALL_ALIAS:
            transform = _ResolveLocalAlias(target_function_name)
            transform.collect(tree)
            tree = transform.visit(tree)
        elif operation == OP_EXPAND_LITERAL_KWARGS:
            tree = _ExpandLiteralKwargs().visit(tree)
        elif operation == OP_EXPAND_LITERAL_STARARGS:
            tree = _ExpandLiteralStarargs().visit(tree)
        else:
            raise ValueError(f"unsupported selector representation primitive: {operation}")
        ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _statement_covering_line(tree: ast.AST, line: int) -> Optional[ast.stmt]:
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        start = int(getattr(node, "lineno", -1))
        end = int(getattr(node, "end_lineno", start))
        if start <= int(line) <= end:
            matches.append(node)
    matches.sort(key=lambda node: (
        int(getattr(node, "end_lineno", node.lineno)) - int(node.lineno),
        -int(node.lineno),
    ))
    return matches[0] if matches else None


def remap_failure_line_after_normalization(
    original_source: str,
    original_failure_line: int,
    normalized_source: str,
) -> Optional[int]:
    """Map the unchanged failed statement through AST normalization by exact structure."""
    original_tree = ast.parse(str(original_source))
    failed = _statement_covering_line(original_tree, int(original_failure_line))
    if failed is None:
        return None
    fingerprint = ast.dump(failed, include_attributes=False)
    normalized_tree = ast.parse(str(normalized_source))
    matches = [
        node for node in ast.walk(normalized_tree)
        if isinstance(node, ast.stmt) and ast.dump(node, include_attributes=False) == fingerprint
    ]
    if len(matches) != 1:
        return None
    return int(matches[0].lineno)


def select_upstream_patch_with_representation_program(
    selector: UpstreamPatchSelector,
    representation: UpstreamSelectorRepresentation,
    program: SelectorRepresentationProgram,
    candidates: Sequence[UpstreamPatchCandidate],
    source: str,
    failure_line: int,
    schema: CallBindingSchema,
) -> Optional[UpstreamPatchCandidate]:
    """Normalize a shadow copy, select there, then return the original candidate identity."""
    normalized_source = normalize_source_with_representation_program(
        source, program, schema.function_name
    )
    normalized_failure_line = remap_failure_line_after_normalization(
        source, failure_line, normalized_source
    )
    if normalized_failure_line is None:
        return None
    normalized_candidates = []
    original_by_id = {}
    for candidate in candidates:
        try:
            patched = normalize_source_with_representation_program(
                candidate.patched_source, program, schema.function_name
            )
        except (SyntaxError, ValueError):
            continue
        shadow = UpstreamPatchCandidate(
            program_id=candidate.program_id,
            candidate_id=candidate.candidate_id,
            patched_source=patched,
            operation_count=candidate.operation_count,
            oracle_fingerprint_sha256=candidate.oracle_fingerprint_sha256,
        )
        normalized_candidates.append(shadow)
        original_by_id[candidate.candidate_id] = candidate
    chosen = select_upstream_patch_with_representation(
        selector,
        representation,
        tuple(normalized_candidates),
        normalized_source,
        normalized_failure_line,
        schema,
    )
    if chosen is None:
        return None
    return original_by_id.get(chosen.candidate_id)


def _parse_program_id(proposal: InterventionProposal) -> Optional[str]:
    reason = str(proposal.reason)
    if PROGRAM_MARKER not in reason:
        return None
    return reason.split(PROGRAM_MARKER, 1)[1].strip().split()[0].rstrip(",;)") or None


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def derive_selector_representation_program_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> SelectorRepresentationProgramPolicy:
    program_by_experiment: Dict[str, str] = {}
    all_programs = set()
    for proposal in proposals:
        program_id = _parse_program_id(proposal)
        if program_id:
            program_by_experiment[proposal.experiment_id] = program_id
            all_programs.add(program_id)
    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if pair.experiment_id not in program_by_experiment or not _authoritative(pair):
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
            program_id = program_by_experiment[experiment_id]
            support.setdefault(program_id, {})[context_id] = score
    eligible = []
    for program_id, contexts in support.items():
        if len(contexts) < max(1, int(min_contexts)):
            continue
        eligible.append((-len(contexts), -sum(contexts.values()) / len(contexts), program_id, tuple(sorted(contexts))))
    eligible.sort()
    if not eligible:
        return SelectorRepresentationProgramPolicy(
            status="NO_REPRODUCED_SELECTOR_REPRESENTATION_PROGRAM",
            program_id=None,
            supporting_contexts=(),
            candidate_program_count=len(all_programs),
            reason="no composed selector representation program has repeated verifier-derived support",
        )
    chosen = eligible[0]
    return SelectorRepresentationProgramPolicy(
        status="REPRODUCED_SELECTOR_REPRESENTATION_PROGRAM",
        program_id=chosen[2],
        supporting_contexts=chosen[3],
        candidate_program_count=len(all_programs),
        reason="composed selector representation program retained by repeated external executable outcomes",
    )


def select_authorized_selector_representation_program(
    programs: Sequence[SelectorRepresentationProgram],
    policy: SelectorRepresentationProgramPolicy,
) -> Optional[SelectorRepresentationProgram]:
    if policy.status != "REPRODUCED_SELECTOR_REPRESENTATION_PROGRAM" or not policy.program_id:
        return None
    return next((item for item in programs if item.program_id == policy.program_id), None)


class SelectorRepresentationProgramOrgan:
    def __init__(self, body) -> None:
        self.body = body

    def remember(self, proposals: Sequence[SelectorRepresentationProgramProposal]) -> None:
        for item in proposals:
            self.body.memory.remember_experiment(item.proposal)

    def policy(self) -> SelectorRepresentationProgramPolicy:
        return derive_selector_representation_program_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )
