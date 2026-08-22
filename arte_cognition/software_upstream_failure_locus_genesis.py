from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .world_coupling import WorldOutcomePair


UPSTREAM_PROGRAM_MARKER = "upstream_failure_program="
UPSTREAM_LOCUS_MARKER = "upstream_failure_locus="
UPSTREAM_EDIT_MARKER = "upstream_failure_edit="

_LOCUS_ASSERTION_BACKSLICE_LIST = "ASSERTION_BACKSLICE_TO_UPSTREAM_LIST"
_EDIT_REWRITE_PEER = "REWRITE_CALL_ARGUMENTS_FROM_PEER"
_EDIT_DUPLICATE_PEER = "DUPLICATE_LIST_CALL_WITH_FRESH_ID"
_EDIT_REWRITE_AND_DUPLICATE = "COMPOSE_REWRITE_AND_DUPLICATE"

_TRACE_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)')


@dataclass(frozen=True)
class UpstreamFailureProgram:
    locus_selector: str
    edit_operator: str

    @property
    def program_id(self) -> str:
        payload = {
            "locus_selector": self.locus_selector,
            "edit_operator": self.edit_operator,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        return f"UPSTREAM_FAILURE_PROGRAM::{digest}"


@dataclass(frozen=True)
class UpstreamPatchCandidate:
    program_id: str
    candidate_id: str
    patched_source: str
    operation_count: int
    oracle_suffix_sha256: str


@dataclass(frozen=True)
class UpstreamFailureProgramProposal:
    program: UpstreamFailureProgram
    proposal: InterventionProposal


@dataclass(frozen=True)
class UpstreamFailureProgramPolicy:
    status: str
    program_id: Optional[str]
    supporting_contexts: Tuple[str, ...]
    candidate_program_count: int
    reason: str


def generate_upstream_failure_programs() -> Tuple[UpstreamFailureProgram, ...]:
    """Outcome-independent bounded programs over a newly reachable upstream locus.

    These programs do not encode file names, observation labels, literal values,
    human fixes, or candidate success. They differ only in a small authored edit
    alphabet. External execution decides whether any program deserves authority.
    """
    return (
        UpstreamFailureProgram(_LOCUS_ASSERTION_BACKSLICE_LIST, _EDIT_REWRITE_PEER),
        UpstreamFailureProgram(_LOCUS_ASSERTION_BACKSLICE_LIST, _EDIT_DUPLICATE_PEER),
        UpstreamFailureProgram(_LOCUS_ASSERTION_BACKSLICE_LIST, _EDIT_REWRITE_AND_DUPLICATE),
    )


def propose_upstream_failure_program(program: UpstreamFailureProgram) -> UpstreamFailureProgramProposal:
    digest = hashlib.sha256(program.program_id.encode("utf-8")).hexdigest()[:20]
    proposal = InterventionProposal(
        experiment_id=f"SOFTWARE_UPSTREAM_FAILURE_PROGRAM::{digest}",
        axis_id=f"AXIS::SOFTWARE_FAILURE_LOCUS::{digest}",
        manipulated_variable=program.program_id,
        held_fixed=(),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="TRACEBACK_LOCAL_REPAIR_LANGUAGE_INEXPRESSIVE",
        predicted_high_side="UPSTREAM_BACKSLICE_REPAIR_SEARCH",
        reason=(
            "outcome-independent upstream failure-locus proposal; "
            f"{UPSTREAM_PROGRAM_MARKER}{program.program_id} "
            f"{UPSTREAM_LOCUS_MARKER}{program.locus_selector} "
            f"{UPSTREAM_EDIT_MARKER}{program.edit_operator}"
        ),
        status="PROPOSAL_ONLY",
    )
    return UpstreamFailureProgramProposal(program=program, proposal=proposal)


def _normalize_path(path: str) -> str:
    return str(path).replace("\\", "/").lstrip("./")


def target_frame_line(stderr: str, target_path: str) -> Optional[int]:
    target = _normalize_path(target_path)
    lines = []
    for path, line in _TRACE_FRAME_RE.findall(str(stderr)):
        if _normalize_path(path).endswith(target):
            lines.append(int(line))
    return lines[-1] if lines else None


def oracle_suffix(source: str, failure_line: int) -> str:
    lines = str(source).splitlines(keepends=True)
    start = max(0, int(failure_line) - 1)
    return "".join(lines[start:])


def oracle_suffix_sha256(source: str, failure_line: int) -> str:
    return hashlib.sha256(oracle_suffix(source, failure_line).encode("utf-8")).hexdigest()


def oracle_preserved(original: str, candidate: str, failure_line: int) -> bool:
    return oracle_suffix(original, failure_line) == oracle_suffix(candidate, failure_line)


def _names_in_node(node: ast.AST) -> Tuple[str, ...]:
    return tuple(sorted({item.id for item in ast.walk(node) if isinstance(item, ast.Name)}))


def _assigned_name(node: ast.Assign) -> Optional[str]:
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return None
    return node.targets[0].id


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


def locate_upstream_list_assignment(source: str, failure_line: int) -> Optional[Tuple[int, int, str]]:
    """Backslice failed statement -> producer assignment -> named list input.

    The algorithm is syntax/dataflow based. It never inspects expected outcomes or
    later fixes. It deliberately stops at one assignment hop to keep the language
    bounded and auditable.
    """
    try:
        tree = ast.parse(str(source))
    except SyntaxError:
        return None
    failed = _statement_covering_line(tree, int(failure_line))
    if failed is None:
        return None
    failed_names = set(_names_in_node(failed))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    producers = []
    for node in assignments:
        name = _assigned_name(node)
        if name in failed_names and int(node.lineno) < int(failure_line) and isinstance(node.value, ast.Call):
            producers.append(node)
    producers.sort(key=lambda node: int(node.lineno), reverse=True)
    for producer in producers:
        referenced = []
        for arg in producer.value.args:
            if isinstance(arg, ast.Name):
                referenced.append(arg.id)
        for keyword in producer.value.keywords:
            if isinstance(keyword.value, ast.Name):
                referenced.append(keyword.value.id)
        for ref in referenced:
            lists = [
                node for node in assignments
                if _assigned_name(node) == ref
                and int(node.lineno) < int(producer.lineno)
                and isinstance(node.value, ast.List)
            ]
            lists.sort(key=lambda node: int(node.lineno), reverse=True)
            if lists:
                chosen = lists[0]
                return int(chosen.lineno), int(chosen.col_offset), ref
    return None


def _find_list_assignment(tree: ast.AST, line: int, col: int) -> Optional[ast.Assign]:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and int(getattr(node, "lineno", -1)) == int(line)
            and int(getattr(node, "col_offset", -1)) == int(col)
            and isinstance(node.value, ast.List)
        ):
            return node
    return None


def _call_signature(call: ast.Call) -> str:
    args = [ast.unparse(item) for item in call.args[1:]] if call.args else []
    keywords = [(item.arg or "**", ast.unparse(item.value)) for item in call.keywords]
    return json.dumps({"args": args, "keywords": keywords}, sort_keys=True, separators=(",", ":"))


def _fresh_id(program_id: str, source: str, element_index: int, duplicate_index: int) -> str:
    digest = hashlib.sha256(
        f"{program_id}|{source}|{element_index}|{duplicate_index}".encode("utf-8")
    ).hexdigest()[:12]
    return f"arte_auto_{digest}"


def _mutated_source(
    source: str,
    locus: Tuple[int, int, str],
    *,
    rewrite: Optional[Tuple[int, int]] = None,
    duplicate_index: Optional[int] = None,
    program_id: str,
) -> Optional[str]:
    tree = ast.parse(str(source))
    assignment = _find_list_assignment(tree, locus[0], locus[1])
    if assignment is None:
        return None
    elements = assignment.value.elts
    if rewrite is not None:
        target_index, peer_index = rewrite
        if not (0 <= target_index < len(elements) and 0 <= peer_index < len(elements)):
            return None
        target = elements[target_index]
        peer = elements[peer_index]
        if not isinstance(target, ast.Call) or not isinstance(peer, ast.Call):
            return None
        if ast.dump(target.func, include_attributes=False) != ast.dump(peer.func, include_attributes=False):
            return None
        first = copy.deepcopy(target.args[:1])
        target.args = first + [copy.deepcopy(item) for item in peer.args[1:]]
        target.keywords = [copy.deepcopy(item) for item in peer.keywords]
    if duplicate_index is not None:
        if not (0 <= duplicate_index < len(elements)):
            return None
        original = elements[duplicate_index]
        if not isinstance(original, ast.Call):
            return None
        duplicate = copy.deepcopy(original)
        if duplicate.args and isinstance(duplicate.args[0], ast.Constant) and isinstance(duplicate.args[0].value, str):
            duplicate.args[0] = ast.Constant(
                value=_fresh_id(program_id, source, duplicate_index, len(elements))
            )
        elements.insert(duplicate_index + 1, duplicate)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def generate_upstream_patch_candidates(
    program: UpstreamFailureProgram,
    stderr: str,
    source: str,
    target_path: str,
    max_candidates: int = 256,
) -> Tuple[UpstreamPatchCandidate, ...]:
    if program.locus_selector != _LOCUS_ASSERTION_BACKSLICE_LIST:
        return ()
    failure_line = target_frame_line(stderr, target_path)
    if failure_line is None:
        return ()
    locus = locate_upstream_list_assignment(source, failure_line)
    if locus is None:
        return ()
    tree = ast.parse(str(source))
    assignment = _find_list_assignment(tree, locus[0], locus[1])
    if assignment is None:
        return ()
    calls = [isinstance(item, ast.Call) for item in assignment.value.elts]
    call_indices = [index for index, is_call in enumerate(calls) if is_call]
    signatures = {
        index: _call_signature(assignment.value.elts[index])  # type: ignore[arg-type]
        for index in call_indices
    }
    edits = []
    if program.edit_operator in {_EDIT_REWRITE_PEER, _EDIT_REWRITE_AND_DUPLICATE}:
        rewrites = [
            (target, peer)
            for target in call_indices
            for peer in call_indices
            if target != peer and signatures[target] != signatures[peer]
        ]
    else:
        rewrites = []
    if program.edit_operator == _EDIT_REWRITE_PEER:
        edits = [(rewrite, None) for rewrite in rewrites]
    elif program.edit_operator == _EDIT_DUPLICATE_PEER:
        edits = [(None, index) for index in call_indices]
    elif program.edit_operator == _EDIT_REWRITE_AND_DUPLICATE:
        edits = [(rewrite, duplicate) for rewrite in rewrites for duplicate in call_indices]
    else:
        return ()

    original_oracle = oracle_suffix_sha256(source, failure_line)
    seen = set()
    candidates = []
    for rewrite, duplicate in edits:
        patched = _mutated_source(
            source, locus, rewrite=rewrite, duplicate_index=duplicate,
            program_id=program.program_id,
        )
        if patched is None or not oracle_preserved(source, patched, failure_line):
            continue
        source_digest = hashlib.sha256(patched.encode("utf-8")).hexdigest()
        if source_digest in seen:
            continue
        seen.add(source_digest)
        payload = f"{program.program_id}|{rewrite}|{duplicate}|{source_digest}"
        candidate_id = f"UPSTREAM_PATCH::{hashlib.sha256(payload.encode()).hexdigest()[:20]}"
        candidates.append(UpstreamPatchCandidate(
            program_id=program.program_id,
            candidate_id=candidate_id,
            patched_source=patched,
            operation_count=int(rewrite is not None) + int(duplicate is not None),
            oracle_suffix_sha256=original_oracle,
        ))
        if len(candidates) >= max(1, int(max_candidates)):
            break
    return tuple(candidates)


def _parse_program_id(proposal: InterventionProposal) -> Optional[str]:
    reason = str(proposal.reason)
    if UPSTREAM_PROGRAM_MARKER not in reason:
        return None
    return reason.split(UPSTREAM_PROGRAM_MARKER, 1)[1].strip().split()[0].rstrip(",;)") or None


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def derive_upstream_failure_program_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> UpstreamFailureProgramPolicy:
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
    support: Dict[str, Dict[str, float]] = {}
    required_classes = max(1, int(min_independent_classes))
    for (experiment_id, context_id), by_class in grouped.items():
        if len(by_class) < required_classes:
            continue
        score = sum(max(0.0, float(pair.effect)) for pair in by_class.values()) / len(by_class)
        if score < float(strong_effect_threshold):
            continue
        support.setdefault(program_by_experiment[experiment_id], {})[context_id] = score
    eligible = []
    for program_id, contexts in support.items():
        if len(contexts) < max(1, int(min_contexts)):
            continue
        mean_score = sum(contexts.values()) / len(contexts)
        eligible.append((-len(contexts), -mean_score, program_id, tuple(sorted(contexts))))
    eligible.sort()
    if not eligible:
        return UpstreamFailureProgramPolicy(
            status="NO_REPRODUCED_UPSTREAM_FAILURE_PROGRAM",
            program_id=None,
            supporting_contexts=(),
            candidate_program_count=len(all_programs),
            reason="no upstream failure program has repeated verifier-derived executable support",
        )
    chosen = eligible[0]
    return UpstreamFailureProgramPolicy(
        status="REPRODUCED_UPSTREAM_FAILURE_PROGRAM",
        program_id=chosen[2],
        supporting_contexts=chosen[3],
        candidate_program_count=len(all_programs),
        reason="upstream failure-locus program retained by repeated external executable outcomes",
    )


def select_authorized_upstream_failure_program(
    programs: Sequence[UpstreamFailureProgram],
    policy: UpstreamFailureProgramPolicy,
) -> Optional[UpstreamFailureProgram]:
    if policy.status != "REPRODUCED_UPSTREAM_FAILURE_PROGRAM" or not policy.program_id:
        return None
    return next((item for item in programs if item.program_id == policy.program_id), None)


class UpstreamFailureProgramOrgan:
    def __init__(self, body) -> None:
        self.body = body

    def remember(self, proposals: Sequence[UpstreamFailureProgramProposal]) -> None:
        for item in proposals:
            self.body.memory.remember_experiment(item.proposal)

    def policy(self) -> UpstreamFailureProgramPolicy:
        return derive_upstream_failure_program_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )
