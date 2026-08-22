from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .software_repair_constructor_genesis import RelationalConstructorPrimitive
from .world_coupling import WorldOutcomePair


EXTRACTOR_PROGRAM_MARKER = "failure_extractor_program="
EXTRACTOR_FRAME_MARKER = "failure_extractor_frame="
EXTRACTOR_LOCUS_MARKER = "failure_extractor_locus="
EXTRACTOR_EDIT_MARKER = "failure_extractor_edit="

_FRAME_TARGET_LAST = "TARGET_PATH_LAST_FRAME"
_LOCUS_ENCLOSING_CALL = "ENCLOSING_AST_CALL"
_EDIT_DROP_POSITIONAL = "ENUMERATE_DROP_POSITIONAL_ARGUMENT"
_EDIT_DROP_KEYWORD = "ENUMERATE_DROP_KEYWORD_ARGUMENT"

_TRACE_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)')
_EXCEPTION_RE = re.compile(r"(?m)^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):\s*(.*)$")


@dataclass(frozen=True)
class FailureExtractorProgram:
    frame_selector: str
    locus_selector: str
    edit_enumerator: str

    @property
    def program_id(self) -> str:
        payload = {
            "frame_selector": self.frame_selector,
            "locus_selector": self.locus_selector,
            "edit_enumerator": self.edit_enumerator,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        return f"FAILURE_EXTRACTOR_PROGRAM::{digest}"


@dataclass(frozen=True)
class ExtractorPatchCandidate:
    program_id: str
    edit_id: str
    source_hash: str
    patched_source: str


@dataclass(frozen=True)
class FailureExtractorProgramProposal:
    program: FailureExtractorProgram
    proposal: InterventionProposal


@dataclass(frozen=True)
class FailureExtractorProgramPolicy:
    status: str
    program_id: Optional[str]
    supporting_contexts: Tuple[str, ...]
    candidate_program_count: int
    reason: str


@dataclass(frozen=True)
class ExtractedRepairInterpretation:
    exception_family: str
    program_id: str
    constructor_primitive: RelationalConstructorPrimitive
    patch_candidates: Tuple[ExtractorPatchCandidate, ...]


def generate_failure_extractor_programs() -> Tuple[FailureExtractorProgram, ...]:
    """Generate a bounded extractor-program shadow language without outcome access.

    The alphabet is deliberately generic: select a traceback frame belonging to the
    supplied target source, locate the AST call covering that frame line, then
    enumerate one edit family. No exception name, API name, argument ordinal, hidden
    test result, or later human fix is encoded in the program definitions.
    """
    return (
        FailureExtractorProgram(
            frame_selector=_FRAME_TARGET_LAST,
            locus_selector=_LOCUS_ENCLOSING_CALL,
            edit_enumerator=_EDIT_DROP_POSITIONAL,
        ),
        FailureExtractorProgram(
            frame_selector=_FRAME_TARGET_LAST,
            locus_selector=_LOCUS_ENCLOSING_CALL,
            edit_enumerator=_EDIT_DROP_KEYWORD,
        ),
    )


def propose_failure_extractor_program(program: FailureExtractorProgram) -> FailureExtractorProgramProposal:
    digest = hashlib.sha256(program.program_id.encode("utf-8")).hexdigest()[:20]
    proposal = InterventionProposal(
        experiment_id=f"SOFTWARE_FAILURE_EXTRACTOR_PROGRAM::{digest}",
        axis_id=f"AXIS::SOFTWARE_FAILURE_EXTRACTOR::{digest}",
        manipulated_variable=program.program_id,
        held_fixed=(),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="CURRENT_RELATION_EXTRACTOR_INEXPRESSIVE",
        predicted_high_side="GENERATED_EXTRACTOR_PROGRAM_REPAIR_SEARCH",
        reason=(
            "outcome-independent compositional failure-extractor program proposal; "
            f"{EXTRACTOR_PROGRAM_MARKER}{program.program_id} "
            f"{EXTRACTOR_FRAME_MARKER}{program.frame_selector} "
            f"{EXTRACTOR_LOCUS_MARKER}{program.locus_selector} "
            f"{EXTRACTOR_EDIT_MARKER}{program.edit_enumerator}"
        ),
        status="PROPOSAL_ONLY",
    )
    return FailureExtractorProgramProposal(program=program, proposal=proposal)


def _normalize_path(path: str) -> str:
    return str(path).replace("\\", "/").lstrip("./")


def _target_frame_line(stderr: str, target_path: str) -> Optional[int]:
    target = _normalize_path(target_path)
    matches = []
    for path, line in _TRACE_FRAME_RE.findall(str(stderr)):
        normalized = _normalize_path(path)
        if normalized.endswith(target):
            matches.append(int(line))
    return matches[-1] if matches else None


def _covering_calls(tree: ast.AST, line: int) -> Tuple[ast.Call, ...]:
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        start = int(getattr(node, "lineno", -1))
        end = int(getattr(node, "end_lineno", start))
        if start <= int(line) <= end:
            calls.append(node)
    calls.sort(key=lambda node: (
        int(getattr(node, "end_lineno", node.lineno)) - int(node.lineno),
        int(node.col_offset),
    ))
    return tuple(calls)


class _CallEditTransformer(ast.NodeTransformer):
    def __init__(self, target_line: int, target_col: int, mode: str, item_index: int) -> None:
        self.target_line = int(target_line)
        self.target_col = int(target_col)
        self.mode = str(mode)
        self.item_index = int(item_index)
        self.changed = 0

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if int(node.lineno) != self.target_line or int(node.col_offset) != self.target_col:
            return node
        if self.mode == _EDIT_DROP_POSITIONAL and 0 <= self.item_index < len(node.args):
            node.args = [arg for index, arg in enumerate(node.args) if index != self.item_index]
            self.changed += 1
        elif self.mode == _EDIT_DROP_KEYWORD and 0 <= self.item_index < len(node.keywords):
            node.keywords = [item for index, item in enumerate(node.keywords) if index != self.item_index]
            self.changed += 1
        return node


def apply_failure_extractor_program(
    program: FailureExtractorProgram,
    stderr: str,
    source: str,
    target_path: str,
) -> Tuple[ExtractorPatchCandidate, ...]:
    if program.frame_selector != _FRAME_TARGET_LAST or program.locus_selector != _LOCUS_ENCLOSING_CALL:
        return ()
    line = _target_frame_line(stderr, target_path)
    if line is None:
        return ()
    try:
        tree = ast.parse(str(source))
    except SyntaxError:
        return ()
    calls = _covering_calls(tree, line)
    if not calls:
        return ()
    call = calls[0]
    count = len(call.args) if program.edit_enumerator == _EDIT_DROP_POSITIONAL else len(call.keywords)
    if program.edit_enumerator not in {_EDIT_DROP_POSITIONAL, _EDIT_DROP_KEYWORD}:
        return ()
    source_hash = hashlib.sha256(str(source).encode("utf-8")).hexdigest()
    candidates = []
    for item_index in range(count):
        cloned = copy.deepcopy(tree)
        transformer = _CallEditTransformer(
            target_line=int(call.lineno),
            target_col=int(call.col_offset),
            mode=program.edit_enumerator,
            item_index=item_index,
        )
        changed = transformer.visit(cloned)
        ast.fix_missing_locations(changed)
        if transformer.changed != 1:
            continue
        patched = ast.unparse(changed) + "\n"
        digest = hashlib.sha256(
            f"{program.program_id}|{source_hash}|{target_path}|{item_index}|{patched}".encode("utf-8")
        ).hexdigest()[:16]
        candidates.append(ExtractorPatchCandidate(
            program_id=program.program_id,
            edit_id=f"{program.edit_enumerator}::{item_index}::{digest}",
            source_hash=source_hash,
            patched_source=patched,
        ))
    return tuple(candidates)


def terminal_exception_family(stderr: str) -> Optional[str]:
    matches = tuple(_EXCEPTION_RE.finditer(str(stderr)))
    if not matches:
        return None
    raw = matches[-1].group(1)
    return re.sub(r"(?<!^)(?=[A-Z])", "_", raw).upper()


def interpret_with_extractor_program(
    program: FailureExtractorProgram,
    stderr: str,
    source: str,
    target_path: str,
) -> Optional[ExtractedRepairInterpretation]:
    patches = apply_failure_extractor_program(program, stderr, source, target_path)
    exception = terminal_exception_family(stderr)
    if exception is None or not patches:
        return None
    primitive = RelationalConstructorPrimitive(
        exception_family=exception,
        locus_kind="CALL",
        binding_relation=f"EXTRACTOR_PROGRAM::{program.program_id}",
        repair_goal="RESTORE_EXECUTABLE_CONTRACT",
    )
    return ExtractedRepairInterpretation(
        exception_family=exception,
        program_id=program.program_id,
        constructor_primitive=primitive,
        patch_candidates=patches,
    )


def _parse_program_id(proposal: InterventionProposal) -> Optional[str]:
    reason = str(proposal.reason)
    if EXTRACTOR_PROGRAM_MARKER not in reason:
        return None
    return reason.split(EXTRACTOR_PROGRAM_MARKER, 1)[1].strip().split()[0].rstrip(",;)") or None


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def derive_failure_extractor_program_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> FailureExtractorProgramPolicy:
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
    required_classes = max(1, int(min_independent_classes))
    support: Dict[str, Dict[str, float]] = {}
    for (experiment_id, context_id), by_class in grouped.items():
        if len(by_class) < required_classes:
            continue
        score = sum(abs(pair.effect) for pair in by_class.values()) / len(by_class)
        if score < float(strong_effect_threshold):
            continue
        program_id = program_by_experiment[experiment_id]
        support.setdefault(program_id, {})[context_id] = float(score)
    eligible = []
    required_contexts = max(1, int(min_contexts))
    for program_id, contexts in support.items():
        if len(contexts) < required_contexts:
            continue
        mean_score = sum(contexts.values()) / len(contexts)
        eligible.append((-len(contexts), -mean_score, program_id, tuple(sorted(contexts))))
    eligible.sort()
    if not eligible:
        return FailureExtractorProgramPolicy(
            status="NO_REPRODUCED_FAILURE_EXTRACTOR_PROGRAM",
            program_id=None,
            supporting_contexts=(),
            candidate_program_count=len(all_programs),
            reason="no generated extractor program has repeated verifier-derived executable success",
        )
    chosen = eligible[0]
    return FailureExtractorProgramPolicy(
        status="REPRODUCED_FAILURE_EXTRACTOR_PROGRAM",
        program_id=chosen[2],
        supporting_contexts=chosen[3],
        candidate_program_count=len(all_programs),
        reason="generated traceback-to-AST extractor program retained by repeated external executable repair success",
    )


def select_authorized_failure_extractor_program(
    programs: Sequence[FailureExtractorProgram],
    policy: FailureExtractorProgramPolicy,
) -> Optional[FailureExtractorProgram]:
    if policy.status != "REPRODUCED_FAILURE_EXTRACTOR_PROGRAM" or not policy.program_id:
        return None
    for program in programs:
        if program.program_id == policy.program_id:
            return program
    return None


class FailureExtractorProgramOrgan:
    def __init__(self, body) -> None:
        self.body = body

    def remember(self, proposals: Sequence[FailureExtractorProgramProposal]) -> None:
        for item in proposals:
            self.body.memory.remember_experiment(item.proposal)

    def policy(self) -> FailureExtractorProgramPolicy:
        return derive_failure_extractor_program_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )
