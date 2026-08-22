from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .software_task_acquisition import PythonASTRepairGenerator, SoftwarePatchCandidate
from .world_coupling import WorldOutcomePair


CODE_CONTEXT_MARKER = "software_code_context="
REPOSITORY_FILE_MARKER = "repository_file_path_hash="
SOFTWARE_REPAIR_MARKER = "software_repair_operator="


@dataclass(frozen=True)
class RepositoryWidePatchCandidate:
    task_id: str
    repository_hash: str
    file_path: str
    file_path_hash: str
    site_index: int
    operator_id: str
    context_fingerprint: str
    patched_source: str
    proposal: InterventionProposal


@dataclass(frozen=True)
class RepositoryWideRepairPolicy:
    status: str
    context_fingerprint: Optional[str]
    operator_id: Optional[str]
    supporting_contexts: Tuple[str, ...]
    candidate_signature_count: int
    reason: str


@dataclass(frozen=True)
class RepositoryWideRepairSelection:
    status: str
    candidates: Tuple[RepositoryWidePatchCandidate, ...]
    policy_context_fingerprint: Optional[str]
    policy_operator_id: Optional[str]
    total_candidate_count: int
    reason: str


def repository_source_hash(files: Mapping[str, str]) -> str:
    payload = [(str(path), str(files[path])) for path in sorted(files)]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _normalized_shape(node: ast.AST, depth: int = 7) -> str:
    """Return a bounded syntax-only shape with names, values and filenames erased.

    The representation deliberately retains program structure while erasing identifier
    spellings, literal values, exact comparison direction and source position. It is
    therefore a source-derived localization phenotype, not a filename or symbol-name
    lookup table. The shape grammar/depth are still human-authored and remain an
    explicit claim boundary.
    """
    if depth <= 0:
        return type(node).__name__
    if isinstance(node, ast.Name):
        return "Name"
    if isinstance(node, ast.Attribute):
        return f"Attribute({_normalized_shape(node.value, depth - 1)})"
    if isinstance(node, ast.Constant):
        return f"Constant<{type(node.value).__name__}>"
    if isinstance(node, ast.cmpop):
        return "CmpOp"
    if isinstance(node, ast.boolop):
        return "BoolOp"
    if isinstance(node, ast.operator):
        return "BinOp"
    if isinstance(node, ast.unaryop):
        return "UnaryOp"
    if isinstance(node, (ast.Load, ast.Store, ast.Del)):
        return type(node).__name__

    children = [_normalized_shape(child, depth - 1) for child in ast.iter_child_nodes(node)]
    return f"{type(node).__name__}({','.join(children)})"


class _ContextSiteCollector(ast.NodeVisitor):
    """Enumerate repair sites in the same post-order as PythonASTRepairGenerator."""

    def __init__(self) -> None:
        self.function_stack: List[ast.AST] = []
        self.records: List[Tuple[str, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.function_stack.append(node)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.function_stack.append(node)
        self.generic_visit(node)
        self.function_stack.pop()

    def _context(self, node: ast.AST) -> str:
        anchor = self.function_stack[-1] if self.function_stack else node
        shape = _normalized_shape(anchor)
        return hashlib.sha256(shape.encode("utf-8")).hexdigest()[:20]

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        if len(node.ops) != 1:
            return
        mapping = {
            ast.Eq: "COMPARE::Eq->NotEq",
            ast.NotEq: "COMPARE::NotEq->Eq",
            ast.Gt: "COMPARE::Gt->GtE",
            ast.GtE: "COMPARE::GtE->Gt",
            ast.Lt: "COMPARE::Lt->LtE",
            ast.LtE: "COMPARE::LtE->Lt",
        }
        operator_id = mapping.get(type(node.ops[0]))
        if operator_id is not None:
            self.records.append((operator_id, self._context(node)))

    def visit_BoolOp(self, node: ast.BoolOp):
        self.generic_visit(node)
        mapping = {ast.And: "BOOL::And->Or", ast.Or: "BOOL::Or->And"}
        operator_id = mapping.get(type(node.op))
        if operator_id is not None:
            self.records.append((operator_id, self._context(node)))


def source_context_records(source: str) -> Tuple[Tuple[str, str], ...]:
    collector = _ContextSiteCollector()
    collector.visit(ast.parse(source))
    return tuple(collector.records)


class RepositoryWidePythonRepairGenerator:
    """Generate exact one-file patches across a repository without hidden outcomes."""

    def __init__(self, base_generator: Optional[PythonASTRepairGenerator] = None) -> None:
        self.base_generator = base_generator or PythonASTRepairGenerator()

    def generate(
        self,
        task_id: str,
        files: Mapping[str, str],
    ) -> Tuple[RepositoryWidePatchCandidate, ...]:
        repo_hash = repository_source_hash(files)
        generated: List[RepositoryWidePatchCandidate] = []
        for file_path in sorted(files):
            source = str(files[file_path])
            base_candidates = self.base_generator.generate(f"{task_id}::{file_path}", source)
            records = source_context_records(source)
            if len(records) != len(base_candidates):
                raise AssertionError("repository-wide context site identity drifted from base repair generator")
            path_hash = hashlib.sha256(str(file_path).encode("utf-8")).hexdigest()[:16]
            for base, (operator_id, context_fingerprint) in zip(base_candidates, records):
                if base.operator_id != operator_id:
                    raise AssertionError("context fingerprint attached to the wrong AST repair site")
                payload = {
                    "task_id": str(task_id),
                    "repository_hash": repo_hash,
                    "file_path_hash": path_hash,
                    "site_index": int(base.site_index),
                    "operator_id": base.operator_id,
                    "context_fingerprint": context_fingerprint,
                    "patched_source_hash": hashlib.sha256(base.patched_source.encode("utf-8")).hexdigest(),
                }
                digest = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()[:20]
                proposal = InterventionProposal(
                    experiment_id=f"REPO_WIDE_PATCH::{repo_hash[:12]}::{path_hash}::{base.site_index}::{digest}",
                    axis_id=f"AXIS::REPO_WIDE_SOFTWARE_REPAIR::{repo_hash[:16]}",
                    manipulated_variable=f"{context_fingerprint}|{base.operator_id}",
                    held_fixed=(),
                    low_value=0.0,
                    high_value=1.0,
                    predicted_low_side="BUGGY_REPOSITORY",
                    predicted_high_side="ONE_FILE_PATCHED_REPOSITORY",
                    reason=(
                        "execute repository-wide source-derived Python repair candidate; "
                        f"{SOFTWARE_REPAIR_MARKER}{base.operator_id} "
                        f"{CODE_CONTEXT_MARKER}{context_fingerprint} "
                        f"{REPOSITORY_FILE_MARKER}{path_hash} "
                        f"site_index={base.site_index} repository_hash={repo_hash}"
                    ),
                    status="PROPOSAL_ONLY",
                )
                generated.append(RepositoryWidePatchCandidate(
                    task_id=str(task_id),
                    repository_hash=repo_hash,
                    file_path=str(file_path),
                    file_path_hash=path_hash,
                    site_index=base.site_index,
                    operator_id=base.operator_id,
                    context_fingerprint=context_fingerprint,
                    patched_source=base.patched_source,
                    proposal=proposal,
                ))
        return tuple(generated)


def _parse_marker(reason: str, marker: str) -> Optional[str]:
    if marker not in reason:
        return None
    return reason.split(marker, 1)[1].strip().split()[0].rstrip(",;)") or None


def parse_repository_wide_signature(proposal: InterventionProposal) -> Tuple[Optional[str], Optional[str]]:
    reason = str(proposal.reason)
    return _parse_marker(reason, CODE_CONTEXT_MARKER), _parse_marker(reason, SOFTWARE_REPAIR_MARKER)


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def derive_repository_wide_repair_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> RepositoryWideRepairPolicy:
    signature_by_experiment: Dict[str, Tuple[str, str]] = {}
    for proposal in proposals:
        context_fingerprint, operator_id = parse_repository_wide_signature(proposal)
        if context_fingerprint is not None and operator_id is not None:
            signature_by_experiment[proposal.experiment_id] = (context_fingerprint, operator_id)

    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair) or pair.experiment_id not in signature_by_experiment:
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(
            pair.independence_class_id, pair
        )

    minimum_classes = max(1, int(min_independent_classes))
    support: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (experiment_id, context_id), classes in grouped.items():
        if len(classes) < minimum_classes:
            continue
        score = sum(abs(pair.effect) for pair in classes.values()) / len(classes)
        if score < float(strong_effect_threshold):
            continue
        signature = signature_by_experiment[experiment_id]
        support.setdefault(signature, {})[context_id] = float(score)

    required = max(1, int(min_contexts))
    eligible = []
    for (fingerprint, operator_id), contexts in support.items():
        if len(contexts) < required:
            continue
        mean_score = sum(contexts.values()) / len(contexts)
        eligible.append((-len(contexts), -mean_score, fingerprint, operator_id, tuple(sorted(contexts))))
    eligible.sort()
    if not eligible:
        return RepositoryWideRepairPolicy(
            status="NO_REPRODUCED_REPOSITORY_WIDE_LOCALIZATION",
            context_fingerprint=None,
            operator_id=None,
            supporting_contexts=(),
            candidate_signature_count=len(set(signature_by_experiment.values())),
            reason="no filename-independent AST context plus repair operator reproduced across world contexts",
        )
    chosen = eligible[0]
    return RepositoryWideRepairPolicy(
        status="REPRODUCED_REPOSITORY_WIDE_LOCALIZATION",
        context_fingerprint=chosen[2],
        operator_id=chosen[3],
        supporting_contexts=chosen[4],
        candidate_signature_count=len(set(signature_by_experiment.values())),
        reason="filename-independent AST context fingerprint plus repair operator reproduced across authenticated contexts",
    )


def select_repository_wide_candidates(
    candidates: Sequence[RepositoryWidePatchCandidate],
    policy: Optional[RepositoryWideRepairPolicy],
    max_candidates: Optional[int] = None,
    operator_only: bool = False,
) -> RepositoryWideRepairSelection:
    ordered = tuple(candidates)
    policy_fingerprint = None
    policy_operator = None
    status = "FULL_REPOSITORY_WIDE_REPAIR_SEARCH"
    reason = "no learned repository-wide localization applied"
    if policy is not None and policy.operator_id:
        policy_operator = policy.operator_id
        if operator_only:
            matching = tuple(candidate for candidate in ordered if candidate.operator_id == policy.operator_id)
            nonmatching = tuple(candidate for candidate in ordered if candidate not in matching)
            ordered = matching + nonmatching
            status = "OPERATOR_ONLY_REPAIR_PRIORITIZED"
            reason = "repair operator retained while learned localization phenotype is removed"
        elif policy.status == "REPRODUCED_REPOSITORY_WIDE_LOCALIZATION" and policy.context_fingerprint:
            policy_fingerprint = policy.context_fingerprint
            matching = tuple(
                candidate for candidate in ordered
                if candidate.operator_id == policy.operator_id
                and candidate.context_fingerprint == policy.context_fingerprint
            )
            nonmatching = tuple(candidate for candidate in ordered if candidate not in matching)
            ordered = matching + nonmatching
            status = "LEARNED_REPOSITORY_WIDE_LOCALIZATION_PRIORITIZED"
            reason = "reproduced filename-independent AST context and repair operator prioritized"
    if max_candidates is not None:
        ordered = ordered[: max(0, int(max_candidates))]
    return RepositoryWideRepairSelection(
        status=status,
        candidates=ordered,
        policy_context_fingerprint=policy_fingerprint,
        policy_operator_id=policy_operator,
        total_candidate_count=len(candidates),
        reason=reason,
    )


class RepositoryWideSoftwareTaskAcquisitionOrgan:
    """Repository-wide patch generation whose authority is reconstructed from BODY evidence."""

    def __init__(self, body, generator: Optional[RepositoryWidePythonRepairGenerator] = None) -> None:
        self.body = body
        self.generator = generator or RepositoryWidePythonRepairGenerator()

    def propose(self, task_id: str, files: Mapping[str, str]) -> Tuple[RepositoryWidePatchCandidate, ...]:
        candidates = self.generator.generate(task_id, files)
        for candidate in candidates:
            self.body.memory.remember_experiment(candidate.proposal)
        return candidates

    def policy(self) -> RepositoryWideRepairPolicy:
        return derive_repository_wide_repair_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )

    def select(
        self,
        candidates: Sequence[RepositoryWidePatchCandidate],
        max_candidates: Optional[int] = None,
        apply_learned_policy: bool = True,
        operator_only: bool = False,
    ) -> RepositoryWideRepairSelection:
        policy = self.policy() if apply_learned_policy else None
        return select_repository_wide_candidates(
            candidates,
            policy=policy,
            max_candidates=max_candidates,
            operator_only=operator_only,
        )
