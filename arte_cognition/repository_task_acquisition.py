from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import subprocess
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .world_coupling import WorldOutcomePair, WorldOutcomeReceipt


REPOSITORY_REPAIR_OPERATOR_MARKER = "repository_repair_operator="
REPOSITORY_FILE_ROLE_MARKER = "repository_file_role="


_COMPARE_MUTATIONS = {
    ast.Eq: (ast.NotEq, "COMPARE::Eq->NotEq"),
    ast.NotEq: (ast.Eq, "COMPARE::NotEq->Eq"),
    ast.Gt: (ast.GtE, "COMPARE::Gt->GtE"),
    ast.GtE: (ast.Gt, "COMPARE::GtE->Gt"),
    ast.Lt: (ast.LtE, "COMPARE::Lt->LtE"),
    ast.LtE: (ast.Lt, "COMPARE::LtE->Lt"),
}
_BOOL_MUTATIONS = {
    ast.And: (ast.Or, "BOOL::And->Or"),
    ast.Or: (ast.And, "BOOL::Or->And"),
}
_BINOP_MUTATIONS = {
    ast.Add: (ast.Sub, "BINOP::Add->Sub"),
    ast.Sub: (ast.Add, "BINOP::Sub->Add"),
    ast.Mult: (ast.Add, "BINOP::Mult->Add"),
}


@dataclass(frozen=True)
class RepositoryPatchCandidate:
    task_id: str
    repository_hash: str
    file_path: str
    file_role: str
    site_index: int
    operator_id: str
    patched_source: str
    proposal: InterventionProposal


@dataclass(frozen=True)
class RepositoryRepairPolicy:
    status: str
    file_role: Optional[str]
    operator_id: Optional[str]
    supporting_contexts: Tuple[str, ...]
    candidate_signature_count: int
    reason: str


@dataclass(frozen=True)
class RepositoryRepairSelection:
    status: str
    candidates: Tuple[RepositoryPatchCandidate, ...]
    policy_file_role: Optional[str]
    policy_operator_id: Optional[str]
    total_candidate_count: int
    reason: str


def _module_name(path: str) -> str:
    clean = str(path).replace("\\", "/")
    if clean.endswith(".py"):
        clean = clean[:-3]
    return clean.replace("/", ".")


def repository_hash(files: Mapping[str, str]) -> str:
    payload = [[str(path), str(source)] for path, source in sorted(files.items())]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def derive_repository_file_roles(files: Mapping[str, str]) -> Dict[str, str]:
    """Derive filename-independent structural roles from the Python import graph."""
    python_files = {str(path): str(source) for path, source in files.items() if str(path).endswith(".py")}
    module_to_path = {_module_name(path): path for path in python_files}
    outgoing: Dict[str, set[str]] = {path: set() for path in python_files}
    incoming: Dict[str, set[str]] = {path: set() for path in python_files}

    for path, source in python_files.items():
        tree = ast.parse(source)
        targets: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                targets.add(str(node.module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    targets.add(str(alias.name))
        for target in targets:
            resolved = module_to_path.get(target)
            if resolved is None and "." in target:
                resolved = module_to_path.get(target.split(".", 1)[0])
            if resolved is None or resolved == path:
                continue
            outgoing[path].add(resolved)
            incoming[resolved].add(path)

    roles: Dict[str, str] = {}
    for path in sorted(python_files):
        in_count = len(incoming[path])
        out_count = len(outgoing[path])
        if in_count > 0 and out_count == 0:
            role = "IMPORTED_LEAF"
        elif in_count == 0 and out_count > 0:
            role = "ROOT_IMPORTER"
        elif in_count > 0 and out_count > 0:
            role = "INTERMEDIATE"
        else:
            role = "ISOLATED"
        roles[path] = role
    return roles


class _RepositorySingleMutationTransformer(ast.NodeTransformer):
    def __init__(self, target_index: int) -> None:
        self.target_index = int(target_index)
        self.current_index = -1
        self.applied_operator_id: Optional[str] = None

    def _maybe_apply(self, replacement, operator_id: str, node, attribute: str) -> None:
        self.current_index += 1
        if self.current_index == self.target_index:
            setattr(node, attribute, replacement())
            self.applied_operator_id = operator_id

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        if len(node.ops) == 1:
            mutation = _COMPARE_MUTATIONS.get(type(node.ops[0]))
            if mutation is not None:
                replacement, operator_id = mutation
                self.current_index += 1
                if self.current_index == self.target_index:
                    node.ops[0] = replacement()
                    self.applied_operator_id = operator_id
        return node

    def visit_BoolOp(self, node: ast.BoolOp):
        self.generic_visit(node)
        mutation = _BOOL_MUTATIONS.get(type(node.op))
        if mutation is not None:
            replacement, operator_id = mutation
            self._maybe_apply(replacement, operator_id, node, "op")
        return node

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        mutation = _BINOP_MUTATIONS.get(type(node.op))
        if mutation is not None:
            replacement, operator_id = mutation
            self._maybe_apply(replacement, operator_id, node, "op")
        return node


class PythonRepositoryRepairGenerator:
    """Generate exact one-site repairs across every Python file without test outcomes."""

    @staticmethod
    def _operator_ids(source: str) -> Tuple[str, ...]:
        ids: List[str] = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                mutation = _COMPARE_MUTATIONS.get(type(node.ops[0]))
                if mutation is not None:
                    ids.append(mutation[1])
            elif isinstance(node, ast.BoolOp):
                mutation = _BOOL_MUTATIONS.get(type(node.op))
                if mutation is not None:
                    ids.append(mutation[1])
            elif isinstance(node, ast.BinOp):
                mutation = _BINOP_MUTATIONS.get(type(node.op))
                if mutation is not None:
                    ids.append(mutation[1])
        return tuple(ids)

    def generate(
        self,
        task_id: str,
        files: Mapping[str, str],
    ) -> Tuple[RepositoryPatchCandidate, ...]:
        repo_hash = repository_hash(files)
        roles = derive_repository_file_roles(files)
        candidates: List[RepositoryPatchCandidate] = []
        for file_path in sorted(roles):
            source = str(files[file_path])
            for site_index, expected_operator_id in enumerate(self._operator_ids(source)):
                tree = ast.parse(source)
                transformer = _RepositorySingleMutationTransformer(site_index)
                mutated = transformer.visit(tree)
                ast.fix_missing_locations(mutated)
                if transformer.applied_operator_id != expected_operator_id:
                    raise AssertionError("repository repair site identity drifted during regeneration")
                patched_source = ast.unparse(mutated) + "\n"
                payload = {
                    "task_id": str(task_id),
                    "repository_hash": repo_hash,
                    "file_path": str(file_path),
                    "file_role": roles[file_path],
                    "site_index": int(site_index),
                    "operator_id": expected_operator_id,
                    "patched_source": patched_source,
                }
                digest = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()[:20]
                proposal = InterventionProposal(
                    experiment_id=(
                        f"REPOSITORY_PATCH::{repo_hash[:12]}::{hashlib.sha256(file_path.encode()).hexdigest()[:8]}::"
                        f"{site_index}::{digest}"
                    ),
                    axis_id=f"AXIS::REPOSITORY_REPAIR::{repo_hash[:16]}",
                    manipulated_variable=f"{roles[file_path]}::{expected_operator_id}",
                    held_fixed=(),
                    low_value=0.0,
                    high_value=1.0,
                    predicted_low_side="BUGGY_REPOSITORY",
                    predicted_high_side="PATCHED_REPOSITORY",
                    reason=(
                        "execute exact multi-file repository patch candidate; "
                        f"{REPOSITORY_REPAIR_OPERATOR_MARKER}{expected_operator_id} "
                        f"{REPOSITORY_FILE_ROLE_MARKER}{roles[file_path]} "
                        f"repository_hash={repo_hash} file_hash={hashlib.sha256(file_path.encode()).hexdigest()} "
                        f"site_index={site_index}"
                    ),
                    status="PROPOSAL_ONLY",
                )
                candidates.append(RepositoryPatchCandidate(
                    task_id=str(task_id),
                    repository_hash=repo_hash,
                    file_path=str(file_path),
                    file_role=roles[file_path],
                    site_index=site_index,
                    operator_id=expected_operator_id,
                    patched_source=patched_source,
                    proposal=proposal,
                ))
        return tuple(candidates)


def _parse_marker(reason: str, marker: str) -> Optional[str]:
    if marker not in reason:
        return None
    return reason.split(marker, 1)[1].strip().split()[0].rstrip(",;)") or None


def parse_repository_repair_signature(proposal: InterventionProposal) -> Tuple[Optional[str], Optional[str]]:
    reason = str(proposal.reason)
    return (
        _parse_marker(reason, REPOSITORY_FILE_ROLE_MARKER),
        _parse_marker(reason, REPOSITORY_REPAIR_OPERATOR_MARKER),
    )


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def derive_repository_repair_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> RepositoryRepairPolicy:
    signatures = {
        proposal.experiment_id: parse_repository_repair_signature(proposal)
        for proposal in proposals
    }
    signatures = {
        experiment_id: signature
        for experiment_id, signature in signatures.items()
        if signature[0] is not None and signature[1] is not None
    }
    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair) or pair.experiment_id not in signatures:
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(
            pair.independence_class_id, pair
        )

    support: Dict[Tuple[str, str], Dict[str, float]] = {}
    minimum_classes = max(1, int(min_independent_classes))
    for (experiment_id, context_id), by_class in grouped.items():
        if len(by_class) < minimum_classes:
            continue
        score = sum(abs(pair.effect) for pair in by_class.values()) / len(by_class)
        if score < float(strong_effect_threshold):
            continue
        role, operator_id = signatures[experiment_id]
        if role is None or operator_id is None:
            continue
        support.setdefault((role, operator_id), {})[context_id] = float(score)

    eligible = []
    required_contexts = max(1, int(min_contexts))
    for (role, operator_id), contexts in support.items():
        if len(contexts) < required_contexts:
            continue
        mean_score = sum(contexts.values()) / len(contexts)
        eligible.append((-len(contexts), -mean_score, role, operator_id, tuple(sorted(contexts))))
    eligible.sort()
    signature_space = set(signatures.values())
    if not eligible:
        return RepositoryRepairPolicy(
            status="NO_REPRODUCED_REPOSITORY_REPAIR_LOCALIZATION",
            file_role=None,
            operator_id=None,
            supporting_contexts=(),
            candidate_signature_count=len(signature_space),
            reason="no file-role and repair-operator pair has repeated authenticated repository success",
        )
    chosen = eligible[0]
    return RepositoryRepairPolicy(
        status="REPRODUCED_REPOSITORY_REPAIR_LOCALIZATION",
        file_role=chosen[2],
        operator_id=chosen[3],
        supporting_contexts=chosen[4],
        candidate_signature_count=len(signature_space),
        reason="static import-graph file role plus repair operator reproduced across source-disjoint repositories",
    )


def select_repository_patch_candidates(
    candidates: Sequence[RepositoryPatchCandidate],
    policy: Optional[RepositoryRepairPolicy],
    max_candidates: Optional[int] = None,
) -> RepositoryRepairSelection:
    ordered = tuple(candidates)
    policy_role = None
    policy_operator = None
    status = "FULL_REPOSITORY_REPAIR_SEARCH"
    reason = "no learned repository localization applied"
    if (
        policy is not None
        and policy.status == "REPRODUCED_REPOSITORY_REPAIR_LOCALIZATION"
        and policy.file_role
        and policy.operator_id
    ):
        policy_role = policy.file_role
        policy_operator = policy.operator_id
        matching = tuple(
            candidate for candidate in ordered
            if candidate.file_role == policy.file_role and candidate.operator_id == policy.operator_id
        )
        nonmatching = tuple(candidate for candidate in ordered if candidate not in matching)
        ordered = matching + nonmatching
        status = "LEARNED_REPOSITORY_LOCALIZATION_PRIORITIZED"
        reason = "reproduced import-graph role and repair operator prioritized on fresh repository"
    if max_candidates is not None:
        ordered = ordered[: max(0, int(max_candidates))]
    return RepositoryRepairSelection(
        status=status,
        candidates=ordered,
        policy_file_role=policy_role,
        policy_operator_id=policy_operator,
        total_candidate_count=len(candidates),
        reason=reason,
    )


class SubprocessRepositoryRepairExecutor:
    """Materialize a multi-file repository and execute hidden tests in a fresh process."""

    _HARNESS = r'''
import importlib, json, pathlib, sys, tempfile
payload = json.loads(sys.stdin.read())
try:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        for relative, source in payload["files"].items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        sys.path.insert(0, str(root))
        module = importlib.import_module(payload["entry_module"])
        function = getattr(module, payload["function_name"])
        ok = True
        for case in payload["cases"]:
            try:
                result = function(*case["args"])
            except Exception:
                ok = False
                break
            if result != case["expected"]:
                ok = False
                break
        print(json.dumps({"ok": bool(ok)}))
except Exception:
    print(json.dumps({"ok": False}))
'''

    def __init__(
        self,
        baseline_files: Mapping[str, str],
        candidate: RepositoryPatchCandidate,
        entry_module: str,
        function_name: str,
        hidden_cases: Sequence[Mapping[str, object]],
        signer,
        source_id: str,
        context_id: str,
        challenge_id: str,
        epoch: int,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.baseline_files = {str(path): str(source) for path, source in baseline_files.items()}
        self.candidate = candidate
        self.entry_module = str(entry_module)
        self.function_name = str(function_name)
        self.hidden_cases = tuple(dict(case) for case in hidden_cases)
        self.signer = signer
        self.source_id = str(source_id)
        self.context_id = str(context_id)
        self.challenge_id = str(challenge_id)
        self.epoch = int(epoch)
        self.timeout_seconds = max(0.5, float(timeout_seconds))

    def _files_for_arm(self, arm: str) -> Dict[str, str]:
        files = dict(self.baseline_files)
        if str(arm).upper() == "HIGH":
            files[self.candidate.file_path] = self.candidate.patched_source
        return files

    def _run(self, files: Mapping[str, str]) -> float:
        payload = {
            "files": dict(files),
            "entry_module": self.entry_module,
            "function_name": self.function_name,
            "cases": self.hidden_cases,
        }
        try:
            completed = subprocess.run(
                [sys.executable, "-c", self._HARNESS],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                return 0.0
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            return 1.0 if bool(result.get("ok")) else 0.0
        except Exception:
            return 0.0

    def execute(self, proposal: InterventionProposal, arm: str, value: float) -> WorldOutcomeReceipt:
        outcome = self._run(self._files_for_arm(arm))
        return self.signer.sign(WorldOutcomeReceipt(
            receipt_id=f"{self.challenge_id}::{proposal.experiment_id}::{arm}",
            experiment_id=proposal.experiment_id,
            axis_id=proposal.axis_id,
            arm=str(arm).upper(),
            intervention_value=float(value),
            outcome=float(outcome),
            source_id=self.source_id,
            context_id=self.context_id,
            challenge_id=self.challenge_id,
            epoch=self.epoch,
            budget_token=f"repository-hidden-tests::{self.challenge_id}",
            externally_generated=True,
        ))


class RepositoryTaskAcquisitionOrgan:
    """Stateless multi-file localization organ backed only by canonical BODY evidence."""

    def __init__(self, body, generator: Optional[PythonRepositoryRepairGenerator] = None) -> None:
        self.body = body
        self.generator = generator or PythonRepositoryRepairGenerator()

    def propose(self, task_id: str, files: Mapping[str, str]) -> Tuple[RepositoryPatchCandidate, ...]:
        candidates = self.generator.generate(task_id, files)
        for candidate in candidates:
            self.body.memory.remember_experiment(candidate.proposal)
        return candidates

    def policy(self) -> RepositoryRepairPolicy:
        return derive_repository_repair_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )

    def select(
        self,
        candidates: Sequence[RepositoryPatchCandidate],
        max_candidates: Optional[int] = None,
        apply_learned_policy: bool = True,
    ) -> RepositoryRepairSelection:
        policy = self.policy() if apply_learned_policy else None
        return select_repository_patch_candidates(candidates, policy, max_candidates=max_candidates)
