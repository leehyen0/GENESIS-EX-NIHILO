from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import PurePosixPath
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .software_task_acquisition import PythonASTRepairGenerator
from .world_coupling import WorldOutcomePair, WorldOutcomeReceipt


REPOSITORY_LOCALIZATION_MARKER = "repository_localization_signature="
REPOSITORY_TARGET_MARKER = "repository_target_path="


@dataclass(frozen=True)
class RepositoryPatchCandidate:
    task_id: str
    repository_hash: str
    target_path: str
    location_signature: str
    operator_id: str
    patched_files: Tuple[Tuple[str, str], ...]
    proposal: InterventionProposal


@dataclass(frozen=True)
class RepositoryLocalizationPolicy:
    status: str
    location_signature: Optional[str]
    supporting_contexts: Tuple[str, ...]
    candidate_signature_count: int
    reason: str


@dataclass(frozen=True)
class RepositoryRepairSelection:
    status: str
    candidates: Tuple[RepositoryPatchCandidate, ...]
    policy_location_signature: Optional[str]
    total_candidate_count: int
    reason: str


def repository_hash(files: Mapping[str, str]) -> str:
    material = json.dumps(
        {str(path): str(files[path]) for path in sorted(files)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _repair_site_count(source: str) -> int:
    try:
        return len(PythonASTRepairGenerator._site_operator_ids(source))
    except Exception:
        return 0


def _module_name(path: str) -> str:
    return PurePosixPath(path).stem


def _local_import_modules(source: str) -> Tuple[str, ...]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and int(node.level or 0) >= 1 and node.module:
            out.append(str(node.module).split(".")[-1])
    return tuple(sorted(out))


def repository_location_signatures(files: Mapping[str, str]) -> Dict[str, str]:
    python_files = {
        str(path): str(source)
        for path, source in files.items()
        if str(path).endswith(".py") and PurePosixPath(str(path)).name != "__init__.py"
    }
    module_to_path = {_module_name(path): path for path in python_files}
    imported_by = {path: 0 for path in python_files}
    imports_count = {path: 0 for path in python_files}
    for path, source in python_files.items():
        local = _local_import_modules(source)
        imports_count[path] = len(local)
        for module in local:
            target = module_to_path.get(module)
            if target is not None:
                imported_by[target] += 1

    signatures: Dict[str, str] = {}
    for path, source in python_files.items():
        signatures[path] = (
            f"IB{min(imported_by[path], 2)}"
            f"|IM{min(imports_count[path], 2)}"
            f"|RS{min(_repair_site_count(source), 3)}"
        )
    return signatures


def generate_repository_patch_candidates(
    task_id: str,
    files: Mapping[str, str],
) -> Tuple[RepositoryPatchCandidate, ...]:
    repo_hash = repository_hash(files)
    signatures = repository_location_signatures(files)
    generator = PythonASTRepairGenerator()
    out = []
    for path in sorted(signatures):
        source = str(files[path])
        for local in generator.generate(f"{task_id}:{path}", source):
            patched = {str(p): str(s) for p, s in files.items()}
            patched[path] = local.patched_source
            payload = {
                "task_id": str(task_id),
                "repository_hash": repo_hash,
                "target_path": path,
                "location_signature": signatures[path],
                "operator_id": local.operator_id,
                "patched_source_hash": hashlib.sha256(local.patched_source.encode("utf-8")).hexdigest(),
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:20]
            proposal = InterventionProposal(
                experiment_id=f"REPOSITORY_PATCH::{repo_hash[:12]}::{digest}",
                axis_id=f"AXIS::REPOSITORY_REPAIR::{repo_hash[:16]}",
                manipulated_variable=f"{path}::{local.operator_id}",
                held_fixed=(),
                low_value=0.0,
                high_value=1.0,
                predicted_low_side="BUGGY_REPOSITORY",
                predicted_high_side="PATCHED_REPOSITORY",
                reason=(
                    "execute repository patch candidate under hidden CI; "
                    f"{REPOSITORY_LOCALIZATION_MARKER}{signatures[path]} "
                    f"{REPOSITORY_TARGET_MARKER}{path} "
                    f"software_repair_operator={local.operator_id} "
                    f"repository_hash={repo_hash}"
                ),
                status="PROPOSAL_ONLY",
            )
            out.append(RepositoryPatchCandidate(
                task_id=str(task_id),
                repository_hash=repo_hash,
                target_path=path,
                location_signature=signatures[path],
                operator_id=local.operator_id,
                patched_files=tuple(sorted(patched.items())),
                proposal=proposal,
            ))
    return tuple(out)


def parse_repository_localization_signature(proposal: InterventionProposal) -> Optional[str]:
    reason = str(proposal.reason)
    if REPOSITORY_LOCALIZATION_MARKER not in reason:
        return None
    tail = reason.split(REPOSITORY_LOCALIZATION_MARKER, 1)[1].strip().split()[0]
    return tail or None


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def derive_repository_localization_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> RepositoryLocalizationPolicy:
    signature_by_experiment = {}
    for proposal in proposals:
        signature = parse_repository_localization_signature(proposal)
        if signature is not None:
            signature_by_experiment[proposal.experiment_id] = signature
    signature_space = tuple(sorted(set(signature_by_experiment.values())))

    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair) or pair.experiment_id not in signature_by_experiment:
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(
            pair.independence_class_id, pair
        )

    support: Dict[str, Dict[str, float]] = {}
    minimum = max(1, int(min_independent_classes))
    for (experiment_id, context_id), by_class in grouped.items():
        if len(by_class) < minimum:
            continue
        score = sum(abs(item.effect) for item in by_class.values()) / len(by_class)
        if score < float(strong_effect_threshold):
            continue
        signature = signature_by_experiment[experiment_id]
        contexts = support.setdefault(signature, {})
        contexts[context_id] = max(contexts.get(context_id, 0.0), float(score))

    eligible = []
    required = max(1, int(min_contexts))
    for signature, contexts in support.items():
        if len(contexts) < required:
            continue
        mean_score = sum(contexts.values()) / len(contexts)
        eligible.append((-len(contexts), -mean_score, signature, tuple(sorted(contexts))))
    eligible.sort()
    if not eligible:
        return RepositoryLocalizationPolicy(
            status="NO_REPRODUCED_REPOSITORY_LOCALIZATION",
            location_signature=None,
            supporting_contexts=(),
            candidate_signature_count=len(signature_space),
            reason="no repository location signature has repeated authenticated patch success",
        )
    chosen = eligible[0]
    return RepositoryLocalizationPolicy(
        status="REPRODUCED_REPOSITORY_LOCALIZATION",
        location_signature=chosen[2],
        supporting_contexts=chosen[3],
        candidate_signature_count=len(signature_space),
        reason="source-disjoint repositories retained a structural localization signature",
    )


def select_repository_repairs(
    candidates: Sequence[RepositoryPatchCandidate],
    policy: Optional[RepositoryLocalizationPolicy],
    max_candidates: Optional[int] = None,
) -> RepositoryRepairSelection:
    ordered = tuple(sorted(candidates, key=lambda c: (c.target_path, c.operator_id, c.proposal.experiment_id)))
    policy_signature = None
    status = "FULL_REPOSITORY_PATCH_SEARCH"
    reason = "no learned repository localization applied"
    if (
        policy is not None
        and policy.status == "REPRODUCED_REPOSITORY_LOCALIZATION"
        and policy.location_signature
    ):
        policy_signature = policy.location_signature
        matching = tuple(c for c in ordered if c.location_signature == policy_signature)
        nonmatching = tuple(c for c in ordered if c.location_signature != policy_signature)
        ordered = matching + nonmatching
        status = "LEARNED_REPOSITORY_LOCALIZATION_PRIORITIZED"
        reason = "reproduced structural file signature prioritized before patch execution"
    if max_candidates is not None:
        ordered = ordered[: max(0, int(max_candidates))]
    return RepositoryRepairSelection(
        status=status,
        candidates=ordered,
        policy_location_signature=policy_signature,
        total_candidate_count=len(candidates),
        reason=reason,
    )


class SubprocessRepositoryRepairExecutor:
    """Execute a multi-file Python repository under hidden tests in a child process."""

    _HARNESS = r'''
import importlib, json, sys
payload = json.loads(sys.stdin.read())
sys.path.insert(0, payload["root"])
module = importlib.import_module(payload["module"])
fn = getattr(module, payload["function"])
ok = True
for case in payload["cases"]:
    try:
        result = fn(*case["args"])
    except Exception:
        ok = False
        break
    if result != case["expected"]:
        ok = False
        break
print(json.dumps({"ok": bool(ok)}))
'''

    def __init__(
        self,
        baseline_files: Mapping[str, str],
        patched_files: Mapping[str, str],
        module: str,
        function_name: str,
        hidden_cases: Sequence[Mapping[str, object]],
        signer,
        source_id: str,
        context_id: str,
        challenge_id: str,
        epoch: int,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.baseline_files = {str(k): str(v) for k, v in baseline_files.items()}
        self.patched_files = {str(k): str(v) for k, v in patched_files.items()}
        self.module = str(module)
        self.function_name = str(function_name)
        self.hidden_cases = tuple(dict(case) for case in hidden_cases)
        self.signer = signer
        self.source_id = str(source_id)
        self.context_id = str(context_id)
        self.challenge_id = str(challenge_id)
        self.epoch = int(epoch)
        self.timeout_seconds = max(0.5, float(timeout_seconds))

    def _run(self, files: Mapping[str, str]) -> float:
        with tempfile.TemporaryDirectory() as root:
            for path, source in files.items():
                target = os.path.join(root, *PurePosixPath(path).parts)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "w", encoding="utf-8") as handle:
                    handle.write(source)
            payload = {
                "root": root,
                "module": self.module,
                "function": self.function_name,
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
        files = self.baseline_files if str(arm).upper() == "LOW" else self.patched_files
        outcome = self._run(files)
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
            budget_token=f"repository-hidden-ci::{self.challenge_id}",
            externally_generated=True,
        ))


class RepositoryRepairOrgan:
    """Stateless repository localization/patch organ backed by canonical BODY evidence."""

    def __init__(self, body) -> None:
        self.body = body

    def propose(self, task_id: str, files: Mapping[str, str]) -> Tuple[RepositoryPatchCandidate, ...]:
        candidates = generate_repository_patch_candidates(task_id, files)
        for candidate in candidates:
            self.body.memory.remember_experiment(candidate.proposal)
        return candidates

    def policy(self) -> RepositoryLocalizationPolicy:
        return derive_repository_localization_policy(
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
        apply_learned_localization: bool = True,
    ) -> RepositoryRepairSelection:
        policy = self.policy() if apply_learned_localization else None
        return select_repository_repairs(candidates, policy, max_candidates=max_candidates)
