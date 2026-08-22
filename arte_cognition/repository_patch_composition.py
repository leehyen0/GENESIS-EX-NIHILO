from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
import subprocess
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .repository_task_acquisition import RepositoryPatchCandidate, repository_hash
from .world_coupling import WorldOutcomePair, WorldOutcomeReceipt


PAIR_SIGNATURE_MARKER = "repository_patch_pair_signature="


@dataclass(frozen=True)
class SingleEditFailureAssessment:
    status: str
    complete_contexts: Tuple[str, ...]
    falsified_contexts: Tuple[str, ...]
    supported_contexts: Tuple[str, ...]
    missing_experiment_ids: Tuple[str, ...]
    evaluated_candidate_count: int
    reason: str


@dataclass(frozen=True)
class RepositoryPatchPairCandidate:
    task_id: str
    repository_hash: str
    members: Tuple[RepositoryPatchCandidate, RepositoryPatchCandidate]
    signature: str
    proposal: InterventionProposal


@dataclass(frozen=True)
class RepositoryPatchPairPolicy:
    status: str
    signature: Optional[str]
    supporting_contexts: Tuple[str, ...]
    candidate_signature_count: int
    reason: str


@dataclass(frozen=True)
class RepositoryPatchPairSelection:
    status: str
    candidates: Tuple[RepositoryPatchPairCandidate, ...]
    policy_signature: Optional[str]
    total_candidate_count: int
    reason: str


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def assess_complete_single_edit_failure(
    candidates_by_context: Mapping[str, Sequence[RepositoryPatchCandidate]],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> SingleEditFailureAssessment:
    """Open pair composition only after complete repeated failure of every single edit."""
    minimum_classes = max(1, int(min_independent_classes))
    by_key: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair):
            continue
        by_key.setdefault((pair.context_id, pair.experiment_id), {}).setdefault(
            pair.independence_class_id, pair
        )

    complete: List[str] = []
    falsified: List[str] = []
    supported: List[str] = []
    missing: List[str] = []
    evaluated = 0
    for context_id, candidates in candidates_by_context.items():
        if not candidates:
            continue
        context_complete = True
        context_supported = False
        for candidate in candidates:
            classes = by_key.get((str(context_id), candidate.proposal.experiment_id), {})
            if len(classes) < minimum_classes:
                context_complete = False
                missing.append(candidate.proposal.experiment_id)
                continue
            evaluated += 1
            score = sum(abs(pair.effect) for pair in classes.values()) / len(classes)
            if score >= float(strong_effect_threshold):
                context_supported = True
        if context_complete:
            complete.append(str(context_id))
            if context_supported:
                supported.append(str(context_id))
            else:
                falsified.append(str(context_id))

    if supported:
        return SingleEditFailureAssessment(
            status="SINGLE_EDIT_REPAIR_RETAINS_SUPPORTED_PATCH",
            complete_contexts=tuple(sorted(complete)),
            falsified_contexts=tuple(sorted(falsified)),
            supported_contexts=tuple(sorted(supported)),
            missing_experiment_ids=tuple(sorted(set(missing))),
            evaluated_candidate_count=evaluated,
            reason="at least one completely evaluated repository admits a strong single-file repair",
        )
    if len(falsified) < max(1, int(min_contexts)):
        return SingleEditFailureAssessment(
            status="INSUFFICIENT_COMPLETE_SINGLE_EDIT_FAILURE",
            complete_contexts=tuple(sorted(complete)),
            falsified_contexts=tuple(sorted(falsified)),
            supported_contexts=(),
            missing_experiment_ids=tuple(sorted(set(missing))),
            evaluated_candidate_count=evaluated,
            reason="pair composition requires repeated complete single-edit failure; absence is not refutation",
        )
    return SingleEditFailureAssessment(
        status="SINGLE_EDIT_REPAIR_SPACE_FALSIFIED_OPEN_PAIR_COMPOSITION",
        complete_contexts=tuple(sorted(complete)),
        falsified_contexts=tuple(sorted(falsified)),
        supported_contexts=(),
        missing_experiment_ids=tuple(sorted(set(missing))),
        evaluated_candidate_count=evaluated,
        reason="all single-file candidates were independently executed and weak in repeated repositories",
    )


def canonical_pair_signature(
    members: Sequence[RepositoryPatchCandidate],
) -> str:
    parts = sorted(f"{member.file_role}@{member.operator_id}" for member in members)
    return "|".join(parts)


class RepositoryPatchPairGenerator:
    """Compose two distinct-file repair candidates without consuming hidden outcomes."""

    def generate(
        self,
        task_id: str,
        files: Mapping[str, str],
        single_candidates: Sequence[RepositoryPatchCandidate],
        frontier_open: bool,
    ) -> Tuple[RepositoryPatchPairCandidate, ...]:
        if not frontier_open:
            return ()
        repo_hash = repository_hash(files)
        candidates: List[RepositoryPatchPairCandidate] = []
        for first, second in combinations(tuple(single_candidates), 2):
            if first.file_path == second.file_path:
                continue
            members = tuple(sorted((first, second), key=lambda item: (item.file_path, item.site_index, item.operator_id)))
            signature = canonical_pair_signature(members)
            payload = {
                "task_id": str(task_id),
                "repository_hash": repo_hash,
                "member_experiment_ids": [member.proposal.experiment_id for member in members],
                "signature": signature,
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:20]
            proposal = InterventionProposal(
                experiment_id=f"REPOSITORY_PATCH_PAIR::{repo_hash[:12]}::{digest}",
                axis_id=f"AXIS::REPOSITORY_PATCH_PAIR::{repo_hash[:16]}",
                manipulated_variable=signature,
                held_fixed=(),
                low_value=0.0,
                high_value=1.0,
                predicted_low_side="BUGGY_REPOSITORY",
                predicted_high_side="COORDINATED_PATCHED_REPOSITORY",
                reason=(
                    "execute exact two-file coordinated repository patch; "
                    f"{PAIR_SIGNATURE_MARKER}{signature} repository_hash={repo_hash}"
                ),
                status="PROPOSAL_ONLY",
            )
            candidates.append(RepositoryPatchPairCandidate(
                task_id=str(task_id),
                repository_hash=repo_hash,
                members=(members[0], members[1]),
                signature=signature,
                proposal=proposal,
            ))
        return tuple(candidates)


def parse_pair_signature(proposal: InterventionProposal) -> Optional[str]:
    reason = str(proposal.reason)
    if PAIR_SIGNATURE_MARKER not in reason:
        return None
    return reason.split(PAIR_SIGNATURE_MARKER, 1)[1].strip().split()[0].rstrip(",;)") or None


def derive_repository_patch_pair_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> RepositoryPatchPairPolicy:
    signature_by_experiment = {
        proposal.experiment_id: parse_pair_signature(proposal)
        for proposal in proposals
        if parse_pair_signature(proposal) is not None
    }
    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair) or pair.experiment_id not in signature_by_experiment:
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(
            pair.independence_class_id, pair
        )

    minimum_classes = max(1, int(min_independent_classes))
    support: Dict[str, Dict[str, float]] = {}
    for (experiment_id, context_id), classes in grouped.items():
        if len(classes) < minimum_classes:
            continue
        score = sum(abs(pair.effect) for pair in classes.values()) / len(classes)
        if score < float(strong_effect_threshold):
            continue
        signature = signature_by_experiment[experiment_id]
        if signature is not None:
            support.setdefault(signature, {})[context_id] = float(score)

    eligible = []
    required = max(1, int(min_contexts))
    for signature, contexts in support.items():
        if len(contexts) < required:
            continue
        mean_score = sum(contexts.values()) / len(contexts)
        eligible.append((-len(contexts), -mean_score, signature, tuple(sorted(contexts))))
    eligible.sort()
    signature_space = set(value for value in signature_by_experiment.values() if value is not None)
    if not eligible:
        return RepositoryPatchPairPolicy(
            status="NO_REPRODUCED_COORDINATED_PATCH_SIGNATURE",
            signature=None,
            supporting_contexts=(),
            candidate_signature_count=len(signature_space),
            reason="no coordinated two-file repair signature has repeated authenticated repository success",
        )
    chosen = eligible[0]
    return RepositoryPatchPairPolicy(
        status="REPRODUCED_COORDINATED_PATCH_SIGNATURE",
        signature=chosen[2],
        supporting_contexts=chosen[3],
        candidate_signature_count=len(signature_space),
        reason="two-file structural repair signature reproduced across source-disjoint repositories",
    )


def select_repository_patch_pairs(
    candidates: Sequence[RepositoryPatchPairCandidate],
    policy: Optional[RepositoryPatchPairPolicy],
    max_candidates: Optional[int] = None,
) -> RepositoryPatchPairSelection:
    ordered = tuple(candidates)
    policy_signature = None
    status = "FULL_COORDINATED_PATCH_SEARCH"
    reason = "no learned coordinated patch signature applied"
    if policy is not None and policy.status == "REPRODUCED_COORDINATED_PATCH_SIGNATURE" and policy.signature:
        policy_signature = policy.signature
        matching = tuple(candidate for candidate in ordered if candidate.signature == policy.signature)
        nonmatching = tuple(candidate for candidate in ordered if candidate.signature != policy.signature)
        ordered = matching + nonmatching
        status = "LEARNED_COORDINATED_PATCH_SIGNATURE_PRIORITIZED"
        reason = "reproduced two-file structural repair signature prioritized on fresh repository"
    if max_candidates is not None:
        ordered = ordered[: max(0, int(max_candidates))]
    return RepositoryPatchPairSelection(
        status=status,
        candidates=ordered,
        policy_signature=policy_signature,
        total_candidate_count=len(candidates),
        reason=reason,
    )


class SubprocessRepositoryPatchPairExecutor:
    """Apply both exact member patches and run hidden tests in a fresh repository process."""

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
        candidate: RepositoryPatchPairCandidate,
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
            for member in self.candidate.members:
                files[member.file_path] = member.patched_source
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
            budget_token=f"repository-pair-hidden-tests::{self.challenge_id}",
            externally_generated=True,
        ))


class RepositoryPatchCompositionOrgan:
    """Stateless coordinated-patch organ deriving authority from canonical BODY evidence."""

    def __init__(self, body, generator: Optional[RepositoryPatchPairGenerator] = None) -> None:
        self.body = body
        self.generator = generator or RepositoryPatchPairGenerator()

    def policy(self) -> RepositoryPatchPairPolicy:
        return derive_repository_patch_pair_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )

    def assess_single_failure(self, candidates_by_context) -> SingleEditFailureAssessment:
        return assess_complete_single_edit_failure(
            candidates_by_context=candidates_by_context,
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )

    def propose(
        self,
        task_id: str,
        files: Mapping[str, str],
        single_candidates: Sequence[RepositoryPatchCandidate],
        training_single_candidates_by_context=None,
    ) -> Tuple[RepositoryPatchPairCandidate, ...]:
        learned = self.policy()
        frontier_open = bool(
            learned.status == "REPRODUCED_COORDINATED_PATCH_SIGNATURE" and learned.signature
        )
        if not frontier_open and training_single_candidates_by_context is not None:
            assessment = self.assess_single_failure(training_single_candidates_by_context)
            frontier_open = assessment.status == "SINGLE_EDIT_REPAIR_SPACE_FALSIFIED_OPEN_PAIR_COMPOSITION"
        candidates = self.generator.generate(task_id, files, single_candidates, frontier_open=frontier_open)
        for candidate in candidates:
            self.body.memory.remember_experiment(candidate.proposal)
        return candidates

    def select(
        self,
        candidates: Sequence[RepositoryPatchPairCandidate],
        max_candidates: Optional[int] = None,
        apply_learned_policy: bool = True,
    ) -> RepositoryPatchPairSelection:
        policy = self.policy() if apply_learned_policy else None
        return select_repository_patch_pairs(candidates, policy, max_candidates=max_candidates)
