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


PATCH_SET_SIGNATURE_MARKER = "repository_patch_set_signature="
PATCH_SET_CARDINALITY_MARKER = "repository_patch_set_cardinality="


@dataclass(frozen=True)
class PatchSetFailureAssessment:
    cardinality: int
    status: str
    complete_contexts: Tuple[str, ...]
    falsified_contexts: Tuple[str, ...]
    supported_contexts: Tuple[str, ...]
    missing_experiment_ids: Tuple[str, ...]
    evaluated_candidate_count: int
    reason: str


@dataclass(frozen=True)
class RepositoryPatchSetCandidate:
    task_id: str
    repository_hash: str
    cardinality: int
    members: Tuple[RepositoryPatchCandidate, ...]
    signature: str
    proposal: InterventionProposal


@dataclass(frozen=True)
class RepositoryPatchSetPolicy:
    status: str
    cardinality: Optional[int]
    signature: Optional[str]
    supporting_contexts: Tuple[str, ...]
    candidate_signature_count: int
    reason: str


@dataclass(frozen=True)
class RepositoryPatchSetSelection:
    status: str
    candidates: Tuple[RepositoryPatchSetCandidate, ...]
    policy_cardinality: Optional[int]
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


def canonical_patch_set_signature(members: Sequence[RepositoryPatchCandidate]) -> str:
    return "|".join(sorted(f"{member.file_role}@{member.operator_id}" for member in members))


class RepositoryPatchSetGenerator:
    """Generate exact distinct-file patch sets of a requested cardinality.

    Candidate *content* depends only on the source-derived single-edit universe.
    Outcomes may authorize a cardinality frontier, but never choose the members of
    generated sets.
    """

    def generate(
        self,
        task_id: str,
        files: Mapping[str, str],
        single_candidates: Sequence[RepositoryPatchCandidate],
        cardinality: int,
        frontier_open: bool,
    ) -> Tuple[RepositoryPatchSetCandidate, ...]:
        requested = max(1, int(cardinality))
        if not frontier_open:
            return ()
        repo_hash = repository_hash(files)
        generated: List[RepositoryPatchSetCandidate] = []
        for raw_members in combinations(tuple(single_candidates), requested):
            if len({member.file_path for member in raw_members}) != requested:
                continue
            members = tuple(sorted(
                raw_members,
                key=lambda item: (item.file_path, item.site_index, item.operator_id),
            ))
            signature = canonical_patch_set_signature(members)
            payload = {
                "task_id": str(task_id),
                "repository_hash": repo_hash,
                "cardinality": requested,
                "member_experiment_ids": [member.proposal.experiment_id for member in members],
                "signature": signature,
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:20]
            proposal = InterventionProposal(
                experiment_id=f"REPOSITORY_PATCH_SET::{requested}::{repo_hash[:12]}::{digest}",
                axis_id=f"AXIS::REPOSITORY_PATCH_SET::{requested}::{repo_hash[:16]}",
                manipulated_variable=signature,
                held_fixed=(),
                low_value=0.0,
                high_value=1.0,
                predicted_low_side="BUGGY_REPOSITORY",
                predicted_high_side="PATCH_SET_REPOSITORY",
                reason=(
                    "execute exact coordinated repository patch set; "
                    f"{PATCH_SET_CARDINALITY_MARKER}{requested} "
                    f"{PATCH_SET_SIGNATURE_MARKER}{signature} repository_hash={repo_hash}"
                ),
                status="PROPOSAL_ONLY",
            )
            generated.append(RepositoryPatchSetCandidate(
                task_id=str(task_id),
                repository_hash=repo_hash,
                cardinality=requested,
                members=members,
                signature=signature,
                proposal=proposal,
            ))
        return tuple(generated)


def _parse_marker(reason: str, marker: str) -> Optional[str]:
    if marker not in reason:
        return None
    return reason.split(marker, 1)[1].strip().split()[0].rstrip(",;)") or None


def parse_patch_set_identity(proposal: InterventionProposal) -> Tuple[Optional[int], Optional[str]]:
    reason = str(proposal.reason)
    cardinality_text = _parse_marker(reason, PATCH_SET_CARDINALITY_MARKER)
    signature = _parse_marker(reason, PATCH_SET_SIGNATURE_MARKER)
    if cardinality_text is None or signature is None:
        return None, None
    try:
        cardinality = int(cardinality_text)
    except ValueError:
        return None, None
    return cardinality, signature


def assess_complete_patch_set_failure(
    candidates_by_context: Mapping[str, Sequence[RepositoryPatchSetCandidate]],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> PatchSetFailureAssessment:
    cardinalities = {
        candidate.cardinality
        for candidates in candidates_by_context.values()
        for candidate in candidates
    }
    if len(cardinalities) != 1:
        raise ValueError("failure assessment requires exactly one patch-set cardinality")
    cardinality = next(iter(cardinalities))
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
        status = "PATCH_SET_CARDINALITY_HAS_SUPPORTED_REPAIR"
        reason = "at least one completely evaluated context retains a strong repair at this cardinality"
    elif len(falsified) >= max(1, int(min_contexts)):
        status = "PATCH_SET_SPACE_FALSIFIED_OPEN_NEXT_CARDINALITY"
        reason = "every patch set at this cardinality was independently executed and weak in repeated contexts"
    else:
        status = "INSUFFICIENT_COMPLETE_PATCH_SET_FAILURE"
        reason = "higher cardinality requires repeated complete lower-cardinality failure; absence is not refutation"
    return PatchSetFailureAssessment(
        cardinality=cardinality,
        status=status,
        complete_contexts=tuple(sorted(complete)),
        falsified_contexts=tuple(sorted(falsified)),
        supported_contexts=tuple(sorted(supported)),
        missing_experiment_ids=tuple(sorted(set(missing))),
        evaluated_candidate_count=evaluated,
        reason=reason,
    )


def derive_repository_patch_set_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> RepositoryPatchSetPolicy:
    identity_by_experiment = {}
    for proposal in proposals:
        identity = parse_patch_set_identity(proposal)
        if identity[0] is not None and identity[1] is not None:
            identity_by_experiment[proposal.experiment_id] = identity

    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair) or pair.experiment_id not in identity_by_experiment:
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(
            pair.independence_class_id, pair
        )

    minimum_classes = max(1, int(min_independent_classes))
    support: Dict[Tuple[int, str], Dict[str, float]] = {}
    for (experiment_id, context_id), classes in grouped.items():
        if len(classes) < minimum_classes:
            continue
        score = sum(abs(pair.effect) for pair in classes.values()) / len(classes)
        if score < float(strong_effect_threshold):
            continue
        cardinality, signature = identity_by_experiment[experiment_id]
        support.setdefault((int(cardinality), str(signature)), {})[context_id] = float(score)

    required = max(1, int(min_contexts))
    eligible = []
    for (cardinality, signature), contexts in support.items():
        if len(contexts) < required:
            continue
        mean_score = sum(contexts.values()) / len(contexts)
        # Minimal causally sufficient repair cardinality wins before larger sets.
        eligible.append((cardinality, -len(contexts), -mean_score, signature, tuple(sorted(contexts))))
    eligible.sort()
    signature_space = set(identity_by_experiment.values())
    if not eligible:
        return RepositoryPatchSetPolicy(
            status="NO_REPRODUCED_PATCH_SET_POLICY",
            cardinality=None,
            signature=None,
            supporting_contexts=(),
            candidate_signature_count=len(signature_space),
            reason="no patch-set cardinality and structural signature has repeated authenticated success",
        )
    chosen = eligible[0]
    return RepositoryPatchSetPolicy(
        status="REPRODUCED_MINIMAL_PATCH_SET_POLICY",
        cardinality=int(chosen[0]),
        signature=str(chosen[3]),
        supporting_contexts=chosen[4],
        candidate_signature_count=len(signature_space),
        reason="smallest repeatedly successful authenticated patch-set cardinality retained",
    )


def select_repository_patch_sets(
    candidates: Sequence[RepositoryPatchSetCandidate],
    policy: Optional[RepositoryPatchSetPolicy],
    max_candidates: Optional[int] = None,
) -> RepositoryPatchSetSelection:
    ordered = tuple(candidates)
    policy_cardinality = None
    policy_signature = None
    status = "FULL_PATCH_SET_SEARCH"
    reason = "no learned patch-set cardinality/signature applied"
    if (
        policy is not None
        and policy.status == "REPRODUCED_MINIMAL_PATCH_SET_POLICY"
        and policy.cardinality is not None
        and policy.signature
    ):
        policy_cardinality = int(policy.cardinality)
        policy_signature = str(policy.signature)
        matching = tuple(
            candidate for candidate in ordered
            if candidate.cardinality == policy_cardinality and candidate.signature == policy_signature
        )
        nonmatching = tuple(candidate for candidate in ordered if candidate not in matching)
        ordered = matching + nonmatching
        status = "LEARNED_MINIMAL_PATCH_SET_PRIORITIZED"
        reason = "reproduced minimal patch cardinality and structural signature prioritized"
    if max_candidates is not None:
        ordered = ordered[: max(0, int(max_candidates))]
    return RepositoryPatchSetSelection(
        status=status,
        candidates=ordered,
        policy_cardinality=policy_cardinality,
        policy_signature=policy_signature,
        total_candidate_count=len(candidates),
        reason=reason,
    )


class SubprocessRepositoryPatchSetExecutor:
    """Apply an exact N-file patch set and run hidden tests in a fresh repository process."""

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
        candidate: RepositoryPatchSetCandidate,
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
            budget_token=f"repository-patch-set-hidden-tests::{self.challenge_id}",
            externally_generated=True,
        ))


class RepositoryPatchCardinalityOrgan:
    """Stateless developmental organ for minimal coordinated repair cardinality."""

    def __init__(self, body, generator: Optional[RepositoryPatchSetGenerator] = None) -> None:
        self.body = body
        self.generator = generator or RepositoryPatchSetGenerator()

    def policy(self) -> RepositoryPatchSetPolicy:
        return derive_repository_patch_set_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )

    def assess(self, candidates_by_context) -> PatchSetFailureAssessment:
        return assess_complete_patch_set_failure(
            candidates_by_context=candidates_by_context,
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )

    @staticmethod
    def _prerequisites_close_gap(cardinality: int, assessments: Sequence[PatchSetFailureAssessment]) -> bool:
        if cardinality <= 1:
            return True
        by_cardinality = {assessment.cardinality: assessment for assessment in assessments}
        for lower in range(1, cardinality):
            assessment = by_cardinality.get(lower)
            if assessment is None or assessment.status != "PATCH_SET_SPACE_FALSIFIED_OPEN_NEXT_CARDINALITY":
                return False
            if assessment.missing_experiment_ids:
                return False
        return True

    def propose(
        self,
        task_id: str,
        files: Mapping[str, str],
        single_candidates: Sequence[RepositoryPatchCandidate],
        cardinality: int,
        prerequisite_assessments: Sequence[PatchSetFailureAssessment] = (),
    ) -> Tuple[RepositoryPatchSetCandidate, ...]:
        requested = max(1, int(cardinality))
        learned = self.policy()
        inherited_authority = bool(
            learned.status == "REPRODUCED_MINIMAL_PATCH_SET_POLICY"
            and learned.cardinality is not None
            and int(learned.cardinality) == requested
        )
        frontier_open = inherited_authority or self._prerequisites_close_gap(
            requested, prerequisite_assessments
        )
        candidates = self.generator.generate(
            task_id=task_id,
            files=files,
            single_candidates=single_candidates,
            cardinality=requested,
            frontier_open=frontier_open,
        )
        for candidate in candidates:
            self.body.memory.remember_experiment(candidate.proposal)
        return candidates

    def select(
        self,
        candidates: Sequence[RepositoryPatchSetCandidate],
        max_candidates: Optional[int] = None,
        apply_learned_policy: bool = True,
    ) -> RepositoryPatchSetSelection:
        policy = self.policy() if apply_learned_policy else None
        return select_repository_patch_sets(candidates, policy, max_candidates=max_candidates)
