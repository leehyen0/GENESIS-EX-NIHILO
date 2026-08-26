from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple
import hashlib
import json
import re


_HASH64 = re.compile(r"^[0-9a-f]{64}$")
_HASH40 = re.compile(r"^[0-9a-f]{40}$")

STAGE_ORDER: Tuple[str, ...] = (
    "FROZEN",
    "TASK_ACQUIRED",
    "WORLD_PINNED",
    "BASELINE_RECORDED",
    "CANDIDATE_FROZEN",
    "HIDDEN_EVALUATED",
    "CREDIT_RECORDED",
)
_STAGE_INDEX = {stage: index for index, stage in enumerate(STAGE_ORDER)}


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _valid_sha256(value: str) -> bool:
    return bool(_HASH64.fullmatch(str(value)))


@dataclass(frozen=True)
class GitHubSourceBinding:
    """Immutable GitHub provenance consumed by the BODY.

    GitHub is a source/evidence surface, never evaluator, action, promotion or claim
    authority merely because a commit/blob exists.
    """

    repository: str
    ref: str
    commit_sha: str
    path: str
    blob_sha: str
    role: str

    def validate(self) -> Tuple[str, ...]:
        errors = []
        if "/" not in self.repository or self.repository.startswith("/") or self.repository.endswith("/"):
            errors.append("github_repository_invalid")
        if not self.ref:
            errors.append("github_ref_missing")
        if not _HASH40.fullmatch(self.commit_sha):
            errors.append("github_commit_sha_invalid")
        if not self.path or self.path.startswith("/"):
            errors.append("github_path_invalid")
        if not _HASH40.fullmatch(self.blob_sha):
            errors.append("github_blob_sha_invalid")
        if not self.role:
            errors.append("github_role_missing")
        return tuple(errors)

    def to_dict(self) -> Dict[str, str]:
        return {
            "repository": self.repository,
            "ref": self.ref,
            "commit_sha": self.commit_sha,
            "path": self.path,
            "blob_sha": self.blob_sha,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GitHubSourceBinding":
        binding = cls(
            repository=str(payload.get("repository", "")),
            ref=str(payload.get("ref", "")),
            commit_sha=str(payload.get("commit_sha", "")),
            path=str(payload.get("path", "")),
            blob_sha=str(payload.get("blob_sha", "")),
            role=str(payload.get("role", "")),
        )
        errors = binding.validate()
        if errors:
            raise ValueError("invalid GitHub source binding: " + ",".join(errors))
        return binding


@dataclass(frozen=True)
class CausalExperimentGermline:
    """Monotonic, atomic identity for one prospective causal experiment.

    The initial FROZEN state contains only information legally available before a fresh
    task is exposed. Task/world/baseline/candidate/outcome/credit identities are bound
    exactly when their causal stage is reached. Once bound, a descendant may not change
    them. This preserves zero-shot chronology while preventing pointer-by-pointer mixing
    across morphology/checkpoint epochs.

    The germline stores identity/provenance, not authority. External verifier/world
    authority must be re-established after every restore.
    """

    experiment_id: str
    benchmark_family: str
    operator_sha256: str
    evaluator_sha256: str
    source_receipt_sha256: str
    freeze_sha256: str
    stage: str = "FROZEN"
    task_ref: str = ""
    task_sha256: str = ""
    world_sha256: str = ""
    baseline_receipt_sha256: str = ""
    candidate_sha256: str = ""
    outcome_receipt_sha256: str = ""
    credit_receipt_sha256: str = ""
    github_sources: Tuple[GitHubSourceBinding, ...] = ()
    parent_germline_sha256: str = ""
    authority_reverification_required: bool = True

    def validate(self) -> Tuple[str, ...]:
        errors = []
        if not self.experiment_id:
            errors.append("experiment_id_missing")
        if not self.benchmark_family:
            errors.append("benchmark_family_missing")
        for name, value in (
            ("operator_sha256", self.operator_sha256),
            ("evaluator_sha256", self.evaluator_sha256),
            ("source_receipt_sha256", self.source_receipt_sha256),
            ("freeze_sha256", self.freeze_sha256),
        ):
            if not _valid_sha256(value):
                errors.append(f"{name}_invalid")
        if self.parent_germline_sha256 and not _valid_sha256(self.parent_germline_sha256):
            errors.append("parent_germline_sha256_invalid")
        if self.stage not in _STAGE_INDEX:
            errors.append("stage_invalid")
            return tuple(sorted(set(errors)))
        if not self.authority_reverification_required:
            errors.append("serialized_authority_forbidden")

        stage = _STAGE_INDEX[self.stage]
        staged_fields = (
            ("TASK_ACQUIRED", "task_sha256", self.task_sha256),
            ("WORLD_PINNED", "world_sha256", self.world_sha256),
            ("BASELINE_RECORDED", "baseline_receipt_sha256", self.baseline_receipt_sha256),
            ("CANDIDATE_FROZEN", "candidate_sha256", self.candidate_sha256),
            ("HIDDEN_EVALUATED", "outcome_receipt_sha256", self.outcome_receipt_sha256),
            ("CREDIT_RECORDED", "credit_receipt_sha256", self.credit_receipt_sha256),
        )
        for required_stage, name, value in staged_fields:
            boundary = _STAGE_INDEX[required_stage]
            if stage >= boundary and not _valid_sha256(value):
                errors.append(f"{name}_missing_or_invalid")
            if stage < boundary and value:
                errors.append(f"{name}_bound_before_{required_stage.lower()}")

        if stage >= _STAGE_INDEX["TASK_ACQUIRED"] and not self.task_ref:
            errors.append("task_ref_missing")
        if stage < _STAGE_INDEX["TASK_ACQUIRED"] and self.task_ref:
            errors.append("task_ref_bound_before_task_acquired")

        source_keys = set()
        for source in self.github_sources:
            errors.extend(source.validate())
            key = (source.repository, source.commit_sha, source.path, source.blob_sha, source.role)
            if key in source_keys:
                errors.append("duplicate_github_source_binding")
            source_keys.add(key)
        return tuple(sorted(set(errors)))

    def pretask_identity_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "benchmark_family": self.benchmark_family,
            "operator_sha256": self.operator_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "source_receipt_sha256": self.source_receipt_sha256,
            "freeze_sha256": self.freeze_sha256,
            "github_sources": [source.to_dict() for source in self.github_sources],
            "authority_reverification_required": True,
        }

    def binding_dict(self) -> Dict[str, str]:
        return {
            "task_ref": self.task_ref,
            "task_sha256": self.task_sha256,
            "world_sha256": self.world_sha256,
            "baseline_receipt_sha256": self.baseline_receipt_sha256,
            "candidate_sha256": self.candidate_sha256,
            "outcome_receipt_sha256": self.outcome_receipt_sha256,
            "credit_receipt_sha256": self.credit_receipt_sha256,
        }

    def fingerprint(self) -> str:
        payload: Dict[str, Any] = self.pretask_identity_dict()
        payload.update(self.binding_dict())
        payload["stage"] = self.stage
        payload["parent_germline_sha256"] = self.parent_germline_sha256
        return _sha256(payload)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = self.pretask_identity_dict()
        payload.update(self.binding_dict())
        payload.update(
            {
                "stage": self.stage,
                "parent_germline_sha256": self.parent_germline_sha256,
                "germline_sha256": self.fingerprint(),
            }
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CausalExperimentGermline":
        germline = cls(
            experiment_id=str(payload.get("experiment_id", "")),
            benchmark_family=str(payload.get("benchmark_family", "")),
            operator_sha256=str(payload.get("operator_sha256", "")),
            evaluator_sha256=str(payload.get("evaluator_sha256", "")),
            source_receipt_sha256=str(payload.get("source_receipt_sha256", "")),
            freeze_sha256=str(payload.get("freeze_sha256", "")),
            stage=str(payload.get("stage", "FROZEN")),
            task_ref=str(payload.get("task_ref", "")),
            task_sha256=str(payload.get("task_sha256", "")),
            world_sha256=str(payload.get("world_sha256", "")),
            baseline_receipt_sha256=str(payload.get("baseline_receipt_sha256", "")),
            candidate_sha256=str(payload.get("candidate_sha256", "")),
            outcome_receipt_sha256=str(payload.get("outcome_receipt_sha256", "")),
            credit_receipt_sha256=str(payload.get("credit_receipt_sha256", "")),
            github_sources=tuple(
                GitHubSourceBinding.from_dict(dict(row))
                for row in payload.get("github_sources", ())
            ),
            parent_germline_sha256=str(payload.get("parent_germline_sha256", "")),
            authority_reverification_required=bool(
                payload.get("authority_reverification_required", True)
            ),
        )
        errors = germline.validate()
        if errors:
            raise ValueError("invalid causal experiment germline: " + ",".join(errors))
        expected = str(payload.get("germline_sha256", ""))
        if expected and expected != germline.fingerprint():
            raise ValueError("causal experiment germline fingerprint mismatch")
        return germline

    def advance(
        self,
        next_stage: str,
        *,
        task_ref: str = "",
        task_sha256: str = "",
        world_sha256: str = "",
        baseline_receipt_sha256: str = "",
        candidate_sha256: str = "",
        outcome_receipt_sha256: str = "",
        credit_receipt_sha256: str = "",
    ) -> "CausalExperimentGermline":
        if self.stage not in _STAGE_INDEX or next_stage not in _STAGE_INDEX:
            raise ValueError("unknown causal experiment stage")
        if _STAGE_INDEX[next_stage] != _STAGE_INDEX[self.stage] + 1:
            raise ValueError(f"illegal causal experiment stage transition: {self.stage}->{next_stage}")

        supplied = {
            "task_ref": task_ref,
            "task_sha256": task_sha256,
            "world_sha256": world_sha256,
            "baseline_receipt_sha256": baseline_receipt_sha256,
            "candidate_sha256": candidate_sha256,
            "outcome_receipt_sha256": outcome_receipt_sha256,
            "credit_receipt_sha256": credit_receipt_sha256,
        }
        opened_at = {
            "TASK_ACQUIRED": {"task_ref", "task_sha256"},
            "WORLD_PINNED": {"world_sha256"},
            "BASELINE_RECORDED": {"baseline_receipt_sha256"},
            "CANDIDATE_FROZEN": {"candidate_sha256"},
            "HIDDEN_EVALUATED": {"outcome_receipt_sha256"},
            "CREDIT_RECORDED": {"credit_receipt_sha256"},
        }[next_stage]
        unexpected = [name for name, value in supplied.items() if value and name not in opened_at]
        if unexpected:
            raise ValueError("binding supplied at wrong causal stage: " + ",".join(sorted(unexpected)))

        current = self.binding_dict()
        for name in opened_at:
            if not supplied[name]:
                raise ValueError(f"missing binding for {next_stage}: {name}")
            current[name] = supplied[name]

        child = CausalExperimentGermline(
            experiment_id=self.experiment_id,
            benchmark_family=self.benchmark_family,
            operator_sha256=self.operator_sha256,
            evaluator_sha256=self.evaluator_sha256,
            source_receipt_sha256=self.source_receipt_sha256,
            freeze_sha256=self.freeze_sha256,
            stage=next_stage,
            task_ref=current["task_ref"],
            task_sha256=current["task_sha256"],
            world_sha256=current["world_sha256"],
            baseline_receipt_sha256=current["baseline_receipt_sha256"],
            candidate_sha256=current["candidate_sha256"],
            outcome_receipt_sha256=current["outcome_receipt_sha256"],
            credit_receipt_sha256=current["credit_receipt_sha256"],
            github_sources=self.github_sources,
            parent_germline_sha256=self.fingerprint(),
            authority_reverification_required=True,
        )
        errors = child.validate()
        if errors:
            raise ValueError("invalid causal experiment stage binding: " + ",".join(errors))
        return child


@dataclass(frozen=True)
class GermlineVerification:
    passed: bool
    status: str
    errors: Tuple[str, ...]


def verify_descendant_germline(
    parent: CausalExperimentGermline,
    child: CausalExperimentGermline,
) -> GermlineVerification:
    """Fail closed unless the descendant preserves old bindings and adds only the next one."""

    errors = list(parent.validate()) + list(child.validate())
    if parent.pretask_identity_dict() != child.pretask_identity_dict():
        errors.append("pretask_experiment_identity_changed")
    if child.parent_germline_sha256 != parent.fingerprint():
        errors.append("parent_germline_binding_mismatch")
    if parent.stage not in _STAGE_INDEX or child.stage not in _STAGE_INDEX:
        errors.append("stage_invalid")
    elif _STAGE_INDEX[child.stage] != _STAGE_INDEX[parent.stage] + 1:
        errors.append("nonmonotonic_or_skipped_stage_transition")

    parent_bindings = parent.binding_dict()
    child_bindings = child.binding_dict()
    for name, old_value in parent_bindings.items():
        if old_value and child_bindings[name] != old_value:
            errors.append(f"inherited_binding_changed::{name}")

    errors = sorted(set(errors))
    return GermlineVerification(
        passed=not errors,
        status=(
            "PASS_MONOTONIC_ATOMIC_CAUSAL_EXPERIMENT_HEREDITY"
            if not errors
            else "FAIL_CLOSED_CAUSAL_EXPERIMENT_HEREDITY"
        ),
        errors=tuple(errors),
    )


def github_source_set_sha256(sources: Sequence[GitHubSourceBinding]) -> str:
    """Stable provenance digest. A matching digest still grants no authority."""

    rows = [source.to_dict() for source in sources]
    rows.sort(key=lambda row: (row["repository"], row["commit_sha"], row["path"], row["role"]))
    return _sha256({"github_sources": rows})
