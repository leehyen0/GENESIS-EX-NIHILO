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
    "FOSSILIZED",
)

_ALLOWED_STAGE_TRANSITIONS = {
    stage: {stage, STAGE_ORDER[index + 1]} if index + 1 < len(STAGE_ORDER) else {stage}
    for index, stage in enumerate(STAGE_ORDER)
}


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class GitHubSourceBinding:
    """Immutable GitHub provenance consumed by the BODY.

    GitHub is a source/evidence surface, not evaluator or promotion authority.
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
    """Atomic hereditary identity of one prospective causal experiment.

    Operator, task, world, evaluator, source receipt, freeze contract and GitHub
    provenance are inherited as one object so descendant epochs cannot silently mix
    pointers from different experiments. The germline preserves identity, not
    authority: external evaluator/world authority must be re-established after restore.
    """

    experiment_id: str
    benchmark_family: str
    operator_sha256: str
    task_ref: str
    task_sha256: str
    world_sha256: str
    evaluator_sha256: str
    source_receipt_sha256: str
    freeze_sha256: str
    stage: str = "FROZEN"
    github_sources: Tuple[GitHubSourceBinding, ...] = ()
    parent_germline_sha256: str = ""
    authority_reverification_required: bool = True

    def validate(self) -> Tuple[str, ...]:
        errors = []
        if not self.experiment_id:
            errors.append("experiment_id_missing")
        if not self.benchmark_family:
            errors.append("benchmark_family_missing")
        if not self.task_ref:
            errors.append("task_ref_missing")
        for name, value in (
            ("operator_sha256", self.operator_sha256),
            ("task_sha256", self.task_sha256),
            ("world_sha256", self.world_sha256),
            ("evaluator_sha256", self.evaluator_sha256),
            ("source_receipt_sha256", self.source_receipt_sha256),
            ("freeze_sha256", self.freeze_sha256),
        ):
            if not _HASH64.fullmatch(value):
                errors.append(f"{name}_invalid")
        if self.parent_germline_sha256 and not _HASH64.fullmatch(self.parent_germline_sha256):
            errors.append("parent_germline_sha256_invalid")
        if self.stage not in STAGE_ORDER:
            errors.append("stage_invalid")
        if not self.authority_reverification_required:
            errors.append("serialized_authority_forbidden")
        source_keys = set()
        for source in self.github_sources:
            errors.extend(source.validate())
            key = (source.repository, source.commit_sha, source.path, source.blob_sha, source.role)
            if key in source_keys:
                errors.append("duplicate_github_source_binding")
            source_keys.add(key)
        return tuple(sorted(set(errors)))

    def immutable_identity_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "benchmark_family": self.benchmark_family,
            "operator_sha256": self.operator_sha256,
            "task_ref": self.task_ref,
            "task_sha256": self.task_sha256,
            "world_sha256": self.world_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "source_receipt_sha256": self.source_receipt_sha256,
            "freeze_sha256": self.freeze_sha256,
            "github_sources": [source.to_dict() for source in self.github_sources],
            "authority_reverification_required": True,
        }

    def fingerprint(self) -> str:
        payload = self.immutable_identity_dict()
        payload["stage"] = self.stage
        payload["parent_germline_sha256"] = self.parent_germline_sha256
        return _sha256(payload)

    def to_dict(self) -> Dict[str, Any]:
        payload = self.immutable_identity_dict()
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
            task_ref=str(payload.get("task_ref", "")),
            task_sha256=str(payload.get("task_sha256", "")),
            world_sha256=str(payload.get("world_sha256", "")),
            evaluator_sha256=str(payload.get("evaluator_sha256", "")),
            source_receipt_sha256=str(payload.get("source_receipt_sha256", "")),
            freeze_sha256=str(payload.get("freeze_sha256", "")),
            stage=str(payload.get("stage", "FROZEN")),
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

    def advance(self, next_stage: str) -> "CausalExperimentGermline":
        if next_stage not in _ALLOWED_STAGE_TRANSITIONS.get(self.stage, set()):
            raise ValueError(f"illegal causal experiment stage transition: {self.stage}->{next_stage}")
        return CausalExperimentGermline(
            experiment_id=self.experiment_id,
            benchmark_family=self.benchmark_family,
            operator_sha256=self.operator_sha256,
            task_ref=self.task_ref,
            task_sha256=self.task_sha256,
            world_sha256=self.world_sha256,
            evaluator_sha256=self.evaluator_sha256,
            source_receipt_sha256=self.source_receipt_sha256,
            freeze_sha256=self.freeze_sha256,
            stage=next_stage,
            github_sources=self.github_sources,
            parent_germline_sha256=self.fingerprint(),
            authority_reverification_required=True,
        )


@dataclass(frozen=True)
class GermlineVerification:
    passed: bool
    status: str
    errors: Tuple[str, ...]


def verify_descendant_germline(
    parent: CausalExperimentGermline,
    child: CausalExperimentGermline,
) -> GermlineVerification:
    """Fail closed unless a descendant preserves the whole experiment atomically."""

    errors = list(parent.validate()) + list(child.validate())
    if parent.immutable_identity_dict() != child.immutable_identity_dict():
        errors.append("immutable_experiment_identity_changed")
    if child.parent_germline_sha256 != parent.fingerprint():
        errors.append("parent_germline_binding_mismatch")
    if child.stage not in _ALLOWED_STAGE_TRANSITIONS.get(parent.stage, set()):
        errors.append("nonmonotonic_or_skipped_stage_transition")
    errors = sorted(set(errors))
    return GermlineVerification(
        passed=not errors,
        status=(
            "PASS_ATOMIC_CAUSAL_EXPERIMENT_HEREDITY"
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
