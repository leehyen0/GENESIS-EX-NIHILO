from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple
import hashlib
import json
import re

from .executable_morphology import MorphologyGenome, OrganKind
from .morphology_genesis import MorphologyCandidate


_NATIVE_REF = re.compile(
    r"^native-meta://(?P<kind>generator|mutator)/(?P<residual>[A-Za-z0-9_.:-]+)/(?P<suffix>[0-9a-f]{16})$"
)


@dataclass(frozen=True)
class NativeMetaPolicyProgram:
    policy_id: str
    target_kind: OrganKind
    residual_id: str
    source_ref: str
    source_suffix: str
    preferred_operation_family: str
    candidate_budget_bonus: int
    tie_break_seed: int
    current_outcomes_required: bool = False

    def fingerprint(self) -> str:
        payload = {
            "policy_id": self.policy_id,
            "target_kind": self.target_kind.value,
            "residual_id": self.residual_id,
            "source_ref": self.source_ref,
            "source_suffix": self.source_suffix,
            "preferred_operation_family": self.preferred_operation_family,
            "candidate_budget_bonus": self.candidate_budget_bonus,
            "tie_break_seed": self.tie_break_seed,
            "current_outcomes_required": self.current_outcomes_required,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class NativeMetaPolicyExecution:
    policy_fingerprint: str
    input_candidate_ids: Tuple[str, ...]
    selected_candidate_ids: Tuple[str, ...]
    effective_candidate_budget: int
    preferred_operation_family: str
    current_outcomes_consumed: bool = False

    def fingerprint(self) -> str:
        payload = {
            "policy_fingerprint": self.policy_fingerprint,
            "input_candidate_ids": list(self.input_candidate_ids),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "effective_candidate_budget": self.effective_candidate_budget,
            "preferred_operation_family": self.preferred_operation_family,
            "current_outcomes_consumed": self.current_outcomes_consumed,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def compile_native_meta_policy(
    implementation_ref: str,
    *,
    target_kind: OrganKind,
    expected_residual_id: Optional[str] = None,
) -> NativeMetaPolicyProgram:
    """Cold-compile one generated L3 implementation ref into typed search semantics.

    The compiler has no outcome input. Family and residual identity are checked before
    any behavior is authorized. Unknown/malformed refs fail closed.
    """
    ref = str(implementation_ref)
    match = _NATIVE_REF.fullmatch(ref)
    if match is None:
        raise ValueError("INVALID_NATIVE_META_POLICY_REF")

    ref_kind = OrganKind.GENERATOR if match.group("kind") == "generator" else OrganKind.MUTATOR
    if ref_kind != target_kind:
        raise ValueError("NATIVE_META_POLICY_KIND_MISMATCH")

    residual_id = match.group("residual")
    if expected_residual_id is not None and residual_id != str(expected_residual_id):
        raise ValueError("NATIVE_META_POLICY_RESIDUAL_MISMATCH")

    suffix = match.group("suffix")
    seed = int(suffix, 16)
    if ref_kind == OrganKind.GENERATOR:
        family = "CHANGE_GENERATOR_POLICY"
        # A generator-policy descendant can widen a bounded future candidate frontier.
        # The exact bounded bonus is inherited from its generated ref and is restart-stable.
        bonus = 1 + (seed % 3)
    else:
        family = "CHANGE_MUTATOR_POLICY"
        # Mutator semantics alter family priority rather than silently increasing budget.
        bonus = 0

    policy_id = "NATIVE_META_POLICY::" + hashlib.sha256(
        f"{ref_kind.value}|{residual_id}|{suffix}|{family}|{bonus}".encode("utf-8")
    ).hexdigest()[:24]
    return NativeMetaPolicyProgram(
        policy_id=policy_id,
        target_kind=ref_kind,
        residual_id=residual_id,
        source_ref=ref,
        source_suffix=suffix,
        preferred_operation_family=family,
        candidate_budget_bonus=bonus,
        tie_break_seed=seed,
        current_outcomes_required=False,
    )


def compile_genome_native_meta_policies(
    genome: MorphologyGenome,
    *,
    expected_residual_id: Optional[str] = None,
) -> Tuple[NativeMetaPolicyProgram, ...]:
    """Compile active generated policies directly from descendant OrganSpec refs."""
    programs = []
    for organ in sorted(genome.organs, key=lambda row: row.organ_id):
        if not organ.enabled or organ.kind not in {OrganKind.GENERATOR, OrganKind.MUTATOR}:
            continue
        if not str(organ.implementation_ref).startswith("native-meta://"):
            continue
        programs.append(
            compile_native_meta_policy(
                organ.implementation_ref,
                target_kind=organ.kind,
                expected_residual_id=expected_residual_id,
            )
        )
    return tuple(programs)


def _candidate_rank(program: NativeMetaPolicyProgram, candidate: MorphologyCandidate) -> tuple:
    preferred = 0 if candidate.operation_family == program.preferred_operation_family else 1
    digest = hashlib.sha256(
        f"{program.tie_break_seed}|{candidate.candidate_id}".encode("utf-8")
    ).hexdigest()
    if program.target_kind == OrganKind.MUTATOR:
        # A mutator policy gives the intended mutation family first priority.
        return (preferred, digest, candidate.candidate_id)
    # Generator policy expands frontier and uses a stable generated ordering within it.
    return (digest, preferred, candidate.candidate_id)


def execute_native_meta_policy(
    program: NativeMetaPolicyProgram,
    candidates: Sequence[MorphologyCandidate],
    *,
    parent_candidate_budget: int,
) -> NativeMetaPolicyExecution:
    """Execute generated policy semantics on a future candidate frontier.

    No outcome, score, benchmark result or hidden evaluator value is accepted by this
    function. Its behavior is therefore pre-outcome and reconstructible from the ref.
    """
    budget = max(1, int(parent_candidate_budget)) + max(0, int(program.candidate_budget_bonus))
    unique = {}
    for candidate in candidates:
        unique.setdefault(candidate.candidate_id, candidate)
    ordered = sorted(unique.values(), key=lambda row: _candidate_rank(program, row))
    selected = tuple(row.candidate_id for row in ordered[:budget])
    return NativeMetaPolicyExecution(
        policy_fingerprint=program.fingerprint(),
        input_candidate_ids=tuple(row.candidate_id for row in candidates),
        selected_candidate_ids=selected,
        effective_candidate_budget=budget,
        preferred_operation_family=program.preferred_operation_family,
        current_outcomes_consumed=False,
    )


def parent_candidate_selection(
    candidates: Sequence[MorphologyCandidate],
    *,
    candidate_budget: int,
) -> Tuple[str, ...]:
    """Frozen no-policy/REMOVE control for the same candidate frontier."""
    budget = max(1, int(candidate_budget))
    unique = {}
    for candidate in candidates:
        unique.setdefault(candidate.candidate_id, candidate)
    return tuple(row.candidate_id for row in sorted(unique.values(), key=lambda row: row.candidate_id)[:budget])
