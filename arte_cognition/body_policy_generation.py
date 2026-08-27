from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from .executable_morphology import MorphologyCompiler, MorphologyGenome, PressureVector
from .morphology_genesis import MorphologyCandidate, MorphologyResidual
from .native_recursive_research import NativeMetaMorphologyGenesisEngine
from .native_meta_policy_runtime import compile_native_meta_policy, execute_native_meta_policy


@dataclass(frozen=True)
class BodyCandidateGeneration:
    body_fingerprint: str
    compiled_runtime_fingerprint: str
    fresh_residual_ids: Tuple[str, ...]
    nominal_budget: int
    effective_budget: int
    policy_fingerprints: Tuple[str, ...]
    candidate_ids: Tuple[str, ...]
    operation_families: Tuple[str, ...]
    current_outcomes_consumed: bool = False


@dataclass(frozen=True)
class ContextualBodySelection:
    body_fingerprint: str
    compiled_runtime_fingerprint: str
    fresh_residual_id: str
    pressure_signature: Tuple[Tuple[str, float], ...]
    selected_candidate_id: str
    selected_operation_family: str
    policy_fingerprints: Tuple[str, ...]
    selected_candidate_budget: int
    raw_candidate_budget: int
    current_outcomes_consumed: bool = False


def _parent_order(candidates: Sequence[MorphologyCandidate]) -> Tuple[MorphologyCandidate, ...]:
    unique = {}
    for candidate in candidates:
        unique.setdefault(candidate.candidate_id, candidate)
    return tuple(sorted(unique.values(), key=lambda row: (row.operation_family, row.candidate_id)))


def _context_family(pressure: PressureVector) -> str:
    p = pressure.normalized()
    if p.human_dependency > p.novelty_pressure:
        return "CHANGE_MUTATOR_POLICY"
    if p.novelty_pressure > p.human_dependency:
        return "CHANGE_GENERATOR_POLICY"
    raise ValueError("AMBIGUOUS_CONTEXTUAL_POLICY_PRESSURE")


def generate_body_candidates(
    genome: MorphologyGenome,
    fresh_residuals: Sequence[MorphologyResidual],
    *,
    nominal_budget: int = 1,
    raw_candidate_budget: int = 256,
    expected_policy_origin_residual_id: str | None = None,
) -> BodyCandidateGeneration:
    """Generate future candidates while consuming inherited compiled L3 policy.

    `expected_policy_origin_residual_id` authenticates where the inherited policy came
    from. `fresh_residuals` are the new problems on which that policy is used. No
    outcome or evaluator result is accepted by this function.
    """
    if not fresh_residuals:
        raise ValueError("BODY_POLICY_GENERATION_REQUIRES_FRESH_RESIDUAL")
    nominal_budget = max(1, int(nominal_budget))
    raw_candidate_budget = max(nominal_budget, int(raw_candidate_budget))

    runtime = MorphologyCompiler.compile_runtime(
        genome,
        expected_residual_id=expected_policy_origin_residual_id,
    )
    raw = NativeMetaMorphologyGenesisEngine(candidate_budget=raw_candidate_budget).generate(
        genome,
        fresh_residuals,
    )
    ordered = list(_parent_order(raw))
    selected = ordered[:nominal_budget]
    effective_budget = nominal_budget
    fingerprints = []

    for binding in runtime.native_meta_policies:
        program = compile_native_meta_policy(
            binding.implementation_ref,
            target_kind=binding.target_kind,
            expected_residual_id=expected_policy_origin_residual_id,
        )
        execution = execute_native_meta_policy(
            program,
            ordered,
            parent_candidate_budget=effective_budget,
        )
        by_id = {row.candidate_id: row for row in ordered}
        selected = [by_id[candidate_id] for candidate_id in execution.selected_candidate_ids]
        effective_budget = execution.effective_candidate_budget
        fingerprints.append(program.fingerprint())

    return BodyCandidateGeneration(
        body_fingerprint=genome.fingerprint(),
        compiled_runtime_fingerprint=runtime.fingerprint(),
        fresh_residual_ids=tuple(row.residual_id for row in fresh_residuals),
        nominal_budget=nominal_budget,
        effective_budget=effective_budget,
        policy_fingerprints=tuple(fingerprints),
        candidate_ids=tuple(row.candidate_id for row in selected),
        operation_families=tuple(row.operation_family for row in selected),
        current_outcomes_consumed=False,
    )


def generate_contextual_body_candidate(
    genome: MorphologyGenome,
    fresh_residual: MorphologyResidual,
    *,
    raw_candidate_budget: int = 64,
    expected_policy_origin_residual_id: str | None = None,
) -> ContextualBodySelection:
    """Select exactly one candidate using only inherited policy + fresh pressure.

    The no-policy parent keeps the frozen family/id order. A policy-bearing descendant
    may route one evaluation slot to a context-relevant family. No score, outcome,
    verifier target, or benchmark result is accepted by this function.
    """
    raw_candidate_budget = max(2, int(raw_candidate_budget))
    runtime = MorphologyCompiler.compile_runtime(
        genome,
        expected_residual_id=expected_policy_origin_residual_id,
    )
    raw = NativeMetaMorphologyGenesisEngine(candidate_budget=raw_candidate_budget).generate(
        genome,
        (fresh_residual,),
    )
    ordered = list(_parent_order(raw))
    if not ordered:
        raise ValueError("NO_BODY_POLICY_CANDIDATES")

    fingerprints = []
    if runtime.native_meta_policies:
        preferred = _context_family(fresh_residual.pressure)
        for binding in runtime.native_meta_policies:
            program = compile_native_meta_policy(
                binding.implementation_ref,
                target_kind=binding.target_kind,
                expected_residual_id=expected_policy_origin_residual_id,
            )
            fingerprints.append(program.fingerprint())
        preferred_rows = [row for row in ordered if row.operation_family == preferred]
        if not preferred_rows:
            raise ValueError("CONTEXTUAL_POLICY_TARGET_FAMILY_UNAVAILABLE")
        selected = preferred_rows[0]
    else:
        selected = ordered[0]

    return ContextualBodySelection(
        body_fingerprint=genome.fingerprint(),
        compiled_runtime_fingerprint=runtime.fingerprint(),
        fresh_residual_id=fresh_residual.residual_id,
        pressure_signature=fresh_residual.pressure.dominant(),
        selected_candidate_id=selected.candidate_id,
        selected_operation_family=selected.operation_family,
        policy_fingerprints=tuple(fingerprints),
        selected_candidate_budget=1,
        raw_candidate_budget=raw_candidate_budget,
        current_outcomes_consumed=False,
    )
