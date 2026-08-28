from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from arte_cognition.executable_morphology import (
    MorphologyGenome,
    MorphologyMutator,
    MorphologyMutation,
    MutationLevel,
    OrganKind,
    OrganSpec,
    PressureVector,
)
from arte_cognition.morphology_genesis import MorphologyCandidate, MorphologyResidual
from arte_cognition.native_recursive_research import NativeMetaMorphologyGenesisEngine
from arte_cognition.native_meta_policy_runtime import (
    compile_genome_native_meta_policies,
    compile_native_meta_policy,
    execute_native_meta_policy,
    parent_candidate_selection,
)


PARENT_TRACKED = (
    "arte_cognition/meta_acceleration.py",
    "arte_cognition/executable_morphology.py",
    "arte_cognition/morphology_genesis.py",
    "arte_cognition/native_recursive_research.py",
)
CANDIDATE_TRACKED = PARENT_TRACKED + ("arte_cognition/native_meta_policy_runtime.py",)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bundle(sources: dict[str, str]) -> str:
    return _sha(json.dumps(sources, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _git_show(ref: str, path: str) -> str:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _genome() -> MorphologyGenome:
    organs = (
        OrganSpec("generator", OrganKind.GENERATOR, produces=("candidate",), implementation_ref="bootstrap://generator"),
        OrganSpec("mutator", OrganKind.MUTATOR, consumes=("candidate",), produces=("mutation",), implementation_ref="bootstrap://mutator"),
        OrganSpec("governor", OrganKind.GOVERNOR),
        OrganSpec("archive", OrganKind.ARCHIVE),
    )
    return MorphologyGenome(organs=organs, edges=(), event_order=tuple(row.organ_id for row in organs))


def _probe_candidates(by_family: dict[str, MorphologyCandidate]) -> tuple[MorphologyCandidate, ...]:
    generator = by_family["CHANGE_GENERATOR_POLICY"]
    mutator = by_family["CHANGE_MUTATOR_POLICY"]
    return (
        MorphologyCandidate(
            candidate_id="A_PARENT_DEFAULT_GENERATOR",
            mutation=generator.mutation,
            descendant_fingerprint=generator.descendant_fingerprint,
            origin_residual_ids=generator.origin_residual_ids,
            operation_family=generator.operation_family,
            generation_uses_outcomes=False,
        ),
        MorphologyCandidate(
            candidate_id="B_MUTATOR_TARGET",
            mutation=mutator.mutation,
            descendant_fingerprint=mutator.descendant_fingerprint,
            origin_residual_ids=mutator.origin_residual_ids,
            operation_family=mutator.operation_family,
            generation_uses_outcomes=False,
        ),
    )


def _expect_error(fn, token: str) -> str:
    try:
        fn()
    except ValueError as exc:
        text = str(exc)
        if token not in text:
            raise SystemExit(f"wrong fail-closed error: expected token={token} actual={text}")
        return text
    raise SystemExit(f"control failed open: expected {token}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-head", required=True)
    parser.add_argument("--precommit", required=True)
    parser.add_argument("--cycle1-receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    precommit = json.loads(Path(args.precommit).read_text(encoding="utf-8"))
    cycle1 = json.loads(Path(args.cycle1_receipt).read_text(encoding="utf-8"))
    if precommit["problem_detector"] != "L3_POLICY_SEMANTICS_UNBOUND":
        raise SystemExit("cycle2 precommit problem changed")
    if precommit["official_benchmark_used"] is not False:
        raise SystemExit("cycle2 unexpectedly authorized official benchmark")
    if cycle1["eligible_for_recursive_acceleration_ledger"] is not False:
        raise SystemExit("cycle1 reverify claim boundary changed")

    parent_sources = {path: _git_show(args.parent_head, path) for path in PARENT_TRACKED}
    parent_hash = _bundle(parent_sources)
    if parent_hash != precommit["parent_core_body_sha256"]:
        raise SystemExit(
            f"cycle2 parent core continuity mismatch: expected={precommit['parent_core_body_sha256']} actual={parent_hash}"
        )
    if parent_hash != cycle1["candidate_body_hash"]:
        raise SystemExit("cycle1 receipt and cycle2 precommit disagree on parent core BODY")

    candidate_sources = {path: _read(path) for path in CANDIDATE_TRACKED}
    candidate_hash = _bundle(candidate_sources)

    genome = _genome()
    residual_id = "cycle2-semantics-probe"
    residual = MorphologyResidual(
        residual_id,
        PressureVector(human_dependency=1.0, theory_blindspot=0.25),
    )
    generated = NativeMetaMorphologyGenesisEngine(candidate_budget=16).generate(genome, (residual,))
    by_family = {row.operation_family: row for row in generated if row.operation_family.startswith("CHANGE_")}
    expected_families = {"CHANGE_GENERATOR_POLICY", "CHANGE_MUTATOR_POLICY"}
    if set(by_family) != expected_families:
        raise SystemExit(f"cycle2 could not reconstruct L3 refs: {sorted(by_family)}")

    programs = {}
    descendant_bindings = {}
    for family in sorted(expected_families):
        candidate = by_family[family]
        descendant = MorphologyMutator().apply(genome, candidate.mutation)
        compiled = compile_genome_native_meta_policies(descendant, expected_residual_id=residual_id)
        if len(compiled) != 1:
            raise SystemExit(f"descendant policy binding count mismatch for {family}: {len(compiled)}")
        program = compiled[0]
        if program.preferred_operation_family != family:
            raise SystemExit(f"descendant policy family mismatch: expected={family} actual={program.preferred_operation_family}")
        if program.current_outcomes_required:
            raise SystemExit("generated policy compilation requires current outcomes")
        programs[family] = program
        descendant_bindings[family] = {
            "descendant_fingerprint": descendant.fingerprint(),
            "policy_fingerprint": program.fingerprint(),
            "source_ref": program.source_ref,
        }

    probe = _probe_candidates(by_family)
    remove_selection = parent_candidate_selection(probe, candidate_budget=1)
    if remove_selection != ("A_PARENT_DEFAULT_GENERATOR",):
        raise SystemExit(f"REMOVE parent probe changed unexpectedly: {remove_selection}")

    generator_program = programs["CHANGE_GENERATOR_POLICY"]
    generator_full = execute_native_meta_policy(generator_program, probe, parent_candidate_budget=1)
    if len(generator_full.selected_candidate_ids) <= len(remove_selection):
        raise SystemExit("FULL generator policy did not widen future candidate frontier")
    if generator_full.current_outcomes_consumed:
        raise SystemExit("generator policy consumed current outcomes")

    mutator_program = programs["CHANGE_MUTATOR_POLICY"]
    mutator_full = execute_native_meta_policy(mutator_program, probe, parent_candidate_budget=1)
    if mutator_full.selected_candidate_ids != ("B_MUTATOR_TARGET",):
        raise SystemExit(f"FULL mutator policy did not change family priority: {mutator_full.selected_candidate_ids}")
    if mutator_full.selected_candidate_ids == remove_selection:
        raise SystemExit("FULL mutator behavior aliased REMOVE parent behavior")
    if mutator_full.current_outcomes_consumed:
        raise SystemExit("mutator policy consumed current outcomes")

    generator_ref = generator_program.source_ref
    mutator_ref = mutator_program.source_ref
    wrong_generator_to_mutator = _expect_error(
        lambda: compile_native_meta_policy(
            generator_ref,
            target_kind=OrganKind.MUTATOR,
            expected_residual_id=residual_id,
        ),
        "KIND_MISMATCH",
    )
    wrong_mutator_to_generator = _expect_error(
        lambda: compile_native_meta_policy(
            mutator_ref,
            target_kind=OrganKind.GENERATOR,
            expected_residual_id=residual_id,
        ),
        "KIND_MISMATCH",
    )

    shuffle_residual = MorphologyResidual(
        "cycle2-shuffle-other-residual",
        PressureVector(human_dependency=1.0, theory_blindspot=0.25),
    )
    shuffled_rows = NativeMetaMorphologyGenesisEngine(candidate_budget=16).generate(genome, (shuffle_residual,))
    shuffled_generator = next(row for row in shuffled_rows if row.operation_family == "CHANGE_GENERATOR_POLICY")
    shuffled_ref = shuffled_generator.mutation.payload["organ"]["implementation_ref"]
    shuffle_error = _expect_error(
        lambda: compile_native_meta_policy(
            shuffled_ref,
            target_kind=OrganKind.GENERATOR,
            expected_residual_id=residual_id,
        ),
        "RESIDUAL_MISMATCH",
    )
    malformed_error = _expect_error(
        lambda: compile_native_meta_policy("native-meta://generator/broken", target_kind=OrganKind.GENERATOR),
        "INVALID_NATIVE_META_POLICY_REF",
    )

    generator_restart = compile_native_meta_policy(
        generator_ref,
        target_kind=OrganKind.GENERATOR,
        expected_residual_id=residual_id,
    )
    mutator_restart = compile_native_meta_policy(
        mutator_ref,
        target_kind=OrganKind.MUTATOR,
        expected_residual_id=residual_id,
    )
    generator_restart_exec = execute_native_meta_policy(generator_restart, probe, parent_candidate_budget=1)
    mutator_restart_exec = execute_native_meta_policy(mutator_restart, probe, parent_candidate_budget=1)
    restart_equal = bool(
        generator_program.fingerprint() == generator_restart.fingerprint()
        and mutator_program.fingerprint() == mutator_restart.fingerprint()
        and generator_full.fingerprint() == generator_restart_exec.fingerprint()
        and mutator_full.fingerprint() == mutator_restart_exec.fingerprint()
    )
    if not restart_equal:
        raise SystemExit("native meta policy cold restart reconstruction changed semantics")

    # This generation establishes a bounded executable resolver, not canonical compiler integration.
    executable_source = _read("arte_cognition/executable_morphology.py")
    canonical_integration = "compile_genome_native_meta_policies" in executable_source or "compile_native_meta_policy" in executable_source
    if canonical_integration:
        raise SystemExit("cycle2 scope unexpectedly crossed into canonical MorphologyCompiler integration")

    receipt_material = {
        "parent_core_body_hash": parent_hash,
        "candidate_core_body_hash": candidate_hash,
        "descendant_bindings": descendant_bindings,
        "remove_selection": list(remove_selection),
        "generator_full": list(generator_full.selected_candidate_ids),
        "mutator_full": list(mutator_full.selected_candidate_ids),
        "wrong_errors": [wrong_generator_to_mutator, wrong_mutator_to_generator],
        "shuffle_error": shuffle_error,
        "malformed_error": malformed_error,
        "restart_equal": restart_equal,
        "canonical_compiler_integration": canonical_integration,
        "official_benchmark_used": False,
    }
    outcome_hash = _sha(json.dumps(receipt_material, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    receipt = {
        "schema": "arte.native_recursive_research_cycle2/v1",
        "generation": 2,
        "parent_evidence_head_sha": args.parent_head,
        "parent_core_body_hash": parent_hash,
        "candidate_core_body_hash": candidate_hash,
        "problem_detector": "L3_POLICY_SEMANTICS_UNBOUND",
        "generated_policy_program_count": len(programs),
        "generated_policy_families": sorted(programs),
        "descendant_organ_ref_binding_pass": True,
        "generator_full_changes_future_search": True,
        "generator_remove_parent_selection": list(remove_selection),
        "generator_full_selection": list(generator_full.selected_candidate_ids),
        "mutator_full_changes_future_search": True,
        "mutator_remove_parent_selection": list(remove_selection),
        "mutator_full_selection": list(mutator_full.selected_candidate_ids),
        "wrong_family_fail_closed": True,
        "shuffle_residual_fail_closed": True,
        "malformed_ref_fail_closed": True,
        "restart_reconstruction_equal": restart_equal,
        "current_outcomes_consumed": False,
        "runtime_semantics_established": True,
        "canonical_morphology_compiler_integration": False,
        "next_problem_detectors": ["L3_POLICY_COMPILER_INTEGRATION_UNBOUND"],
        "eligible_claim": "EXECUTABLE_L3_POLICY_BINDING_ESTABLISHED_BOUNDED_NATIVE_RUNTIME",
        "capability_improvement_established": False,
        "descendant_generative_improvement_established": False,
        "recursive_acceleration_established": False,
        "official_benchmark_used": False,
        "external_claim_authority": False,
        "AGI": False,
        "ASI": False,
        "outcome_receipt_sha256": outcome_hash,
    }
    Path(args.output).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
