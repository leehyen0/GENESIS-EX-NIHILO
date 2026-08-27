from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from arte_cognition.executable_morphology import MorphologyGenome, OrganKind, OrganSpec, PressureVector, MutationLevel
from arte_cognition.morphology_genesis import MorphologyGenesisEngine, MorphologyResidual
from arte_cognition.native_recursive_research import NativeMetaMorphologyGenesisEngine, discover_native_research_problems


TRACKED = (
    "arte_cognition/meta_acceleration.py",
    "arte_cognition/executable_morphology.py",
    "arte_cognition/morphology_genesis.py",
    "arte_cognition/native_recursive_research.py",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_bundle_hash(sources: dict[str, str]) -> str:
    material = json.dumps(sources, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(material)


def git_show(ref: str, path: str) -> str:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)


def read_current(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def morphology() -> MorphologyGenome:
    organs = (
        OrganSpec("generator", OrganKind.GENERATOR, produces=("candidate",), implementation_ref="bootstrap://generator"),
        OrganSpec(
            "mutator",
            OrganKind.MUTATOR,
            consumes=("candidate",),
            produces=("mutation",),
            implementation_ref="bootstrap://mutator",
        ),
        OrganSpec("governor", OrganKind.GOVERNOR),
        OrganSpec("archive", OrganKind.ARCHIVE),
    )
    return MorphologyGenome(organs=organs, edges=(), event_order=tuple(o.organ_id for o in organs))


def _class_function(source: str, class_name: str, function_name: str) -> ast.AST | None:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == function_name:
                    return child
    return None


def compiler_compile_reads_implementation_ref(executable_source: str) -> bool:
    compile_node = _class_function(executable_source, "MorphologyCompiler", "compile")
    if compile_node is None:
        raise RuntimeError("MorphologyCompiler.compile missing")
    return any(isinstance(node, ast.Attribute) and node.attr == "implementation_ref" for node in ast.walk(compile_node))


def executable_native_meta_resolver_exists(sources: dict[str, str]) -> bool:
    resolver_names = {"compile_native_meta_policy", "execute_native_meta_policy", "resolve_native_meta_policy"}
    for source in sources.values():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in resolver_names:
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle0-head", required=True)
    parser.add_argument("--cycle0-receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cycle0 = json.loads(Path(args.cycle0_receipt).read_text(encoding="utf-8"))
    if cycle0["recursive_acceleration_established"] is not False or cycle0["official_benchmark_used"] is not False:
        raise SystemExit("cycle0 claim boundary changed")

    parent_sources = {path: git_show(args.cycle0_head, path) for path in TRACKED}
    current_sources = {path: read_current(path) for path in TRACKED}
    parent_body_hash = source_bundle_hash(parent_sources)
    candidate_body_hash = source_bundle_hash(current_sources)
    if parent_body_hash != cycle0["candidate_body_hash"]:
        raise SystemExit(
            f"cycle0 BODY continuity mismatch: expected={cycle0['candidate_body_hash']} actual={parent_body_hash}"
        )

    parent_problems = discover_native_research_problems(parent_sources, body_hash=parent_body_hash)
    parent_detectors = {row.detector_id for row in parent_problems}
    if parent_detectors != {"GENERATOR_MUTATOR_PRESSURE_UNREACHABLE"}:
        raise SystemExit(f"unexpected frozen parent frontier: {sorted(parent_detectors)}")

    # Reproduce attempt-0 detector behavior without modifying the frozen L3 implementation.
    legacy_candidate_problems = discover_native_research_problems(current_sources, body_hash=candidate_body_hash)
    legacy_candidate_detectors = {row.detector_id for row in legacy_candidate_problems}
    if "GENERATOR_MUTATOR_PRESSURE_UNREACHABLE" in legacy_candidate_detectors:
        raise SystemExit("candidate no longer closes L3 reachability")
    if "L3_POLICY_SEMANTICS_UNBOUND" in legacy_candidate_detectors:
        raise SystemExit("attempt-0 lexical false negative no longer reproduced; evaluator surface changed")

    genome = morphology()
    residual = MorphologyResidual(
        "cycle1-human-dependency",
        PressureVector(human_dependency=1.0, theory_blindspot=0.25),
    )
    full_rows = NativeMetaMorphologyGenesisEngine(candidate_budget=64).generate(genome, (residual,))
    full_l3 = [row for row in full_rows if row.mutation.level == MutationLevel.GENERATOR_MUTATOR]
    remove_rows = MorphologyGenesisEngine(candidate_budget=64).generate(genome, (residual,))
    remove_l3 = [row for row in remove_rows if row.mutation.level == MutationLevel.GENERATOR_MUTATOR]
    wrong_rows = NativeMetaMorphologyGenesisEngine(candidate_budget=64).generate(
        genome,
        (MorphologyResidual("cycle1-wrong-pressure", PressureVector()),),
    )
    wrong_l3 = [row for row in wrong_rows if row.mutation.level == MutationLevel.GENERATOR_MUTATOR]

    families = {row.operation_family for row in full_l3}
    if families != {"CHANGE_GENERATOR_POLICY", "CHANGE_MUTATOR_POLICY"}:
        raise SystemExit(f"FULL missing expected L3 candidate families: {sorted(families)}")
    if remove_l3:
        raise SystemExit("REMOVE control retained L3 candidates")
    if wrong_l3:
        raise SystemExit("WRONG pressure control retained L3 candidates")
    if any(row.generation_uses_outcomes for row in full_l3):
        raise SystemExit("L3 candidate generation used current outcomes")

    executable_source = current_sources["arte_cognition/executable_morphology.py"]
    compiler_reads_ref = compiler_compile_reads_implementation_ref(executable_source)
    resolver_exists = executable_native_meta_resolver_exists(current_sources)
    semantics_unbound = (not compiler_reads_ref) and (not resolver_exists)
    if not semantics_unbound:
        raise SystemExit(
            "AST re-verifier found executable L3 semantics; expected unbound semantics before cycle 2: "
            f"compiler_reads_ref={compiler_reads_ref} resolver_exists={resolver_exists}"
        )

    receipt_material = {
        "parent_body_hash": parent_body_hash,
        "candidate_body_hash": candidate_body_hash,
        "full_l3_candidate_ids": sorted(row.candidate_id for row in full_l3),
        "full_l3_operation_families": sorted(families),
        "remove_l3_count": len(remove_l3),
        "wrong_l3_count": len(wrong_l3),
        "legacy_candidate_detectors": sorted(legacy_candidate_detectors),
        "compiler_compile_reads_implementation_ref": compiler_reads_ref,
        "native_meta_resolver_exists": resolver_exists,
        "next_problem": "L3_POLICY_SEMANTICS_UNBOUND",
        "official_benchmark_used": False,
    }
    outcome_receipt_sha256 = sha256_bytes(
        json.dumps(receipt_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )

    receipt = {
        "schema": "arte.native_recursive_research_cycle1_reverify/v1",
        "generation": 1,
        "attempt": 1,
        "measurement_scope": "POST_OUTCOME_EVALUATOR_REPAIR_STRUCTURAL_REVERIFY_NOT_ACCELERATION_EVIDENCE",
        "parent_head_sha": args.cycle0_head,
        "parent_body_hash": parent_body_hash,
        "candidate_body_hash": candidate_body_hash,
        "attempt0_failure_preserved": "research/native_research_cycle_001_attempt0_failure.json",
        "legacy_detector_false_negative_reproduced": legacy_candidate_detectors == set(),
        "structural_l3_reachability_reverified": True,
        "full_l3_candidate_count": len(full_l3),
        "full_l3_operation_families": sorted(families),
        "remove_l3_candidate_count": len(remove_l3),
        "wrong_l3_candidate_count": len(wrong_l3),
        "generation_uses_current_outcomes": any(row.generation_uses_outcomes for row in full_l3),
        "compiler_compile_reads_implementation_ref": compiler_reads_ref,
        "native_meta_resolver_exists": resolver_exists,
        "next_problem_detectors": ["L3_POLICY_SEMANTICS_UNBOUND"],
        "eligible_claim": "BOUNDED_L3_STRUCTURAL_REACHABILITY_REVERIFIED_ONLY",
        "post_outcome_evaluator_repair": True,
        "eligible_for_recursive_acceleration_ledger": False,
        "runtime_semantics_established": False,
        "capability_improvement_established": False,
        "recursive_acceleration_established": False,
        "official_benchmark_used": False,
        "external_claim_authority": False,
        "AGI": False,
        "ASI": False,
        "outcome_receipt_sha256": outcome_receipt_sha256,
    }
    Path(args.output).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
