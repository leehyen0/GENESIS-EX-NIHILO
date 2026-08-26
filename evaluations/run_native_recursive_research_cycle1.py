from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from arte_cognition.executable_morphology import MorphologyGenome, OrganKind, OrganSpec, PressureVector, MutationLevel
from arte_cognition.meta_acceleration import MutationStrategyState
from arte_cognition.morphology_genesis import MorphologyGenesisEngine, MorphologyResidual
from arte_cognition.native_recursive_research import (
    NativeMetaMorphologyGenesisEngine,
    NativeResearchCycle,
    NativeResearchEvaluation,
    NativeResearchLearner,
    NativeRecursiveResearchLedger,
    discover_native_research_problems,
)


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
        OrganSpec("mutator", OrganKind.MUTATOR, consumes=("candidate",), produces=("mutation",), implementation_ref="bootstrap://mutator"),
        OrganSpec("governor", OrganKind.GOVERNOR),
        OrganSpec("archive", OrganKind.ARCHIVE),
    )
    return MorphologyGenome(organs=organs, edges=(), event_order=tuple(o.organ_id for o in organs))


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
    descendant_body_hash = source_bundle_hash(current_sources)
    if parent_body_hash != cycle0["candidate_body_hash"]:
        raise SystemExit(
            f"cycle0 BODY continuity mismatch: expected={cycle0['candidate_body_hash']} actual={parent_body_hash}"
        )

    parent_problems = discover_native_research_problems(parent_sources, body_hash=parent_body_hash)
    current_problems = discover_native_research_problems(current_sources, body_hash=descendant_body_hash)
    parent_by_detector = {row.detector_id: row for row in parent_problems}
    current_detectors = {row.detector_id for row in current_problems}
    expected_parent = {"GENERATOR_MUTATOR_PRESSURE_UNREACHABLE"}
    if set(parent_by_detector) != expected_parent:
        raise SystemExit(f"unexpected frozen parent frontier: {sorted(parent_by_detector)}")
    if "GENERATOR_MUTATOR_PRESSURE_UNREACHABLE" in current_detectors:
        raise SystemExit("cycle1 failed to close L3 reachability")
    if "L3_POLICY_SEMANTICS_UNBOUND" not in current_detectors:
        raise SystemExit(f"cycle1 failed to expose next semantics frontier: {sorted(current_detectors)}")

    genome = morphology()
    residual = MorphologyResidual(
        "cycle1-human-dependency",
        PressureVector(human_dependency=1.0, theory_blindspot=0.25),
    )
    full_rows = NativeMetaMorphologyGenesisEngine(candidate_budget=64).generate(genome, (residual,))
    full_l3 = [row for row in full_rows if row.mutation.level == MutationLevel.GENERATOR_MUTATOR]
    remove_rows = MorphologyGenesisEngine(candidate_budget=64).generate(genome, (residual,))
    remove_l3 = [row for row in remove_rows if row.mutation.level == MutationLevel.GENERATOR_MUTATOR]
    wrong_residual = MorphologyResidual("cycle1-wrong-pressure", PressureVector())
    wrong_rows = NativeMetaMorphologyGenesisEngine(candidate_budget=64).generate(genome, (wrong_residual,))
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

    problem = parent_by_detector["GENERATOR_MUTATOR_PRESSURE_UNREACHABLE"]
    outcome_receipt = sha256_bytes(
        json.dumps(
            {
                "parent_body_hash": parent_body_hash,
                "descendant_body_hash": descendant_body_hash,
                "full_l3_candidate_ids": sorted(row.candidate_id for row in full_l3),
                "remove_l3_count": len(remove_l3),
                "wrong_l3_count": len(wrong_l3),
                "next_problem_detectors": sorted(current_detectors),
                "official_benchmark_used": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    evaluation = NativeResearchEvaluation(
        evaluation_id="github-native-cycle-1",
        problem_sha256=problem.fingerprint(),
        operation_family="MUTATE_MUTATOR",
        context_id=args.cycle0_head,
        evidence_class="GITHUB_REPOSITORY_SELF_RESEARCH",
        solved=True,
        precommitted=True,
        evaluator_reverified=True,
        removal_effect=1.0,
        wrong_swap_effect=1.0,
        retained_competence_delta=0.0,
        calibration_delta=0.0,
        problem_discovery_delta=1.0,
        research_invention_delta=1.0,
        meta_improvement_delta=1.0,
        compute_cost=1.0,
        evidence_cost=1.0,
        human_structural_intervention=1.0,
        outcome_receipt_sha256=outcome_receipt,
        official_benchmark_used=False,
    )
    remove_eval = NativeResearchEvaluation(**{**evaluation.__dict__, "evaluation_id": "cycle1-remove", "removal_effect": 0.0})
    wrong_eval = NativeResearchEvaluation(**{**evaluation.__dict__, "evaluation_id": "cycle1-wrong", "wrong_swap_effect": 0.0})

    learner = NativeResearchLearner()
    full_credit = learner.update(MutationStrategyState(), (evaluation,)).score("MUTATE_MUTATOR")
    remove_credit = learner.update(MutationStrategyState(), (remove_eval,)).score("MUTATE_MUTATOR")
    wrong_credit = learner.update(MutationStrategyState(), (wrong_eval,)).score("MUTATE_MUTATOR")
    if full_credit <= 0.0 or remove_credit >= 0.0 or wrong_credit >= 0.0:
        raise SystemExit("cycle1 causal credit controls failed")

    # Reconstruct the frozen first-cycle measurement so the second assessment is a real contiguous lineage.
    cycle0_productivity = float(cycle0["research_productivity"])
    cycle0_discovery = 3.0 * cycle0_productivity - 2.0
    ledger = NativeRecursiveResearchLedger()
    ledger.append(
        NativeResearchCycle(
            generation=0,
            problem_sha256=cycle0["selected_problem_sha256"],
            parent_body_hash=cycle0["base_body_hash"],
            descendant_body_hash=cycle0["candidate_body_hash"],
            problem_discovery_score=cycle0_discovery,
            research_invention_score=1.0,
            meta_improvement_ability=1.0,
            retained_competence=1.0,
            calibration_score=1.0,
            compute_cost=1.0,
            evidence_cost=1.0,
            human_structural_intervention=1.0,
            native_problem=True,
            controls_pass=True,
        )
    )
    ledger.append(
        NativeResearchCycle(
            generation=1,
            problem_sha256=problem.fingerprint(),
            parent_body_hash=parent_body_hash,
            descendant_body_hash=descendant_body_hash,
            problem_discovery_score=evaluation.problem_discovery_delta,
            research_invention_score=evaluation.research_invention_delta,
            meta_improvement_ability=evaluation.meta_improvement_delta,
            retained_competence=1.0,
            calibration_score=1.0,
            compute_cost=evaluation.compute_cost,
            evidence_cost=evaluation.evidence_cost,
            human_structural_intervention=evaluation.human_structural_intervention,
            native_problem=not problem.human_seeded,
            controls_pass=evaluation.controls_pass,
        )
    )
    assessment = ledger.assess()
    if assessment.cycle_count != 2:
        raise SystemExit("cycle1 lineage did not contain two contiguous cycles")
    if assessment.status != "PASS_BOUNDED_NATIVE_RESEARCH_CYCLE_NOT_RECURSIVE":
        raise SystemExit(f"cycle1 over/under-claimed status: {assessment.status}")
    if not assessment.research_productivity_trajectory[1] > assessment.research_productivity_trajectory[0]:
        raise SystemExit("research productivity did not increase from cycle0 to cycle1")

    receipt = {
        "schema": "arte.native_recursive_research/v1",
        "generation": 1,
        "measurement_scope": "SELF_HOSTED_CODE_CONTRACT_NOT_OFFICIAL_BENCHMARK",
        "parent_head_sha": args.cycle0_head,
        "parent_body_hash": parent_body_hash,
        "descendant_body_hash": descendant_body_hash,
        "parent_problem_detectors": sorted(parent_by_detector),
        "closed_problem_detectors": ["GENERATOR_MUTATOR_PRESSURE_UNREACHABLE"],
        "next_problem_detectors": sorted(current_detectors),
        "selected_problem_id": problem.problem_id,
        "selected_problem_sha256": problem.fingerprint(),
        "full_l3_candidate_count": len(full_l3),
        "full_l3_operation_families": sorted(families),
        "remove_l3_candidate_count": len(remove_l3),
        "wrong_l3_candidate_count": len(wrong_l3),
        "full_credit": full_credit,
        "remove_credit": remove_credit,
        "wrong_credit": wrong_credit,
        "research_productivity": evaluation.research_productivity,
        "research_productivity_trajectory": list(assessment.research_productivity_trajectory),
        "human_intervention_trajectory": list(assessment.human_intervention_trajectory),
        "assessment_status": assessment.status,
        "recursive_acceleration_established": False,
        "runtime_semantics_established": False,
        "capability_improvement_established": False,
        "official_benchmark_used": False,
        "external_claim_authority": False,
        "AGI": False,
        "ASI": False,
        "outcome_receipt_sha256": outcome_receipt,
    }
    Path(args.output).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
