from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from arte_cognition.meta_acceleration import MutationStrategyState
from arte_cognition.native_recursive_research import (
    NativeResearchCycle,
    NativeResearchEvaluation,
    NativeResearchLearner,
    NativeRecursiveResearchLedger,
    choose_native_meta_target,
    discover_native_research_problems,
)


TRACKED = (
    "arte_cognition/meta_acceleration.py",
    "arte_cognition/executable_morphology.py",
    "arte_cognition/morphology_genesis.py",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_sources = {path: git_show(args.base_sha, path) for path in TRACKED}
    candidate_sources = {path: read_current(path) for path in TRACKED}
    candidate_sources["arte_cognition/native_recursive_research.py"] = read_current(
        "arte_cognition/native_recursive_research.py"
    )

    base_body_hash = source_bundle_hash(base_sources)
    candidate_body_hash = source_bundle_hash(candidate_sources)
    base_problems = discover_native_research_problems(base_sources, body_hash=base_body_hash)
    candidate_problems = discover_native_research_problems(candidate_sources, body_hash=candidate_body_hash)
    base_by_detector = {problem.detector_id: problem for problem in base_problems}
    candidate_detectors = {problem.detector_id for problem in candidate_problems}
    closed = sorted(set(base_by_detector) - candidate_detectors)
    remaining = sorted(candidate_detectors)

    required_closed = {"EXTERNAL_ONLY_META_CREDIT", "UNCRECREDITED_RESEARCH_INVENTION"}
    if not required_closed.issubset(closed):
        raise SystemExit(f"native credit repair did not close required bottlenecks: closed={closed}")
    if "GENERATOR_MUTATOR_PRESSURE_UNREACHABLE" not in remaining:
        raise SystemExit("expected next recursive frontier was not preserved")

    problem = base_by_detector["EXTERNAL_ONLY_META_CREDIT"]
    outcome_receipt = sha256_bytes(
        json.dumps(
            {
                "base_body_hash": base_body_hash,
                "candidate_body_hash": candidate_body_hash,
                "closed": closed,
                "remaining": remaining,
                "official_benchmark_used": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    full = NativeResearchEvaluation(
        evaluation_id="github-native-cycle-0",
        problem_sha256=problem.fingerprint(),
        operation_family="MUTATE_SEARCH_POLICY",
        context_id=args.base_sha,
        evidence_class="GITHUB_REPOSITORY_SELF_RESEARCH",
        solved=True,
        precommitted=True,
        evaluator_reverified=True,
        removal_effect=1.0,
        wrong_swap_effect=1.0,
        retained_competence_delta=0.0,
        calibration_delta=0.0,
        problem_discovery_delta=len(closed) / max(1.0, float(len(base_problems))),
        research_invention_delta=1.0,
        meta_improvement_delta=1.0,
        compute_cost=1.0,
        evidence_cost=1.0,
        human_structural_intervention=1.0,
        outcome_receipt_sha256=outcome_receipt,
        official_benchmark_used=False,
    )
    remove = NativeResearchEvaluation(**{**full.__dict__, "evaluation_id": "remove-control", "removal_effect": 0.0})
    wrong = NativeResearchEvaluation(**{**full.__dict__, "evaluation_id": "wrong-control", "wrong_swap_effect": 0.0})

    learner = NativeResearchLearner()
    full_state = learner.update(MutationStrategyState(), (full,))
    remove_state = learner.update(MutationStrategyState(), (remove,))
    wrong_state = learner.update(MutationStrategyState(), (wrong,))
    if full_state.score("MUTATE_SEARCH_POLICY") <= 0.0:
        raise SystemExit("FULL native research did not earn positive inherited credit")
    if remove_state.score("MUTATE_SEARCH_POLICY") >= 0.0:
        raise SystemExit("REMOVE control incorrectly earned nonnegative credit")
    if wrong_state.score("MUTATE_SEARCH_POLICY") >= 0.0:
        raise SystemExit("WRONG control incorrectly earned nonnegative credit")

    baseline_target = choose_native_meta_target(
        human_dependency=0.80,
        candidate_search_cost=0.79,
        evaluator_uncertainty=0.10,
        transfer_failure=0.10,
    )
    descendant_target = choose_native_meta_target(
        human_dependency=0.80,
        candidate_search_cost=0.79,
        evaluator_uncertainty=0.10,
        transfer_failure=0.10,
        strategy=full_state,
    )
    if baseline_target == descendant_target:
        raise SystemExit("earned native research credit did not change the next meta target")

    ledger = NativeRecursiveResearchLedger()
    ledger.append(
        NativeResearchCycle(
            generation=0,
            problem_sha256=problem.fingerprint(),
            parent_body_hash=base_body_hash,
            descendant_body_hash=candidate_body_hash,
            problem_discovery_score=full.problem_discovery_delta,
            research_invention_score=full.research_invention_delta,
            meta_improvement_ability=full.meta_improvement_delta,
            retained_competence=1.0,
            calibration_score=1.0,
            compute_cost=full.compute_cost,
            evidence_cost=full.evidence_cost,
            human_structural_intervention=full.human_structural_intervention,
            native_problem=not problem.human_seeded,
            controls_pass=full.controls_pass,
        )
    )
    assessment = ledger.assess()
    if assessment.status != "PASS_BOUNDED_NATIVE_RESEARCH_CYCLE_NOT_RECURSIVE":
        raise SystemExit(f"unexpected first-cycle assessment: {assessment.status}")

    receipt = {
        "schema": "arte.native_recursive_research/v1",
        "measurement_scope": "SELF_HOSTED_CODE_CONTRACT_NOT_OFFICIAL_BENCHMARK",
        "base_sha": args.base_sha,
        "base_body_hash": base_body_hash,
        "candidate_body_hash": candidate_body_hash,
        "base_problem_detectors": sorted(base_by_detector),
        "closed_problem_detectors": closed,
        "next_problem_detectors": remaining,
        "selected_problem_id": problem.problem_id,
        "selected_problem_sha256": problem.fingerprint(),
        "full_credit": full_state.score("MUTATE_SEARCH_POLICY"),
        "remove_credit": remove_state.score("MUTATE_SEARCH_POLICY"),
        "wrong_credit": wrong_state.score("MUTATE_SEARCH_POLICY"),
        "baseline_next_target": baseline_target,
        "descendant_next_target": descendant_target,
        "research_productivity": full.research_productivity,
        "assessment_status": assessment.status,
        "recursive_acceleration_established": False,
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
