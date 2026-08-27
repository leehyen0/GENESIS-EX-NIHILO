from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from arte_cognition.body_policy_generation import generate_body_candidates
from arte_cognition.executable_morphology import (
    MorphologyGenome,
    MorphologyMutator,
    OrganKind,
    OrganSpec,
    PressureVector,
)
from arte_cognition.morphology_genesis import MorphologyResidual
from arte_cognition.native_recursive_research import NativeMetaMorphologyGenesisEngine


PARENT_TRACKED = (
    "arte_cognition/meta_acceleration.py",
    "arte_cognition/executable_morphology.py",
    "arte_cognition/morphology_genesis.py",
    "arte_cognition/native_recursive_research.py",
    "arte_cognition/native_meta_policy_runtime.py",
)
CANDIDATE_TRACKED = PARENT_TRACKED + ("arte_cognition/body_policy_generation.py",)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bundle(sources: dict[str, str]) -> str:
    return _sha(json.dumps(sources, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _git_show(ref: str, path: str) -> str:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _parent() -> MorphologyGenome:
    organs = (
        OrganSpec("generator", OrganKind.GENERATOR, produces=("candidate",), implementation_ref="bootstrap://generator"),
        OrganSpec("mutator", OrganKind.MUTATOR, consumes=("candidate",), produces=("mutation",), implementation_ref="bootstrap://mutator"),
        OrganSpec("governor", OrganKind.GOVERNOR),
        OrganSpec("archive", OrganKind.ARCHIVE),
    )
    return MorphologyGenome(organs=organs, edges=(), event_order=("generator", "mutator", "governor", "archive"))


def _child(parent: MorphologyGenome, origin: str, family: str) -> MorphologyGenome:
    residual = MorphologyResidual(origin, PressureVector(human_dependency=1.0, theory_blindspot=0.25))
    rows = NativeMetaMorphologyGenesisEngine(candidate_budget=16).generate(parent, (residual,))
    candidate = next(row for row in rows if row.operation_family == family)
    return MorphologyMutator().apply(parent, candidate.mutation)


def _fresh() -> MorphologyResidual:
    return MorphologyResidual("cycle4-fresh-B", PressureVector(human_dependency=1.0, theory_blindspot=0.25))


def _expect_error(fn, token: str) -> str:
    try:
        fn()
    except ValueError as exc:
        text = str(exc)
        if token not in text:
            raise SystemExit(f"wrong fail-closed error: expected={token} actual={text}")
        return text
    raise SystemExit(f"control failed open: expected {token}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-head", required=True)
    parser.add_argument("--precommit", required=True)
    parser.add_argument("--cycle3-receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    precommit = json.loads(Path(args.precommit).read_text(encoding="utf-8"))
    cycle3 = json.loads(Path(args.cycle3_receipt).read_text(encoding="utf-8"))
    if precommit["problem_detector"] != "COMPILED_L3_POLICY_BODY_EXECUTION_UNBOUND":
        raise SystemExit("cycle4 precommit problem changed")
    if cycle3["body_candidate_generation_consumes_compiled_policy"] is not False:
        raise SystemExit("cycle3 already claimed BODY policy execution")

    parent_sources = {path: _git_show(args.parent_head, path) for path in PARENT_TRACKED}
    parent_hash = _bundle(parent_sources)
    if parent_hash != precommit["parent_core_body_sha256"]:
        raise SystemExit(f"cycle4 parent mismatch: expected={precommit['parent_core_body_sha256']} actual={parent_hash}")
    if parent_hash != cycle3["candidate_core_body_hash"]:
        raise SystemExit("cycle3 receipt and cycle4 precommit disagree on parent BODY")

    candidate_sources = {path: _read(path) for path in CANDIDATE_TRACKED}
    candidate_hash = _bundle(candidate_sources)

    origin = "cycle4-policy-origin-A"
    fresh = _fresh()
    parent = _parent()
    generator_child = _child(parent, origin, "CHANGE_GENERATOR_POLICY")
    mutator_child = _child(parent, origin, "CHANGE_MUTATOR_POLICY")

    remove = generate_body_candidates(parent, (fresh,), nominal_budget=1)
    generator_full = generate_body_candidates(
        generator_child,
        (fresh,),
        nominal_budget=1,
        expected_policy_origin_residual_id=origin,
    )
    mutator_full = generate_body_candidates(
        mutator_child,
        (fresh,),
        nominal_budget=1,
        expected_policy_origin_residual_id=origin,
    )

    if remove.policy_fingerprints or remove.effective_budget != 1 or len(remove.candidate_ids) != 1:
        raise SystemExit("REMOVE/no-policy parent contract failed")
    if generator_full.effective_budget <= remove.effective_budget:
        raise SystemExit("generator child did not widen BODY future candidate frontier")
    if len(generator_full.candidate_ids) <= len(remove.candidate_ids):
        raise SystemExit("generator child did not expose additional fresh candidates")
    if not generator_full.policy_fingerprints:
        raise SystemExit("generator child did not consume canonical compiled policy")
    if remove.operation_families != ("CHANGE_GENERATOR_POLICY",):
        raise SystemExit(f"unexpected frozen REMOVE family: {remove.operation_families}")
    if mutator_full.operation_families != ("CHANGE_MUTATOR_POLICY",):
        raise SystemExit(f"mutator child did not change fresh family priority: {mutator_full.operation_families}")
    if mutator_full.candidate_ids == remove.candidate_ids:
        raise SystemExit("mutator child aliased REMOVE parent candidate selection")
    if any((remove.current_outcomes_consumed, generator_full.current_outcomes_consumed, mutator_full.current_outcomes_consumed)):
        raise SystemExit("BODY candidate generation consumed current outcomes")

    # WRONG: a generator ref placed into the mutator organ must fail at canonical compile.
    generator_ref = next(o.implementation_ref for o in generator_child.organs if o.organ_id == "generator")
    wrong_organs = tuple(
        OrganSpec(
            organ_id=o.organ_id,
            kind=o.kind,
            consumes=o.consumes,
            produces=o.produces,
            implementation_ref=generator_ref if o.organ_id == "mutator" else o.implementation_ref,
            version=o.version,
            cost_hint=o.cost_hint,
            provenance=o.provenance,
            enabled=o.enabled,
        )
        for o in parent.organs
    )
    wrong = MorphologyGenome(wrong_organs, parent.edges, parent.event_order, parent.constitution_epoch)
    wrong_error = _expect_error(
        lambda: generate_body_candidates(
            wrong,
            (fresh,),
            nominal_budget=1,
            expected_policy_origin_residual_id=origin,
        ),
        "KIND_MISMATCH",
    )

    # SHUFFLE: policy generated from origin C cannot masquerade as origin A.
    shuffled_child = _child(parent, "cycle4-policy-origin-C", "CHANGE_GENERATOR_POLICY")
    shuffle_error = _expect_error(
        lambda: generate_body_candidates(
            shuffled_child,
            (fresh,),
            nominal_budget=1,
            expected_policy_origin_residual_id=origin,
        ),
        "RESIDUAL_MISMATCH",
    )

    # Cold reconstruction by deterministic regeneration must give identical future candidates.
    generator_child_restart = _child(_parent(), origin, "CHANGE_GENERATOR_POLICY")
    generator_restart = generate_body_candidates(
        generator_child_restart,
        (_fresh(),),
        nominal_budget=1,
        expected_policy_origin_residual_id=origin,
    )
    mutator_child_restart = _child(_parent(), origin, "CHANGE_MUTATOR_POLICY")
    mutator_restart = generate_body_candidates(
        mutator_child_restart,
        (_fresh(),),
        nominal_budget=1,
        expected_policy_origin_residual_id=origin,
    )
    restart_equal = generator_full == generator_restart and mutator_full == mutator_restart
    if not restart_equal:
        raise SystemExit("cycle4 cold reconstruction changed future candidate generation")

    material = {
        "parent_core_body_hash": parent_hash,
        "candidate_core_body_hash": candidate_hash,
        "policy_origin": origin,
        "fresh_residual": fresh.residual_id,
        "remove": remove.__dict__,
        "generator_full": generator_full.__dict__,
        "mutator_full": mutator_full.__dict__,
        "wrong_error": wrong_error,
        "shuffle_error": shuffle_error,
        "restart_equal": restart_equal,
        "official_benchmark_used": False,
    }
    outcome_hash = _sha(json.dumps(material, sort_keys=True, separators=(",", ":"), default=list).encode("utf-8"))
    receipt = {
        "schema": "arte.native_recursive_research_cycle4/v1",
        "generation": 4,
        "parent_evidence_head_sha": args.parent_head,
        "parent_core_body_hash": parent_hash,
        "candidate_core_body_hash": candidate_hash,
        "problem_detector": "COMPILED_L3_POLICY_BODY_EXECUTION_UNBOUND",
        "policy_origin_residual_id": origin,
        "fresh_application_residual_id": fresh.residual_id,
        "origin_and_fresh_residual_disjoint": origin != fresh.residual_id,
        "remove_candidate_count": len(remove.candidate_ids),
        "remove_operation_families": list(remove.operation_families),
        "generator_full_candidate_count": len(generator_full.candidate_ids),
        "generator_full_effective_budget": generator_full.effective_budget,
        "generator_full_policy_consumed": bool(generator_full.policy_fingerprints),
        "mutator_full_operation_families": list(mutator_full.operation_families),
        "mutator_full_policy_consumed": bool(mutator_full.policy_fingerprints),
        "wrong_kind_fail_closed": True,
        "shuffle_policy_origin_fail_closed": True,
        "restart_candidate_generation_equal": restart_equal,
        "current_outcomes_consumed": False,
        "body_candidate_generation_consumes_compiled_policy": True,
        "next_problem_detectors": ["INHERITED_L3_POLICY_USEFULNESS_UNPROVEN"],
        "eligible_claim": "BODY_CANDIDATE_GENERATION_CONSUMES_INHERITED_L3_POLICY",
        "useful_candidate_quality_improvement_established": False,
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
