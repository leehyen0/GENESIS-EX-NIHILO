from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from arte_cognition.executable_morphology import (
    MorphologyCompiler,
    MorphologyGenome,
    MorphologyMutator,
    OrganKind,
    OrganSpec,
    PressureVector,
)
from arte_cognition.morphology_genesis import MorphologyResidual
from arte_cognition.native_recursive_research import NativeMetaMorphologyGenesisEngine


TRACKED = (
    "arte_cognition/meta_acceleration.py",
    "arte_cognition/executable_morphology.py",
    "arte_cognition/morphology_genesis.py",
    "arte_cognition/native_recursive_research.py",
    "arte_cognition/native_meta_policy_runtime.py",
)


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
    return MorphologyGenome(organs=organs, edges=(), event_order=("generator", "mutator", "governor", "archive"))


def _l3(genome: MorphologyGenome, residual_id: str):
    residual = MorphologyResidual(residual_id, PressureVector(human_dependency=1.0, theory_blindspot=0.25))
    rows = NativeMetaMorphologyGenesisEngine(candidate_budget=16).generate(genome, (residual,))
    by_family = {row.operation_family: row for row in rows if row.operation_family.startswith("CHANGE_")}
    if set(by_family) != {"CHANGE_GENERATOR_POLICY", "CHANGE_MUTATOR_POLICY"}:
        raise SystemExit(f"could not reconstruct generation-1 L3 family: {sorted(by_family)}")
    return by_family


def _expect_value_error(fn, token: str) -> str:
    try:
        fn()
    except ValueError as exc:
        text = str(exc)
        if token not in text:
            raise SystemExit(f"wrong canonical compiler failure: expected={token} actual={text}")
        return text
    raise SystemExit(f"canonical compiler failed open: expected={token}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-head", required=True)
    parser.add_argument("--precommit", required=True)
    parser.add_argument("--cycle2-receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    precommit = json.loads(Path(args.precommit).read_text(encoding="utf-8"))
    cycle2 = json.loads(Path(args.cycle2_receipt).read_text(encoding="utf-8"))
    if precommit["problem_detector"] != "L3_POLICY_COMPILER_INTEGRATION_UNBOUND":
        raise SystemExit("cycle3 precommit problem changed")
    if cycle2["canonical_morphology_compiler_integration"] is not False:
        raise SystemExit("cycle2 already claimed canonical compiler integration")

    parent_sources = {path: _git_show(args.parent_head, path) for path in TRACKED}
    parent_hash = _bundle(parent_sources)
    if parent_hash != precommit["parent_core_body_sha256"]:
        raise SystemExit(
            f"cycle3 parent continuity mismatch: expected={precommit['parent_core_body_sha256']} actual={parent_hash}"
        )
    if parent_hash != cycle2["candidate_core_body_hash"]:
        raise SystemExit("cycle2 receipt and cycle3 precommit disagree on parent BODY")

    current_sources = {path: _read(path) for path in TRACKED}
    candidate_hash = _bundle(current_sources)

    parent = _genome()
    legacy_order = parent.event_order
    if MorphologyCompiler.compile(parent) != legacy_order:
        raise SystemExit("legacy MorphologyCompiler.compile event-order contract regressed")
    remove_runtime = MorphologyCompiler.compile_runtime(parent)
    if remove_runtime.event_order != legacy_order or remove_runtime.native_meta_policies:
        raise SystemExit("REMOVE/bootstrap parent unexpectedly compiled native-meta policies")

    residual_id = "cycle3-canonical-full"
    families = _l3(parent, residual_id)
    full_bindings = {}
    restart_equal = True
    for family in sorted(families):
        descendant = MorphologyMutator().apply(parent, families[family].mutation)
        runtime_a = MorphologyCompiler.compile_runtime(descendant, expected_residual_id=residual_id)
        runtime_b = MorphologyCompiler.compile_runtime(descendant, expected_residual_id=residual_id)
        if len(runtime_a.native_meta_policies) != 1:
            raise SystemExit(f"FULL canonical binding count mismatch for {family}")
        binding = runtime_a.native_meta_policies[0]
        if binding.preferred_operation_family != family:
            raise SystemExit(f"FULL canonical binding family mismatch: expected={family} actual={binding.preferred_operation_family}")
        if MorphologyCompiler.compile(descendant) != descendant.event_order:
            raise SystemExit("canonical compile compatibility output changed for policy descendant")
        restart_equal = restart_equal and (
            runtime_a.fingerprint() == runtime_b.fingerprint()
            and binding.fingerprint() == runtime_b.native_meta_policies[0].fingerprint()
        )
        full_bindings[family] = {
            "organ_id": binding.organ_id,
            "target_kind": binding.target_kind.value,
            "implementation_ref": binding.implementation_ref,
            "policy_fingerprint": binding.policy_fingerprint,
            "binding_fingerprint": binding.fingerprint(),
        }
    if not restart_equal:
        raise SystemExit("canonical compiler restart reconstruction mismatch")

    generator_ref = families["CHANGE_GENERATOR_POLICY"].mutation.payload["organ"]["implementation_ref"]
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
    wrong_error = _expect_value_error(lambda: MorphologyCompiler.compile(wrong), "KIND_MISMATCH")

    shuffle_families = _l3(parent, "cycle3-canonical-other")
    shuffled_descendant = MorphologyMutator().apply(parent, shuffle_families["CHANGE_GENERATOR_POLICY"].mutation)
    shuffle_error = _expect_value_error(
        lambda: MorphologyCompiler.compile_runtime(shuffled_descendant, expected_residual_id=residual_id),
        "RESIDUAL_MISMATCH",
    )

    malformed_organs = tuple(
        OrganSpec(
            organ_id=o.organ_id,
            kind=o.kind,
            consumes=o.consumes,
            produces=o.produces,
            implementation_ref="native-meta://generator/broken" if o.organ_id == "generator" else o.implementation_ref,
            version=o.version,
            cost_hint=o.cost_hint,
            provenance=o.provenance,
            enabled=o.enabled,
        )
        for o in parent.organs
    )
    malformed = MorphologyGenome(malformed_organs, parent.edges, parent.event_order, parent.constitution_epoch)
    malformed_error = _expect_value_error(lambda: MorphologyCompiler.compile(malformed), "INVALID_NATIVE_META_POLICY_REF")

    # Canonical compile now binds semantics, but its binding is not yet consumed by the
    # BODY's normal candidate-generation path. That is the next causal frontier.
    morphology_source = _read("arte_cognition/morphology_genesis.py")
    body_generation_consumes_binding = "CompiledNativeMetaPolicyBinding" in morphology_source or "compile_runtime(" in morphology_source
    if body_generation_consumes_binding:
        raise SystemExit("cycle3 scope unexpectedly crossed into BODY candidate-generation policy execution")

    material = {
        "parent_core_body_hash": parent_hash,
        "candidate_core_body_hash": candidate_hash,
        "legacy_order": list(legacy_order),
        "remove_policy_count": len(remove_runtime.native_meta_policies),
        "full_bindings": full_bindings,
        "wrong_error": wrong_error,
        "shuffle_error": shuffle_error,
        "malformed_error": malformed_error,
        "restart_equal": restart_equal,
        "body_generation_consumes_binding": body_generation_consumes_binding,
        "official_benchmark_used": False,
    }
    outcome_hash = _sha(json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    receipt = {
        "schema": "arte.native_recursive_research_cycle3/v1",
        "generation": 3,
        "parent_evidence_head_sha": args.parent_head,
        "parent_core_body_hash": parent_hash,
        "candidate_core_body_hash": candidate_hash,
        "problem_detector": "L3_POLICY_COMPILER_INTEGRATION_UNBOUND",
        "legacy_compile_contract_preserved": True,
        "remove_native_policy_count": 0,
        "canonical_full_binding_count": len(full_bindings),
        "canonical_full_binding_families": sorted(full_bindings),
        "wrong_kind_fail_closed": True,
        "shuffle_residual_fail_closed": True,
        "malformed_ref_fail_closed": True,
        "restart_reconstruction_equal": restart_equal,
        "canonical_morphology_compiler_integration": True,
        "body_candidate_generation_consumes_compiled_policy": False,
        "next_problem_detectors": ["COMPILED_L3_POLICY_BODY_EXECUTION_UNBOUND"],
        "eligible_claim": "CANONICAL_L3_POLICY_COMPILER_BINDING_ESTABLISHED",
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
