from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.software_failure_extractor_program_genesis import (
    FailureExtractorProgramOrgan,
    apply_failure_extractor_program,
    generate_failure_extractor_programs,
    interpret_with_extractor_program,
    propose_failure_extractor_program,
    select_authorized_failure_extractor_program,
)
from arte_cognition.software_repair_constructor_genesis import (
    ConstructorInexpressivityContext,
    assess_constructor_inexpressivity,
    infer_relational_constructor_primitive,
    propose_relational_repair_class,
)
from evaluations.run_natural_repair_constructor_genesis import (
    FreshRepositoryEnvironment,
    _authority,
    _execute_candidate,
    _git_blob_sha,
    _repository_sources,
)

G5_COMMIT = "194dda71c313516b1eda141311ab3781c09ca5d8"
G5_FIX_COMMIT = "0610bbaa8937fe845f9c7afed525682dc226f1db"
G5_BLOB = "c40f511b8b27c94028fa553c3821db4597af4ce1"
G5_PATH = "arte_cognition/test_causal_primitive_genesis.py"
G5_FIXTURE = ROOT / "evaluations/fixtures/historical_194dda71/test_causal_primitive_genesis.py"
G5_TEST = (
    "arte_cognition.test_causal_primitive_genesis."
    "RawThresholdPrimitiveGenesisTests."
    "test_runtime_cannot_open_g5_before_g4_is_falsified"
)

G6_COMMIT = "0610bbaa8937fe845f9c7afed525682dc226f1db"
G6_FIX_COMMIT = "c540a4871fdce63b3639852ee81ecf647ada82d0"
G6_BLOB = "d5a2ff9319eb7fbf03586ac517b07400d993d747"
G6_PATH = "arte_cognition/test_causal_linear_primitive_genesis.py"
G6_FIXTURE = ROOT / "evaluations/fixtures/historical_0610bbaa/test_causal_linear_primitive_genesis.py"
G6_TEST = (
    "arte_cognition.test_causal_linear_primitive_genesis."
    "LinearFormPrimitiveGenesisTests."
    "test_runtime_cannot_open_g6_before_g5_is_falsified"
)

G7_COMMIT = "c540a4871fdce63b3639852ee81ecf647ada82d0"
G7_FIX_COMMIT = "bc03f733ba6386447a3e163053a33beb1f6e816f"
G7_BLOB = "2722ce1e5e5d2705481c76324facae75c8f655f5"
G7_PATH = "arte_cognition/test_causal_symbolic_primitive_genesis.py"
G7_FIXTURE = ROOT / "evaluations/fixtures/historical_c540a487/test_causal_symbolic_primitive_genesis.py"
G7_TEST = (
    "arte_cognition.test_causal_symbolic_primitive_genesis."
    "SymbolicPrimitiveGenesisTests."
    "test_runtime_cannot_open_symbolic_search_before_g6_falsification"
)


def _old_relation_count(stderr: str, source: str, path: str, sources) -> int:
    assessment = assess_constructor_inexpressivity((ConstructorInexpressivityContext(
        context_id="old-relation-probe",
        baseline_capability=0.0,
        old_constructor_candidate_count=0,
        failure_signature=stderr,
    ),), min_contexts=1)
    primitive = infer_relational_constructor_primitive(
        stderr=stderr,
        source=source,
        target_path=path,
        repository_sources=sources,
        assessment=assessment,
    )
    return 1 if primitive is not None else 0


def _successful_patches(program, stderr, source, path, environment):
    patches = apply_failure_extractor_program(program, stderr, source, path)
    successful = []
    for patch in patches:
        capability, _, _ = environment.run(patch.patched_source)
        if capability == 1.0:
            successful.append(patch)
    return patches, tuple(successful)


def _train_context(parent, programs, proposals, *, source, path, selector, context,
                   signers, verifier, epoch_base, sources):
    environment = FreshRepositoryEnvironment(source, path, "UNITTEST", selector=selector)
    baseline, _, stderr = environment.run()
    if baseline != 0.0 or "ValueError" not in stderr or "non-authoritative" not in stderr:
        raise AssertionError(f"natural authority-contract failure did not reproduce: {stderr[-1000:]}")
    if _old_relation_count(stderr, source, path, sources) != 0:
        raise AssertionError("merged #81 fixed relation extractor unexpectedly represented ValueError")

    rows = []
    for item in proposals:
        program = item.program
        patches, successful = _successful_patches(program, stderr, source, path, environment)
        rows.append((program, patches, successful))
        if not successful:
            continue
        effects = _execute_candidate(
            parent,
            item,
            environment,
            successful[0].patched_source,
            context,
            signers,
            verifier,
            epoch_base,
        )
        if min(effects) < 0.9:
            raise AssertionError("generated extractor program lacked replicated executable world effect")
    successful_programs = [row for row in rows if row[2]]
    if len(successful_programs) != 1:
        raise AssertionError(
            f"natural context must isolate exactly one extractor-program family; got {len(successful_programs)}"
        )
    return {
        "baseline": baseline,
        "stderr": stderr,
        "program": successful_programs[0][0],
        "candidate_count": len(successful_programs[0][1]),
        "successful_patch_count": len(successful_programs[0][2]),
    }


def main() -> None:
    g5_source = G5_FIXTURE.read_text(encoding="utf-8")
    g6_source = G6_FIXTURE.read_text(encoding="utf-8")
    g7_source = G7_FIXTURE.read_text(encoding="utf-8")
    observed_blobs = tuple(_git_blob_sha(source) for source in (g5_source, g6_source, g7_source))
    if observed_blobs != (G5_BLOB, G6_BLOB, G7_BLOB):
        raise AssertionError(f"historical extractor fixtures diverged from exact Git blobs: {observed_blobs}")

    base_sources = _repository_sources()
    programs = generate_failure_extractor_programs()
    proposals = tuple(propose_failure_extractor_program(program) for program in programs)
    if len(programs) != 2 or len({program.program_id for program in programs}) != 2:
        raise AssertionError("extractor program shadow language was not a two-program bounded search")

    signers, verifier = _authority()
    parent = PersistentCognitiveRuntime()
    organ = FailureExtractorProgramOrgan(parent)
    organ.remember(proposals)

    g5_sources = dict(base_sources)
    g5_sources[G5_PATH] = g5_source
    g5 = _train_context(
        parent, programs, proposals,
        source=g5_source, path=G5_PATH, selector=G5_TEST,
        context="natural-g5-authority-contract", signers=signers, verifier=verifier,
        epoch_base=41000, sources=g5_sources,
    )
    if organ.policy().status == "REPRODUCED_FAILURE_EXTRACTOR_PROGRAM":
        raise AssertionError("one natural context incorrectly authorized an extractor program")

    g6_sources = dict(base_sources)
    g6_sources[G6_PATH] = g6_source
    g6 = _train_context(
        parent, programs, proposals,
        source=g6_source, path=G6_PATH, selector=G6_TEST,
        context="natural-g6-authority-contract", signers=signers, verifier=verifier,
        epoch_base=42000, sources=g6_sources,
    )
    if g5["program"].program_id != g6["program"].program_id:
        raise AssertionError("natural G5/G6 did not converge on the same extractor program")

    parent_policy = organ.policy()
    if (
        parent_policy.status != "REPRODUCED_FAILURE_EXTRACTOR_PROGRAM"
        or parent_policy.program_id != g5["program"].program_id
        or set(parent_policy.supporting_contexts) != {
            "natural-g5-authority-contract", "natural-g6-authority-contract"
        }
    ):
        raise AssertionError(f"world evidence failed to authorize extractor program: {parent_policy}")

    checkpoint = checkpoint_dict(parent)
    verifierless = restore_runtime(checkpoint)
    verifierless_policy = FailureExtractorProgramOrgan(verifierless).policy()
    if verifierless_policy.status == "REPRODUCED_FAILURE_EXTRACTOR_PROGRAM":
        raise AssertionError("extractor-program authority leaked through checkpoint")
    reverified = restore_runtime(checkpoint, world_verifier=verifier)
    reverified_policy = FailureExtractorProgramOrgan(reverified).policy()
    if reverified_policy != parent_policy:
        raise AssertionError("external reverification did not reconstruct extractor-program policy")

    heldout_sources = dict(base_sources)
    heldout_sources[G7_PATH] = g7_source
    heldout_environment = FreshRepositoryEnvironment(g7_source, G7_PATH, "UNITTEST", selector=G7_TEST)
    baseline, _, heldout_stderr = heldout_environment.run()
    if baseline != 0.0 or "ValueError" not in heldout_stderr or "non-authoritative" not in heldout_stderr:
        raise AssertionError(f"natural G7 heldout failure did not reproduce: {heldout_stderr[-1000:]}")

    old_counts = tuple(
        _old_relation_count(heldout_stderr, g7_source, G7_PATH, heldout_sources)
        for _ in range(16)
    )
    if any(old_counts):
        raise AssertionError("OLD+MORE_COMPUTE escaped the merged #81 fixed relationship extractor")

    if select_authorized_failure_extractor_program(programs, verifierless_policy) is not None:
        raise AssertionError("verifierless descendant selected an extractor program")
    learned_program = select_authorized_failure_extractor_program(programs, reverified_policy)
    if learned_program is None or learned_program.program_id != parent_policy.program_id:
        raise AssertionError("reverified descendant failed to recover learned extractor program")

    interpretation = interpret_with_extractor_program(
        learned_program, heldout_stderr, g7_source, G7_PATH
    )
    if interpretation is None or interpretation.exception_family != "VALUE_ERROR":
        raise AssertionError(f"learned extractor program failed to interpret natural G7: {interpretation}")
    if "generate_world_driven_symbolic_primitive_models" in g5_source or "generate_world_driven_symbolic_primitive_models" in g6_source:
        raise AssertionError("heldout concrete target call leaked into natural training sources")

    heldout_successes = []
    for patch in interpretation.patch_candidates:
        capability, _, _ = heldout_environment.run(patch.patched_source)
        if capability == 1.0:
            heldout_successes.append(patch)
    if not heldout_successes:
        raise AssertionError("learned extractor program produced no executable G7 repair")

    # Downstream chain: generated interpretation becomes a new constructor primitive
    # and repair class. The concrete patch still comes from the learned extractor
    # program, not from the old #81 fixed exception-specific mechanism table.
    heldout_class = propose_relational_repair_class(interpretation.constructor_primitive)
    treatment = restore_runtime(checkpoint, world_verifier=verifier)
    treatment_effects = _execute_candidate(
        treatment,
        heldout_class,
        heldout_environment,
        heldout_successes[0].patched_source,
        "natural-g7-heldout",
        signers,
        verifier,
        43000,
    )
    treatment_capability = 1.0 if min(treatment_effects) >= 0.9 else 0.0

    # Same external-execution budget controls. Removing the learned extractor program
    # yields no patch candidates, so the budget is spent replaying the unchanged
    # failing source rather than silently receiving a different repair language.
    budget = max(1, len(interpretation.patch_candidates))
    remove = restore_runtime(checkpoint, world_verifier=verifier)
    remove_results = tuple(heldout_environment.run()[0] for _ in range(budget))
    remove_capability = max(remove_results)
    wrong_program = next(program for program in programs if program.program_id != learned_program.program_id)
    wrong_patches = apply_failure_extractor_program(wrong_program, heldout_stderr, g7_source, G7_PATH)
    wrong_results = tuple(heldout_environment.run()[0] for _ in range(budget)) if not wrong_patches else tuple(
        heldout_environment.run(patch.patched_source)[0] for patch in wrong_patches
    )
    wrong_capability = max(wrong_results) if wrong_results else 0.0

    if treatment_capability != 1.0 or remove_capability != 0.0 or wrong_capability != 0.0:
        raise AssertionError("Treatment/REMOVE/WRONG extractor-program causal isolation failed")

    # Add the heldout success to extractor-program lineage only after the transfer
    # already succeeded; this cannot be used to choose the program for the heldout.
    treatment_organ = FailureExtractorProgramOrgan(treatment)
    learned_proposal = next(item for item in proposals if item.program.program_id == learned_program.program_id)
    _execute_candidate(
        treatment,
        learned_proposal,
        heldout_environment,
        heldout_successes[0].patched_source,
        "natural-g7-heldout-program-confirmation",
        signers,
        verifier,
        44000,
    )
    if "natural-g7-heldout-program-confirmation" not in treatment_organ.policy().supporting_contexts:
        raise AssertionError("heldout transfer did not enter extractor-program world-evidence lineage")

    result = {
        "status": "PASS_BOUNDED_WORLD_AUTHORIZED_FAILURE_EXTRACTOR_PROGRAM_GENESIS_AND_NATURAL_G7_TRANSFER",
        "repository": "leehyen0/GENESIS-EX-NIHILO",
        "natural_training": True,
        "training_commits": [G5_COMMIT, G6_COMMIT],
        "training_fix_commits": [G5_FIX_COMMIT, G6_FIX_COMMIT],
        "training_blobs": [G5_BLOB, G6_BLOB],
        "training_fixtures_exact_git_blobs": True,
        "heldout_commit": G7_COMMIT,
        "heldout_fix_commit": G7_FIX_COMMIT,
        "heldout_blob": G7_BLOB,
        "heldout_fixture_exact_git_blob": True,
        "heldout_source_disjoint_file": True,
        "heldout_concrete_target_call_absent_from_training": True,
        "training_exception_family": "VALUE_ERROR",
        "fixed_relationship_extractor_training_candidate_counts": [0, 0],
        "old_more_compute_attempts": len(old_counts),
        "old_more_compute_total_candidates": sum(old_counts),
        "old_more_compute_capability": baseline,
        "generated_extractor_program_count": len(programs),
        "learned_program_id": learned_program.program_id,
        "learned_frame_selector": learned_program.frame_selector,
        "learned_locus_selector": learned_program.locus_selector,
        "learned_edit_enumerator": learned_program.edit_enumerator,
        "learned_program_supporting_contexts": list(parent_policy.supporting_contexts),
        "one_context_insufficient_for_program_authority": True,
        "meta_policy_rederived_after_external_reverification": True,
        "verifierless_extractor_program_authority": False,
        "training_program_candidate_counts": [g5["candidate_count"], g6["candidate_count"]],
        "training_successful_patch_counts": [g5["successful_patch_count"], g6["successful_patch_count"]],
        "heldout_exception_family": interpretation.exception_family,
        "heldout_constructor_primitive_id": interpretation.constructor_primitive.primitive_id,
        "heldout_generated_class_id": heldout_class.class_id,
        "heldout_binding_relation": interpretation.constructor_primitive.binding_relation,
        "heldout_patch_candidate_count": len(interpretation.patch_candidates),
        "heldout_successful_patch_count": len(heldout_successes),
        "treatment_candidate_count": len(interpretation.patch_candidates),
        "treatment_external_execution_budget": budget,
        "treatment_capability": treatment_capability,
        "remove_same_checkpoint_candidate_count": 0,
        "remove_same_checkpoint_external_execution_budget": budget,
        "remove_same_checkpoint_capability": remove_capability,
        "wrong_program_candidate_count": len(wrong_patches),
        "wrong_program_external_execution_budget": budget,
        "wrong_program_capability": wrong_capability,
        "fresh_heldout_entered_program_lineage_after_transfer": True,
        "candidate_generation_uses_hidden_success_outcomes": False,
        "later_human_fix_exposed_to_body": False,
        "fixed_relationship_extractor_causally_bypassed": True,
        "extractor_program_grammar_human_authored": True,
        "trace_frame_selector_alphabet_human_authored": True,
        "ast_locus_selector_alphabet_human_authored": True,
        "edit_enumerator_alphabet_human_authored": True,
        "unrestricted_extractor_operator_genesis": False,
        "unrestricted_repair_class_genesis": False,
        "foundation_weight_change": False,
        "physical_world": False,
        "independent_organizational_custody": False,
        "global_recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
