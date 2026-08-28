from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.software_repair_class_genesis import (
    FixedRepairClassContextResult,
    GeneratedRepairClassOrgan,
    assess_fixed_repair_class_frontier,
    generate_repair_class_from_failure,
    generate_repair_mechanisms,
)
from evaluations.run_natural_repair_class_genesis import (
    TRAINING_CONTEXTS,
    TRAINING_FIXTURE,
    TRAINING_PATH,
    HistoricalExecutionEnvironment,
    _execute_fixed_content_context,
    _execute_generated_class_context,
    _git_blob_sha,
    _make_authority,
    _repository_python_paths,
    _run_candidate_set,
    existing_content_candidates,
)


NATURAL_COMMIT = "fb9bb533d9c70452a16e3fbc24f3a35ce19a43ee"
NATURAL_FIX_COMMIT = "fae7966316207a01ccbee591ec7890b22b4ab57f"
NATURAL_BLOB = "34f335bd8229404f701208d9c6e2b3e373998732"
NATURAL_PATH = "evaluations/run_compositional_transform_generator_genesis.py"
NATURAL_FIXTURE = (
    ROOT
    / "evaluations/fixtures/historical_fb9bb533/run_compositional_transform_generator_genesis.py"
)


def _reconstruct_generated_class_authority():
    source = TRAINING_FIXTURE.read_text(encoding="utf-8")
    repository_paths = _repository_python_paths()
    signers, verifier = _make_authority()
    body = PersistentCognitiveRuntime()
    content = existing_content_candidates(source, "third-natural-transfer-training")
    fixed_results = []
    failures = {}
    for index, (context_id, seed) in enumerate(TRAINING_CONTEXTS.items()):
        environment = HistoricalExecutionEnvironment(source, TRAINING_PATH, "SCRIPT", seed=seed)
        baseline, _, stderr = environment.run()
        if baseline != 0.0 or "ModuleNotFoundError" not in stderr:
            raise AssertionError(f"training import failure no longer reproduces: {stderr[-500:]}")
        failures[context_id] = stderr
        strong = _execute_fixed_content_context(
            body, environment, content, context_id, signers, verifier, 110000 + index * 10000
        )
        if strong:
            raise AssertionError("old CONTENT class unexpectedly repaired training import failure")
        fixed_results.append(FixedRepairClassContextResult(
            context_id=context_id,
            applicable_class_ids=("CONTENT",),
            evaluated_candidate_count=len(content),
            missing_candidate_count=0,
            capability=0.0,
        ))
    frontier = assess_fixed_repair_class_frontier(fixed_results, min_contexts=2)
    candidates = generate_repair_class_from_failure(
        stderr=failures[next(iter(TRAINING_CONTEXTS))],
        target_path=TRAINING_PATH,
        repository_paths=repository_paths,
        frontier=frontier,
    )
    if len(candidates) != 1:
        raise AssertionError(f"failed to reconstruct generated repair class: {len(candidates)}")
    generated_class = candidates[0]
    counter = {}
    training_success_sets = []
    for index, (context_id, seed) in enumerate(TRAINING_CONTEXTS.items()):
        environment = HistoricalExecutionEnvironment(source, TRAINING_PATH, "SCRIPT", seed=seed)
        mechanisms = generate_repair_mechanisms(
            generated_class.phenotype.class_id,
            source,
            TRAINING_PATH,
            failures[context_id],
            repository_paths,
        )
        outcomes = {item.mechanism_id: environment.run(item.patched_source)[0] for item in mechanisms}
        success = tuple(sorted(key for key, value in outcomes.items() if value == 1.0))
        training_success_sets.append(success)
        effects = _execute_generated_class_context(
            body,
            generated_class,
            environment,
            mechanisms,
            context_id,
            signers,
            verifier,
            150000 + index * 10000,
            counter,
        )
        if min(effects) < 0.9:
            raise AssertionError("generated class lost training world support")
    policy = GeneratedRepairClassOrgan(body).policy()
    if policy.class_id != generated_class.phenotype.class_id:
        raise AssertionError(f"generated class authority reconstruction failed: {policy}")
    return body, generated_class, repository_paths, signers, verifier, tuple(training_success_sets)


def main() -> None:
    natural_source = NATURAL_FIXTURE.read_text(encoding="utf-8")
    if _git_blob_sha(natural_source) != NATURAL_BLOB:
        raise AssertionError("third natural fixture diverged from exact historical Git blob")

    body, generated_class, repository_paths, signers, verifier, training_success_sets = (
        _reconstruct_generated_class_authority()
    )
    checkpoint = checkpoint_dict(body)
    verifierless = restore_runtime(checkpoint)
    if GeneratedRepairClassOrgan(verifierless).policy().class_id is not None:
        raise AssertionError("generated repair-class authority leaked through checkpoint")
    descendant = restore_runtime(checkpoint, world_verifier=verifier)
    policy = GeneratedRepairClassOrgan(descendant).policy()
    if policy.class_id != generated_class.phenotype.class_id:
        raise AssertionError("external reverification failed to reconstruct generated class")

    environment = HistoricalExecutionEnvironment(
        natural_source,
        NATURAL_PATH,
        "SCRIPT",
        seed=9283711,
    )
    baseline, _, stderr = environment.run()
    if baseline != 0.0 or "ModuleNotFoundError" not in stderr:
        raise AssertionError(f"unused natural transform-evaluator failure did not reproduce: {stderr[-700:]}")

    mechanisms = generate_repair_mechanisms(
        policy.class_id,
        natural_source,
        NATURAL_PATH,
        stderr,
        repository_paths,
    )
    if not mechanisms:
        raise AssertionError("inherited generated class produced no repair mechanism on third natural source")
    mechanism_outcomes = {
        item.mechanism_id: environment.run(item.patched_source)[0] for item in mechanisms
    }
    successes = tuple(sorted(key for key, value in mechanism_outcomes.items() if value == 1.0))
    if successes != ("SEARCH_CONTEXT::FILE_PARENT_DEPTH::1::PREPEND",):
        raise AssertionError(f"unexpected third-natural success set: {successes}")

    treatment_capability, treatment_count = _run_candidate_set(environment, mechanisms)
    if treatment_capability != 1.0:
        raise AssertionError("inherited generated repair class failed unused natural source")

    same_budget_wrong = tuple(
        item for item in mechanisms
        if item.mechanism_id != "SEARCH_CONTEXT::FILE_PARENT_DEPTH::1::PREPEND"
    )
    wrong_scores = [environment.run(item.patched_source)[0] for item in same_budget_wrong]
    if any(score == 1.0 for score in wrong_scores):
        raise AssertionError("wrong search-context mechanism unexpectedly solved natural source")

    # No generated-class mechanism: exact historical source remains failing. This is
    # the same descendant checkpoint with class application removed, not a claim that
    # every conceivable software repair language was exhausted on this source.
    remove_capability = environment.run(natural_source)[0]
    if remove_capability != 0.0:
        raise AssertionError("REMOVE generated class unexpectedly solved natural source")

    result = {
        "status": "PASS_BOUNDED_THIRD_NATURAL_HISTORICAL_REPAIR_CLASS_TRANSFER",
        "repository": "leehyen0/GENESIS-EX-NIHILO",
        "historical_commit": NATURAL_COMMIT,
        "historical_fix_commit": NATURAL_FIX_COMMIT,
        "historical_blob": NATURAL_BLOB,
        "historical_fixture_exact_git_blob": True,
        "historical_path": NATURAL_PATH,
        "natural_historical_bug": True,
        "later_human_fix_exposed_to_body": False,
        "baseline_failure": "ModuleNotFoundError",
        "generated_class_id": generated_class.phenotype.class_id,
        "generated_class_supporting_contexts": list(policy.supporting_contexts),
        "training_success_sets": [list(item) for item in training_success_sets],
        "fresh_natural_generated_mechanism_count": len(mechanisms),
        "fresh_natural_successful_mechanisms": list(successes),
        "fresh_natural_capability": treatment_capability,
        "remove_generated_class_capability": remove_capability,
        "wrong_mechanism_count": len(same_budget_wrong),
        "wrong_mechanism_capabilities": wrong_scores,
        "source_disjoint_from_training": True,
        "same_concrete_training_mechanism_transferred": True,
        "new_repair_class_generated_on_heldout": False,
        "new_capability_frontier_gain": False,
        "evidence_interpretation": "natural transfer/retention strengthening, not a new capability-genesis event",
        "verifierless_generated_class_authority": False,
        "reverified_generated_class_authority": True,
        "class_constructor_primitives_human_authored": True,
        "unrestricted_repair_class_genesis": False,
        "foundation_weight_change": False,
        "global_recursive_acceleration": False,
        "independent_organizational_custody": False,
        "physical_world": False,
        "AGI": False,
        "ASI": False,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
