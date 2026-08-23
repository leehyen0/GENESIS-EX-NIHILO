from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.software_upstream_failure_locus_genesis import (
    UpstreamFailureProgramOrgan,
    generate_upstream_failure_programs,
    generate_upstream_patch_candidates,
    propose_upstream_failure_program,
    select_authorized_upstream_failure_program,
)
from arte_cognition.software_upstream_patch_discrimination import (
    UpstreamPatchSelectorOrgan,
    generate_upstream_patch_selectors,
    propose_upstream_patch_selector,
    select_authorized_upstream_patch_selector,
    select_upstream_patch,
)
from evaluations.run_natural_repair_constructor_genesis import _authority, _execute_candidate
from evaluations.run_upstream_failure_locus_genesis import (
    HISTORICAL_BLOB,
    HISTORICAL_FIXTURE,
    HISTORICAL_PATH,
    HISTORICAL_SELECTOR,
    MinimalUpstreamEnvironment,
    _baseline_environment,
    _git_blob_sha,
    _randomized_source,
    _search_program,
    _train_context,
)


def _selector_training_context(
    body,
    selector_proposals,
    learned_program,
    *,
    source: str,
    path: str,
    selector: str,
    context: str,
    signers,
    verifier,
    epoch_base: int,
):
    _, stderr, failure_line = _baseline_environment(source, path, selector)
    environment = MinimalUpstreamEnvironment(source, path, selector, failure_line)
    upstream_candidates = generate_upstream_patch_candidates(
        learned_program, stderr, source, path, max_candidates=256
    )
    if not upstream_candidates:
        raise AssertionError("authorized upstream program produced no selector frontier")

    rows = []
    for index, item in enumerate(selector_proposals):
        selected = select_upstream_patch(
            item.selector, upstream_candidates, source, failure_line
        )
        if selected is None:
            rows.append((item, None, ()))
            continue
        # Selection occurs before any candidate execution outcome. Two independent
        # verifier classes then execute the single selected patch.
        effects = _execute_candidate(
            body,
            item,
            environment,
            selected.patched_source,
            context,
            signers,
            verifier,
            epoch_base + index * 10,
        )
        rows.append((item, selected, effects))
    strong = [row for row in rows if row[1] is not None and row[2] and min(row[2]) >= 0.9]
    if len(strong) != 1:
        detail = [
            (
                row[0].selector.selection_rule,
                None if row[1] is None else row[1].candidate_id,
                list(row[2]),
            )
            for row in rows
        ]
        raise AssertionError(f"world evidence did not isolate one upstream patch selector: {detail}")
    return {
        "stderr": stderr,
        "failure_line": failure_line,
        "environment": environment,
        "upstream_candidates": upstream_candidates,
        "selector": strong[0][0].selector,
        "selected_candidate": strong[0][1],
        "rows": rows,
    }


def main() -> None:
    historical_source = HISTORICAL_FIXTURE.read_text(encoding="utf-8")
    if _git_blob_sha(historical_source) != HISTORICAL_BLOB:
        raise AssertionError("natural historical source diverged from frozen Git blob")

    signers, verifier = _authority()
    body = PersistentCognitiveRuntime()

    # Reconstruct the merged #84 upstream-program authority from world evidence.
    programs = generate_upstream_failure_programs()
    program_proposals = tuple(propose_upstream_failure_program(item) for item in programs)
    program_organ = UpstreamFailureProgramOrgan(body)
    program_organ.remember(program_proposals)

    natural_program = _train_context(
        body,
        program_proposals,
        source=historical_source,
        path=HISTORICAL_PATH,
        selector=HISTORICAL_SELECTOR,
        context="selector-natural-program-base",
        signers=signers,
        verifier=verifier,
        epoch_base=61000,
    )
    train_token = secrets.token_hex(4)
    train_source, train_path, train_selector = _randomized_source(train_token)
    randomized_program = _train_context(
        body,
        program_proposals,
        source=train_source,
        path=train_path,
        selector=train_selector,
        context="selector-randomized-program-base",
        signers=signers,
        verifier=verifier,
        epoch_base=62000,
    )
    if natural_program["program"].program_id != randomized_program["program"].program_id:
        raise AssertionError("upstream program authority did not reconstruct consistently")
    learned_program = select_authorized_upstream_failure_program(programs, program_organ.policy())
    if learned_program is None:
        raise AssertionError("upstream program failed to become authoritative before selector learning")

    selectors = generate_upstream_patch_selectors()
    selector_proposals = tuple(propose_upstream_patch_selector(item) for item in selectors)
    selector_organ = UpstreamPatchSelectorOrgan(body)
    selector_organ.remember(selector_proposals)

    natural = _selector_training_context(
        body,
        selector_proposals,
        learned_program,
        source=historical_source,
        path=HISTORICAL_PATH,
        selector=HISTORICAL_SELECTOR,
        context="natural-upstream-selector",
        signers=signers,
        verifier=verifier,
        epoch_base=63000,
    )
    if selector_organ.policy().status == "REPRODUCED_UPSTREAM_PATCH_SELECTOR":
        raise AssertionError("one natural selector context incorrectly created authority")

    randomized = _selector_training_context(
        body,
        selector_proposals,
        learned_program,
        source=train_source,
        path=train_path,
        selector=train_selector,
        context="randomized-upstream-selector",
        signers=signers,
        verifier=verifier,
        epoch_base=64000,
    )
    if natural["selector"].selector_id != randomized["selector"].selector_id:
        raise AssertionError("natural and randomized worlds selected different patch discriminators")

    selector_policy = selector_organ.policy()
    if (
        selector_policy.status != "REPRODUCED_UPSTREAM_PATCH_SELECTOR"
        or selector_policy.selector_id != natural["selector"].selector_id
        or set(selector_policy.supporting_contexts) != {
            "natural-upstream-selector", "randomized-upstream-selector"
        }
    ):
        raise AssertionError(f"selector policy failed world authorization: {selector_policy}")

    checkpoint = checkpoint_dict(body)
    verifierless = restore_runtime(checkpoint)
    verifierless_policy = UpstreamPatchSelectorOrgan(verifierless).policy()
    if verifierless_policy.status == "REPRODUCED_UPSTREAM_PATCH_SELECTOR":
        raise AssertionError("patch-selector authority leaked through checkpoint")
    if select_authorized_upstream_patch_selector(selectors, verifierless_policy) is not None:
        raise AssertionError("verifierless descendant selected an upstream patch selector")

    reverified = restore_runtime(checkpoint, world_verifier=verifier)
    reverified_policy = UpstreamPatchSelectorOrgan(reverified).policy()
    learned_selector = select_authorized_upstream_patch_selector(selectors, reverified_policy)
    if reverified_policy != selector_policy or learned_selector is None:
        raise AssertionError("external reverification failed to reconstruct selector authority")

    # Fresh randomized heldout. Generate the 10-patch frontier, apply learned selector,
    # and freeze the one candidate BEFORE executing any heldout candidate outcome.
    heldout_token = secrets.token_hex(4)
    while heldout_token == train_token:
        heldout_token = secrets.token_hex(4)
    heldout_source, heldout_path, heldout_selector = _randomized_source(heldout_token)
    if hashlib.sha256(heldout_source.encode()).hexdigest() == hashlib.sha256(train_source.encode()).hexdigest():
        raise AssertionError("randomized selector heldout was not source-disjoint")
    _, heldout_stderr, heldout_failure_line = _baseline_environment(
        heldout_source, heldout_path, heldout_selector
    )
    heldout_environment = MinimalUpstreamEnvironment(
        heldout_source, heldout_path, heldout_selector, heldout_failure_line
    )
    heldout_frontier = generate_upstream_patch_candidates(
        learned_program, heldout_stderr, heldout_source, heldout_path, max_candidates=256
    )
    selected_before_world = select_upstream_patch(
        learned_selector, heldout_frontier, heldout_source, heldout_failure_line
    )
    if selected_before_world is None:
        raise AssertionError("learned selector produced no heldout candidate before world execution")

    learned_selector_proposal = next(
        item for item in selector_proposals if item.selector.selector_id == learned_selector.selector_id
    )
    treatment = restore_runtime(checkpoint, world_verifier=verifier)
    treatment_effects = _execute_candidate(
        treatment,
        learned_selector_proposal,
        heldout_environment,
        selected_before_world.patched_source,
        "heldout-upstream-selector-confirmation",
        signers,
        verifier,
        65000,
    )
    treatment_capability = 1.0 if min(treatment_effects) >= 0.9 else 0.0

    # Only after the one-candidate treatment is frozen and executed do we enumerate
    # world outcomes for the full inherited #84 frontier, strictly as evaluator audit.
    full_candidates, full_effects, full_successes = _search_program(
        learned_program, heldout_stderr, heldout_environment
    )
    if selected_before_world.candidate_id not in {item.candidate_id for item in full_successes}:
        raise AssertionError("pre-world selector did not choose a genuinely successful heldout patch")

    # Same-checkpoint REMOVE: selector application disabled, one candidate budget is
    # spent on unchanged source. The selector evidence remains fossilized in BODY but
    # is causally removed from behavior.
    remove = restore_runtime(checkpoint, world_verifier=verifier)
    del remove
    remove_effects = tuple(heldout_environment.run()[0] for _ in range(len(treatment_effects)))
    remove_capability = max(remove_effects) if remove_effects else 0.0

    wrong_selector = next(item for item in selectors if item.selector_id != learned_selector.selector_id)
    wrong_candidate = select_upstream_patch(
        wrong_selector, heldout_frontier, heldout_source, heldout_failure_line
    )
    if wrong_candidate is None:
        raise AssertionError("wrong selector produced no matched one-candidate control")
    wrong_proposal = next(
        item for item in selector_proposals if item.selector.selector_id == wrong_selector.selector_id
    )
    wrong_body = restore_runtime(checkpoint, world_verifier=verifier)
    wrong_effects = _execute_candidate(
        wrong_body,
        wrong_proposal,
        heldout_environment,
        wrong_candidate.patched_source,
        "heldout-upstream-selector-wrong",
        signers,
        verifier,
        66000,
    )
    wrong_capability = 1.0 if min(wrong_effects) >= 0.9 else 0.0

    if treatment_capability != 1.0 or remove_capability != 0.0 or wrong_capability != 0.0:
        raise AssertionError("upstream selector Treatment/REMOVE/WRONG causal isolation failed")

    result = {
        "status": "PASS_BOUNDED_WORLD_AUTHORIZED_UPSTREAM_PATCH_SELECTOR_AND_PRE_OUTCOME_SOURCE_DISJOINT_TRANSFER",
        "repository": "leehyen0/GENESIS-EX-NIHILO",
        "parent_main": "658400289763e1a32a47cabb6c5c0c43d1e60755",
        "historical_fixture_exact_git_blob": True,
        "inherited_upstream_program_id": learned_program.program_id,
        "inherited_upstream_edit_operator": learned_program.edit_operator,
        "generated_selector_count": len(selectors),
        "learned_selector_id": learned_selector.selector_id,
        "learned_selection_rule": learned_selector.selection_rule,
        "selector_supporting_contexts": list(selector_policy.supporting_contexts),
        "one_natural_context_insufficient_for_selector_authority": True,
        "selector_candidate_generation_uses_world_outcomes": False,
        "selector_uses_literal_feature_or_outcome_values": False,
        "selector_uses_later_human_fix": False,
        "selector_ast_relation_grammar_human_authored": True,
        "natural_frontier_count": len(natural["upstream_candidates"]),
        "natural_selected_candidate_id": natural["selected_candidate"].candidate_id,
        "randomized_training_post_checkout": True,
        "randomized_training_source_hash": hashlib.sha256(train_source.encode()).hexdigest(),
        "randomized_training_frontier_count": len(randomized["upstream_candidates"]),
        "verifierless_selector_authority": False,
        "selector_policy_rederived_after_external_reverification": True,
        "heldout_post_checkout_randomized": True,
        "heldout_source_disjoint_from_training": True,
        "heldout_source_hash": hashlib.sha256(heldout_source.encode()).hexdigest(),
        "heldout_full_candidate_count": len(full_candidates),
        "heldout_full_successful_candidate_count": len(full_successes),
        "heldout_selector_candidate_count": 1,
        "heldout_selected_candidate_id": selected_before_world.candidate_id,
        "heldout_candidate_outcomes_exposed_before_selection": False,
        "selected_candidate_verified_successful_after_freeze": True,
        "candidate_reduction_vs_full": 1.0 - (1.0 / max(1, len(full_candidates))),
        "treatment_external_pair_count": len(treatment_effects),
        "treatment_capability": treatment_capability,
        "remove_definition": "same checkpoint and inherited upstream program; disable learned selector application",
        "remove_candidate_count": 0,
        "remove_external_execution_count": len(remove_effects),
        "remove_same_checkpoint_capability": remove_capability,
        "wrong_selector_id": wrong_selector.selector_id,
        "wrong_selection_rule": wrong_selector.selection_rule,
        "wrong_candidate_count": 1,
        "wrong_external_pair_count": len(wrong_effects),
        "wrong_capability": wrong_capability,
        "heldout_world_search_needed_for_action_selection": False,
        "training_world_outcomes_used_to_select_rule": True,
        "unrestricted_semantic_selector_genesis": False,
        "unrestricted_failure_locus_genesis": False,
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
