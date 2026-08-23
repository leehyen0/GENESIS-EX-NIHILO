from __future__ import annotations

import ast
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
from arte_cognition.semantic_genesis import ResidualObservation
from arte_cognition.software_call_binding_representation import (
    MODE_SIGNATURE_FALSE_DEFAULT_TRUE,
    SelectorRepresentationContext,
    UpstreamSelectorRepresentationOrgan,
    assess_selector_representation_inexpressivity,
    derive_call_binding_schema,
    generate_selector_representations,
    propose_selector_representation,
    select_authorized_selector_representation,
    select_upstream_patch_with_representation,
)
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
from evaluations.run_natural_repair_constructor_genesis import _authority, _execute_candidate, _git_blob_sha
from evaluations.run_upstream_failure_locus_genesis import (
    HISTORICAL_BLOB,
    HISTORICAL_FIXTURE,
    HISTORICAL_PATH,
    HISTORICAL_SELECTOR,
    MinimalUpstreamEnvironment,
    _baseline_environment,
    _randomized_source,
    _search_program,
    _train_context,
)
from evaluations.run_upstream_patch_semantic_selection import _selector_training_context


class _HeldoutKeywordToPositional(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = 0

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id != "ResidualObservation":
            return node
        heldout = None
        kept = []
        for keyword in node.keywords:
            if keyword.arg == "heldout" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                heldout = keyword.value
            else:
                kept.append(keyword)
        if heldout is None:
            return node
        # ResidualObservation(residual_id, features, outcome, source_class='OBSERVATION', heldout=False)
        # Encode the same binding through positional syntax. This is an evaluator-created
        # syntax-equivalent counterfactual, not a claimed natural historical commit.
        if len(node.args) != 3:
            raise AssertionError("syntax-shift probe expected three positional ResidualObservation arguments")
        node.args = list(node.args) + [ast.Constant(value="OBSERVATION"), ast.Constant(value=True)]
        node.keywords = kept
        self.changed += 1
        return node


def _syntax_shift_heldout_binding(source: str) -> str:
    tree = ast.parse(source)
    transformer = _HeldoutKeywordToPositional()
    shifted = transformer.visit(tree)
    ast.fix_missing_locations(shifted)
    if transformer.changed != 1:
        raise AssertionError(f"expected exactly one heldout syntax shift, got {transformer.changed}")
    return ast.unparse(shifted) + "\n"


def _reconstruct_inherited_authority(body, signers, verifier):
    historical_source = HISTORICAL_FIXTURE.read_text(encoding="utf-8")
    if _git_blob_sha(historical_source) != HISTORICAL_BLOB:
        raise AssertionError("historical parent fixture diverged from exact Git blob")

    programs = generate_upstream_failure_programs()
    program_proposals = tuple(propose_upstream_failure_program(item) for item in programs)
    program_organ = UpstreamFailureProgramOrgan(body)
    program_organ.remember(program_proposals)
    natural_program = _train_context(
        body, program_proposals,
        source=historical_source, path=HISTORICAL_PATH, selector=HISTORICAL_SELECTOR,
        context="binding-parent-natural-program", signers=signers, verifier=verifier,
        epoch_base=71000,
    )
    token = secrets.token_hex(4)
    random_source, random_path, random_selector = _randomized_source(token)
    random_program = _train_context(
        body, program_proposals,
        source=random_source, path=random_path, selector=random_selector,
        context="binding-parent-random-program", signers=signers, verifier=verifier,
        epoch_base=72000,
    )
    if natural_program["program"].program_id != random_program["program"].program_id:
        raise AssertionError("inherited upstream program did not reconstruct consistently")
    learned_program = select_authorized_upstream_failure_program(programs, program_organ.policy())
    if learned_program is None:
        raise AssertionError("failed to reconstruct inherited upstream program authority")

    selectors = generate_upstream_patch_selectors()
    selector_proposals = tuple(propose_upstream_patch_selector(item) for item in selectors)
    selector_organ = UpstreamPatchSelectorOrgan(body)
    selector_organ.remember(selector_proposals)
    natural_selector = _selector_training_context(
        body, selector_proposals, learned_program,
        source=historical_source, path=HISTORICAL_PATH, selector=HISTORICAL_SELECTOR,
        context="binding-parent-natural-selector", signers=signers, verifier=verifier,
        epoch_base=73000,
    )
    random_selector_row = _selector_training_context(
        body, selector_proposals, learned_program,
        source=random_source, path=random_path, selector=random_selector,
        context="binding-parent-random-selector", signers=signers, verifier=verifier,
        epoch_base=74000,
    )
    if natural_selector["selector"].selector_id != random_selector_row["selector"].selector_id:
        raise AssertionError("inherited selector did not reconstruct consistently")
    learned_selector = select_authorized_upstream_patch_selector(selectors, selector_organ.policy())
    if learned_selector is None:
        raise AssertionError("failed to reconstruct inherited selector authority")
    return {
        "historical_source": historical_source,
        "random_source": random_source,
        "random_path": random_path,
        "random_selector": random_selector,
        "learned_program": learned_program,
        "learned_selector": learned_selector,
        "selector_proposals": selector_proposals,
    }


def _inexpressivity_context(learned_program, learned_selector, source: str, path: str, selector: str, context: str):
    _, stderr, failure_line = _baseline_environment(source, path, selector)
    environment = MinimalUpstreamEnvironment(source, path, selector, failure_line)
    frontier = generate_upstream_patch_candidates(
        learned_program, stderr, source, path, max_candidates=256
    )
    if not frontier:
        raise AssertionError("representation probe lost inherited repair frontier")
    old_selected = select_upstream_patch(
        learned_selector, frontier, source, failure_line
    )
    return {
        "context": SelectorRepresentationContext(
            context_id=context,
            inherited_frontier_count=len(frontier),
            old_selected_candidate_count=0 if old_selected is None else 1,
        ),
        "stderr": stderr,
        "failure_line": failure_line,
        "environment": environment,
        "frontier": frontier,
        "old_selected": old_selected,
    }


def _train_representation_context(
    body,
    proposals,
    learned_selector,
    schema,
    row,
    *,
    context: str,
    signers,
    verifier,
    epoch_base: int,
):
    outcomes = []
    for index, item in enumerate(proposals):
        selected = select_upstream_patch_with_representation(
            learned_selector,
            item.representation,
            row["frontier"],
            row["environment"].source,
            row["failure_line"],
            schema,
        )
        if selected is None:
            outcomes.append((item, None, ()))
            continue
        effects = _execute_candidate(
            body, item, row["environment"], selected.patched_source,
            context, signers, verifier, epoch_base + index * 10,
        )
        outcomes.append((item, selected, effects))
    strong = [item for item in outcomes if item[1] is not None and item[2] and min(item[2]) >= 0.9]
    if len(strong) != 1:
        detail = [
            (item[0].representation.mode, None if item[1] is None else item[1].candidate_id, list(item[2]))
            for item in outcomes
        ]
        raise AssertionError(f"representation world evidence did not isolate one candidate: {detail}")
    return {"rows": outcomes, "strong": strong[0]}


def main() -> None:
    signers, verifier = _authority()
    body = PersistentCognitiveRuntime()
    inherited = _reconstruct_inherited_authority(body, signers, verifier)
    learned_program = inherited["learned_program"]
    learned_selector = inherited["learned_selector"]

    # Create a syntax-equivalent counterfactual from the exact historical source and
    # a source-disjoint randomized context. Both retain the 10-candidate repair frontier,
    # but #85's surface keyword marker cannot denote any candidate.
    shifted_historical = _syntax_shift_heldout_binding(inherited["historical_source"])
    shifted_random = _syntax_shift_heldout_binding(inherited["random_source"])
    historical_gap = _inexpressivity_context(
        learned_program, learned_selector,
        shifted_historical, HISTORICAL_PATH, HISTORICAL_SELECTOR,
        "syntax-shifted-historical-derived",
    )
    random_gap = _inexpressivity_context(
        learned_program, learned_selector,
        shifted_random, inherited["random_path"], inherited["random_selector"],
        "syntax-shifted-randomized",
    )
    if historical_gap["old_selected"] is not None or random_gap["old_selected"] is not None:
        raise AssertionError("#85 surface selector unexpectedly survived positional-binding syntax shift")

    old_frontier_hashes = tuple(sorted(item.candidate_id for item in historical_gap["frontier"]))
    for _ in range(16):
        repeated = _inexpressivity_context(
            learned_program, learned_selector,
            shifted_historical, HISTORICAL_PATH, HISTORICAL_SELECTOR,
            "repeat-not-authority",
        )
        if repeated["old_selected"] is not None:
            raise AssertionError("OLD+MORE_COMPUTE unexpectedly created surface selector applicability")
        if tuple(sorted(item.candidate_id for item in repeated["frontier"])) != old_frontier_hashes:
            raise AssertionError("OLD+MORE_COMPUTE changed inherited repair frontier")

    assessment = assess_selector_representation_inexpressivity(
        (historical_gap["context"], random_gap["context"]), min_contexts=2
    )
    if assessment.status != "SELECTOR_REPRESENTATION_INEXPRESSIVE_OPEN_BINDING":
        raise AssertionError(f"representation escape did not open: {assessment}")

    schema = derive_call_binding_schema(ResidualObservation)
    if not schema.false_default_parameters:
        raise AssertionError("call signature exposed no false-default semantic marker")
    representations = generate_selector_representations(assessment)
    proposals = tuple(propose_selector_representation(item) for item in representations)
    organ = UpstreamSelectorRepresentationOrgan(body)
    organ.remember(proposals)

    first = _train_representation_context(
        body, proposals, learned_selector, schema, historical_gap,
        context="binding-representation-historical-derived", signers=signers,
        verifier=verifier, epoch_base=75000,
    )
    if organ.policy().status == "REPRODUCED_SELECTOR_REPRESENTATION":
        raise AssertionError("one representation context incorrectly created authority")
    second = _train_representation_context(
        body, proposals, learned_selector, schema, random_gap,
        context="binding-representation-randomized", signers=signers,
        verifier=verifier, epoch_base=76000,
    )
    if first["strong"][0].representation.representation_id != second["strong"][0].representation.representation_id:
        raise AssertionError("representation support did not reproduce across source-disjoint contexts")

    policy = organ.policy()
    learned_representation = select_authorized_selector_representation(representations, policy)
    if learned_representation is None or learned_representation.mode != MODE_SIGNATURE_FALSE_DEFAULT_TRUE:
        raise AssertionError(f"call-binding representation failed authorization: {policy}")

    checkpoint = checkpoint_dict(body)
    verifierless = restore_runtime(checkpoint)
    verifierless_policy = UpstreamSelectorRepresentationOrgan(verifierless).policy()
    if select_authorized_selector_representation(representations, verifierless_policy) is not None:
        raise AssertionError("selector representation authority leaked through checkpoint")
    reverified = restore_runtime(checkpoint, world_verifier=verifier)
    reverified_policy = UpstreamSelectorRepresentationOrgan(reverified).policy()
    reverified_representation = select_authorized_selector_representation(representations, reverified_policy)
    if reverified_policy != policy or reverified_representation is None:
        raise AssertionError("external reverification failed to reconstruct representation authority")

    # Compatibility: call-binding normalization must preserve #85's keyword-surface choice.
    _, keyword_stderr, keyword_failure_line = _baseline_environment(
        inherited["historical_source"], HISTORICAL_PATH, HISTORICAL_SELECTOR
    )
    keyword_frontier = generate_upstream_patch_candidates(
        learned_program, keyword_stderr, inherited["historical_source"], HISTORICAL_PATH, max_candidates=256
    )
    keyword_old = select_upstream_patch(
        learned_selector, keyword_frontier, inherited["historical_source"], keyword_failure_line
    )
    keyword_new = select_upstream_patch_with_representation(
        learned_selector, learned_representation, keyword_frontier,
        inherited["historical_source"], keyword_failure_line, schema
    )
    if keyword_old is None or keyword_new is None or keyword_old.candidate_id != keyword_new.candidate_id:
        raise AssertionError("binding normalization regressed the inherited keyword-surface selector")

    # Fresh positional-binding heldout. The inherited #85 selector remains inapplicable;
    # the learned representation freezes one candidate before heldout candidate outcomes.
    heldout_token = secrets.token_hex(4)
    heldout_source_raw, heldout_path, heldout_selector = _randomized_source(heldout_token)
    heldout_source = _syntax_shift_heldout_binding(heldout_source_raw)
    if hashlib.sha256(heldout_source.encode()).hexdigest() in {
        hashlib.sha256(shifted_historical.encode()).hexdigest(),
        hashlib.sha256(shifted_random.encode()).hexdigest(),
    }:
        raise AssertionError("fresh representation heldout was not source-disjoint")
    heldout_gap = _inexpressivity_context(
        learned_program, learned_selector, heldout_source, heldout_path, heldout_selector,
        "binding-representation-heldout",
    )
    if heldout_gap["old_selected"] is not None:
        raise AssertionError("old surface selector unexpectedly selected heldout before representation")
    selected_before_world = select_upstream_patch_with_representation(
        learned_selector, reverified_representation, heldout_gap["frontier"],
        heldout_source, heldout_gap["failure_line"], schema
    )
    if selected_before_world is None:
        raise AssertionError("authorized binding representation selected no heldout patch")

    learned_proposal = next(
        item for item in proposals if item.representation.representation_id == learned_representation.representation_id
    )
    treatment = restore_runtime(checkpoint, world_verifier=verifier)
    treatment_effects = _execute_candidate(
        treatment, learned_proposal, heldout_gap["environment"], selected_before_world.patched_source,
        "binding-heldout-treatment", signers, verifier, 77000,
    )
    treatment_capability = 1.0 if min(treatment_effects) >= 0.9 else 0.0

    # Full frontier is evaluated only after the treatment was frozen/executed, for audit.
    full_candidates, _, full_successes = _search_program(
        learned_program, heldout_gap["stderr"], heldout_gap["environment"]
    )
    if selected_before_world.candidate_id not in {item.candidate_id for item in full_successes}:
        raise AssertionError("binding-normalized pre-world selection was not genuinely successful")

    # REMOVE: same checkpoint, learned selector/program retained, representation application removed.
    remove_effects = tuple(heldout_gap["environment"].run()[0] for _ in range(len(treatment_effects)))
    remove_capability = max(remove_effects) if remove_effects else 0.0

    wrong_representation = next(
        item for item in representations if item.representation_id != learned_representation.representation_id
    )
    wrong_candidate = select_upstream_patch_with_representation(
        learned_selector, wrong_representation, heldout_gap["frontier"],
        heldout_source, heldout_gap["failure_line"], schema
    )
    if wrong_candidate is None:
        raise AssertionError("wrong representation failed to produce matched one-candidate control")
    wrong_proposal = next(
        item for item in proposals if item.representation.representation_id == wrong_representation.representation_id
    )
    wrong_body = restore_runtime(checkpoint, world_verifier=verifier)
    wrong_effects = _execute_candidate(
        wrong_body, wrong_proposal, heldout_gap["environment"], wrong_candidate.patched_source,
        "binding-heldout-wrong", signers, verifier, 78000,
    )
    wrong_capability = 1.0 if min(wrong_effects) >= 0.9 else 0.0

    if treatment_capability != 1.0 or remove_capability != 0.0 or wrong_capability != 0.0:
        raise AssertionError("binding representation Treatment/REMOVE/WRONG causal isolation failed")

    result = {
        "status": "PASS_BOUNDED_SELECTOR_CALL_BINDING_REPRESENTATION_ESCAPE_AND_PRE_OUTCOME_TRANSFER",
        "repository": "leehyen0/GENESIS-EX-NIHILO",
        "parent_main": "d72d000b8dc6f6ad7f1d37fcc593ad0071fee7bd",
        "historical_parent_fixture_exact_git_blob": True,
        "historical_source_derived_syntax_counterfactual": True,
        "natural_historical_new_failure": False,
        "syntax_shift_semantics": "heldout=True keyword -> source_class default plus heldout True positional binding",
        "inherited_upstream_program_id": learned_program.program_id,
        "inherited_selector_id": learned_selector.selector_id,
        "inherited_selector_rule": learned_selector.selection_rule,
        "representation_inexpressivity_not_candidate_refutation": True,
        "inexpressive_contexts": list(assessment.inexpressive_contexts),
        "old_frontier_counts": [
            historical_gap["context"].inherited_frontier_count,
            random_gap["context"].inherited_frontier_count,
        ],
        "old_selected_candidate_counts": [0, 0],
        "old_more_compute_attempts": 16,
        "old_more_compute_selected_candidate_count": 0,
        "old_more_compute_frontier_identical": True,
        "call_binding_schema_function": schema.function_name,
        "call_binding_false_default_parameter_count": len(schema.false_default_parameters),
        "call_binding_schema_derived_from_runtime_signature": True,
        "generated_representation_count": len(representations),
        "representation_candidate_generation_uses_world_outcomes": False,
        "learned_representation_id": learned_representation.representation_id,
        "learned_representation_mode": learned_representation.mode,
        "representation_supporting_contexts": list(policy.supporting_contexts),
        "one_context_insufficient_for_representation_authority": True,
        "verifierless_representation_authority": False,
        "representation_policy_rederived_after_external_reverification": True,
        "keyword_surface_backward_compatible_candidate_identity": True,
        "heldout_post_checkout_randomized": True,
        "heldout_source_disjoint_from_training": True,
        "heldout_old_selector_candidate_count": 0,
        "heldout_full_candidate_count": len(full_candidates),
        "heldout_full_successful_candidate_count": len(full_successes),
        "heldout_binding_selected_candidate_count": 1,
        "heldout_candidate_outcomes_exposed_before_selection": False,
        "heldout_world_search_needed_for_action_selection": False,
        "candidate_reduction_vs_full": 1.0 - (1.0 / max(1, len(full_candidates))),
        "treatment_external_pair_count": len(treatment_effects),
        "treatment_capability": treatment_capability,
        "remove_definition": "same checkpoint and inherited selector/program; remove call-binding representation application",
        "remove_external_execution_count": len(remove_effects),
        "remove_same_checkpoint_capability": remove_capability,
        "wrong_representation_mode": wrong_representation.mode,
        "wrong_candidate_count": 1,
        "wrong_external_pair_count": len(wrong_effects),
        "wrong_capability": wrong_capability,
        "binding_normalizer_human_authored": True,
        "representation_candidate_modes_human_authored": True,
        "unrestricted_representation_language_genesis": False,
        "unrestricted_semantic_selector_genesis": False,
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
