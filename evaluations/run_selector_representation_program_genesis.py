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
    UpstreamSelectorRepresentationOrgan,
    assess_selector_representation_inexpressivity,
    derive_call_binding_schema,
    generate_selector_representations,
    propose_selector_representation,
    select_authorized_selector_representation,
    select_upstream_patch_with_representation,
)
from arte_cognition.software_selector_representation_program_genesis import (
    OP_EXPAND_LITERAL_KWARGS,
    OP_EXPAND_LITERAL_STARARGS,
    OP_RESOLVE_LOCAL_CALL_ALIAS,
    SelectorRepresentationProgramOrgan,
    assess_representation_program_inexpressivity,
    generate_selector_representation_programs,
    propose_selector_representation_program,
    select_authorized_selector_representation_program,
    select_upstream_patch_with_representation_program,
)
from evaluations.run_call_binding_selector_representation import (
    _inexpressivity_context,
    _reconstruct_inherited_authority,
    _syntax_shift_heldout_binding,
    _train_representation_context,
)
from evaluations.run_natural_repair_constructor_genesis import _authority, _execute_candidate
from evaluations.run_upstream_failure_locus_genesis import (
    HISTORICAL_PATH,
    HISTORICAL_SELECTOR,
    _randomized_source,
    _search_program,
)


class _AliasAndKwargsSurface(ast.NodeTransformer):
    def __init__(self, alias: str) -> None:
        self.alias = str(alias)
        self.changed_calls = 0
        self.changed_heldout = 0

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id != "ResidualObservation":
            return node
        node.func = ast.copy_location(ast.Name(id=self.alias, ctx=ast.Load()), node.func)
        self.changed_calls += 1
        kept = []
        heldout_value = None
        for keyword in node.keywords:
            if (
                keyword.arg == "heldout"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                heldout_value = keyword.value
            else:
                kept.append(keyword)
        if heldout_value is not None:
            mapping = ast.Dict(
                keys=[ast.Constant(value="heldout")],
                values=[ast.Constant(value=True)],
            )
            kept.append(ast.keyword(arg=None, value=mapping))
            self.changed_heldout += 1
        node.keywords = kept
        return node


def _alias_kwargs_surface(source: str, alias: str) -> str:
    tree = ast.parse(str(source))
    transform = _AliasAndKwargsSurface(alias)
    tree = transform.visit(tree)
    if transform.changed_calls < 1 or transform.changed_heldout < 1:
        raise AssertionError("alias+kwargs probe found no target ResidualObservation/heldout surface")
    assignment = ast.Assign(
        targets=[ast.Name(id=str(alias), ctx=ast.Store())],
        value=ast.Name(id="ResidualObservation", ctx=ast.Load()),
    )
    insert_at = 0
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            insert_at = index + 1
    tree.body.insert(insert_at, assignment)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _reconstruct_binding_representation(body, inherited, signers, verifier):
    learned_program = inherited["learned_program"]
    learned_selector = inherited["learned_selector"]
    shifted_historical = _syntax_shift_heldout_binding(inherited["historical_source"])
    shifted_random = _syntax_shift_heldout_binding(inherited["random_source"])
    historical_gap = _inexpressivity_context(
        learned_program,
        learned_selector,
        shifted_historical,
        HISTORICAL_PATH,
        HISTORICAL_SELECTOR,
        "program-parent-binding-historical",
    )
    random_gap = _inexpressivity_context(
        learned_program,
        learned_selector,
        shifted_random,
        inherited["random_path"],
        inherited["random_selector"],
        "program-parent-binding-random",
    )
    assessment = assess_selector_representation_inexpressivity(
        (historical_gap["context"], random_gap["context"]), min_contexts=2
    )
    representations = generate_selector_representations(assessment)
    proposals = tuple(propose_selector_representation(item) for item in representations)
    organ = UpstreamSelectorRepresentationOrgan(body)
    organ.remember(proposals)
    _train_representation_context(
        body,
        proposals,
        learned_selector,
        derive_call_binding_schema(ResidualObservation),
        historical_gap,
        context="program-parent-binding-authority-historical",
        signers=signers,
        verifier=verifier,
        epoch_base=81000,
    )
    _train_representation_context(
        body,
        proposals,
        learned_selector,
        derive_call_binding_schema(ResidualObservation),
        random_gap,
        context="program-parent-binding-authority-random",
        signers=signers,
        verifier=verifier,
        epoch_base=82000,
    )
    policy = organ.policy()
    learned = select_authorized_selector_representation(representations, policy)
    if learned is None or learned.mode != MODE_SIGNATURE_FALSE_DEFAULT_TRUE:
        raise AssertionError(f"failed to reconstruct inherited #86 representation: {policy}")
    wrong = next(item for item in representations if item.representation_id != learned.representation_id)
    return learned, wrong, representations


def _program_gap(
    inherited_program,
    inherited_selector,
    inherited_representation,
    schema,
    *,
    source: str,
    path: str,
    selector: str,
    context: str,
):
    row = _inexpressivity_context(
        inherited_program,
        inherited_selector,
        source,
        path,
        selector,
        context,
    )
    old_selected = select_upstream_patch_with_representation(
        inherited_selector,
        inherited_representation,
        row["frontier"],
        source,
        row["failure_line"],
        schema,
    )
    row["binding_selected"] = old_selected
    return row


def _train_program_context(
    body,
    proposals,
    inherited_selector,
    inherited_representation,
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
        selected = select_upstream_patch_with_representation_program(
            inherited_selector,
            inherited_representation,
            item.program,
            row["frontier"],
            row["environment"].source,
            row["failure_line"],
            schema,
        )
        patched_source = row["environment"].source if selected is None else selected.patched_source
        effects = _execute_candidate(
            body,
            item,
            row["environment"],
            patched_source,
            context,
            signers,
            verifier,
            epoch_base + index * 10,
        )
        outcomes.append((item, selected, effects))
    strong = [item for item in outcomes if item[1] is not None and item[2] and min(item[2]) >= 0.9]
    if len(strong) != 1:
        detail = [
            (item[0].program.operations, None if item[1] is None else item[1].candidate_id, list(item[2]))
            for item in outcomes
        ]
        raise AssertionError(f"program world evidence did not isolate one composed program: {detail}")
    return {"rows": outcomes, "strong": strong[0]}


def main() -> None:
    signers, verifier = _authority()
    body = PersistentCognitiveRuntime()
    inherited = _reconstruct_inherited_authority(body, signers, verifier)
    learned_upstream = inherited["learned_program"]
    learned_selector = inherited["learned_selector"]
    learned_representation, wrong_representation, parent_representations = _reconstruct_binding_representation(
        body, inherited, signers, verifier
    )
    schema = derive_call_binding_schema(ResidualObservation)

    alias_h = f"ArteObsAlias_{secrets.token_hex(3)}"
    alias_r = f"ArteObsAlias_{secrets.token_hex(3)}"
    historical_surface = _alias_kwargs_surface(inherited["historical_source"], alias_h)
    randomized_surface = _alias_kwargs_surface(inherited["random_source"], alias_r)

    historical_gap = _program_gap(
        learned_upstream,
        learned_selector,
        learned_representation,
        schema,
        source=historical_surface,
        path=HISTORICAL_PATH,
        selector=HISTORICAL_SELECTOR,
        context="program-gap-historical-derived",
    )
    random_gap = _program_gap(
        learned_upstream,
        learned_selector,
        learned_representation,
        schema,
        source=randomized_surface,
        path=inherited["random_path"],
        selector=inherited["random_selector"],
        context="program-gap-randomized",
    )
    if historical_gap["binding_selected"] is not None or random_gap["binding_selected"] is not None:
        raise AssertionError("#86 binding representation unexpectedly survived alias+kwargs surface")
    if len(historical_gap["frontier"]) < 2 or len(random_gap["frontier"]) < 2:
        raise AssertionError("representation-program probe lost inherited repair frontier")

    old_ids = tuple(sorted(item.candidate_id for item in historical_gap["frontier"]))
    for _ in range(16):
        repeated = _program_gap(
            learned_upstream,
            learned_selector,
            learned_representation,
            schema,
            source=historical_surface,
            path=HISTORICAL_PATH,
            selector=HISTORICAL_SELECTOR,
            context="program-gap-old-more-compute",
        )
        if tuple(sorted(item.candidate_id for item in repeated["frontier"])) != old_ids:
            raise AssertionError("OLD+MORE_COMPUTE changed inherited alias+kwargs frontier")
        if repeated["binding_selected"] is not None:
            raise AssertionError("OLD+MORE_COMPUTE unexpectedly made #86 representation expressive")

    assessment = assess_representation_program_inexpressivity(
        (
            ("alias-kwargs-historical-derived", len(historical_gap["frontier"]), 0),
            ("alias-kwargs-randomized", len(random_gap["frontier"]), 0),
        ),
        min_contexts=2,
    )
    programs = generate_selector_representation_programs(assessment, max_depth=2)
    if len(programs) != 6:
        raise AssertionError(f"expected six bounded primitive-composition programs, got {len(programs)}")
    proposals = tuple(propose_selector_representation_program(item) for item in programs)
    organ = SelectorRepresentationProgramOrgan(body)
    organ.remember(proposals)

    first = _train_program_context(
        body,
        proposals,
        learned_selector,
        learned_representation,
        schema,
        historical_gap,
        context="selector-program-training-historical-derived",
        signers=signers,
        verifier=verifier,
        epoch_base=83000,
    )
    if organ.policy().status == "REPRODUCED_SELECTOR_REPRESENTATION_PROGRAM":
        raise AssertionError("one context incorrectly authorized representation program")
    second = _train_program_context(
        body,
        proposals,
        learned_selector,
        learned_representation,
        schema,
        random_gap,
        context="selector-program-training-randomized",
        signers=signers,
        verifier=verifier,
        epoch_base=84000,
    )
    first_program = first["strong"][0].program
    second_program = second["strong"][0].program
    if first_program.program_id != second_program.program_id:
        raise AssertionError("composed program support did not reproduce across source-disjoint contexts")
    expected_ops = (OP_RESOLVE_LOCAL_CALL_ALIAS, OP_EXPAND_LITERAL_KWARGS)
    if first_program.operations != expected_ops:
        raise AssertionError(f"unexpected learned program operations: {first_program.operations}")

    policy = organ.policy()
    learned_program = select_authorized_selector_representation_program(programs, policy)
    if learned_program is None or learned_program.operations != expected_ops:
        raise AssertionError(f"composed selector representation program failed authorization: {policy}")

    checkpoint = checkpoint_dict(body)
    verifierless = restore_runtime(checkpoint)
    if select_authorized_selector_representation_program(
        programs, SelectorRepresentationProgramOrgan(verifierless).policy()
    ) is not None:
        raise AssertionError("representation-program authority leaked through checkpoint")
    reverified = restore_runtime(checkpoint, world_verifier=verifier)
    reverified_policy = SelectorRepresentationProgramOrgan(reverified).policy()
    reverified_program = select_authorized_selector_representation_program(programs, reverified_policy)
    if reverified_program is None or reverified_program.program_id != learned_program.program_id:
        raise AssertionError("external reverification failed to reconstruct program authority")

    heldout_token = secrets.token_hex(4)
    heldout_raw, heldout_path, heldout_selector = _randomized_source(heldout_token)
    heldout_alias = f"ArteObsAlias_{secrets.token_hex(3)}"
    heldout_surface = _alias_kwargs_surface(heldout_raw, heldout_alias)
    training_hashes = {
        hashlib.sha256(historical_surface.encode()).hexdigest(),
        hashlib.sha256(randomized_surface.encode()).hexdigest(),
    }
    if hashlib.sha256(heldout_surface.encode()).hexdigest() in training_hashes:
        raise AssertionError("fresh program heldout was not source-disjoint")
    heldout_gap = _program_gap(
        learned_upstream,
        learned_selector,
        learned_representation,
        schema,
        source=heldout_surface,
        path=heldout_path,
        selector=heldout_selector,
        context="selector-program-heldout",
    )
    if heldout_gap["binding_selected"] is not None:
        raise AssertionError("old #86 representation unexpectedly selected alias+kwargs heldout")

    selected_before_world = select_upstream_patch_with_representation_program(
        learned_selector,
        learned_representation,
        reverified_program,
        heldout_gap["frontier"],
        heldout_surface,
        heldout_gap["failure_line"],
        schema,
    )
    if selected_before_world is None:
        raise AssertionError("authorized composed program selected no heldout patch")

    learned_proposal = next(item for item in proposals if item.program.program_id == learned_program.program_id)
    treatment = restore_runtime(checkpoint, world_verifier=verifier)
    treatment_effects = _execute_candidate(
        treatment,
        learned_proposal,
        heldout_gap["environment"],
        selected_before_world.patched_source,
        "selector-program-heldout-treatment",
        signers,
        verifier,
        85000,
    )
    treatment_capability = 1.0 if treatment_effects and min(treatment_effects) >= 0.9 else 0.0

    full_candidates, _, full_successes = _search_program(
        learned_upstream, heldout_gap["stderr"], heldout_gap["environment"]
    )
    if selected_before_world.candidate_id not in {item.candidate_id for item in full_successes}:
        raise AssertionError("pre-outcome program selection was not genuinely successful")

    remove_effects = tuple(heldout_gap["environment"].run()[0] for _ in range(len(treatment_effects)))
    remove_capability = max(remove_effects) if remove_effects else 0.0

    wrong_program = next(
        item for item in programs
        if len(item.operations) == 2
        and item.operations != learned_program.operations
        and OP_EXPAND_LITERAL_STARARGS in item.operations
    )
    wrong_selected = select_upstream_patch_with_representation_program(
        learned_selector,
        learned_representation,
        wrong_program,
        heldout_gap["frontier"],
        heldout_surface,
        heldout_gap["failure_line"],
        schema,
    )
    wrong_effects = tuple(heldout_gap["environment"].run()[0] for _ in range(len(treatment_effects)))
    if wrong_selected is not None:
        wrong_proposal = next(item for item in proposals if item.program.program_id == wrong_program.program_id)
        wrong_body = restore_runtime(checkpoint, world_verifier=verifier)
        wrong_effects = _execute_candidate(
            wrong_body,
            wrong_proposal,
            heldout_gap["environment"],
            wrong_selected.patched_source,
            "selector-program-heldout-wrong-program",
            signers,
            verifier,
            86000,
        )
    wrong_capability = 1.0 if wrong_effects and min(wrong_effects) >= 0.9 else 0.0

    wrong_rep_selected = select_upstream_patch_with_representation_program(
        learned_selector,
        wrong_representation,
        learned_program,
        heldout_gap["frontier"],
        heldout_surface,
        heldout_gap["failure_line"],
        schema,
    )
    if wrong_rep_selected is None:
        raise AssertionError("wrong-representation swap failed to produce matched one-candidate control")
    wrong_rep_body = restore_runtime(checkpoint, world_verifier=verifier)
    wrong_rep_effects = _execute_candidate(
        wrong_rep_body,
        learned_proposal,
        heldout_gap["environment"],
        wrong_rep_selected.patched_source,
        "selector-program-heldout-wrong-representation",
        signers,
        verifier,
        87000,
    )
    wrong_rep_capability = 1.0 if wrong_rep_effects and min(wrong_rep_effects) >= 0.9 else 0.0

    if (
        treatment_capability != 1.0
        or remove_capability != 0.0
        or wrong_capability != 0.0
        or wrong_rep_capability != 0.0
    ):
        raise AssertionError("selector representation program Treatment/REMOVE/WRONG causal isolation failed")

    result = {
        "status": "PASS_BOUNDED_COMPOSITIONAL_SELECTOR_REPRESENTATION_PROGRAM_GENESIS_AND_PRE_OUTCOME_TRANSFER",
        "repository": "leehyen0/GENESIS-EX-NIHILO",
        "parent_main": "303dc4a54dcb27747f017febca00aea17445d947",
        "inherited_upstream_program_id": learned_upstream.program_id,
        "inherited_selector_id": learned_selector.selector_id,
        "inherited_representation_id": learned_representation.representation_id,
        "inherited_representation_mode": learned_representation.mode,
        "representation_program_inexpressivity_not_candidate_refutation": True,
        "training_old_frontier_counts": [len(historical_gap["frontier"]), len(random_gap["frontier"])],
        "training_old_representation_selected_counts": [0, 0],
        "old_more_compute_attempts": 16,
        "old_more_compute_frontier_identical": True,
        "old_more_compute_selected_candidate_count": 0,
        "primitive_alphabet": [
            OP_RESOLVE_LOCAL_CALL_ALIAS,
            OP_EXPAND_LITERAL_KWARGS,
            OP_EXPAND_LITERAL_STARARGS,
        ],
        "primitive_alphabet_human_authored": True,
        "program_max_depth": 2,
        "program_depth_bound_human_authored": True,
        "generated_program_count": len(programs),
        "candidate_program_generation_uses_world_outcomes": False,
        "learned_program_id": learned_program.program_id,
        "learned_program_operations": list(learned_program.operations),
        "learned_program_is_composition_of_smaller_primitives": True,
        "named_complete_alias_kwargs_mode_authored": False,
        "one_context_insufficient_for_program_authority": True,
        "program_supporting_contexts": list(policy.supporting_contexts),
        "verifierless_program_authority": False,
        "program_policy_rederived_after_external_reverification": True,
        "historical_source_derived_syntax_counterfactual": True,
        "natural_historical_new_failure": False,
        "alias_names_fresh": True,
        "heldout_post_checkout_randomized": True,
        "heldout_source_disjoint_from_training": True,
        "heldout_old_representation_selected_candidate_count": 0,
        "heldout_full_candidate_count": len(full_candidates),
        "heldout_full_successful_candidate_count": len(full_successes),
        "heldout_program_selected_candidate_count": 1,
        "heldout_candidate_outcomes_exposed_before_selection": False,
        "heldout_world_search_needed_for_action_selection": False,
        "candidate_reduction_vs_full": 1.0 - (1.0 / max(1, len(full_candidates))),
        "treatment_external_pair_count": len(treatment_effects),
        "treatment_capability": treatment_capability,
        "remove_definition": "same checkpoint and inherited upstream/selector/#86 representation; remove generated normalizer program",
        "remove_external_execution_count": len(remove_effects),
        "remove_same_checkpoint_capability": remove_capability,
        "wrong_program_operations": list(wrong_program.operations),
        "wrong_program_selected_candidate_count": 0 if wrong_selected is None else 1,
        "wrong_program_external_execution_count": len(wrong_effects),
        "wrong_program_capability": wrong_capability,
        "wrong_representation_mode": wrong_representation.mode,
        "wrong_representation_selected_candidate_count": 1,
        "wrong_representation_external_pair_count": len(wrong_rep_effects),
        "wrong_representation_capability": wrong_rep_capability,
        "unrestricted_representation_language_genesis": False,
        "unrestricted_selector_operator_genesis": False,
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
