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
from arte_cognition.software_call_binding_representation import derive_call_binding_schema
from arte_cognition.software_selector_relation_schema_genesis import (
    BindingSchemaOrgan,
    assess_binding_schema_inexpressivity,
    generate_binding_schema_candidates,
    propose_binding_schema,
    select_authorized_binding_schema,
    select_upstream_patch_with_binding_schema,
)
from arte_cognition.software_selector_representation_program_genesis import (
    assess_representation_program_inexpressivity,
    generate_selector_representation_programs,
    select_upstream_patch_with_representation_program,
)
from evaluations.run_call_binding_selector_representation import _reconstruct_inherited_authority
from evaluations.run_natural_repair_constructor_genesis import _authority, _execute_candidate
from evaluations.run_selector_representation_program_genesis import (
    _program_gap,
    _reconstruct_binding_representation,
)
from evaluations.run_upstream_failure_locus_genesis import (
    HISTORICAL_PATH,
    HISTORICAL_SELECTOR,
    _randomized_source,
    _search_program,
)


class _AliasLocalKwargsSurface(ast.NodeTransformer):
    def __init__(self, alias: str, mapping: str) -> None:
        self.alias = str(alias)
        self.mapping = str(mapping)
        self.changed_calls = 0
        self.changed_heldout = 0

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id != "ResidualObservation":
            return node
        node.func = ast.copy_location(ast.Name(id=self.alias, ctx=ast.Load()), node.func)
        self.changed_calls += 1
        kept = []
        found_heldout = False
        for keyword in node.keywords:
            if (
                keyword.arg == "heldout"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                found_heldout = True
            else:
                kept.append(keyword)
        if found_heldout:
            kept.append(ast.keyword(arg=None, value=ast.Name(id=self.mapping, ctx=ast.Load())))
            self.changed_heldout += 1
        node.keywords = kept
        return node


def _alias_local_kwargs_surface(source: str, alias: str, mapping: str) -> str:
    tree = ast.parse(str(source))
    transform = _AliasLocalKwargsSurface(alias, mapping)
    tree = transform.visit(tree)
    if transform.changed_calls < 1 or transform.changed_heldout < 1:
        raise AssertionError("local-binding probe found no ResidualObservation heldout=True surface")

    alias_stmt = ast.Assign(
        targets=[ast.Name(id=str(alias), ctx=ast.Store())],
        value=ast.Name(id="ResidualObservation", ctx=ast.Load()),
    )
    mapping_stmt = ast.Assign(
        targets=[ast.Name(id=str(mapping), ctx=ast.Store())],
        value=ast.Dict(keys=[ast.Constant(value="heldout")], values=[ast.Constant(value=True)]),
    )
    insert_at = 0
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            insert_at = index + 1
    tree.body.insert(insert_at, alias_stmt)
    tree.body.insert(insert_at + 1, mapping_stmt)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _old_fixed_programs():
    assessment = assess_representation_program_inexpressivity(
        (("fixed-program-gap-a", 10, 0), ("fixed-program-gap-b", 10, 0)),
        min_contexts=2,
    )
    programs = generate_selector_representation_programs(assessment, max_depth=2)
    if len(programs) != 6:
        raise AssertionError(f"expected exact inherited #87 six-program family, got {len(programs)}")
    return programs


def _assert_old_family_inexpressive(
    programs,
    learned_selector,
    learned_representation,
    row,
    call_schema,
) -> None:
    for program in programs:
        selected = select_upstream_patch_with_representation_program(
            learned_selector,
            learned_representation,
            program,
            row["frontier"],
            row["environment"].source,
            row["failure_line"],
            call_schema,
        )
        if selected is not None:
            raise AssertionError(
                f"inherited #87 fixed program unexpectedly represented local-binding surface: {program.operations}"
            )


def _train_schema_context(
    body,
    proposals,
    learned_selector,
    learned_representation,
    call_schema,
    row,
    *,
    context: str,
    signers,
    verifier,
    epoch_base: int,
):
    outcomes = []
    for index, item in enumerate(proposals):
        selected = select_upstream_patch_with_binding_schema(
            learned_selector,
            learned_representation,
            item.schema,
            row["frontier"],
            row["environment"].source,
            row["failure_line"],
            call_schema,
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
    strong = [
        item for item in outcomes
        if item[1] is not None and item[2] and min(item[2]) >= 0.9
    ]
    if len(strong) != 1:
        detail = [
            (entry[0].schema.relation_signatures, None if entry[1] is None else entry[1].candidate_id, list(entry[2]))
            for entry in outcomes
        ]
        raise AssertionError(f"source-derived schema evidence did not isolate one schema: {detail}")
    return outcomes, strong[0]


def main() -> None:
    signers, verifier = _authority()
    body = PersistentCognitiveRuntime()
    inherited = _reconstruct_inherited_authority(body, signers, verifier)
    learned_upstream = inherited["learned_program"]
    learned_selector = inherited["learned_selector"]
    learned_representation, _, _ = _reconstruct_binding_representation(
        body, inherited, signers, verifier
    )
    call_schema = derive_call_binding_schema(ResidualObservation)
    old_programs = _old_fixed_programs()

    alias_h = f"ArteObsAlias_{secrets.token_hex(3)}"
    map_h = f"arte_opts_{secrets.token_hex(3)}"
    alias_r = f"ArteObsAlias_{secrets.token_hex(3)}"
    map_r = f"arte_opts_{secrets.token_hex(3)}"
    historical_surface = _alias_local_kwargs_surface(inherited["historical_source"], alias_h, map_h)
    randomized_surface = _alias_local_kwargs_surface(inherited["random_source"], alias_r, map_r)

    historical_gap = _program_gap(
        learned_upstream,
        learned_selector,
        learned_representation,
        call_schema,
        source=historical_surface,
        path=HISTORICAL_PATH,
        selector=HISTORICAL_SELECTOR,
        context="binding-schema-gap-historical-derived",
    )
    random_gap = _program_gap(
        learned_upstream,
        learned_selector,
        learned_representation,
        call_schema,
        source=randomized_surface,
        path=inherited["random_path"],
        selector=inherited["random_selector"],
        context="binding-schema-gap-randomized",
    )
    if len(historical_gap["frontier"]) < 2 or len(random_gap["frontier"]) < 2:
        raise AssertionError("binding-schema probe lost inherited nonempty repair frontier")
    _assert_old_family_inexpressive(old_programs, learned_selector, learned_representation, historical_gap, call_schema)
    _assert_old_family_inexpressive(old_programs, learned_selector, learned_representation, random_gap, call_schema)

    historical_ids = tuple(sorted(item.candidate_id for item in historical_gap["frontier"]))
    for _ in range(16):
        repeated = _program_gap(
            learned_upstream,
            learned_selector,
            learned_representation,
            call_schema,
            source=historical_surface,
            path=HISTORICAL_PATH,
            selector=HISTORICAL_SELECTOR,
            context="binding-schema-old-more-compute",
        )
        if tuple(sorted(item.candidate_id for item in repeated["frontier"])) != historical_ids:
            raise AssertionError("OLD+MORE_COMPUTE changed inherited repair frontier")
        _assert_old_family_inexpressive(old_programs, learned_selector, learned_representation, repeated, call_schema)

    assessment = assess_binding_schema_inexpressivity(
        (
            ("source-relations-historical-derived", len(historical_gap["frontier"]), 0),
            ("source-relations-randomized", len(random_gap["frontier"]), 0),
        ),
        min_contexts=2,
    )
    schemas = generate_binding_schema_candidates(
        assessment,
        (historical_surface, randomized_surface),
        call_schema.function_name,
        max_relations=3,
    )
    if len(schemas) < 3:
        raise AssertionError(f"expected a nontrivial source-derived schema shadow, got {len(schemas)}")
    proposals = tuple(propose_binding_schema(item) for item in schemas)
    organ = BindingSchemaOrgan(body)
    organ.remember(proposals)

    _, first_strong = _train_schema_context(
        body,
        proposals,
        learned_selector,
        learned_representation,
        call_schema,
        historical_gap,
        context="binding-schema-training-historical-derived",
        signers=signers,
        verifier=verifier,
        epoch_base=91000,
    )
    if organ.policy().status == "REPRODUCED_BINDING_SCHEMA":
        raise AssertionError("one context incorrectly authorized source-derived schema")
    _, second_strong = _train_schema_context(
        body,
        proposals,
        learned_selector,
        learned_representation,
        call_schema,
        random_gap,
        context="binding-schema-training-randomized",
        signers=signers,
        verifier=verifier,
        epoch_base=92000,
    )
    if first_strong[0].schema.schema_id != second_strong[0].schema.schema_id:
        raise AssertionError("source-derived schema did not reproduce across contexts")

    policy = organ.policy()
    learned_schema = select_authorized_binding_schema(schemas, policy)
    if learned_schema is None or len(learned_schema.relation_signatures) < 2:
        raise AssertionError(f"failed to authorize composed source-derived schema: {policy}")

    checkpoint = checkpoint_dict(body)
    verifierless = restore_runtime(checkpoint)
    if select_authorized_binding_schema(schemas, BindingSchemaOrgan(verifierless).policy()) is not None:
        raise AssertionError("binding-schema authority leaked through checkpoint")
    reverified = restore_runtime(checkpoint, world_verifier=verifier)
    reverified_schema = select_authorized_binding_schema(schemas, BindingSchemaOrgan(reverified).policy())
    if reverified_schema is None or reverified_schema.schema_id != learned_schema.schema_id:
        raise AssertionError("external reverification failed to reconstruct binding-schema authority")

    token = secrets.token_hex(4)
    heldout_raw, heldout_path, heldout_selector = _randomized_source(token)
    heldout_alias = f"ArteObsAlias_{secrets.token_hex(3)}"
    heldout_map = f"arte_opts_{secrets.token_hex(3)}"
    heldout_surface = _alias_local_kwargs_surface(heldout_raw, heldout_alias, heldout_map)
    training_hashes = {
        hashlib.sha256(historical_surface.encode()).hexdigest(),
        hashlib.sha256(randomized_surface.encode()).hexdigest(),
    }
    if hashlib.sha256(heldout_surface.encode()).hexdigest() in training_hashes:
        raise AssertionError("binding-schema heldout was not source-disjoint")

    heldout_gap = _program_gap(
        learned_upstream,
        learned_selector,
        learned_representation,
        call_schema,
        source=heldout_surface,
        path=heldout_path,
        selector=heldout_selector,
        context="binding-schema-heldout",
    )
    _assert_old_family_inexpressive(old_programs, learned_selector, learned_representation, heldout_gap, call_schema)

    selected = select_upstream_patch_with_binding_schema(
        learned_selector,
        learned_representation,
        reverified_schema,
        heldout_gap["frontier"],
        heldout_surface,
        heldout_gap["failure_line"],
        call_schema,
    )
    if selected is None:
        raise AssertionError("authorized source-derived schema selected no heldout patch")

    learned_proposal = next(item for item in proposals if item.schema.schema_id == learned_schema.schema_id)
    treatment = restore_runtime(checkpoint, world_verifier=verifier)
    treatment_effects = _execute_candidate(
        treatment,
        learned_proposal,
        heldout_gap["environment"],
        selected.patched_source,
        "binding-schema-heldout-treatment",
        signers,
        verifier,
        93000,
    )
    treatment_capability = 1.0 if treatment_effects and min(treatment_effects) >= 0.9 else 0.0

    remove_body = restore_runtime(checkpoint, world_verifier=verifier)
    remove_effects = _execute_candidate(
        remove_body,
        learned_proposal,
        heldout_gap["environment"],
        heldout_surface,
        "binding-schema-heldout-remove",
        signers,
        verifier,
        94000,
    )
    remove_capability = 1.0 if remove_effects and min(remove_effects) >= 0.9 else 0.0

    wrong_schema = next(
        item for item in schemas
        if item.schema_id != learned_schema.schema_id and len(item.relation_signatures) == 1
    )
    wrong_proposal = next(item for item in proposals if item.schema.schema_id == wrong_schema.schema_id)
    wrong_selected = select_upstream_patch_with_binding_schema(
        learned_selector,
        learned_representation,
        wrong_schema,
        heldout_gap["frontier"],
        heldout_surface,
        heldout_gap["failure_line"],
        call_schema,
    )
    wrong_body = restore_runtime(checkpoint, world_verifier=verifier)
    wrong_effects = _execute_candidate(
        wrong_body,
        wrong_proposal,
        heldout_gap["environment"],
        heldout_surface if wrong_selected is None else wrong_selected.patched_source,
        "binding-schema-heldout-wrong",
        signers,
        verifier,
        95000,
    )
    wrong_capability = 1.0 if wrong_effects and min(wrong_effects) >= 0.9 else 0.0

    full_candidates, _, full_successes = _search_program(
        learned_upstream, heldout_gap["stderr"], heldout_gap["environment"]
    )
    if selected.candidate_id not in {item.candidate_id for item in full_successes}:
        raise AssertionError("pre-outcome schema-selected heldout patch was not genuinely successful")
    if treatment_capability != 1.0 or remove_capability != 0.0 or wrong_capability != 0.0:
        raise AssertionError("binding-schema Treatment/REMOVE/WRONG causal isolation failed")

    result = {
        "status": "PASS_BOUNDED_SOURCE_DERIVED_SELECTOR_BINDING_SCHEMA_GENESIS_AND_PRE_OUTCOME_TRANSFER",
        "repository": "leehyen0/GENESIS-EX-NIHILO",
        "parent_main": "70a7cc80026e7168b14a85a86db33fd9717d7e46",
        "old_fixed_program_candidate_count": len(old_programs),
        "old_fixed_program_selected_counts": [0, 0, 0],
        "old_more_compute_attempts": 16,
        "old_more_compute_selected_count": 0,
        "source_relation_derived_candidate_vocabulary": True,
        "candidate_schema_generation_uses_world_outcomes": False,
        "generated_schema_count": len(schemas),
        "learned_schema_id": learned_schema.schema_id,
        "learned_relation_signatures": list(learned_schema.relation_signatures),
        "one_context_insufficient_for_authority": True,
        "supporting_contexts": list(policy.supporting_contexts),
        "verifierless_schema_authority": False,
        "schema_policy_rederived_after_external_reverification": True,
        "heldout_post_checkout_randomized": True,
        "heldout_source_disjoint_from_training": True,
        "heldout_candidate_outcomes_exposed_before_selection": False,
        "heldout_full_candidate_count": len(full_candidates),
        "heldout_schema_selected_candidate_count": 1,
        "treatment_external_pair_count": len(treatment_effects),
        "treatment_capability": treatment_capability,
        "remove_external_pair_count": len(remove_effects),
        "remove_same_checkpoint_capability": remove_capability,
        "wrong_external_pair_count": len(wrong_effects),
        "wrong_schema_relations": list(wrong_schema.relation_signatures),
        "wrong_capability": wrong_capability,
        "relation_signature_metalanguage_human_authored": True,
        "static_binding_interpreter_human_authored": True,
        "max_relation_count_human_authored": True,
        "natural_historical_new_failure": False,
        "unrestricted_representation_language_genesis": False,
        "unrestricted_operator_genesis": False,
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
