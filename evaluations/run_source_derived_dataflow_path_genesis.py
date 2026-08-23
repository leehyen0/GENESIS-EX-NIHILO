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
from arte_cognition.software_selector_dataflow_path_genesis import (
    DataflowPathOrgan,
    assess_dataflow_path_inexpressivity,
    generate_dataflow_path_schema_candidates,
    propose_dataflow_path_schema,
    select_authorized_dataflow_path_schema,
    select_upstream_patch_with_dataflow_path_schema,
)
from arte_cognition.software_selector_relation_schema_genesis import (
    assess_binding_schema_inexpressivity,
    generate_binding_schema_candidates,
    select_upstream_patch_with_binding_schema,
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


class _DynamicBuilderSurface(ast.NodeTransformer):
    def __init__(self, alias: str, builder: str, mapping: str) -> None:
        self.alias = str(alias)
        self.builder = str(builder)
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


def _dynamic_builder_surface(source: str, alias: str, builder: str, mapping: str) -> str:
    tree = ast.parse(str(source))
    transform = _DynamicBuilderSurface(alias, builder, mapping)
    tree = transform.visit(tree)
    if transform.changed_calls < 1 or transform.changed_heldout < 1:
        raise AssertionError("dynamic builder probe found no ResidualObservation heldout=True surface")

    alias_stmt = ast.Assign(
        targets=[ast.Name(id=str(alias), ctx=ast.Store())],
        value=ast.Name(id="ResidualObservation", ctx=ast.Load()),
    )
    builder_stmt = ast.FunctionDef(
        name=str(builder),
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=[ast.Return(value=ast.Dict(
            keys=[ast.Constant(value="heldout")],
            values=[ast.Constant(value=True)],
        ))],
        decorator_list=[],
    )
    mapping_stmt = ast.Assign(
        targets=[ast.Name(id=str(mapping), ctx=ast.Store())],
        value=ast.Call(func=ast.Name(id=str(builder), ctx=ast.Load()), args=[], keywords=[]),
    )
    insert_at = 0
    for index, stmt in enumerate(tree.body):
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            insert_at = index + 1
    tree.body.insert(insert_at, alias_stmt)
    tree.body.insert(insert_at + 1, builder_stmt)
    tree.body.insert(insert_at + 2, mapping_stmt)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _assert_static_schema_inexpressive(
    static_schemas,
    learned_selector,
    learned_representation,
    row,
    call_schema,
) -> None:
    for schema in static_schemas:
        selected = select_upstream_patch_with_binding_schema(
            learned_selector,
            learned_representation,
            schema,
            row["frontier"],
            row["environment"].source,
            row["failure_line"],
            call_schema,
        )
        if selected is not None:
            raise AssertionError(
                f"merged PR89 static relation schema unexpectedly solved deeper dataflow surface: {schema.relation_signatures}"
            )


def _train_context(
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
        selected = select_upstream_patch_with_dataflow_path_schema(
            learned_selector,
            learned_representation,
            item.schema,
            row["frontier"],
            row["environment"].source,
            row["failure_line"],
            call_schema,
        )
        patched = row["environment"].source if selected is None else selected.patched_source
        effects = _execute_candidate(
            body,
            item,
            row["environment"],
            patched,
            context,
            signers,
            verifier,
            epoch_base + index * 10,
        )
        outcomes.append((item, selected, effects))
    strong = [
        entry for entry in outcomes
        if entry[1] is not None and entry[2] and min(entry[2]) >= 0.9
    ]
    if len(strong) != 1:
        detail = [
            (entry[0].schema.path_signatures, None if entry[1] is None else entry[1].candidate_id, list(entry[2]))
            for entry in outcomes
        ]
        raise AssertionError(f"dataflow path evidence did not isolate one schema: {detail}")
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

    h_alias = f"ArteObsAlias_{secrets.token_hex(3)}"
    h_builder = f"build_opts_{secrets.token_hex(3)}"
    h_map = f"arte_opts_{secrets.token_hex(3)}"
    r_alias = f"ArteObsAlias_{secrets.token_hex(3)}"
    r_builder = f"build_opts_{secrets.token_hex(3)}"
    r_map = f"arte_opts_{secrets.token_hex(3)}"
    historical_surface = _dynamic_builder_surface(
        inherited["historical_source"], h_alias, h_builder, h_map
    )
    random_surface = _dynamic_builder_surface(
        inherited["random_source"], r_alias, r_builder, r_map
    )

    historical_gap = _program_gap(
        learned_upstream,
        learned_selector,
        learned_representation,
        call_schema,
        source=historical_surface,
        path=HISTORICAL_PATH,
        selector=HISTORICAL_SELECTOR,
        context="dataflow-gap-historical-derived",
    )
    random_gap = _program_gap(
        learned_upstream,
        learned_selector,
        learned_representation,
        call_schema,
        source=random_surface,
        path=inherited["random_path"],
        selector=inherited["random_selector"],
        context="dataflow-gap-randomized",
    )
    if len(historical_gap["frontier"]) < 2 or len(random_gap["frontier"]) < 2:
        raise AssertionError("deeper dataflow probe lost inherited nonempty repair frontier")

    static_assessment = assess_binding_schema_inexpressivity(
        (
            ("static-h", len(historical_gap["frontier"]), 0),
            ("static-r", len(random_gap["frontier"]), 0),
        ),
        min_contexts=2,
    )
    static_schemas = generate_binding_schema_candidates(
        static_assessment,
        (historical_surface, random_surface),
        call_schema.function_name,
        max_relations=3,
    )
    if not static_schemas:
        raise AssertionError("expected inherited PR89 static schema shadow to remain present")
    _assert_static_schema_inexpressive(
        static_schemas, learned_selector, learned_representation, historical_gap, call_schema
    )
    _assert_static_schema_inexpressive(
        static_schemas, learned_selector, learned_representation, random_gap, call_schema
    )

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
            context="dataflow-old-more-compute",
        )
        if tuple(sorted(item.candidate_id for item in repeated["frontier"])) != historical_ids:
            raise AssertionError("OLD+MORE_COMPUTE changed deeper dataflow repair frontier")
        _assert_static_schema_inexpressive(
            static_schemas, learned_selector, learned_representation, repeated, call_schema
        )

    path_assessment = assess_dataflow_path_inexpressivity(
        (
            ("path-h", len(historical_gap["frontier"]), 0),
            ("path-r", len(random_gap["frontier"]), 0),
        ),
        min_contexts=2,
    )
    schemas = generate_dataflow_path_schema_candidates(
        path_assessment,
        (historical_surface, random_surface),
        call_schema.function_name,
        max_paths=3,
    )
    if len(schemas) < 3:
        raise AssertionError(f"expected nontrivial dataflow path shadow, got {len(schemas)}")
    proposals = tuple(propose_dataflow_path_schema(schema) for schema in schemas)
    organ = DataflowPathOrgan(body)
    organ.remember(proposals)

    _, first = _train_context(
        body,
        proposals,
        learned_selector,
        learned_representation,
        call_schema,
        historical_gap,
        context="dataflow-training-historical-derived",
        signers=signers,
        verifier=verifier,
        epoch_base=101000,
    )
    if organ.policy().status == "REPRODUCED_DATAFLOW_PATH_SCHEMA":
        raise AssertionError("one context incorrectly authorized dataflow path schema")
    _, second = _train_context(
        body,
        proposals,
        learned_selector,
        learned_representation,
        call_schema,
        random_gap,
        context="dataflow-training-randomized",
        signers=signers,
        verifier=verifier,
        epoch_base=102000,
    )
    if first[0].schema.schema_id != second[0].schema.schema_id:
        raise AssertionError("dataflow path schema did not reproduce across contexts")

    policy = organ.policy()
    learned_schema = select_authorized_dataflow_path_schema(schemas, policy)
    if learned_schema is None or len(learned_schema.path_signatures) < 2:
        raise AssertionError(f"failed to authorize composed dataflow path schema: {policy}")

    checkpoint = checkpoint_dict(body)
    verifierless = restore_runtime(checkpoint)
    if select_authorized_dataflow_path_schema(schemas, DataflowPathOrgan(verifierless).policy()) is not None:
        raise AssertionError("dataflow path authority leaked through checkpoint")
    reverified = restore_runtime(checkpoint, world_verifier=verifier)
    reverified_schema = select_authorized_dataflow_path_schema(
        schemas, DataflowPathOrgan(reverified).policy()
    )
    if reverified_schema is None or reverified_schema.schema_id != learned_schema.schema_id:
        raise AssertionError("external reverification failed to reconstruct dataflow path authority")

    token = secrets.token_hex(4)
    heldout_raw, heldout_path, heldout_selector = _randomized_source(token)
    heldout_alias = f"ArteObsAlias_{secrets.token_hex(3)}"
    heldout_builder = f"build_opts_{secrets.token_hex(3)}"
    heldout_map = f"arte_opts_{secrets.token_hex(3)}"
    heldout_surface = _dynamic_builder_surface(
        heldout_raw, heldout_alias, heldout_builder, heldout_map
    )
    training_hashes = {
        hashlib.sha256(historical_surface.encode()).hexdigest(),
        hashlib.sha256(random_surface.encode()).hexdigest(),
    }
    if hashlib.sha256(heldout_surface.encode()).hexdigest() in training_hashes:
        raise AssertionError("dataflow heldout was not source-disjoint")

    heldout_gap = _program_gap(
        learned_upstream,
        learned_selector,
        learned_representation,
        call_schema,
        source=heldout_surface,
        path=heldout_path,
        selector=heldout_selector,
        context="dataflow-heldout",
    )
    heldout_static = generate_binding_schema_candidates(
        static_assessment,
        (heldout_surface, historical_surface),
        call_schema.function_name,
        max_relations=3,
    )
    _assert_static_schema_inexpressive(
        heldout_static, learned_selector, learned_representation, heldout_gap, call_schema
    )

    selected = select_upstream_patch_with_dataflow_path_schema(
        learned_selector,
        learned_representation,
        reverified_schema,
        heldout_gap["frontier"],
        heldout_surface,
        heldout_gap["failure_line"],
        call_schema,
    )
    if selected is None:
        raise AssertionError("authorized dataflow path selected no heldout patch")

    learned_proposal = next(
        item for item in proposals if item.schema.schema_id == learned_schema.schema_id
    )
    treatment = restore_runtime(checkpoint, world_verifier=verifier)
    treatment_effects = _execute_candidate(
        treatment,
        learned_proposal,
        heldout_gap["environment"],
        selected.patched_source,
        "dataflow-heldout-treatment",
        signers,
        verifier,
        103000,
    )
    treatment_capability = 1.0 if treatment_effects and min(treatment_effects) >= 0.9 else 0.0

    remove = restore_runtime(checkpoint, world_verifier=verifier)
    remove_effects = _execute_candidate(
        remove,
        learned_proposal,
        heldout_gap["environment"],
        heldout_surface,
        "dataflow-heldout-remove",
        signers,
        verifier,
        104000,
    )
    remove_capability = 1.0 if remove_effects and min(remove_effects) >= 0.9 else 0.0

    wrong_schema = next(
        schema for schema in schemas
        if schema.schema_id != learned_schema.schema_id and len(schema.path_signatures) == 1
    )
    wrong_proposal = next(item for item in proposals if item.schema.schema_id == wrong_schema.schema_id)
    wrong_selected = select_upstream_patch_with_dataflow_path_schema(
        learned_selector,
        learned_representation,
        wrong_schema,
        heldout_gap["frontier"],
        heldout_surface,
        heldout_gap["failure_line"],
        call_schema,
    )
    wrong = restore_runtime(checkpoint, world_verifier=verifier)
    wrong_effects = _execute_candidate(
        wrong,
        wrong_proposal,
        heldout_gap["environment"],
        heldout_surface if wrong_selected is None else wrong_selected.patched_source,
        "dataflow-heldout-wrong",
        signers,
        verifier,
        105000,
    )
    wrong_capability = 1.0 if wrong_effects and min(wrong_effects) >= 0.9 else 0.0

    full_candidates, _, full_successes = _search_program(
        learned_upstream, heldout_gap["stderr"], heldout_gap["environment"]
    )
    if selected.candidate_id not in {item.candidate_id for item in full_successes}:
        raise AssertionError("pre-outcome dataflow-selected patch was not genuinely successful")
    if treatment_capability != 1.0 or remove_capability != 0.0 or wrong_capability != 0.0:
        raise AssertionError("dataflow Treatment/REMOVE/WRONG causal isolation failed")

    result = {
        "status": "PASS_BOUNDED_SOURCE_DERIVED_SELECTOR_DATAFLOW_PATH_GENESIS_AND_PRE_OUTCOME_TRANSFER",
        "repository": "leehyen0/GENESIS-EX-NIHILO",
        "parent_main": "cac9d0b26067da4d2b3766cb1da93dfa2f68a761",
        "inherited_static_schema_candidate_count": len(static_schemas),
        "inherited_static_schema_selected_counts": [0, 0, 0],
        "old_more_compute_attempts": 16,
        "old_more_compute_selected_count": 0,
        "source_dataflow_path_candidate_vocabulary": True,
        "candidate_generation_uses_world_outcomes": False,
        "generated_schema_count": len(schemas),
        "learned_schema_id": learned_schema.schema_id,
        "learned_path_signatures": list(learned_schema.path_signatures),
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
        "wrong_path_signatures": list(wrong_schema.path_signatures),
        "wrong_capability": wrong_capability,
        "abstract_resolver_human_authored": True,
        "path_edge_vocabulary_human_authored": True,
        "max_resolution_depth_human_authored": True,
        "max_path_count_human_authored": True,
        "natural_historical_new_failure": False,
        "unrestricted_dataflow_language_genesis": False,
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
