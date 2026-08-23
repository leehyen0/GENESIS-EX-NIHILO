from __future__ import annotations

import ast
import unittest

from arte_cognition.software_selector_dataflow_path_genesis import (
    GeneratedDataflowPathSchema,
    assess_dataflow_path_inexpressivity,
    derive_dataflow_path_policy,
    discover_dataflow_path_signatures,
    generate_dataflow_path_schema_candidates,
    normalize_source_with_dataflow_path_schema,
    propose_dataflow_path_schema,
    select_authorized_dataflow_path_schema,
)
from arte_cognition.software_selector_relation_schema_genesis import (
    assess_binding_schema_inexpressivity,
    generate_binding_schema_candidates,
    normalize_source_with_binding_schema,
)
from arte_cognition.world_coupling import WorldOutcomePair
from evaluations.run_source_derived_dataflow_path_genesis import (
    main as run_external_dataflow_path,
)


TARGET = "ResidualObservation"


def _surface(alias: str, builder: str, mapping: str, task: str) -> str:
    return (
        f"{alias} = {TARGET}\n"
        f"def {builder}():\n"
        f"    return {{'heldout': True}}\n"
        f"{mapping} = {builder}()\n"
        f"value = {alias}(task_id='{task}', **{mapping})\n"
    )


def _normalized_target_call(source: str) -> bool:
    tree = ast.parse(source)
    target_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TARGET
    ]
    if len(target_calls) != 1:
        return False
    call = target_calls[0]
    return bool(
        not any(item.arg is None for item in call.keywords)
        and "heldout" in {item.arg for item in call.keywords}
    )


def _pair(experiment_id: str, context: str, cls: str, source: str) -> WorldOutcomePair:
    return WorldOutcomePair(
        pair_id=f"{experiment_id}:{context}:{cls}:{source}",
        experiment_id=experiment_id,
        axis_id="AXIS::TEST",
        source_id=source,
        context_id=context,
        challenge_id=f"challenge-{context}",
        epoch=1,
        low_outcome=0.0,
        high_outcome=1.0,
        low_value=0.0,
        high_value=1.0,
        matched_budget=True,
        externally_generated=True,
        issuer_id=f"issuer-{cls}",
        independence_class_id=cls,
        authority_verified=True,
    )


class SelectorDataflowPathGenesisTests(unittest.TestCase):
    def test_requires_repeated_inexpressivity(self):
        one = assess_dataflow_path_inexpressivity((("a", 10, 0),), min_contexts=2)
        self.assertNotEqual(one.status, "DATAFLOW_PATH_INEXPRESSIVE_OPEN_PATH_GENESIS")
        two = assess_dataflow_path_inexpressivity((("a", 10, 0), ("b", 9, 0)), min_contexts=2)
        self.assertEqual(two.status, "DATAFLOW_PATH_INEXPRESSIVE_OPEN_PATH_GENESIS")

    def test_dataflow_paths_are_identifier_invariant(self):
        first = _surface("AliasA", "builder_a", "opts_a", "x")
        second = _surface("AliasB", "builder_b", "opts_b", "y")
        paths_a = discover_dataflow_path_signatures(first, TARGET)
        paths_b = discover_dataflow_path_signatures(second, TARGET)
        self.assertEqual(paths_a, paths_b)
        self.assertEqual(
            paths_a,
            (
                "MODULE_BINDING->LOCAL_ZEROARG_CALL->FUNCTION_RETURN->VALUE:Dict->CONSUMER:Call.keywords**",
                "MODULE_NAME_ALIAS->TARGET_CALLABLE->CONSUMER:Call.func",
            ),
        )

    def test_merged_static_relation_schema_is_inexpressive_but_path_schema_succeeds(self):
        first = _surface("AliasA", "builder_a", "opts_a", "x")
        second = _surface("AliasB", "builder_b", "opts_b", "y")

        static_assessment = assess_binding_schema_inexpressivity(
            (("a", 10, 0), ("b", 10, 0)), min_contexts=2
        )
        static_schemas = generate_binding_schema_candidates(
            static_assessment, (first, second), TARGET
        )
        self.assertTrue(static_schemas)
        self.assertFalse(any(
            _normalized_target_call(normalize_source_with_binding_schema(first, schema, TARGET))
            for schema in static_schemas
        ))

        path_assessment = assess_dataflow_path_inexpressivity(
            (("a", 10, 0), ("b", 10, 0)), min_contexts=2
        )
        path_schemas = generate_dataflow_path_schema_candidates(
            path_assessment, (first, second), TARGET
        )
        self.assertEqual(len(path_schemas), 3)
        combined = next(schema for schema in path_schemas if len(schema.path_signatures) == 2)
        self.assertTrue(_normalized_target_call(
            normalize_source_with_dataflow_path_schema(first, combined, TARGET)
        ))
        for schema in path_schemas:
            if len(schema.path_signatures) == 1:
                self.assertFalse(_normalized_target_call(
                    normalize_source_with_dataflow_path_schema(first, schema, TARGET)
                ))

    def test_nontrivial_builder_is_retained_counterexample(self):
        source = (
            "Alias = ResidualObservation\n"
            "def build(flag):\n"
            "    if flag:\n"
            "        return {'heldout': True}\n"
            "    return {}\n"
            "opts = build(True)\n"
            "value = Alias(task_id='x', **opts)\n"
        )
        paths = discover_dataflow_path_signatures(source, TARGET)
        self.assertEqual(paths, ("MODULE_NAME_ALIAS->TARGET_CALLABLE->CONSUMER:Call.func",))

    def test_world_policy_requires_repeated_independent_support(self):
        schema = GeneratedDataflowPathSchema((
            "MODULE_BINDING->LOCAL_ZEROARG_CALL->FUNCTION_RETURN->VALUE:Dict->CONSUMER:Call.keywords**",
            "MODULE_NAME_ALIAS->TARGET_CALLABLE->CONSUMER:Call.func",
        ))
        proposal = propose_dataflow_path_schema(schema).proposal
        one = [
            _pair(proposal.experiment_id, "c1", "A", "a"),
            _pair(proposal.experiment_id, "c1", "B", "b"),
        ]
        policy = derive_dataflow_path_policy((proposal,), one, 2)
        self.assertIsNone(select_authorized_dataflow_path_schema((schema,), policy))
        two = one + [
            _pair(proposal.experiment_id, "c2", "A", "c"),
            _pair(proposal.experiment_id, "c2", "B", "d"),
        ]
        policy = derive_dataflow_path_policy((proposal,), two, 2)
        selected = select_authorized_dataflow_path_schema((schema,), policy)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.schema_id, schema.schema_id)

    def test_external_executable_dataflow_path_genesis_and_preoutcome_transfer(self):
        run_external_dataflow_path()


if __name__ == "__main__":
    unittest.main()
