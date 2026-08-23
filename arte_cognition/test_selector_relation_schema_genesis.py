from __future__ import annotations

import ast
import unittest

from arte_cognition.software_selector_relation_schema_genesis import (
    GeneratedBindingSchema,
    assess_binding_schema_inexpressivity,
    derive_binding_schema_policy,
    discover_binding_relation_signatures,
    generate_binding_schema_candidates,
    normalize_source_with_binding_schema,
    propose_binding_schema,
    select_authorized_binding_schema,
)
from arte_cognition.software_selector_representation_program_genesis import (
    assess_representation_program_inexpressivity,
    generate_selector_representation_programs,
    normalize_source_with_representation_program,
)
from arte_cognition.world_coupling import WorldOutcomePair
from evaluations.run_source_derived_binding_schema_genesis import main as run_external_binding_schema


TARGET = "ResidualObservation"


def _surface(alias: str, mapping: str, task: str) -> str:
    return (
        f"{alias} = {TARGET}\n"
        f"{mapping} = {{'heldout': True}}\n"
        f"value = {alias}(task_id='{task}', **{mapping})\n"
    )


def _pair(experiment_id: str, context: str, cls: str, *, effect: float = 1.0, source: str = "src"):
    return WorldOutcomePair(
        pair_id=f"{experiment_id}:{context}:{cls}:{source}",
        experiment_id=experiment_id,
        axis_id="AXIS::TEST",
        source_id=source,
        context_id=context,
        challenge_id=f"challenge-{context}",
        epoch=1,
        low_outcome=0.0,
        high_outcome=float(effect),
        low_value=0.0,
        high_value=1.0,
        matched_budget=True,
        externally_generated=True,
        issuer_id=f"issuer-{cls}",
        independence_class_id=cls,
        authority_verified=True,
    )


def _normalized_target_call(source: str) -> bool:
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    if len(calls) != 1:
        return False
    call = calls[0]
    return bool(
        isinstance(call.func, ast.Name)
        and call.func.id == TARGET
        and not any(item.arg is None for item in call.keywords)
        and "heldout" in {item.arg for item in call.keywords}
    )


class SelectorRelationSchemaGenesisTests(unittest.TestCase):
    def test_requires_repeated_nonempty_inexpressivity(self):
        one = assess_binding_schema_inexpressivity((("a", 10, 0),), min_contexts=2)
        self.assertNotEqual(one.status, "BINDING_SCHEMA_INEXPRESSIVE_OPEN_RELATION_GENESIS")
        two = assess_binding_schema_inexpressivity((("a", 10, 0), ("b", 7, 0)), min_contexts=2)
        self.assertEqual(two.status, "BINDING_SCHEMA_INEXPRESSIVE_OPEN_RELATION_GENESIS")

    def test_relations_are_source_derived_and_identifier_invariant(self):
        first = _surface("AliasA", "opts_a", "x")
        second = _surface("AliasB", "opts_b", "y")
        first_rel = discover_binding_relation_signatures(first, TARGET)
        second_rel = discover_binding_relation_signatures(second, TARGET)
        self.assertEqual(first_rel, second_rel)
        self.assertEqual(first_rel, ("Dict->Call.keywords**", "Name->Call.func"))

        assessment = assess_binding_schema_inexpressivity((("a", 4, 0), ("b", 5, 0)))
        schemas = generate_binding_schema_candidates(assessment, (first, second), TARGET)
        self.assertEqual(len(schemas), 3)
        self.assertEqual(
            {item.relation_signatures for item in schemas},
            {
                ("Dict->Call.keywords**",),
                ("Name->Call.func",),
                ("Dict->Call.keywords**", "Name->Call.func"),
            },
        )

    def test_current_fixed_program_family_is_inexpressive_on_local_binding_surface(self):
        source = _surface("FreshAlias", "fresh_opts", "heldout")
        old_assessment = assess_representation_program_inexpressivity(
            (("old-a", 10, 0), ("old-b", 10, 0)), min_contexts=2
        )
        old_programs = generate_selector_representation_programs(old_assessment, max_depth=2)
        self.assertEqual(len(old_programs), 6)
        self.assertFalse(any(
            _normalized_target_call(normalize_source_with_representation_program(source, program, TARGET))
            for program in old_programs
        ))

        new_schema = GeneratedBindingSchema(("Dict->Call.keywords**", "Name->Call.func"))
        self.assertTrue(_normalized_target_call(
            normalize_source_with_binding_schema(source, new_schema, TARGET)
        ))

    def test_composed_generated_schema_normalizes_new_local_bindings(self):
        source = _surface("FreshAlias", "fresh_opts", "heldout")
        schema = GeneratedBindingSchema(("Dict->Call.keywords**", "Name->Call.func"))
        normalized = normalize_source_with_binding_schema(source, schema, TARGET)
        self.assertTrue(_normalized_target_call(normalized))

        alias_only = normalize_source_with_binding_schema(
            source, GeneratedBindingSchema(("Name->Call.func",)), TARGET
        )
        self.assertIn("**fresh_opts", alias_only)
        kwargs_only = normalize_source_with_binding_schema(
            source, GeneratedBindingSchema(("Dict->Call.keywords**",)), TARGET
        )
        self.assertIn("FreshAlias(", kwargs_only)

    def test_dynamic_kwargs_are_not_invented_as_static_relation(self):
        source = (
            "Alias = ResidualObservation\n"
            "def build():\n    return {'heldout': True}\n"
            "opts = build()\n"
            "value = Alias(task_id='x', **opts)\n"
        )
        relations = discover_binding_relation_signatures(source, TARGET)
        self.assertEqual(relations, ("Name->Call.func",))

    def test_external_policy_requires_two_contexts_and_independence_classes(self):
        schema = GeneratedBindingSchema(("Dict->Call.keywords**", "Name->Call.func"))
        proposal = propose_binding_schema(schema).proposal

        one_context = [
            _pair(proposal.experiment_id, "c1", "A"),
            _pair(proposal.experiment_id, "c1", "B"),
        ]
        policy = derive_binding_schema_policy((proposal,), one_context, min_independent_classes=2)
        self.assertIsNone(select_authorized_binding_schema((schema,), policy))

        repeated = one_context + [
            _pair(proposal.experiment_id, "c2", "A", source="src2a"),
            _pair(proposal.experiment_id, "c2", "B", source="src2b"),
        ]
        policy = derive_binding_schema_policy((proposal,), repeated, min_independent_classes=2)
        chosen = select_authorized_binding_schema((schema,), policy)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.schema_id, schema.schema_id)
        self.assertEqual(policy.supporting_contexts, ("c1", "c2"))

        collapsed = [
            _pair(proposal.experiment_id, "c1", "A", source="one"),
            _pair(proposal.experiment_id, "c1", "A", source="two"),
            _pair(proposal.experiment_id, "c2", "A", source="three"),
            _pair(proposal.experiment_id, "c2", "A", source="four"),
        ]
        collapsed_policy = derive_binding_schema_policy((proposal,), collapsed, min_independent_classes=2)
        self.assertIsNone(select_authorized_binding_schema((schema,), collapsed_policy))

    def test_external_executable_schema_genesis_and_preoutcome_transfer(self):
        run_external_binding_schema()


if __name__ == "__main__":
    unittest.main()
