from __future__ import annotations

import unittest

from arte_cognition.meta_acceleration import apply_mutation_program
from arte_cognition.reflective_relation_genesis import (
    FieldRef,
    ReflectiveRewriteSchema,
    RelationExpression,
    apply_reflective_rewrite_schema,
    generate_reflective_rewrite_schemas,
)
from arte_cognition.relation_access_plan_genesis import (
    apply_relation_access_plan,
    compile_relation_access_plan,
)
from arte_cognition.test_reflective_relation_genesis import (
    capability,
    certificate,
    semantic_wrong_schema,
    training_examples,
    world,
)
from arte_cognition.executable_morphology import OrganKind


class RelationAccessPlanGenesisTests(unittest.TestCase):
    def schema(self):
        reflective, _ = training_examples()
        schemas = generate_reflective_rewrite_schemas(reflective)
        self.assertTrue(schemas)
        return schemas[0]

    def test_generated_schema_compiles_to_outcome_free_index_plan(self):
        schema = self.schema()
        plan = compile_relation_access_plan(schema)
        self.assertIsNotNone(plan)
        self.assertTrue(plan.generated_from_schema_structure)
        self.assertFalse(plan.current_outcomes_required_for_compilation)
        self.assertEqual(len(plan.clauses), len(schema.relations))
        self.assertTrue(all(clause.source_relation_token for clause in plan.clauses))

    def test_current_canonical_parent_reconstructs_same_plan_without_object_reuse(self):
        first_examples, _ = training_examples()
        second_examples, _ = training_examples()
        first_schema = generate_reflective_rewrite_schemas(first_examples)[0]
        second_schema = generate_reflective_rewrite_schemas(second_examples)[0]
        self.assertIsNot(first_schema, second_schema)
        self.assertEqual(first_schema.schema_id, second_schema.schema_id)
        self.assertEqual(
            tuple(relation.token() for relation in first_schema.relations),
            tuple(relation.token() for relation in second_schema.relations),
        )

        first_plan = compile_relation_access_plan(first_schema)
        second_plan = compile_relation_access_plan(second_schema)
        self.assertIsNotNone(first_plan)
        self.assertIsNotNone(second_plan)
        self.assertIsNot(first_plan, second_plan)
        self.assertEqual(first_plan.plan_id, second_plan.plan_id)
        self.assertEqual(first_plan.schema_id, first_schema.schema_id)
        self.assertEqual(second_plan.schema_id, second_schema.schema_id)
        self.assertEqual(
            tuple(clause.token() for clause in first_plan.clauses),
            tuple(clause.token() for clause in second_plan.clauses),
        )

        genome, loci, good, _ = world("canonical-parent-fresh", 9, OrganKind.MEMORY, 731.0)
        cert = certificate("canonical-parent-fresh", loci)
        scan = apply_reflective_rewrite_schema(genome, cert, second_schema)
        indexed = apply_relation_access_plan(genome, cert, second_schema, second_plan)
        self.assertIsNotNone(scan)
        self.assertIsNotNone(indexed)
        self.assertEqual(indexed.outcome_evaluations, 0)
        self.assertEqual(
            capability(apply_mutation_program(genome, scan.mutation_program), loci, good),
            1.0,
        )
        self.assertEqual(
            capability(apply_mutation_program(genome, indexed.mutation_program), loci, good),
            1.0,
        )
        indexed_work = indexed.index_build_field_reads + indexed.index_lookup_count + indexed.candidate_intersection_count
        self.assertLess(indexed_work, scan.candidate_relation_checks)

    def test_index_plan_preserves_capability_and_reduces_structural_work(self):
        schema = self.schema()
        plan = compile_relation_access_plan(schema)
        indexed_costs = []
        scan_costs = []
        for depth, kind, base in (
            (4, OrganKind.MEMORY, 101.0),
            (8, OrganKind.PERCEPTOR, 201.0),
            (16, OrganKind.GENERATOR, 301.0),
        ):
            genome, loci, good, _ = world(f"indexed-{depth}", depth, kind, base)
            cert = certificate(f"indexed-{depth}", loci)
            scan = apply_reflective_rewrite_schema(genome, cert, schema)
            indexed = apply_relation_access_plan(genome, cert, schema, plan)
            self.assertIsNotNone(scan)
            self.assertIsNotNone(indexed)
            scan_desc = apply_mutation_program(genome, scan.mutation_program)
            indexed_desc = apply_mutation_program(genome, indexed.mutation_program)
            self.assertEqual(capability(scan_desc, loci, good), 1.0)
            self.assertEqual(capability(indexed_desc, loci, good), 1.0)
            self.assertEqual(indexed.outcome_evaluations, 0)
            scan_costs.append(scan.candidate_relation_checks)
            indexed_costs.append(indexed.index_build_field_reads + indexed.index_lookup_count + indexed.candidate_intersection_count)
            self.assertLess(indexed_costs[-1], scan_costs[-1])
        self.assertGreater(scan_costs[-1] / indexed_costs[-1], scan_costs[0] / indexed_costs[0])

    def test_wrong_schema_compiles_but_loses_capability(self):
        schema = self.schema()
        wrong = semantic_wrong_schema(schema)
        wrong_plan = compile_relation_access_plan(wrong)
        self.assertIsNotNone(wrong_plan)
        genome, loci, good, bad = world("access-wrong", 8, OrganKind.TOOL, 401.0)
        cert = certificate("access-wrong", loci)
        applied = apply_relation_access_plan(genome, cert, wrong, wrong_plan)
        self.assertIsNotNone(applied)
        descendant = apply_mutation_program(genome, applied.mutation_program)
        self.assertEqual(capability(descendant, loci, good), 0.0)
        self.assertEqual(capability(descendant, loci, bad), 1.0)

    def test_schema_plan_mismatch_fails_closed(self):
        schema = self.schema()
        plan = compile_relation_access_plan(schema)
        wrong = semantic_wrong_schema(schema)
        genome, loci, _, _ = world("mismatch", 4, OrganKind.REPRESENTATION, 501.0)
        cert = certificate("mismatch", loci)
        self.assertIsNone(apply_relation_access_plan(genome, cert, wrong, plan))

    def test_unsupported_relation_orientation_is_not_silently_scanned(self):
        schema = self.schema()
        unsupported = ReflectiveRewriteSchema(
            schema_id="unsupported",
            operation="REWIRE_EDGE",
            relations=(
                RelationExpression("IN", FieldRef("candidate", "cost_hint"), FieldRef("old_target", "consumes")),
            ),
            supporting_contexts=schema.supporting_contexts,
            supporting_source_classes=schema.supporting_source_classes,
            supporting_program_ids=schema.supporting_program_ids,
        )
        self.assertIsNone(compile_relation_access_plan(unsupported))


if __name__ == "__main__":
    unittest.main()
