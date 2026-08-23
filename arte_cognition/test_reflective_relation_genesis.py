from __future__ import annotations

import unittest

from arte_cognition.executable_morphology import EdgeSpec, MorphologyGenome, MutationLevel, OrganKind, OrganSpec
from arte_cognition.meta_acceleration import MutationProgram, MutationTemplate, apply_mutation_program
from arte_cognition.morphology_rewrite_schema_genesis import RewriteSchemaTrainingExample, generate_rewrite_schemas
from arte_cognition.reflective_relation_genesis import (
    FieldRef,
    ReflectiveRewriteSchema,
    ReflectiveTrainingExample,
    RelationExpression,
    apply_reflective_rewrite_schema,
    generate_reflective_rewrite_schemas,
)
from arte_cognition.structural_failure_certificate import StructuralDiagnosticReceipt, derive_structural_failure_certificate


def certificate(label: str, loci):
    receipts = (
        StructuralDiagnosticReceipt(f"{label}-a", label, f"{label}-class-a", tuple(loci), True, True),
        StructuralDiagnosticReceipt(f"{label}-b", label, f"{label}-class-b", tuple(reversed(tuple(loci))), True, True),
    )
    out = derive_structural_failure_certificate(receipts)
    assert out is not None
    return out


def world(label: str, depth: int, kind: OrganKind, base_priority: float):
    organs = []
    edges = []
    good_targets = []
    bad_targets = []
    for index in range(depth):
        artifact = f"{label}-artifact-{index}"
        source = f"{label}-source-{index}"
        old = f"{label}-old-{index}"
        good = f"{label}-good-{index}"
        bad = f"{label}-bad-{index}"
        edge_id = f"{label}-edge-{index}"
        priority = base_priority + 3.0 * index
        shared_output = (f"{label}-shared-output-{index}",)
        shared_impl = f"{label}-impl-{index}"
        organs.extend(
            (
                OrganSpec(source, OrganKind.SOURCE, produces=(artifact,), cost_hint=priority + 0.5),
                # Deliberate cross-locus collision: old[i].cost == edge[i+1].priority.
                # A naked cost==priority rule therefore fails in training and the
                # generator must also infer a local structural relation.
                OrganSpec(old, kind, consumes=(artifact,), produces=shared_output, implementation_ref=shared_impl, version=2, cost_hint=priority + 3.0),
                OrganSpec(good, kind, consumes=(artifact,), produces=shared_output, implementation_ref=shared_impl, version=2, cost_hint=priority),
                OrganSpec(bad, kind, consumes=(artifact,), produces=shared_output, implementation_ref=shared_impl, version=2, cost_hint=priority + 0.5),
            )
        )
        edges.append(EdgeSpec(edge_id, source, old, artifact, priority=priority))
        good_targets.append(good)
        bad_targets.append(bad)
    organs.extend((OrganSpec(f"{label}-gov", OrganKind.GOVERNOR), OrganSpec(f"{label}-archive", OrganKind.ARCHIVE)))
    genome = MorphologyGenome(tuple(organs), tuple(edges), tuple(organ.organ_id for organ in organs))
    loci = tuple(edge.edge_id for edge in edges)
    return genome, loci, tuple(good_targets), tuple(bad_targets)


def program_for(genome: MorphologyGenome, loci, targets, label: str):
    edge_map = {edge.edge_id: edge for edge in genome.edges}
    templates = []
    for locus, target in zip(loci, targets):
        edge = edge_map[locus]
        templates.append(
            MutationTemplate(
                operation="REWIRE_EDGE",
                level=MutationLevel.TOPOLOGY,
                payload={
                    "edge_id": locus,
                    "edge": {
                        "edge_id": edge.edge_id,
                        "source": edge.source,
                        "target": target,
                        "artifact_type": edge.artifact_type,
                        "authority_required": edge.authority_required,
                        "gate": edge.gate,
                        "priority": edge.priority,
                    },
                },
                rationale=("prior-external-success",),
            )
        )
    return MutationProgram(f"program-{label}", tuple(templates), f"strategy-{label}", False)


def capability(genome: MorphologyGenome, loci, targets):
    edge_map = {edge.edge_id: edge for edge in genome.edges}
    return float(all(edge_map[locus].target == target for locus, target in zip(loci, targets)))


def training_examples():
    reflective = []
    named = []
    for label, depth, kind, base in (
        ("alpha", 2, OrganKind.REPRESENTATION, 11.0),
        ("beta", 3, OrganKind.TOOL, 31.0),
    ):
        genome, loci, good, _ = world(label, depth, kind, base)
        cert = certificate(label, loci)
        program = program_for(genome, loci, good, label)
        reflective.append(ReflectiveTrainingExample(label, f"source-{label}", genome, cert, program, 1.0, True, True))
        named.append(RewriteSchemaTrainingExample(label, f"source-{label}", genome, cert, program, 1.0, True, True))
    return tuple(reflective), tuple(named)


class ReflectiveRelationGenesisTests(unittest.TestCase):
    def test_named_relation_atom_predecessor_is_completely_inexpressive(self):
        _, named = training_examples()
        self.assertEqual(generate_rewrite_schemas(named), ())

    def test_reflection_generates_previous_unlisted_cross_object_relations(self):
        reflective, _ = training_examples()
        schemas = generate_reflective_rewrite_schemas(reflective)
        self.assertTrue(schemas)
        tokens = tuple(relation.token() for relation in schemas[0].relations)
        self.assertIn("EQ(candidate.cost_hint,edge.priority)", tokens)
        self.assertIn("IN(edge.artifact_type,candidate.consumes)", tokens)
        self.assertFalse(any("SAME_KIND_AS_OLD_TARGET" in token for token in tokens))

    def test_reflective_relation_transfers_to_unseen_kind_and_values(self):
        reflective, _ = training_examples()
        schema = generate_reflective_rewrite_schemas(reflective)[0]
        genome, loci, good, _ = world("heldout", 7, OrganKind.MEMORY, 101.0)
        cert = certificate("heldout", loci)
        application = apply_reflective_rewrite_schema(genome, cert, schema)
        self.assertIsNotNone(application)
        self.assertEqual(application.outcome_evaluations, 0)
        descendant = apply_mutation_program(genome, application.mutation_program)
        self.assertEqual(capability(descendant, loci, good), 1.0)

    def test_semantic_wrong_reflective_relation_selects_distractor(self):
        reflective, _ = training_examples()
        schema = generate_reflective_rewrite_schemas(reflective)[0]
        genome, loci, good, bad = world("wrong", 4, OrganKind.GENERATOR, 151.0)
        cert = certificate("wrong", loci)
        wrong = ReflectiveRewriteSchema(
            schema_id="wrong-reflective-schema",
            operation="REWIRE_EDGE",
            relations=(
                RelationExpression("EQ", FieldRef("candidate", "cost_hint"), FieldRef("source", "cost_hint")),
                RelationExpression("IN", FieldRef("edge", "artifact_type"), FieldRef("candidate", "consumes")),
            ),
            supporting_contexts=schema.supporting_contexts,
            supporting_source_classes=schema.supporting_source_classes,
            supporting_program_ids=schema.supporting_program_ids,
        )
        application = apply_reflective_rewrite_schema(genome, cert, wrong)
        self.assertIsNotNone(application)
        descendant = apply_mutation_program(genome, application.mutation_program)
        self.assertEqual(capability(descendant, loci, good), 0.0)
        self.assertEqual(capability(descendant, loci, bad), 1.0)

    def test_one_context_and_verifierless_certificate_fail_closed(self):
        reflective, _ = training_examples()
        self.assertEqual(generate_reflective_rewrite_schemas(reflective[:1]), ())
        schema = generate_reflective_rewrite_schemas(reflective)[0]
        genome, loci, _, _ = world("unverified", 2, OrganKind.PERCEPTOR, 181.0)
        cert = certificate("unverified", loci)
        unverified = cert.__class__(
            cert.certificate_id,
            cert.context_id,
            cert.failed_locus_ids,
            cert.supporting_receipt_ids,
            cert.independent_source_classes,
            cert.max_obligations_repaired_per_primitive,
            cert.lower_bound_program_depth,
            cert.evaluator_independent,
            False,
        )
        self.assertIsNone(apply_reflective_rewrite_schema(genome, unverified, schema))


if __name__ == "__main__":
    unittest.main()
