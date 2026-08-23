from __future__ import annotations

import unittest

from arte_cognition.executable_morphology import EdgeSpec, MorphologyGenome, MutationLevel, OrganKind, OrganSpec
from arte_cognition.meta_acceleration import MutationProgram, MutationTemplate, apply_mutation_program
from arte_cognition.morphology_rewrite_schema_genesis import (
    EDGE_FIELDS,
    GeneratedRewriteSchema,
    RewriteSchemaTrainingExample,
    apply_generated_rewrite_schema,
    generate_rewrite_schemas,
)
from arte_cognition.parametric_morphology_macro import ParametricMorphologyMacro, apply_parametric_macro
from arte_cognition.structural_failure_certificate import (
    StructuralDiagnosticReceipt,
    derive_structural_failure_certificate,
)


def certificate(label: str, loci):
    receipts = (
        StructuralDiagnosticReceipt(f"{label}-a", label, f"{label}-class-a", tuple(loci), True, True),
        StructuralDiagnosticReceipt(f"{label}-b", label, f"{label}-class-b", tuple(reversed(tuple(loci))), True, True),
    )
    out = derive_structural_failure_certificate(receipts)
    assert out is not None
    return out


def world(label: str, depth: int, old_kind: OrganKind = OrganKind.REPRESENTATION):
    organs = []
    edges = []
    correct = []
    wrong = []
    for index in range(depth):
        artifact = f"{label}-artifact-{index}"
        source = f"{label}-source-{index}"
        old = f"{label}-old-{index}"
        good = f"{label}-good-{index}"
        bad = f"{label}-bad-{index}"
        edge_id = f"{label}-edge-{index}"
        distractor_kind = OrganKind.PERCEPTOR if old_kind != OrganKind.PERCEPTOR else OrganKind.TOOL
        organs.extend(
            (
                OrganSpec(source, OrganKind.SOURCE, produces=(artifact,)),
                OrganSpec(old, old_kind, consumes=(artifact,), produces=(f"old-out-{index}",)),
                OrganSpec(good, old_kind, consumes=(artifact,), produces=(f"good-out-{index}",)),
                OrganSpec(bad, distractor_kind, consumes=(artifact,), produces=(f"bad-out-{index}",)),
            )
        )
        edges.append(EdgeSpec(edge_id, source, old, artifact))
        correct.append(good)
        wrong.append(bad)
    organs.extend((OrganSpec(f"{label}-gov", OrganKind.GOVERNOR), OrganSpec(f"{label}-archive", OrganKind.ARCHIVE)))
    genome = MorphologyGenome(tuple(organs), tuple(edges), tuple(o.organ_id for o in organs))
    return genome, tuple(e.edge_id for e in edges), tuple(correct), tuple(wrong)


def program_for(genome: MorphologyGenome, loci, targets, label: str):
    edges = {edge.edge_id: edge for edge in genome.edges}
    templates = []
    for locus, target in zip(loci, targets):
        edge = edges[locus]
        replacement = {
            "edge_id": edge.edge_id,
            "source": edge.source,
            "target": target,
            "artifact_type": edge.artifact_type,
            "authority_required": edge.authority_required,
            "gate": edge.gate,
            "priority": edge.priority,
        }
        templates.append(
            MutationTemplate(
                operation="REWIRE_EDGE",
                level=MutationLevel.TOPOLOGY,
                payload={"edge_id": locus, "edge": replacement},
                rationale=("external-success",),
            )
        )
    return MutationProgram(f"program-{label}", tuple(templates), f"strategy-{label}", False)


def capability(genome: MorphologyGenome, loci, targets):
    edges = {edge.edge_id: edge for edge in genome.edges}
    return float(all(edges[locus].target == target for locus, target in zip(loci, targets)))


class RewriteSchemaGenesisTests(unittest.TestCase):
    def training_examples(self):
        examples = []
        for label, depth, kind in (
            ("alpha", 2, OrganKind.REPRESENTATION),
            ("beta", 3, OrganKind.TOOL),
        ):
            genome, loci, good, _ = world(label, depth, kind)
            cert = certificate(label, loci)
            program = program_for(genome, loci, good, label)
            self.assertEqual(capability(apply_mutation_program(genome, program), loci, good), 1.0)
            examples.append(
                RewriteSchemaTrainingExample(
                    context_id=label,
                    source_class=f"source-{label}",
                    genome=genome,
                    certificate=cert,
                    successful_program=program,
                    external_capability=1.0,
                    authority_verified=True,
                    benchmark_disjoint=True,
                )
            )
        return tuple(examples)

    def test_schema_relation_is_generated_from_cross_context_before_after_structure(self):
        schemas = generate_rewrite_schemas(self.training_examples())
        self.assertTrue(schemas)
        schema = schemas[0]
        self.assertIn("SAME_KIND_AS_OLD_TARGET", schema.target_predicates)
        self.assertIn("CONSUMES_EDGE_ARTIFACT", schema.target_predicates)
        self.assertNotIn("DIFFERENT_KIND_FROM_OLD_TARGET", schema.target_predicates)
        self.assertTrue(schema.generated_from_prior_external_successes)
        self.assertFalse(schema.current_outcomes_required_for_application)

    def test_fixed_unique_compatible_macro_fails_but_generated_schema_transfers(self):
        schema = generate_rewrite_schemas(self.training_examples())[0]
        genome, loci, good, _ = world("heldout", 5, OrganKind.MEMORY)
        cert = certificate("heldout", loci)

        predecessor = ParametricMorphologyMacro(
            macro_id="old-macro",
            rule="FOR_EACH_CERTIFIED_FAILED_EDGE_REWIRE_TO_UNIQUE_COMPATIBLE_ALTERNATIVE",
            supporting_contexts=("alpha", "beta"),
            supporting_source_classes=("source-alpha", "source-beta"),
            training_program_ids=("p1", "p2"),
        )
        self.assertIsNone(apply_parametric_macro(genome, cert, predecessor))

        application = apply_generated_rewrite_schema(genome, cert, schema)
        self.assertIsNotNone(application)
        self.assertEqual(application.outcome_evaluations, 0)
        descendant = apply_mutation_program(genome, application.mutation_program)
        self.assertEqual(capability(descendant, loci, good), 1.0)

    def test_one_context_cannot_authorize_schema(self):
        self.assertEqual(generate_rewrite_schemas(self.training_examples()[:1]), ())

    def test_same_form_semantic_wrong_relation_has_zero_capability(self):
        schema = generate_rewrite_schemas(self.training_examples())[0]
        genome, loci, good, bad = world("wrong-heldout", 4, OrganKind.REPRESENTATION)
        cert = certificate("wrong-heldout", loci)
        wrong = GeneratedRewriteSchema(
            schema_id="wrong-schema",
            operation=schema.operation,
            target_predicates=("CONSUMES_EDGE_ARTIFACT", "DIFFERENT_KIND_FROM_OLD_TARGET"),
            preserved_edge_fields=EDGE_FIELDS,
            supporting_contexts=schema.supporting_contexts,
            supporting_source_classes=schema.supporting_source_classes,
            supporting_program_ids=schema.supporting_program_ids,
        )
        application = apply_generated_rewrite_schema(genome, cert, wrong)
        self.assertIsNotNone(application)
        descendant = apply_mutation_program(genome, application.mutation_program)
        self.assertEqual(capability(descendant, loci, good), 0.0)
        self.assertEqual(capability(descendant, loci, bad), 1.0)

    def test_verifierless_certificate_cannot_apply_generated_schema(self):
        schema = generate_rewrite_schemas(self.training_examples())[0]
        genome, loci, _, _ = world("unverified", 2)
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
        self.assertIsNone(apply_generated_rewrite_schema(genome, unverified, schema))


if __name__ == "__main__":
    unittest.main()
