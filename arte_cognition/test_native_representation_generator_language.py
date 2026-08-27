from __future__ import annotations

import unittest

from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.executable_morphology import (
    ExperienceArchive,
    MorphologyGenome,
    MorphologyMutator,
    MutationLevel,
    OrganKind,
    OrganSpec,
    PressureVector,
)
from arte_cognition.meta_acceleration import MutationProgramDevelopmentState, MutationStrategyState
from arte_cognition.morphology_genesis import MorphologyResidual
from arte_cognition.native_representation_genesis import RepresentationSupportExample
from arte_cognition.native_representation_generator_language import (
    CompositionalRepresentationGenesisEngine,
    ExpressionSpec,
    derive_generator_language_mutation,
    expression_representation_programs,
    generator_policies,
    infer_expression_spec,
)
from arte_cognition.self_evolving_body_checkpoint import SelfEvolvingResearchBody, checkpoint_dict, restore_body


def base_genome() -> MorphologyGenome:
    return MorphologyGenome(
        organs=(
            OrganSpec("source", OrganKind.SOURCE, produces=("raw_observation",), implementation_ref="bootstrap://source"),
            OrganSpec("generator", OrganKind.GENERATOR, implementation_ref="bootstrap://generator"),
            OrganSpec("mutator", OrganKind.MUTATOR, implementation_ref="bootstrap://mutator"),
            OrganSpec("governor", OrganKind.GOVERNOR),
            OrganSpec("archive", OrganKind.ARCHIVE),
        ),
        edges=(),
        event_order=(),
    )


def generator_child(origin: str = "cycle7-origin") -> MorphologyGenome:
    mutation = derive_generator_language_mutation(
        base_genome(),
        origin_residual_id=origin,
        failure_fossil="cycle6-fixed-family-reachability-bound",
    )
    return MorphologyMutator().apply(base_genome(), mutation)


def task_residual(name: str = "fresh-expression") -> MorphologyResidual:
    return MorphologyResidual(
        residual_id=name,
        pressure=PressureVector(transfer_failure=1.0, theory_blindspot=1.0),
        same_frozen_phenotype_different_outcome=True,
        more_compute_still_aliased=True,
        missing_artifact_types=("expression_artifact",),
    )


class NativeRepresentationGeneratorLanguageTests(unittest.TestCase):
    def test_generator_language_mutation_is_l3_and_checkpoint_heritable(self):
        mutation = derive_generator_language_mutation(
            base_genome(),
            origin_residual_id="cycle7-origin",
            failure_fossil="cycle6-fixed-family-reachability-bound",
        )
        self.assertEqual(mutation.level, MutationLevel.GENERATOR_MUTATOR)
        self.assertEqual(mutation.operation, "REPLACE_ORGAN")
        child = MorphologyMutator().apply(base_genome(), mutation)
        body = SelfEvolvingResearchBody(
            runtime=PersistentCognitiveRuntime(),
            morphology=child,
            mutation_strategy=MutationStrategyState(),
            mutation_program_state=MutationProgramDevelopmentState(),
            experience_archive=ExperienceArchive(),
        )
        restored = restore_body(checkpoint_dict(body))
        self.assertEqual(restored.morphology.fingerprint(), child.fingerprint())
        policies = generator_policies(restored.morphology, expected_origin_residual_id="cycle7-origin")
        self.assertEqual(len(policies), 1)
        self.assertEqual(policies[0].max_binary_ops, 2)

    def test_support_synthesizes_unnamed_two_operation_expression(self):
        spec = ExpressionSpec("ADD", "XOR", "x")
        pairs = ((1, 2), (4, 3), (7, 1), (10, 5), (2, 8), (12, 3))
        rows = tuple(RepresentationSupportExample(pair, spec.execute(pair)) for pair in pairs)
        self.assertEqual(infer_expression_spec(rows), spec)

        child = generator_child()
        candidate = CompositionalRepresentationGenesisEngine().generate(
            child,
            task_residual(),
            rows,
            expected_generator_origin_residual_id="cycle7-origin",
        )[0]
        self.assertEqual(candidate.mutation.level, MutationLevel.REPRESENTATION_MEMORY_TOOL)
        self.assertEqual(candidate.operation_family, "SYNTHESIZE_REPRESENTATION_EXPRESSION")
        representation_child = MorphologyMutator().apply(child, candidate.mutation)
        program = expression_representation_programs(
            representation_child,
            expected_artifact_type="expression_artifact",
            expected_residual_id="fresh-expression",
        )[0]
        self.assertEqual(program.spec, spec)
        self.assertEqual(program.execute((9, 6)), spec.execute((9, 6)))

    def test_compositional_engine_fails_without_inherited_generator_policy(self):
        spec = ExpressionSpec("ADD", "XOR", "x")
        pairs = ((1, 2), (4, 3), (7, 1), (10, 5), (2, 8), (12, 3))
        rows = tuple(RepresentationSupportExample(pair, spec.execute(pair)) for pair in pairs)
        with self.assertRaisesRegex(ValueError, "COMPOSITIONAL_GENESIS_REQUIRES_INHERITED_GENERATOR_POLICY"):
            CompositionalRepresentationGenesisEngine().generate(
                base_genome(),
                task_residual(),
                rows,
                expected_generator_origin_residual_id="cycle7-origin",
            )


if __name__ == "__main__":
    unittest.main()
