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
from arte_cognition.native_representation_genesis import (
    NativeRepresentationGenesisEngine,
    RepresentationSupportExample,
    executable_representation_programs,
    infer_representation_family,
)
from arte_cognition.self_evolving_body_checkpoint import SelfEvolvingResearchBody, checkpoint_dict, restore_body


def parent_genome() -> MorphologyGenome:
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


def residual(name: str = "fresh-repr") -> MorphologyResidual:
    return MorphologyResidual(
        residual_id=name,
        pressure=PressureVector(transfer_failure=1.0, theory_blindspot=0.4),
        same_frozen_phenotype_different_outcome=True,
        more_compute_still_aliased=True,
        missing_artifact_types=("latent_artifact",),
    )


class NativeRepresentationGenesisTests(unittest.TestCase):
    def test_support_identifies_difference(self):
        rows = (
            RepresentationSupportExample((9, 2), 7),
            RepresentationSupportExample((3, 7), -4),
            RepresentationSupportExample((12, 5), 7),
            RepresentationSupportExample((1, 10), -9),
        )
        self.assertEqual(infer_representation_family(rows), "DIFFERENCE")

    def test_generated_candidate_is_true_l1_representation(self):
        rows = (
            RepresentationSupportExample((4, 1), 5),
            RepresentationSupportExample((7, 3), 4),
            RepresentationSupportExample((8, 2), 10),
            RepresentationSupportExample((5, 5), 0),
        )
        candidate = NativeRepresentationGenesisEngine().generate(parent_genome(), residual(), rows)[0]
        self.assertEqual(candidate.mutation.level, MutationLevel.REPRESENTATION_MEMORY_TOOL)
        self.assertEqual(candidate.operation_family, "ADD_REPRESENTATION_OPERATOR")
        child = MorphologyMutator().apply(parent_genome(), candidate.mutation)
        programs = executable_representation_programs(
            child,
            expected_artifact_type="latent_artifact",
            expected_residual_id="fresh-repr",
        )
        self.assertEqual(len(programs), 1)
        self.assertEqual(programs[0].family, "XOR")
        self.assertEqual(programs[0].execute((11, 6)), 13)

    def test_old_language_alias_evidence_is_required(self):
        bad = MorphologyResidual(
            residual_id="no-escape",
            pressure=PressureVector(transfer_failure=1.0),
            same_frozen_phenotype_different_outcome=False,
            more_compute_still_aliased=True,
            missing_artifact_types=("latent_artifact",),
        )
        rows = (RepresentationSupportExample((9, 2), 7),)
        with self.assertRaisesRegex(ValueError, "REPRESENTATION_ESCAPE_NOT_AUTHORIZED"):
            NativeRepresentationGenesisEngine().generate(parent_genome(), bad, rows)

    def test_representation_survives_body_checkpoint_and_cold_execution(self):
        rows = (
            RepresentationSupportExample((5, 2), 519),
            RepresentationSupportExample((2, 9), 523),
            RepresentationSupportExample((7, 3), 778),
            RepresentationSupportExample((1, 4), 261),
        )
        candidate = NativeRepresentationGenesisEngine().generate(parent_genome(), residual(), rows)[0]
        child = MorphologyMutator().apply(parent_genome(), candidate.mutation)
        body = SelfEvolvingResearchBody(
            runtime=PersistentCognitiveRuntime(),
            morphology=child,
            mutation_strategy=MutationStrategyState(),
            mutation_program_state=MutationProgramDevelopmentState(),
            experience_archive=ExperienceArchive(),
        )
        restored = restore_body(checkpoint_dict(body))
        self.assertEqual(restored.morphology.fingerprint(), child.fingerprint())
        program = executable_representation_programs(
            restored.morphology,
            expected_artifact_type="latent_artifact",
            expected_residual_id="fresh-repr",
        )[0]
        self.assertEqual(program.family, "ORDER_CANONICAL")
        self.assertEqual(program.execute((6, 2)), 520)


if __name__ == "__main__":
    unittest.main()
