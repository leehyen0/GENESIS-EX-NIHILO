from __future__ import annotations

import unittest

from arte_cognition.autonomous_generator_mutation import (
    AutonomousMetaMutationCompiler,
    MetaFailureCertificate,
    apply_autonomous_selection,
    credit_autonomous_selection,
)
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.executable_morphology import ExperienceArchive, MorphologyGenome, OrganKind, OrganSpec
from arte_cognition.meta_acceleration import MutationProgramDevelopmentState, MutationStrategyState
from arte_cognition.native_representation_generator_language import generator_policies
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


def certificate(layer: str = "REPRESENTATION_GENERATOR_LANGUAGE", contaminated: bool = False) -> MetaFailureCertificate:
    genome = parent_genome()
    return MetaFailureCertificate(
        certificate_id="cycle8-native-cert",
        parent_body_hash=genome.fingerprint(),
        failure_layer=layer,
        more_compute_exhausted=True,
        independent_contexts=("cycle6-hidden-representation", "cycle7-hidden-expression"),
        prior_generator_language_gain=1.0,
        source_receipt_hashes=("cycle6-receipt-hash", "cycle7-receipt-hash"),
        current_hidden_task_information_present=contaminated,
    )


class AutonomousGeneratorMutationTests(unittest.TestCase):
    def test_correct_certificate_selects_generator_from_three_proposals(self):
        selection = AutonomousMetaMutationCompiler().propose(parent_genome(), certificate())
        self.assertEqual(len(selection.proposals), 3)
        self.assertEqual(selection.selected.family, "AUTONOMOUS_GENERATOR_LANGUAGE")
        self.assertFalse(selection.generation_uses_current_outcomes)

    def test_wrong_layer_certificate_selects_mutator_not_generator(self):
        selection = AutonomousMetaMutationCompiler().propose(
            parent_genome(), certificate("MUTATOR_SEARCH_POLICY")
        )
        self.assertEqual(selection.selected.family, "AUTONOMOUS_MUTATOR_POLICY")

    def test_current_hidden_task_contamination_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "CONTAMINATED"):
            AutonomousMetaMutationCompiler().propose(parent_genome(), certificate(contaminated=True))

    def test_selected_generator_mutation_survives_body_checkpoint(self):
        selection = AutonomousMetaMutationCompiler().propose(parent_genome(), certificate())
        child = apply_autonomous_selection(parent_genome(), selection)
        body = SelfEvolvingResearchBody(
            runtime=PersistentCognitiveRuntime(),
            morphology=child,
            mutation_strategy=MutationStrategyState(),
            mutation_program_state=MutationProgramDevelopmentState(),
            experience_archive=ExperienceArchive(),
        )
        restored = restore_body(checkpoint_dict(body))
        self.assertEqual(restored.morphology.fingerprint(), child.fingerprint())
        policies = generator_policies(
            restored.morphology,
            expected_origin_residual_id="cycle8-native-cert",
        )
        self.assertEqual(len(policies), 1)

    def test_post_outcome_credit_is_nonvolatile_and_remove_sensitive(self):
        selection = AutonomousMetaMutationCompiler().propose(parent_genome(), certificate())
        child = apply_autonomous_selection(parent_genome(), selection)
        body = SelfEvolvingResearchBody(
            runtime=PersistentCognitiveRuntime(),
            morphology=child,
            mutation_strategy=MutationStrategyState(),
            mutation_program_state=MutationProgramDevelopmentState(),
            experience_archive=ExperienceArchive(),
        )
        effect = credit_autonomous_selection(
            body,
            selection,
            full_useful_rate=1.0,
            remove_useful_rate=0.0,
            wrong_useful_rate=0.0,
            task_ref="fresh-hidden-suite",
        )
        self.assertGreater(effect, 0.0)
        self.assertGreater(body.mutation_strategy.score("AUTONOMOUS_GENERATOR_LANGUAGE"), 0.0)
        self.assertEqual(len(body.experience_archive.episodes), 1)
        restored = restore_body(checkpoint_dict(body))
        self.assertEqual(restored.mutation_strategy, body.mutation_strategy)
        self.assertEqual(restored.experience_archive.episodes, body.experience_archive.episodes)


if __name__ == "__main__":
    unittest.main()
