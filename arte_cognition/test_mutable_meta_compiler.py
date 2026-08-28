from __future__ import annotations

import unittest

from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.executable_morphology import ExperienceArchive, MorphologyGenome, OrganKind, OrganSpec
from arte_cognition.meta_acceleration import MutationProgramDevelopmentState, MutationStrategyState
from arte_cognition.mutable_meta_compiler import (
    FAMILY_ABSTAIN,
    FAMILY_GENERATOR,
    FAMILY_MUTATOR,
    FAMILY_TOPOLOGY,
    initial_meta_compiler_ref,
    learn_meta_compiler_rule,
    meta_compiler_policy_from_body,
)
from arte_cognition.self_evolving_body_checkpoint import SelfEvolvingResearchBody, checkpoint_dict, restore_body


def make_body() -> SelfEvolvingResearchBody:
    morphology = MorphologyGenome(
        organs=(
            OrganSpec("source", OrganKind.SOURCE, produces=("raw_observation",), implementation_ref="bootstrap://source"),
            OrganSpec("generator", OrganKind.GENERATOR, implementation_ref="bootstrap://generator"),
            OrganSpec("mutator", OrganKind.MUTATOR, implementation_ref="bootstrap://mutator"),
            OrganSpec("compiler", OrganKind.COMPILER, implementation_ref=initial_meta_compiler_ref()),
            OrganSpec("governor", OrganKind.GOVERNOR),
            OrganSpec("archive", OrganKind.ARCHIVE),
        ),
        edges=(),
        event_order=(),
    )
    return SelfEvolvingResearchBody(
        runtime=PersistentCognitiveRuntime(),
        morphology=morphology,
        mutation_strategy=MutationStrategyState(),
        mutation_program_state=MutationProgramDevelopmentState(),
        experience_archive=ExperienceArchive(),
    )


class MutableMetaCompilerTests(unittest.TestCase):
    def test_unlearned_signal_fails_closed_to_abstain(self):
        body = make_body()
        family, confidence = meta_compiler_policy_from_body(body).route("opaque-0")
        self.assertEqual(family, FAMILY_ABSTAIN)
        self.assertEqual(confidence, 0.0)

    def test_unique_past_outcome_updates_l6_compiler_policy(self):
        body = make_body()
        receipt = learn_meta_compiler_rule(
            body,
            signal_slot="opaque-0",
            proposal_outcomes={
                FAMILY_GENERATOR: 1.0,
                FAMILY_MUTATOR: 0.0,
                FAMILY_TOPOLOGY: 0.0,
                FAMILY_ABSTAIN: 0.0,
            },
            evidence_ref="past-training://g0",
        )
        self.assertEqual(receipt.winning_family, FAMILY_GENERATOR)
        self.assertFalse(receipt.generation_uses_future_validation_outcomes)
        policy = meta_compiler_policy_from_body(body)
        self.assertEqual(policy.route("opaque-0"), (FAMILY_GENERATOR, 1.0))
        self.assertEqual(policy.route("opaque-1"), (FAMILY_ABSTAIN, 0.0))
        self.assertGreater(body.mutation_strategy.score("META_COMPILER::" + FAMILY_GENERATOR), 0.0)
        self.assertEqual(len(body.experience_archive.episodes), 1)

    def test_checkpoint_preserves_policy_strategy_archive_and_behavior(self):
        body = make_body()
        learn_meta_compiler_rule(
            body,
            signal_slot="opaque-2",
            proposal_outcomes={
                FAMILY_GENERATOR: 0.0,
                FAMILY_MUTATOR: 0.0,
                FAMILY_TOPOLOGY: 1.0,
                FAMILY_ABSTAIN: 0.0,
            },
            evidence_ref="past-training://g1",
        )
        restored = restore_body(checkpoint_dict(body))
        self.assertEqual(restored.morphology.fingerprint(), body.morphology.fingerprint())
        self.assertEqual(restored.mutation_strategy, body.mutation_strategy)
        self.assertEqual(restored.experience_archive.episodes, body.experience_archive.episodes)
        self.assertEqual(meta_compiler_policy_from_body(restored).route("opaque-2"), (FAMILY_TOPOLOGY, 1.0))

    def test_ambiguous_outcomes_fail_closed(self):
        body = make_body()
        with self.assertRaisesRegex(ValueError, "WINNER_NOT_IDENTIFIABLE"):
            learn_meta_compiler_rule(
                body,
                signal_slot="opaque-3",
                proposal_outcomes={
                    FAMILY_GENERATOR: 1.0,
                    FAMILY_MUTATOR: 1.0,
                    FAMILY_TOPOLOGY: 0.0,
                    FAMILY_ABSTAIN: 0.0,
                },
                evidence_ref="past-training://ambiguous",
            )

    def test_inherited_rule_cannot_be_overwritten_by_conflicting_outcome(self):
        body = make_body()
        learn_meta_compiler_rule(
            body,
            signal_slot="opaque-1",
            proposal_outcomes={
                FAMILY_GENERATOR: 0.0,
                FAMILY_MUTATOR: 1.0,
                FAMILY_TOPOLOGY: 0.0,
                FAMILY_ABSTAIN: 0.0,
            },
            evidence_ref="past-training://first",
        )
        with self.assertRaisesRegex(ValueError, "CONTRADICTS_INHERITED_POLICY"):
            learn_meta_compiler_rule(
                body,
                signal_slot="opaque-1",
                proposal_outcomes={
                    FAMILY_GENERATOR: 1.0,
                    FAMILY_MUTATOR: 0.0,
                    FAMILY_TOPOLOGY: 0.0,
                    FAMILY_ABSTAIN: 0.0,
                },
                evidence_ref="past-training://conflict",
            )


if __name__ == "__main__":
    unittest.main()
