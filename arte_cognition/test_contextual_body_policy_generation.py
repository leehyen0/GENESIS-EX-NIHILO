from __future__ import annotations

import unittest

from arte_cognition.body_policy_generation import generate_contextual_body_candidate
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.executable_morphology import (
    ExperienceArchive,
    MorphologyGenome,
    MorphologyMutator,
    OrganKind,
    OrganSpec,
    PressureVector,
)
from arte_cognition.meta_acceleration import MutationProgramDevelopmentState, MutationStrategyState
from arte_cognition.morphology_genesis import MorphologyResidual
from arte_cognition.native_recursive_research import NativeMetaMorphologyGenesisEngine
from arte_cognition.self_evolving_body_checkpoint import SelfEvolvingResearchBody, checkpoint_dict, restore_body


def parent_genome() -> MorphologyGenome:
    organs = (
        OrganSpec("generator", OrganKind.GENERATOR, produces=("candidate",), implementation_ref="bootstrap://generator"),
        OrganSpec("mutator", OrganKind.MUTATOR, consumes=("candidate",), produces=("mutation",), implementation_ref="bootstrap://mutator"),
        OrganSpec("governor", OrganKind.GOVERNOR),
        OrganSpec("archive", OrganKind.ARCHIVE),
    )
    return MorphologyGenome(organs=organs, edges=(), event_order=("generator", "mutator", "governor", "archive"))


def policy_child(origin: str = "context-origin") -> MorphologyGenome:
    parent = parent_genome()
    origin_residual = MorphologyResidual(origin, PressureVector(human_dependency=1.0, theory_blindspot=0.25))
    rows = NativeMetaMorphologyGenesisEngine(candidate_budget=16).generate(parent, (origin_residual,))
    mutation = next(row.mutation for row in rows if row.operation_family == "CHANGE_MUTATOR_POLICY")
    return MorphologyMutator().apply(parent, mutation)


def human_task(task_id: str = "human-task") -> MorphologyResidual:
    return MorphologyResidual(
        task_id,
        PressureVector(human_dependency=1.0, novelty_pressure=0.1, theory_blindspot=0.1),
    )


def novelty_task(task_id: str = "novelty-task") -> MorphologyResidual:
    return MorphologyResidual(
        task_id,
        PressureVector(human_dependency=0.1, novelty_pressure=1.0, theory_blindspot=0.1),
    )


class ContextualBodyPolicyGenerationTests(unittest.TestCase):
    def test_parent_is_context_blind_at_one_selection_slot(self):
        human = generate_contextual_body_candidate(parent_genome(), human_task())
        novelty = generate_contextual_body_candidate(parent_genome(), novelty_task())
        self.assertEqual(human.selected_operation_family, "CHANGE_GENERATOR_POLICY")
        self.assertEqual(novelty.selected_operation_family, "CHANGE_GENERATOR_POLICY")
        self.assertEqual(human.selected_candidate_budget, 1)
        self.assertEqual(novelty.selected_candidate_budget, 1)

    def test_inherited_policy_conditions_same_budget_on_fresh_pressure(self):
        child = policy_child()
        human = generate_contextual_body_candidate(
            child,
            human_task("fresh-human"),
            expected_policy_origin_residual_id="context-origin",
        )
        novelty = generate_contextual_body_candidate(
            child,
            novelty_task("fresh-novelty"),
            expected_policy_origin_residual_id="context-origin",
        )
        self.assertEqual(human.selected_operation_family, "CHANGE_MUTATOR_POLICY")
        self.assertEqual(novelty.selected_operation_family, "CHANGE_GENERATOR_POLICY")
        self.assertEqual(human.selected_candidate_budget, novelty.selected_candidate_budget)
        self.assertFalse(human.current_outcomes_consumed)
        self.assertFalse(novelty.current_outcomes_consumed)

    def test_policy_survives_self_evolving_body_checkpoint_roundtrip(self):
        child = policy_child()
        body = SelfEvolvingResearchBody(
            runtime=PersistentCognitiveRuntime(),
            morphology=child,
            mutation_strategy=MutationStrategyState(),
            mutation_program_state=MutationProgramDevelopmentState(),
            experience_archive=ExperienceArchive(),
        )
        restored = restore_body(checkpoint_dict(body))
        self.assertEqual(restored.morphology.fingerprint(), child.fingerprint())
        result = generate_contextual_body_candidate(
            restored.morphology,
            human_task("fresh-after-restart"),
            expected_policy_origin_residual_id="context-origin",
        )
        self.assertEqual(result.selected_operation_family, "CHANGE_MUTATOR_POLICY")
        self.assertTrue(result.policy_fingerprints)

    def test_ambiguous_fresh_pressure_fails_closed_for_policy_child(self):
        child = policy_child()
        ambiguous = MorphologyResidual(
            "ambiguous",
            PressureVector(human_dependency=0.5, novelty_pressure=0.5, theory_blindspot=0.1),
        )
        with self.assertRaisesRegex(ValueError, "AMBIGUOUS_CONTEXTUAL_POLICY_PRESSURE"):
            generate_contextual_body_candidate(
                child,
                ambiguous,
                expected_policy_origin_residual_id="context-origin",
            )


if __name__ == "__main__":
    unittest.main()
