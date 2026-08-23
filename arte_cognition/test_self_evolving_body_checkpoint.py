from __future__ import annotations

import copy
import unittest

from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.executable_morphology import ExperienceArchive, ExperienceUnit, MorphologyGenome, OrganKind, OrganSpec
from arte_cognition.meta_acceleration import MutationProgramDevelopmentState, MutationStrategyState
from arte_cognition.self_evolving_body_checkpoint import (
    SelfEvolvingResearchBody,
    checkpoint_dict,
    integrity_sha256,
    restore_body,
)


def make_body() -> SelfEvolvingResearchBody:
    runtime = PersistentCognitiveRuntime()
    morphology = MorphologyGenome(
        organs=(
            OrganSpec("governor", OrganKind.GOVERNOR),
            OrganSpec("archive", OrganKind.ARCHIVE),
        ),
        edges=(),
        event_order=("governor", "archive"),
        constitution_epoch=2,
    )
    strategy = MutationStrategyState(
        operation_scores=(("ADD_EDGE", -0.5), ("REWIRE_EDGE", 2.5)),
        operation_support=(("ADD_EDGE", 2), ("REWIRE_EDGE", 4)),
        fossilized_operations=("REMOVE_ORGAN",),
        lineage_hash="strategy-lineage",
    )
    program_state = MutationProgramDevelopmentState(
        max_depth=4,
        complete_failure_receipts=(("ctx-a", "class-a"),),
        lineage_hash="depth-lineage",
    )
    archive = ExperienceArchive()
    archive.append(
        ExperienceUnit(
            episode_id="ep-failure",
            pre_body_hash=morphology.fingerprint(),
            source_refs=("external://source",),
            task_ref="task",
            benchmark_family="EXTERNAL",
            precommitted_hypotheses=("h0",),
            selected_goal_id="goal",
            action_trace_hash="trace",
            outcome_summary="counterexample",
            success=False,
            uncertainty_before=0.3,
            uncertainty_after=0.8,
            mutation_ids=("m1",),
            removal_effect=0.5,
            wrong_swap_effect=0.4,
            heldout_effect=0.0,
            delayed_replay_equal=True,
            descendant_body_hash="desc",
            notes=("preserve-failure",),
        )
    )
    archive.fossilize("bad-mutation", "negative transfer")
    return SelfEvolvingResearchBody(runtime, morphology, strategy, program_state, archive)


class SelfEvolvingBodyCheckpointTests(unittest.TestCase):
    def test_roundtrip_preserves_morphology_strategy_depth_and_failure_archive(self):
        body = make_body()
        payload = checkpoint_dict(body)
        restored = restore_body(payload)
        self.assertIsInstance(restored.runtime, PersistentCognitiveRuntime)
        self.assertEqual(restored.morphology.fingerprint(), body.morphology.fingerprint())
        self.assertEqual(restored.mutation_strategy, body.mutation_strategy)
        self.assertEqual(restored.mutation_program_state, body.mutation_program_state)
        self.assertIn("ep-failure", restored.experience_archive.episodes)
        self.assertFalse(restored.experience_archive.episodes["ep-failure"].success)
        self.assertEqual(restored.experience_archive.fossils["bad-mutation"], "negative transfer")

    def test_wrapper_integrity_detects_tamper(self):
        payload = checkpoint_dict(make_body())
        payload["morphology_genome"]["constitution_epoch"] = 999
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            restore_body(payload)

    def test_required_namespace_cannot_be_removed_even_after_rehash(self):
        payload = checkpoint_dict(make_body())
        payload.pop("experience_archive")
        payload["self_evolving_body"]["integrity_sha256"] = integrity_sha256(payload)
        with self.assertRaisesRegex(ValueError, "missing namespaces"):
            restore_body(payload)

    def test_morphology_fingerprint_fails_closed_even_if_outer_hash_is_recomputed(self):
        payload = checkpoint_dict(make_body())
        payload["morphology_genome"]["constitution_epoch"] = 3
        payload["self_evolving_body"]["integrity_sha256"] = integrity_sha256(payload)
        with self.assertRaisesRegex(ValueError, "morphology fingerprint mismatch"):
            restore_body(payload)

    def test_external_certificate_authority_is_not_a_checkpoint_namespace(self):
        payload = checkpoint_dict(make_body())
        serialized = repr(payload)
        self.assertNotIn("structural_failure_certificates", payload)
        self.assertNotIn("certificate_authority", serialized)
        self.assertIn("must be re-established", payload["self_evolving_body"]["authority_note"])


if __name__ == "__main__":
    unittest.main()
