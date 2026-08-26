from __future__ import annotations

import unittest

from arte_cognition.causal_experiment_germline import (
    CausalExperimentGermline,
    GitHubSourceBinding,
    github_source_set_sha256,
    verify_descendant_germline,
)


H40_A = "a" * 40
H40_B = "b" * 40
H64_A = "a" * 64
H64_B = "b" * 64
H64_C = "c" * 64
H64_D = "d" * 64
H64_E = "e" * 64
H64_F = "f" * 64


def make_source() -> GitHubSourceBinding:
    return GitHubSourceBinding(
        repository="leehyen0/GENESIS-EX-NIHILO",
        ref="main",
        commit_sha=H40_A,
        path="research/ARTE_SELF_EVOLVING_RESEARCH_BODY_20260823.md",
        blob_sha=H40_B,
        role="BODY_BLUEPRINT",
    )


def make_germline() -> CausalExperimentGermline:
    return CausalExperimentGermline(
        experiment_id="fresh-transfer-001",
        benchmark_family="SWE_BENCH",
        operator_sha256=H64_A,
        evaluator_sha256=H64_D,
        source_receipt_sha256=H64_E,
        freeze_sha256=H64_F,
        stage="FROZEN",
        github_sources=(make_source(),),
    )


class CausalExperimentGermlineTests(unittest.TestCase):
    def test_frozen_state_is_zero_shot_and_contains_no_task_or_world_binding(self):
        parent = make_germline()
        self.assertEqual(parent.validate(), ())
        self.assertEqual(parent.task_ref, "")
        self.assertEqual(parent.task_sha256, "")
        self.assertEqual(parent.world_sha256, "")

    def test_roundtrip_preserves_atomic_identity(self):
        parent = make_germline()
        restored = CausalExperimentGermline.from_dict(parent.to_dict())
        self.assertEqual(restored, parent)
        self.assertEqual(restored.fingerprint(), parent.fingerprint())
        self.assertTrue(restored.authority_reverification_required)

    def test_task_then_world_bind_monotonically(self):
        frozen = make_germline()
        task = frozen.advance("TASK_ACQUIRED", task_ref="django__django-fresh", task_sha256=H64_B)
        task_result = verify_descendant_germline(frozen, task)
        self.assertTrue(task_result.passed)
        self.assertEqual(task_result.status, "PASS_MONOTONIC_ATOMIC_CAUSAL_EXPERIMENT_HEREDITY")

        world = task.advance("WORLD_PINNED", world_sha256=H64_C)
        world_result = verify_descendant_germline(task, world)
        self.assertTrue(world_result.passed)
        self.assertEqual(world.task_sha256, H64_B)
        self.assertEqual(world.world_sha256, H64_C)

    def test_task_cannot_be_prebound_before_fresh_acquisition(self):
        frozen = make_germline()
        illegal = CausalExperimentGermline(
            experiment_id=frozen.experiment_id,
            benchmark_family=frozen.benchmark_family,
            operator_sha256=frozen.operator_sha256,
            evaluator_sha256=frozen.evaluator_sha256,
            source_receipt_sha256=frozen.source_receipt_sha256,
            freeze_sha256=frozen.freeze_sha256,
            stage="FROZEN",
            task_ref="leaked-task",
            task_sha256=H64_B,
            github_sources=frozen.github_sources,
        )
        self.assertIn("task_ref_bound_before_task_acquired", illegal.validate())
        self.assertIn("task_sha256_bound_before_task_acquired", illegal.validate())

    def test_inherited_world_change_fails_closed(self):
        frozen = make_germline()
        task = frozen.advance("TASK_ACQUIRED", task_ref="fresh", task_sha256=H64_B)
        world = task.advance("WORLD_PINNED", world_sha256=H64_C)
        baseline = CausalExperimentGermline(
            experiment_id=world.experiment_id,
            benchmark_family=world.benchmark_family,
            operator_sha256=world.operator_sha256,
            evaluator_sha256=world.evaluator_sha256,
            source_receipt_sha256=world.source_receipt_sha256,
            freeze_sha256=world.freeze_sha256,
            stage="BASELINE_RECORDED",
            task_ref=world.task_ref,
            task_sha256=world.task_sha256,
            world_sha256="1" * 64,
            baseline_receipt_sha256="2" * 64,
            github_sources=world.github_sources,
            parent_germline_sha256=world.fingerprint(),
        )
        result = verify_descendant_germline(world, baseline)
        self.assertFalse(result.passed)
        self.assertIn("inherited_binding_changed::world_sha256", result.errors)

    def test_skipped_stage_fails_closed(self):
        frozen = make_germline()
        with self.assertRaisesRegex(ValueError, "illegal causal experiment stage transition"):
            frozen.advance("WORLD_PINNED", world_sha256=H64_C)

    def test_wrong_stage_binding_is_rejected(self):
        frozen = make_germline()
        with self.assertRaisesRegex(ValueError, "wrong causal stage"):
            frozen.advance(
                "TASK_ACQUIRED",
                task_ref="fresh",
                task_sha256=H64_B,
                world_sha256=H64_C,
            )

    def test_serialized_authority_is_forbidden(self):
        parent = make_germline()
        payload = parent.to_dict()
        payload["authority_reverification_required"] = False
        payload.pop("germline_sha256")
        with self.assertRaisesRegex(ValueError, "serialized_authority_forbidden"):
            CausalExperimentGermline.from_dict(payload)

    def test_tampered_fingerprint_is_rejected(self):
        payload = make_germline().to_dict()
        payload["germline_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            CausalExperimentGermline.from_dict(payload)

    def test_github_source_is_provenance_not_authority(self):
        source = make_source()
        self.assertEqual(source.validate(), ())
        digest = github_source_set_sha256((source,))
        self.assertEqual(len(digest), 64)
        self.assertNotIn("authority", source.to_dict())


if __name__ == "__main__":
    unittest.main()
