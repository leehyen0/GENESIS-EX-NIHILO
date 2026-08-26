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
        task_ref="django__django-fresh",
        task_sha256=H64_B,
        world_sha256=H64_C,
        evaluator_sha256=H64_D,
        source_receipt_sha256=H64_E,
        freeze_sha256=H64_F,
        stage="FROZEN",
        github_sources=(make_source(),),
    )


class CausalExperimentGermlineTests(unittest.TestCase):
    def test_roundtrip_preserves_atomic_identity(self):
        parent = make_germline()
        restored = CausalExperimentGermline.from_dict(parent.to_dict())
        self.assertEqual(restored, parent)
        self.assertEqual(restored.fingerprint(), parent.fingerprint())
        self.assertEqual(restored.validate(), ())
        self.assertTrue(restored.authority_reverification_required)

    def test_descendant_may_advance_one_stage_with_exact_parent_binding(self):
        parent = make_germline()
        child = parent.advance("TASK_ACQUIRED")
        result = verify_descendant_germline(parent, child)
        self.assertTrue(result.passed)
        self.assertEqual(result.status, "PASS_ATOMIC_CAUSAL_EXPERIMENT_HEREDITY")

    def test_immutable_world_change_fails_closed(self):
        parent = make_germline()
        child = CausalExperimentGermline(
            experiment_id=parent.experiment_id,
            benchmark_family=parent.benchmark_family,
            operator_sha256=parent.operator_sha256,
            task_ref=parent.task_ref,
            task_sha256=parent.task_sha256,
            world_sha256="1" * 64,
            evaluator_sha256=parent.evaluator_sha256,
            source_receipt_sha256=parent.source_receipt_sha256,
            freeze_sha256=parent.freeze_sha256,
            stage="TASK_ACQUIRED",
            github_sources=parent.github_sources,
            parent_germline_sha256=parent.fingerprint(),
        )
        result = verify_descendant_germline(parent, child)
        self.assertFalse(result.passed)
        self.assertIn("immutable_experiment_identity_changed", result.errors)

    def test_skipped_stage_fails_closed(self):
        parent = make_germline()
        with self.assertRaisesRegex(ValueError, "illegal causal experiment stage transition"):
            parent.advance("WORLD_PINNED")

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
