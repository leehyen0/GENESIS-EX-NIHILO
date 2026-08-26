from __future__ import annotations

import copy
import unittest

from arte_cognition.causal_experiment_germline import CausalExperimentGermline, GitHubSourceBinding
from arte_cognition.self_evolving_body_checkpoint import (
    SELF_EVOLVING_BODY_SCHEMA_V1,
    checkpoint_dict,
    integrity_sha256,
    restore_body,
)
from arte_cognition.test_self_evolving_body_checkpoint import make_body


OLD_REQUIRED = (
    "canonical_runtime",
    "morphology_genome",
    "mutation_strategy",
    "mutation_program_development",
    "experience_archive",
)


def make_germline() -> CausalExperimentGermline:
    source = GitHubSourceBinding(
        repository="leehyen0/GENESIS-EX-NIHILO",
        ref="main",
        commit_sha="f3840760cfc0b5b7eac1e03b4d0f9874b0d1d6b0",
        path="research/ARTE_SELF_EVOLVING_RESEARCH_BODY_20260823.md",
        blob_sha="98ed32e0157926cac5762c429c153d7e180fd1f4",
        role="BODY_BLUEPRINT",
    )
    return CausalExperimentGermline(
        experiment_id="github-native-heredity-contract",
        benchmark_family="SWE_BENCH_FRESH_TRANSFER",
        operator_sha256="1" * 64,
        task_ref="fresh-task",
        task_sha256="2" * 64,
        world_sha256="3" * 64,
        evaluator_sha256="4" * 64,
        source_receipt_sha256="5" * 64,
        freeze_sha256="6" * 64,
        stage="FROZEN",
        github_sources=(source,),
    )


class CausalExperimentBodyIntegrationTests(unittest.TestCase):
    def test_checkpoint_roundtrip_preserves_germline_as_required_namespace(self):
        body = make_body()
        body.causal_experiment_germline = make_germline()
        payload = checkpoint_dict(body)
        self.assertIn("causal_experiment_germline", payload)
        self.assertIn(
            "causal_experiment_germline",
            payload["self_evolving_body"]["required_namespaces"],
        )
        restored = restore_body(payload)
        self.assertEqual(restored.causal_experiment_germline, body.causal_experiment_germline)
        self.assertTrue(restored.causal_experiment_germline.authority_reverification_required)

    def test_germline_namespace_removal_fails_even_after_outer_rehash(self):
        body = make_body()
        body.causal_experiment_germline = make_germline()
        payload = checkpoint_dict(body)
        payload.pop("causal_experiment_germline")
        payload["self_evolving_body"]["integrity_sha256"] = integrity_sha256(payload)
        with self.assertRaisesRegex(ValueError, "missing namespaces"):
            restore_body(payload)

    def test_inner_germline_tamper_fails_even_after_outer_rehash(self):
        body = make_body()
        body.causal_experiment_germline = make_germline()
        payload = checkpoint_dict(body)
        payload["causal_experiment_germline"]["world_sha256"] = "7" * 64
        payload["self_evolving_body"]["integrity_sha256"] = integrity_sha256(payload)
        with self.assertRaisesRegex(ValueError, "germline fingerprint mismatch"):
            restore_body(payload)

    def test_v1_checkpoint_restores_without_retroactive_germline(self):
        payload = checkpoint_dict(make_body())
        legacy = copy.deepcopy(payload)
        legacy.pop("causal_experiment_germline")
        legacy["self_evolving_body"]["schema"] = SELF_EVOLVING_BODY_SCHEMA_V1
        legacy["self_evolving_body"]["required_namespaces"] = list(OLD_REQUIRED)
        legacy["self_evolving_body"]["integrity_sha256"] = integrity_sha256(legacy)
        restored = restore_body(legacy)
        self.assertIsNone(restored.causal_experiment_germline)

    def test_checkpoint_never_serializes_verified_authority_boolean(self):
        body = make_body()
        body.causal_experiment_germline = make_germline()
        payload = checkpoint_dict(body)
        serialized = repr(payload["causal_experiment_germline"])
        self.assertNotIn("authority_verified", serialized)
        self.assertIn("authority_reverification_required", serialized)


if __name__ == "__main__":
    unittest.main()
