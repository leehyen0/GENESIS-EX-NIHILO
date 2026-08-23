from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("classify_external_public_world_g2_daari_residual.py")
MANIFEST = Path(__file__).with_name("external_public_world_g2_daari_manifest.json")


class DaariExternalResidualClassifierTests(unittest.TestCase):
    def run_case(
        self,
        xml: str | None,
        *,
        install_exit: int = 0,
        ollama_exit: int = 0,
        model_pull_exit: int = 0,
        pytest_exit: int = 0,
    ) -> dict:
        with tempfile.TemporaryDirectory() as td:
            junit = Path(td) / "report.xml"
            if xml is not None:
                junit.write_text(xml, encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--manifest",
                    str(MANIFEST),
                    "--junit",
                    str(junit),
                    "--install-exit",
                    str(install_exit),
                    "--ollama-exit",
                    str(ollama_exit),
                    "--model-pull-exit",
                    str(model_pull_exit),
                    "--pytest-exit",
                    str(pytest_exit),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            return json.loads(proc.stdout)

    def test_same_semantic_assertion_is_persistent_cross_environment_residual(self):
        xml = (
            '<testsuite tests="1" failures="1" errors="0" skipped="0">'
            '<testcase classname="tests.integration.test_sampling_live" '
            'name="test_max_tokens_actually_truncates">'
            '<failure message="AssertionError: an 8-token answer cannot have finished counting">'
            'AssertionError: an 8-token answer cannot have finished counting'
            '</failure></testcase></testsuite>'
        )
        result = self.run_case(xml, pytest_exit=1)
        self.assertEqual(result["status"], "EXTERNAL_RESIDUAL_PERSISTS_CROSS_ENVIRONMENT")
        self.assertTrue(result["persistent_residual_observed"])
        self.assertFalse(result["current_body_patch_generated"])

    def test_green_is_non_reproduction_not_world_recovery(self):
        xml = (
            '<testsuite tests="1" failures="0" errors="0" skipped="0">'
            '<testcase classname="tests.integration.test_sampling_live" '
            'name="test_max_tokens_actually_truncates"/>'
            '</testsuite>'
        )
        result = self.run_case(xml, pytest_exit=0)
        self.assertEqual(result["status"], "EXTERNAL_RESIDUAL_NOT_REPRODUCED_CROSS_ENVIRONMENT")
        self.assertFalse(result["persistent_residual_observed"])
        self.assertFalse(result["external_environment_matches_original_exactly"])

    def test_different_failure_signature_is_not_world_evidence(self):
        xml = (
            '<testsuite tests="1" failures="1" errors="0" skipped="0">'
            '<testcase classname="tests.integration.test_sampling_live" '
            'name="test_max_tokens_actually_truncates">'
            '<failure message="Connection refused">Connection refused</failure>'
            '</testcase></testsuite>'
        )
        result = self.run_case(xml, pytest_exit=1)
        self.assertEqual(result["status"], "EXTERNAL_OBSERVATION_UNAVAILABLE")
        self.assertFalse(result["persistent_residual_observed"])

    def test_dependency_or_model_failure_is_not_residual(self):
        result = self.run_case(None, model_pull_exit=1, pytest_exit=99)
        self.assertEqual(result["status"], "EXTERNAL_OBSERVATION_UNAVAILABLE")
        self.assertFalse(result["persistent_residual_observed"])

    def test_no_or_wrong_target_collection_is_unavailable(self):
        xml = '<testsuite tests="0" failures="0" errors="0" skipped="0"/>'
        result = self.run_case(xml, pytest_exit=5)
        self.assertEqual(result["status"], "EXTERNAL_OBSERVATION_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
