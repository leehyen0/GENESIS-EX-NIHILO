from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("classify_external_public_world_residual.py")
MANIFEST = Path(__file__).with_name("external_public_world_g1_swiss_manifest.json")


class ExternalWorldResidualClassifierTests(unittest.TestCase):
    def run_case(self, xml: str | None, install_exit: int, pytest_exit: int) -> dict:
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
                    "--pytest-exit",
                    str(pytest_exit),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            return json.loads(proc.stdout)

    def test_red_valid_live_suite_is_persistent_residual_not_repair_success(self):
        result = self.run_case(
            '<testsuite tests="23" failures="2" errors="0" skipped="0"/>', 0, 1
        )
        self.assertEqual(result["status"], "EXTERNAL_RESIDUAL_PERSISTS")
        self.assertTrue(result["live_comparison_observed"])
        self.assertFalse(result["current_body_patch_generated"])
        self.assertFalse(result["current_body_cognition_promoted"])

    def test_green_live_suite_is_recovery_not_cognition_promotion(self):
        result = self.run_case(
            '<testsuite tests="23" failures="0" errors="0" skipped="0"/>', 0, 0
        )
        self.assertEqual(result["status"], "EXTERNAL_WORLD_RECOVERED_OR_TRANSIENT_FAILURE")
        self.assertTrue(result["live_comparison_observed"])
        self.assertFalse(result["current_body_cognition_promoted"])

    def test_install_failure_is_observation_unavailable(self):
        result = self.run_case(None, 1, 99)
        self.assertEqual(result["status"], "EXTERNAL_OBSERVATION_UNAVAILABLE")
        self.assertFalse(result["live_comparison_observed"])

    def test_collection_or_no_tests_cannot_be_called_world_change(self):
        result = self.run_case(
            '<testsuite tests="0" failures="0" errors="0" skipped="0"/>', 0, 5
        )
        self.assertEqual(result["status"], "EXTERNAL_OBSERVATION_UNAVAILABLE")
        self.assertFalse(result["live_comparison_observed"])


if __name__ == "__main__":
    unittest.main()
