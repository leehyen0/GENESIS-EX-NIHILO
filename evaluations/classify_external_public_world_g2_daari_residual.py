from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


TARGET_TEST = "test_max_tokens_actually_truncates"
EXPECTED_ASSERTION = "an 8-token answer cannot have finished counting"


def _int_attr(root: ET.Element, name: str) -> int:
    raw = root.attrib.get(name, "0")
    try:
        return int(float(raw))
    except ValueError:
        return 0


def _all_suites(root: ET.Element):
    return [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))


def _target_case(root: ET.Element):
    for case in root.findall(".//testcase"):
        if case.attrib.get("name") == TARGET_TEST:
            return case
    if root.tag == "testcase" and root.attrib.get("name") == TARGET_TEST:
        return root
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--junit", required=True)
    parser.add_argument("--install-exit", type=int, required=True)
    parser.add_argument("--ollama-exit", type=int, required=True)
    parser.add_argument("--model-pull-exit", type=int, required=True)
    parser.add_argument("--pytest-exit", type=int, required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    junit_path = Path(args.junit)

    result = {
        "artifact_type": "arte.external_public_world_observation/v2",
        "canonical_parent": manifest["canonical_parent"],
        "external_repository": manifest["external_repository"],
        "external_commit": manifest["external_commit"],
        "external_issue": manifest["external_issue"],
        "target_test": manifest["target_test"],
        "external_environment_matches_original_exactly": False,
        "original_ollama_version_known": False,
        "original_model_digest_known": False,
        "public_exposed_development_experience": True,
        "heldout_authority": False,
        "independent_organizational_custody": False,
        "current_body_patch_generated": False,
        "current_body_cognition_promoted": False,
        "AGI": False,
        "ASI": False,
        "global_recursive_acceleration": False,
        "physical_world_closure": False,
        "foundation_weight_change": False,
        "install_exit": args.install_exit,
        "ollama_exit": args.ollama_exit,
        "model_pull_exit": args.model_pull_exit,
        "pytest_exit": args.pytest_exit,
    }

    if any(code != 0 for code in (args.install_exit, args.ollama_exit, args.model_pull_exit)):
        result.update(
            status="EXTERNAL_OBSERVATION_UNAVAILABLE",
            persistent_residual_observed=False,
            reason="dependency_or_model_setup_failed",
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if not junit_path.exists():
        result.update(
            status="EXTERNAL_OBSERVATION_UNAVAILABLE",
            persistent_residual_observed=False,
            reason="missing_junit",
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    try:
        root = ET.parse(junit_path).getroot()
    except (ET.ParseError, OSError) as exc:
        result.update(
            status="EXTERNAL_OBSERVATION_UNAVAILABLE",
            persistent_residual_observed=False,
            reason=f"invalid_junit:{type(exc).__name__}",
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    suites = _all_suites(root)
    tests = sum(_int_attr(s, "tests") for s in suites)
    failures = sum(_int_attr(s, "failures") for s in suites)
    errors = sum(_int_attr(s, "errors") for s in suites)
    skipped = sum(_int_attr(s, "skipped") for s in suites)
    case = _target_case(root)

    result.update(tests=tests, failures=failures, errors=errors, skipped=skipped)

    if tests != 1 or case is None or args.pytest_exit not in (0, 1):
        result.update(
            status="EXTERNAL_OBSERVATION_UNAVAILABLE",
            persistent_residual_observed=False,
            reason="target_test_not_cleanly_observed",
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    failure = case.find("failure")
    error = case.find("error")
    if error is not None:
        result.update(
            status="EXTERNAL_OBSERVATION_UNAVAILABLE",
            persistent_residual_observed=False,
            reason="target_test_errored_not_semantically_failed",
        )
    elif failure is not None:
        failure_text = " ".join(
            part for part in (failure.attrib.get("message", ""), failure.text or "") if part
        ).lower()
        if EXPECTED_ASSERTION in failure_text:
            result.update(
                status="EXTERNAL_RESIDUAL_PERSISTS_CROSS_ENVIRONMENT",
                persistent_residual_observed=True,
                reason="same_semantic_assertion_failed_on_fresh_github_runner",
            )
        else:
            result.update(
                status="EXTERNAL_OBSERVATION_UNAVAILABLE",
                persistent_residual_observed=False,
                reason="different_failure_signature",
            )
    elif args.pytest_exit == 0:
        result.update(
            status="EXTERNAL_RESIDUAL_NOT_REPRODUCED_CROSS_ENVIRONMENT",
            persistent_residual_observed=False,
            reason="target_passed_on_fresh_github_runner_but_original_environment_not_exactly_reconstructed",
        )
    else:
        result.update(
            status="EXTERNAL_OBSERVATION_UNAVAILABLE",
            persistent_residual_observed=False,
            reason="pytest_red_without_target_failure",
        )

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
