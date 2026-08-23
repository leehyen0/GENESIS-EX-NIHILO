from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def _int_attr(root: ET.Element, name: str) -> int:
    raw = root.attrib.get(name, "0")
    try:
        return int(float(raw))
    except ValueError:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--junit", required=True)
    parser.add_argument("--install-exit", type=int, required=True)
    parser.add_argument("--pytest-exit", type=int, required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    junit_path = Path(args.junit)

    result = {
        "artifact_type": "arte.external_public_world_observation/v1",
        "canonical_parent": manifest["canonical_parent"],
        "external_repository": manifest["external_repository"],
        "external_commit": manifest["external_commit"],
        "external_issue": manifest["external_issue"],
        "prior_external_workflow_run": manifest["prior_external_workflow_run"],
        "install_exit": args.install_exit,
        "pytest_exit": args.pytest_exit,
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
    }

    if args.install_exit != 0 or not junit_path.exists():
        result.update(
            status="EXTERNAL_OBSERVATION_UNAVAILABLE",
            live_comparison_observed=False,
            reason="installation_failed_or_no_junit",
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    try:
        root = ET.parse(junit_path).getroot()
    except (ET.ParseError, OSError) as exc:
        result.update(
            status="EXTERNAL_OBSERVATION_UNAVAILABLE",
            live_comparison_observed=False,
            reason=f"invalid_junit:{type(exc).__name__}",
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    tests = sum(_int_attr(s, "tests") for s in suites)
    failures = sum(_int_attr(s, "failures") for s in suites)
    errors = sum(_int_attr(s, "errors") for s in suites)
    skipped = sum(_int_attr(s, "skipped") for s in suites)

    # If pytest reported a no-tests/collection/infrastructure condition, do not
    # reinterpret it as a changed world contract merely because a partial XML
    # file exists.
    if tests <= 0 or args.pytest_exit not in (0, 1):
        status = "EXTERNAL_OBSERVATION_UNAVAILABLE"
        live_observed = False
        reason = "no_valid_live_test_comparison"
    elif failures > 0 or errors > 0 or args.pytest_exit == 1:
        status = "EXTERNAL_RESIDUAL_PERSISTS"
        live_observed = True
        reason = "live_tests_compared_and_red"
    else:
        status = "EXTERNAL_WORLD_RECOVERED_OR_TRANSIENT_FAILURE"
        live_observed = True
        reason = "live_tests_compared_and_green"

    result.update(
        status=status,
        live_comparison_observed=live_observed,
        reason=reason,
        tests=tests,
        failures=failures,
        errors=errors,
        skipped=skipped,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
