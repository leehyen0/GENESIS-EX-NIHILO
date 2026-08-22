from __future__ import annotations

import argparse
import json
from pathlib import Path

CURRENT = {
    "semantic_good_run": 32471150492,
    "semantic_role_wrong_run": 32471163624,
    "authority_wrong_run": 32470794525,
    "surface_escape_run": 32470929739,
    "vercel_good_head": "af1d95821348ee1ffecdc57d9263eac9134f5de4",
    "vercel_wrong_head": "864ef5726ca966a4957b22fcabd2db85099edcdc"
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    artifact = {
        "schema": "arte.external_epistemic_firewall_artifact/v153",
        "epoch": "2026-08-21T10:07Z",
        "current_evidence": CURRENT,
        "source_classes": {
            "github_actions_audited": "SEMANTIC_AUTHORITY_EVALUATOR",
            "vercel_deployment": "EXECUTION_AVAILABILITY_ONLY"
        },
        "independence_classes": ["github_actions_audited", "github_actions_audited"],
        "authority_sources": ["github_actions_audited"],
        "forbidden_inferences_used": False,
        "claim_boundary": {
            "AGI": False,
            "ASI": False,
            "independent_organization_custody": False,
            "recursive_acceleration_proven": False
        },
        "candidate_can_modify_firewall": False
    }
    p = Path(a.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(artifact, indent=2) + "\n")
    print("V153_FIREWALL_CLONE_WRITTEN")


if __name__ == "__main__":
    main()
