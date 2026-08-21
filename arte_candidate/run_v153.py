from __future__ import annotations

import argparse
import json
from pathlib import Path

STALE = {
    "semantic_good_run": 32470656619,
    "semantic_role_wrong_run": 32470670369,
    "authority_wrong_run": 32470794525,
    "surface_escape_run": 32470929739,
    "vercel_good_head": "8168a56c5d5a3e314df55a54c6d178c41a7f41ac",
    "vercel_wrong_head": "8b2c085f84769c12a7d9688ae2ebe94ff467e7d9"
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    artifact = {
        "schema": "arte.external_epistemic_firewall_artifact/v153",
        "epoch": "2026-08-21T10:07Z",
        "current_evidence": STALE,
        "source_classes": {
            "github_actions_audited": "SEMANTIC_AUTHORITY_EVALUATOR",
            "vercel_deployment": "EXECUTION_AVAILABILITY_ONLY"
        },
        "independence_classes": ["github_actions_audited", "vercel_deployment"],
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
    print("V153_FIREWALL_STALE_WRITTEN")


if __name__ == "__main__":
    main()
