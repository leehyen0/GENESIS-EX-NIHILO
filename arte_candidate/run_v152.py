from __future__ import annotations

import argparse
import json
from pathlib import Path

OBSERVED = {
    "github_actions_audited_semantic": {
        "good_run_id": 32470656619,
        "good_outcome": "success",
        "integrity_wrong_run_id": 32470670369,
        "integrity_wrong_outcome": "failure",
        "authority_wrong_run_id": 32470794525,
        "authority_wrong_outcome": "failure",
        "surface_escape_run_id": 32470929739,
        "surface_escape_outcome": "failure"
    },
    "vercel_deployment": {
        "good_head": "8168a56c5d5a3e314df55a54c6d178c41a7f41ac",
        "integrity_wrong_head": "8b2c085f84769c12a7d9688ae2ebe94ff467e7d9",
        "authority_wrong_head": "e7057622bd45e12beaf3f690814d3d2578dd5ed1",
        "all_observed_state": "success"
    }
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    artifact = {
        "schema": "arte.external_evidence_role_artifact/v152",
        "observed_evidence_fingerprint": OBSERVED,
        "role_map": {
            "github_actions_audited_semantic": ["execution", "semantic_validity", "authority_boundary"],
            "vercel_deployment": ["execution_availability"]
        },
        "forbidden_inference_used": False,
        "claim_boundary": {
            "AGI": False,
            "ASI": False,
            "independent_organization_custody": False,
            "recursive_acceleration_proven": False
        }
    }
    p = Path(a.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(artifact, indent=2) + "\n")
    print("V152_ROLE_GOOD_WRITTEN")


if __name__ == "__main__":
    main()
