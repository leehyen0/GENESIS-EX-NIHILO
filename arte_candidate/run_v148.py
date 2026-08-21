from __future__ import annotations

import argparse
import json
from pathlib import Path

CHALLENGE_ID = "V148-BASE-FROZEN-001"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--candidate-sha", required=True)
    args = ap.parse_args()

    artifact = {
        "schema": "arte.public_candidate_artifact/v148",
        "candidate_sha": args.candidate_sha,
        "challenge_id": CHALLENGE_ID,
        "challenge_digest": "0" * 64,
        "claim_boundary": {
            "AGI": False,
            "ASI": True,
            "independent_organization_custody": False,
            "recursive_acceleration_proven": False,
        },
        "candidate_can_select_verifier_ref": False,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n")
    print("V148_WRONG_CANDIDATE_ARTIFACT_WRITTEN")


if __name__ == "__main__":
    main()
