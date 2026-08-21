from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CHALLENGE = "V149-AUDITED-BASE-001"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--candidate-sha", required=True)
    a = ap.parse_args()
    digest = hashlib.sha256(
        f"{a.candidate_sha}:{CHALLENGE}:audited-base".encode()
    ).hexdigest()
    artifact = {
        "schema": "arte.public_candidate_artifact/v149",
        "candidate_sha": a.candidate_sha,
        "challenge_id": CHALLENGE,
        "challenge_digest": digest,
        "claim_boundary": {
            "AGI": False,
            "ASI": False,
            "independent_organization_custody": False,
            "recursive_acceleration_proven": False
        },
        "candidate_can_modify_evaluator": False
    }
    p = Path(a.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(artifact, indent=2) + "\n")
    print("V149_GOOD_WRITTEN")


if __name__ == "__main__":
    main()
