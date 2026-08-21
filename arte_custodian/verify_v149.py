from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"V149_VERIFY_FAIL: {msg}")
    raise SystemExit(1)


def main() -> None:
    artifact_path = Path(sys.argv[1])
    contract_path = Path(sys.argv[2])
    artifact = json.loads(artifact_path.read_text())
    contract = json.loads(contract_path.read_text())

    if artifact.get("schema") != contract["artifact_schema"]:
        fail("schema mismatch")
    sha = artifact.get("candidate_sha")
    if not isinstance(sha, str) or len(sha) != 40:
        fail("candidate sha malformed")
    expected = hashlib.sha256(
        f"{sha}:{contract['challenge_id']}:audited-base".encode()
    ).hexdigest()
    if artifact.get("challenge_digest") != expected:
        fail("digest mismatch")
    if artifact.get("challenge_id") != contract["challenge_id"]:
        fail("challenge mismatch")
    if artifact.get("claim_boundary") != contract["required_claim_boundary"]:
        fail("claim boundary mismatch")
    if artifact.get("candidate_can_modify_evaluator") is not False:
        fail("candidate evaluator authority invalid")

    receipt = {
        "schema": "arte.audited_base_verification_receipt/v149",
        "verified": True,
        "candidate_sha": sha,
        "challenge_id": contract["challenge_id"],
        "AGI": False,
        "ASI": False,
        "independent_organization_custody": False,
    }
    out = artifact_path.parent / "v149_verification_receipt.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print("V149_VERIFY_PASS")


if __name__ == "__main__":
    main()
