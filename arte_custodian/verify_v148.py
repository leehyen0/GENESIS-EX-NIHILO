from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"V148_VERIFY_FAIL: {msg}")
    raise SystemExit(1)


def main() -> None:
    if len(sys.argv) != 3:
        fail("usage: verify_v148.py ARTIFACT CONTRACT")

    artifact_path = Path(sys.argv[1])
    contract_path = Path(sys.argv[2])
    artifact = json.loads(artifact_path.read_text())
    contract = json.loads(contract_path.read_text())

    if artifact.get("schema") != contract["artifact_schema"]:
        fail("artifact schema mismatch")

    candidate_sha = artifact.get("candidate_sha")
    if not isinstance(candidate_sha, str) or len(candidate_sha) != 40:
        fail("candidate_sha missing or malformed")

    expected = hashlib.sha256(
        f"{candidate_sha}:{contract['challenge_id']}".encode("utf-8")
    ).hexdigest()
    if artifact.get("challenge_digest") != expected:
        fail("challenge digest mismatch")

    if artifact.get("challenge_id") != contract["challenge_id"]:
        fail("challenge id mismatch")

    required = contract["required_claim_boundary"]
    actual = artifact.get("claim_boundary")
    if actual != required:
        fail(f"claim boundary mismatch: expected={required!r} actual={actual!r}")

    if artifact.get("candidate_can_select_verifier_ref") is not False:
        fail("candidate must not select verifier ref")

    result = {
        "schema": "arte.base_frozen_verification_receipt/v148",
        "verified": True,
        "candidate_sha": candidate_sha,
        "challenge_id": contract["challenge_id"],
        "verifier_source": "base-main",
        "contract_source": "base-main",
        "independent_organization_custody": False,
        "AGI": False,
        "ASI": False,
    }
    out = artifact_path.parent / "base_frozen_verification_receipt.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("V148_VERIFY_PASS")


if __name__ == "__main__":
    main()
