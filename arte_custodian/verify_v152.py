from __future__ import annotations

import json
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"V152_VERIFY_FAIL: {msg}")
    raise SystemExit(1)


def main() -> None:
    artifact = json.loads(Path(sys.argv[1]).read_text())
    contract = json.loads(Path(sys.argv[2]).read_text())

    if artifact.get("schema") != "arte.external_evidence_role_artifact/v152":
        fail("schema mismatch")
    if artifact.get("role_map") != contract["required_role_map"]:
        fail("evidence role map mismatch")
    if artifact.get("forbidden_inference_used") is not False:
        fail("forbidden deployment-to-semantic inference used")
    if artifact.get("claim_boundary") != contract["claim_boundary"]:
        fail("claim boundary mismatch")
    if artifact.get("observed_evidence_fingerprint") != contract["observed_external_evidence"]:
        fail("external evidence fingerprint mismatch")

    receipt = {
        "schema": "arte.external_evidence_roles_verification/v152",
        "verified": True,
        "principle": "execution evidence, semantic evidence, and authority evidence are distinct",
        "AGI": False,
        "ASI": False,
        "independent_organization_custody": False
    }
    out = Path(sys.argv[1]).parent / "v152_verification_receipt.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print("V152_VERIFY_PASS")


if __name__ == "__main__":
    main()
