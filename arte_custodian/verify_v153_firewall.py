from __future__ import annotations

import json
import sys
from pathlib import Path


def fail(code: str) -> None:
    print(f"V153_FIREWALL_FAIL:{code}")
    raise SystemExit(1)


def main() -> None:
    artifact = json.loads(Path(sys.argv[1]).read_text())
    contract = json.loads(Path(sys.argv[2]).read_text())

    if artifact.get("schema") != "arte.external_epistemic_firewall_artifact/v153":
        fail("SCHEMA")
    if artifact.get("epoch") != contract["epoch"]:
        fail("STALE_EPOCH")
    if artifact.get("current_evidence") != contract["required_current_evidence"]:
        fail("STALE_OR_ALIASED_EVIDENCE")
    if artifact.get("source_classes") != contract["source_classes"]:
        fail("SOURCE_ROLE")

    classes = artifact.get("independence_classes")
    if classes != contract["required_independence_classes"]:
        fail("CLONED_OR_MISSING_INDEPENDENCE_CLASS")
    if len(classes) != len(set(classes)):
        fail("DUPLICATE_INDEPENDENCE_CLASS")

    if artifact.get("authority_sources") != contract["authority_sources"]:
        fail("AUTHORITY_SOURCE")
    if artifact.get("forbidden_inferences_used"):
        fail("FORBIDDEN_INFERENCE")
    if artifact.get("claim_boundary") != contract["claim_boundary"]:
        fail("CLAIM_BOUNDARY")
    if artifact.get("candidate_can_modify_firewall") is not False:
        fail("SURFACE_AUTHORITY")

    receipt = {
        "schema": "arte.external_epistemic_firewall_receipt/v153",
        "verified": True,
        "epoch": contract["epoch"],
        "gates": ["surface", "freshness", "source_role", "independence", "authority", "claim_boundary"],
        "independent_organization_custody": False,
        "AGI": False,
        "ASI": False
    }
    out = Path(sys.argv[1]).parent / "v153_firewall_receipt.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print("V153_FIREWALL_PASS")


if __name__ == "__main__":
    main()
