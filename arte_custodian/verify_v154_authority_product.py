from __future__ import annotations

import json
import sys
from pathlib import Path


def fail(code: str) -> None:
    print(f"V154_AUTHORITY_PRODUCT_FAIL:{code}")
    raise SystemExit(1)


def main() -> None:
    artifact = json.loads(Path(sys.argv[1]).read_text())
    contract = json.loads(Path(sys.argv[2]).read_text())

    if artifact.get("schema") != "arte.cross_substrate_authority_product_artifact/v154":
        fail("SCHEMA")
    if artifact.get("epoch") != contract["epoch"]:
        fail("STALE_EPOCH")
    if artifact.get("evidence") != contract["required_evidence"]:
        fail("STALE_OR_ALIASED_EVIDENCE")
    if artifact.get("authority_vector") != contract["required_authority_vector"]:
        fail("AUTHORITY_VECTOR")
    if artifact.get("axis_providers") != contract["required_axis_providers"]:
        fail("PROVIDER_ROLE")

    capabilities = contract["provider_capabilities"]
    for axis, provider in artifact["axis_providers"].items():
        if provider not in capabilities:
            fail("UNKNOWN_PROVIDER")
        if axis not in capabilities[provider]:
            fail("PROVIDER_CAPABILITY_ESCAPE")

    if artifact.get("provider_role_collapse"):
        fail("PROVIDER_ROLE_COLLAPSE")
    if artifact.get("source_dropout"):
        fail("SOURCE_DROPOUT")
    if artifact.get("stale_epoch_mix"):
        fail("STALE_EPOCH_MIX")
    if artifact.get("forbidden_inferences_used"):
        fail("FORBIDDEN_INFERENCE")
    if artifact.get("claim_boundary") != contract["claim_boundary"]:
        fail("CLAIM_BOUNDARY")
    if artifact.get("candidate_can_modify_authority_product") is not False:
        fail("SURFACE_AUTHORITY")

    if not all(artifact["authority_vector"].values()):
        fail("INCOMPLETE_PRODUCT")

    receipt = {
        "schema": "arte.cross_substrate_authority_product_receipt/v154",
        "verified": True,
        "epoch": contract["epoch"],
        "authority_axes": list(contract["required_authority_vector"].keys()),
        "providers": sorted(set(contract["required_axis_providers"].values())),
        "promotion": "BOUNDED_DESCENDANT_ONLY",
        "independent_organization_custody": False,
        "AGI": False,
        "ASI": False
    }
    out = Path(sys.argv[1]).parent / "v154_authority_product_receipt.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print("V154_AUTHORITY_PRODUCT_PASS")


if __name__ == "__main__":
    main()
