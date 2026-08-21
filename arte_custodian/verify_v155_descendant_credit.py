from __future__ import annotations

import json
import sys
from pathlib import Path


def fail(code: str) -> None:
    print(f"V155_DESCENDANT_CREDIT_FAIL:{code}")
    raise SystemExit(1)


def main() -> None:
    artifact = json.loads(Path(sys.argv[1]).read_text())
    contract = json.loads(Path(sys.argv[2]).read_text())
    if artifact.get("schema") != "arte.external_outcome_descendant_credit_artifact/v155": fail("SCHEMA")
    if artifact.get("epoch") != contract["epoch"]: fail("EPOCH")
    if artifact.get("observed_v154_outcomes") != contract["v154_external_outcomes"]: fail("OUTCOME_FINGERPRINT")
    if artifact.get("selected_parent") != contract["required_parent"]: fail("WRONG_PARENT")
    if artifact.get("rejected_candidates") != contract["required_rejected"]: fail("REJECTION_SET")
    if artifact.get("cognition_update") != contract["required_cognition_update"]: fail("COGNITION_UPDATE")
    if artifact.get("deployment_only_can_rescue_semantic_failure") is not False: fail("AUTHORITY_COLLAPSE")
    if artifact.get("claim_boundary") != contract["claim_boundary"]: fail("CLAIM_BOUNDARY")
    if artifact.get("candidate_can_modify_credit_verifier") is not False: fail("SURFACE_AUTHORITY")
    receipt = {
      "schema":"arte.external_outcome_descendant_credit_receipt/v155",
      "verified":True,
      "selected_parent":contract["required_parent"],
      "rejected":contract["required_rejected"],
      "world_outcome_consumed_by_descendant":True,
      "AGI":False,"ASI":False
    }
    out=Path(sys.argv[1]).parent/"v155_descendant_credit_receipt.json"
    out.write_text(json.dumps(receipt,indent=2)+"\n")
    print("V155_DESCENDANT_CREDIT_PASS")

if __name__=="__main__":
    main()
