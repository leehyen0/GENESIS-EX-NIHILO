#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def decide(inp):
    if not isinstance(inp, dict):
        return "BLOCK"

    boundary = inp.get("claim_boundary")
    if not isinstance(boundary, dict):
        return "BLOCK"
    for key in ("agi_claimed", "asi_claimed", "independent_custody_claimed"):
        if bool(boundary.get(key, False)):
            return "BLOCK"

    parent = inp.get("parent")
    if not isinstance(parent, dict):
        return "BLOCK"
    if parent.get("external_pass") is not True or parent.get("authority_product_complete") is not True:
        return "BLOCK"

    required_axes = inp.get("required_axes")
    if not isinstance(required_axes, list) or not required_axes or any(not isinstance(x, str) for x in required_axes):
        return "BLOCK"

    providers = inp.get("providers")
    evidence = inp.get("evidence")
    if not isinstance(providers, list) or not isinstance(evidence, list):
        return "BLOCK"

    current = []
    required_evidence_fields = {
        "source_id", "independence_class", "role", "valid",
        "epoch", "required_epoch", "external_verdict"
    }
    for item in evidence:
        if not isinstance(item, dict) or not required_evidence_fields.issubset(item):
            continue
        if item.get("valid") is True and item.get("epoch") == item.get("required_epoch") and item.get("external_verdict") == "pass":
            current.append(item)

    for axis in required_axes:
        if axis == "EXECUTION_AVAILABILITY":
            satisfied = any(
                isinstance(p, dict)
                and p.get("status") == "success"
                and axis in p.get("evidenced_roles", [])
                for p in providers
            )
        else:
            satisfied = any(item.get("role") == axis for item in current)
        if not satisfied:
            return "BLOCK"

    minimum = inp.get("min_independent_classes")
    if not isinstance(minimum, int) or minimum < 1:
        return "BLOCK"
    independent_classes = {
        item.get("independence_class")
        for item in current
        if item.get("role") in required_axes and item.get("role") != "EXECUTION_AVAILABILITY"
    }
    if None in independent_classes:
        return "BLOCK"
    if len(independent_classes) < minimum:
        return "BLOCK"

    return "PROMOTE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if payload.get("schema") != "arte.hidden_candidate_input/v156":
        raise SystemExit("bad input schema")

    decisions = []
    for case in payload.get("cases", []):
        decisions.append({"case_id": case["case_id"], "decision": decide(case.get("input"))})

    out = {
        "schema": "arte.candidate_hidden_decisions/v156",
        "challenge_id": payload.get("challenge_id"),
        "decisions": decisions,
        "claim_boundary": {"AGI": False, "ASI": False, "independent_custody": False}
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
