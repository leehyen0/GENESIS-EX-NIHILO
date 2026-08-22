#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SCHEMA_IN = "arte.hidden_candidate_input/v156"
SCHEMA_OUT = "arte.candidate_hidden_decisions/v156"

ROLE_ALIASES = {
    "EXECUTION": "EXECUTION_AVAILABILITY",
    "EXECUTION_AVAILABILITY": "EXECUTION_AVAILABILITY",
    "DEPLOYABILITY": "EXECUTION_AVAILABILITY",
    "RUNTIME_AVAILABILITY": "EXECUTION_AVAILABILITY",
    "SEMANTIC": "SEMANTIC_CORRECTNESS",
    "SEMANTIC_CORRECTNESS": "SEMANTIC_CORRECTNESS",
    "FRESH": "FRESHNESS",
    "FRESHNESS": "FRESHNESS",
    "INDEPENDENCE": "INDEPENDENCE_FIREWALL",
    "INDEPENDENCE_FIREWALL": "INDEPENDENCE_FIREWALL",
    "CLAIM": "CLAIM_BOUNDARY",
    "BOUNDARY": "CLAIM_BOUNDARY",
    "CLAIM_BOUNDARY": "CLAIM_BOUNDARY",
}

PASS_STATUSES = {"PASS", "SUCCESS", "VERIFIED", "AVAILABLE", "OK", "TRUE"}


def norm_token(value):
    if not isinstance(value, str) or not value.strip():
        return None
    token = re.sub(r"[^A-Z0-9]+", "_", value.strip().upper()).strip("_")
    return token or None


def norm_role(value):
    token = norm_token(value)
    if token is None:
        return None
    return ROLE_ALIASES.get(token, token)


def is_bool(value):
    return type(value) is bool


def provider_is_execution_only(provider_class):
    token = norm_token(provider_class) or ""
    markers = ("VERCEL", "DEPLOYMENT", "DEPLOY", "RUNTIME_AVAILABILITY", "HOSTING")
    return any(m in token for m in markers)


def valid_claim_boundary(boundary):
    if not isinstance(boundary, dict):
        return False
    required = ("agi_claimed", "asi_claimed", "independent_custody_claimed")
    if any(k not in boundary or not is_bool(boundary[k]) for k in required):
        return False
    return not any(boundary[k] for k in required)


def valid_parent(parent):
    if not isinstance(parent, dict):
        return False
    for key in ("external_pass", "authority_product_complete"):
        if key not in parent or not is_bool(parent[key]):
            return False
    return parent["external_pass"] and parent["authority_product_complete"]


def decide_case(obj):
    if not isinstance(obj, dict):
        return "BLOCK"

    required_fields = (
        "providers", "evidence", "required_axes",
        "min_independent_classes", "parent", "claim_boundary",
    )
    if any(k not in obj for k in required_fields):
        return "BLOCK"

    if not valid_claim_boundary(obj["claim_boundary"]):
        return "BLOCK"
    if not valid_parent(obj["parent"]):
        return "BLOCK"

    required_axes_raw = obj["required_axes"]
    if not isinstance(required_axes_raw, list) or not required_axes_raw:
        return "BLOCK"
    required_axes = []
    for axis in required_axes_raw:
        role = norm_role(axis)
        if role is None:
            return "BLOCK"
        required_axes.append(role)
    if len(set(required_axes)) != len(required_axes):
        return "BLOCK"

    minimum = obj["min_independent_classes"]
    if type(minimum) is not int or minimum < 1:
        return "BLOCK"

    providers = obj["providers"]
    if not isinstance(providers, list):
        return "BLOCK"
    provider_map = {}
    for p in providers:
        if not isinstance(p, dict):
            return "BLOCK"
        needed = ("provider_id", "provider_class", "status", "evidenced_roles")
        if any(k not in p for k in needed):
            return "BLOCK"
        pid = p["provider_id"]
        pclass = p["provider_class"]
        status = norm_token(p["status"])
        roles_raw = p["evidenced_roles"]
        if not isinstance(pid, str) or not pid or pid in provider_map:
            return "BLOCK"
        if not isinstance(pclass, str) or not pclass:
            return "BLOCK"
        if status is None or not isinstance(roles_raw, list):
            return "BLOCK"
        roles = []
        for r in roles_raw:
            nr = norm_role(r)
            if nr is None:
                return "BLOCK"
            roles.append(nr)
        if provider_is_execution_only(pclass) and any(r != "EXECUTION_AVAILABILITY" for r in roles):
            return "BLOCK"
        provider_map[pid] = {"status": status, "roles": set(roles), "class": pclass}

    evidence = obj["evidence"]
    if not isinstance(evidence, list):
        return "BLOCK"

    qualifying = []
    for e in evidence:
        if not isinstance(e, dict):
            return "BLOCK"
        needed = (
            "source_id", "independence_class", "role",
            "valid", "epoch", "required_epoch", "external_verdict",
        )
        if any(k not in e for k in needed):
            return "BLOCK"

        source_id = e["source_id"]
        iclass = e["independence_class"]
        role = norm_role(e["role"])
        if not isinstance(source_id, str) or not source_id:
            return "BLOCK"
        if not isinstance(iclass, str) or not iclass:
            return "BLOCK"
        if role is None or not is_bool(e["valid"]):
            return "BLOCK"
        if not isinstance(e["epoch"], (str, int)) or not isinstance(e["required_epoch"], (str, int)):
            return "BLOCK"
        verdict = norm_token(e["external_verdict"])
        if verdict is None:
            return "BLOCK"

        source_provider = provider_map.get(source_id)
        if source_provider is not None:
            if source_provider["status"] not in PASS_STATUSES:
                continue
            if role not in source_provider["roles"]:
                continue
            if provider_is_execution_only(source_provider["class"]) and role != "EXECUTION_AVAILABILITY":
                continue

        if (
            e["valid"] is True
            and e["epoch"] == e["required_epoch"]
            and verdict == "PASS"
        ):
            qualifying.append((role, iclass))

    for axis in required_axes:
        if not any(role == axis for role, _ in qualifying):
            return "BLOCK"

    required_axis_set = set(required_axes)
    independent_classes = {iclass for role, iclass in qualifying if role in required_axis_set}
    if len(independent_classes) < minimum:
        return "BLOCK"

    return "PROMOTE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    doc = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA_IN:
        raise SystemExit("bad input schema")
    challenge_id = doc.get("challenge_id")
    cases = doc.get("cases")
    if not isinstance(challenge_id, str) or not challenge_id or not isinstance(cases, list):
        raise SystemExit("malformed challenge envelope")

    seen = set()
    decisions = []
    for case in cases:
        if not isinstance(case, dict):
            raise SystemExit("malformed case")
        cid = case.get("case_id")
        if not isinstance(cid, str) or not cid or cid in seen:
            raise SystemExit("invalid or duplicate case_id")
        seen.add(cid)
        decisions.append({"case_id": cid, "decision": decide_case(case.get("input"))})

    out = {
        "schema": SCHEMA_OUT,
        "challenge_id": challenge_id,
        "decisions": decisions,
        "claim_boundary": {
            "AGI": False,
            "ASI": False,
            "independent_custody": False,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
