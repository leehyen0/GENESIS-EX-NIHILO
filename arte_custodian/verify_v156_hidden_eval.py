#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def fail(msg: str):
    raise SystemExit(f"V156_HIDDEN_EVAL_FAIL: {msg}")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("expected")
    ap.add_argument("decisions")
    ap.add_argument("freeze_contract")
    ap.add_argument("receipt")
    args = ap.parse_args()

    exp = load(args.expected)
    got = load(args.decisions)
    freeze = load(args.freeze_contract)

    if exp.get("schema") != "arte.hidden_expected/v156":
        fail("bad expected schema")
    if got.get("schema") != "arte.candidate_hidden_decisions/v156":
        fail("bad candidate output schema")
    if freeze.get("schema") != "arte.hidden_candidate_freeze/v156":
        fail("bad freeze schema")

    cid = exp.get("challenge_id")
    if not cid or got.get("challenge_id") != cid or freeze.get("challenge_id") != cid:
        fail("challenge_id mismatch across expected/candidate/freeze")
    if exp.get("generation") != freeze.get("generation"):
        fail("generation mismatch")

    boundary = got.get("claim_boundary")
    required_boundary = {"AGI": False, "ASI": False, "independent_custody": False}
    if boundary != required_boundary:
        fail(f"claim boundary violated: {boundary!r}")

    expected = {x["case_id"]: x for x in exp.get("cases", [])}
    decisions = got.get("decisions")
    if not isinstance(decisions, list):
        fail("decisions must be a list")
    if len(decisions) != len(expected):
        fail("decision count mismatch")

    got_map = {}
    for x in decisions:
        case_id = x.get("case_id")
        decision = x.get("decision")
        if case_id in got_map:
            fail(f"duplicate decision: {case_id}")
        if case_id not in expected:
            fail(f"unknown case_id: {case_id}")
        if decision not in {"PROMOTE", "BLOCK"}:
            fail(f"invalid decision for {case_id}")
        got_map[case_id] = decision

    correct = 0
    failures = []
    tag_total = Counter()
    tag_correct = Counter()
    for case_id, item in expected.items():
        want = item["expected"]
        actual = got_map.get(case_id)
        ok = actual == want
        correct += int(ok)
        for tag in item.get("tags", []):
            tag_total[tag] += 1
            tag_correct[tag] += int(ok)
        if not ok:
            failures.append({"case_id": case_id, "expected": want, "actual": actual, "tags": item.get("tags", [])})

    total = len(expected)
    accuracy = correct / total if total else 0.0
    required_accuracy = float(freeze.get("required_accuracy", 1.0))
    pass_eval = accuracy >= required_accuracy and not failures if required_accuracy >= 1.0 else accuracy >= required_accuracy

    tag_accuracy = {tag: tag_correct[tag] / tag_total[tag] for tag in sorted(tag_total)}
    receipt = {
        "schema": "arte.hidden_external_evaluation_receipt/v156",
        "challenge_id": cid,
        "generation": exp["generation"],
        "candidate_head_sha": freeze.get("candidate_head_sha"),
        "candidate_file_sha256": freeze.get("candidate_file_sha256"),
        "verifier_base_sha": freeze.get("verifier_base_sha"),
        "plaintext_sha256": exp.get("plaintext_sha256"),
        "custodian_id": exp.get("custodian_id"),
        "case_count": total,
        "correct": correct,
        "accuracy": accuracy,
        "required_accuracy": required_accuracy,
        "tag_accuracy": tag_accuracy,
        "failures": failures,
        "verdict": "PASS" if pass_eval else "FAIL",
        "claim_boundary": required_boundary,
    }
    Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.receipt).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": receipt["verdict"], "accuracy": accuracy, "failures": len(failures)}, sort_keys=True))
    if not pass_eval:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
