#!/usr/bin/env python3
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path


def fail(msg: str):
    raise SystemExit(f"V156_COMMIT_REVEAL_FAIL: {msg}")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("commitment")
    ap.add_argument("reveal")
    ap.add_argument("candidate_input")
    ap.add_argument("expected_output")
    args = ap.parse_args()

    c = load_json(args.commitment)
    r = load_json(args.reveal)

    if c.get("schema") != "arte.hidden_challenge_commitment/v156":
        fail("bad commitment schema")
    if r.get("schema") != "arte.hidden_challenge_reveal/v156":
        fail("bad reveal schema")
    if not c.get("challenge_id") or c.get("challenge_id") != r.get("challenge_id"):
        fail("challenge_id mismatch")
    if c.get("generation") not in {"G1", "G2", "G3"}:
        fail("bad commitment generation")
    if c.get("custodian_id") != r.get("custodian_id"):
        fail("custodian_id mismatch")
    if c.get("key_withheld") is not True:
        fail("commitment did not state key_withheld=true")
    if r.get("reveal_only_after_candidate_freeze") is not True:
        fail("reveal missing post-freeze assertion")
    required_claim = {"AGI": False, "ASI": False, "independent_organization_custody": False}
    if c.get("claim_boundary") != required_claim:
        fail("commitment claim boundary mismatch")

    try:
        ciphertext = base64.b64decode(c["ciphertext_b64"], validate=True)
        key = base64.b64decode(r["key_b64"], validate=True)
    except Exception as e:
        fail(f"invalid base64: {e}")

    if len(ciphertext) != int(c.get("byte_length", -1)):
        fail("ciphertext length does not match commitment")
    if len(key) != len(ciphertext):
        fail("one-time-pad key length mismatch")
    if sha256(ciphertext) != c.get("ciphertext_sha256"):
        fail("ciphertext SHA-256 mismatch")
    if r.get("key_sha256") and sha256(key) != r.get("key_sha256"):
        fail("key SHA-256 mismatch")

    plaintext = bytes(a ^ b for a, b in zip(ciphertext, key))
    if sha256(plaintext) != c.get("plaintext_sha256"):
        fail("plaintext commitment mismatch")

    try:
        hidden = json.loads(plaintext.decode("utf-8"))
    except Exception as e:
        fail(f"decrypted hidden challenge is not UTF-8 JSON: {e}")

    if hidden.get("schema") != "arte.hidden_challenge/v156":
        fail("bad hidden challenge schema")
    if hidden.get("challenge_id") != c["challenge_id"]:
        fail("decrypted challenge_id mismatch")
    if hidden.get("generation") != c.get("generation"):
        fail("decrypted generation mismatch")
    if hidden.get("case_weights_committed") is not True:
        fail("case_weights_committed must be true")

    cases = hidden.get("cases")
    if not isinstance(cases, list) or not cases:
        fail("hidden challenge must contain cases")

    seen = set()
    public_cases = []
    expected = []
    explicit_weights = True
    for item in cases:
        cid = item.get("case_id")
        if not isinstance(cid, str) or not cid or cid in seen:
            fail("case_id must be unique non-empty string")
        seen.add(cid)
        if "input" not in item or not isinstance(item["input"], dict):
            fail(f"missing input for {cid}")
        verdict = item.get("expected")
        if verdict not in {"PROMOTE", "BLOCK"}:
            fail(f"invalid expected verdict for {cid}")
        if "weight" in item:
            weight = item["weight"]
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) or float(weight) <= 0:
                fail(f"invalid weight for {cid}")
            weight = float(weight)
        else:
            explicit_weights = False
            weight = 1.0
        public_cases.append({"case_id": cid, "input": item["input"]})
        expected.append({"case_id": cid, "expected": verdict, "tags": item.get("tags", []), "weight": weight})

    candidate_input = {
        "schema": "arte.hidden_candidate_input/v156",
        "challenge_id": hidden["challenge_id"],
        "generation": hidden["generation"],
        "cases": public_cases,
    }
    expected_doc = {
        "schema": "arte.hidden_expected/v156",
        "challenge_id": hidden["challenge_id"],
        "generation": hidden["generation"],
        "plaintext_sha256": c["plaintext_sha256"],
        "custodian_id": c.get("custodian_id"),
        "case_weights_committed": True,
        "weights_explicit": explicit_weights,
        "cases": expected,
    }

    Path(args.candidate_input).parent.mkdir(parents=True, exist_ok=True)
    Path(args.expected_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.candidate_input).write_text(json.dumps(candidate_input, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.expected_output).write_text(json.dumps(expected_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "COMMIT_REVEAL_VERIFIED",
        "challenge_id": hidden["challenge_id"],
        "generation": hidden["generation"],
        "case_count": len(cases),
        "weights_explicit": explicit_weights,
        "plaintext_sha256": c["plaintext_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
