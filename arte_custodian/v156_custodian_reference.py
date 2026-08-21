#!/usr/bin/env python3
"""Reference utility for an EXTERNAL custodian.

Do not run this inside the candidate/ARTE trust boundary for an independent-custody claim.
It creates an information-theoretic one-time-pad commitment from exact UTF-8 challenge bytes.
"""
import argparse
import base64
import hashlib
import json
import secrets
from pathlib import Path


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hidden_challenge_json")
    ap.add_argument("--custodian-id", required=True)
    ap.add_argument("--commitment-out", required=True)
    ap.add_argument("--reveal-out", required=True)
    args = ap.parse_args()

    plaintext = Path(args.hidden_challenge_json).read_bytes()
    hidden = json.loads(plaintext.decode("utf-8"))
    if hidden.get("schema") != "arte.hidden_challenge/v156":
        raise SystemExit("hidden challenge schema must be arte.hidden_challenge/v156")
    challenge_id = hidden.get("challenge_id")
    if not challenge_id:
        raise SystemExit("hidden challenge requires challenge_id")

    key = secrets.token_bytes(len(plaintext))
    ciphertext = bytes(a ^ b for a, b in zip(plaintext, key))

    commitment = {
        "schema": "arte.hidden_challenge_commitment/v156",
        "challenge_id": challenge_id,
        "generation": hidden.get("generation"),
        "custodian_id": args.custodian_id,
        "byte_length": len(plaintext),
        "plaintext_sha256": sha256(plaintext),
        "ciphertext_sha256": sha256(ciphertext),
        "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
        "key_withheld": True,
        "claim_boundary": {"AGI": False, "ASI": False, "independent_organization_custody": False}
    }
    reveal = {
        "schema": "arte.hidden_challenge_reveal/v156",
        "challenge_id": challenge_id,
        "custodian_id": args.custodian_id,
        "key_sha256": sha256(key),
        "key_b64": base64.b64encode(key).decode("ascii"),
        "reveal_only_after_candidate_freeze": True
    }

    Path(args.commitment_out).write_text(json.dumps(commitment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.reveal_out).write_text(json.dumps(reveal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "challenge_id": challenge_id,
        "plaintext_sha256": commitment["plaintext_sha256"],
        "ciphertext_sha256": commitment["ciphertext_sha256"],
        "commitment_out": args.commitment_out,
        "reveal_out": args.reveal_out,
        "IMPORTANT": "Return commitment_out now. Keep reveal_out private until the exact candidate freeze is shown."
    }, sort_keys=True))


if __name__ == "__main__":
    main()
