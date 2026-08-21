#!/usr/bin/env python3
"""Frozen V156 residual-to-descendant compiler.

This compiler is intentionally conservative. After G1 freeze, no human structural
edit is allowed. Given the exact previous candidate source and an external V156
hidden-evaluation receipt, it emits the next candidate source by preserving all
behavioral code and embedding only cryptographic lineage metadata. If the public
semantics require a behavioral repair that was not precommitted before G1, this
compiler MUST NOT invent one; recursive_acceleration_proven remains false instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MARKER = "# ARTE_V156_GENERATED_LINEAGE="


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--previous-candidate", required=True)
    ap.add_argument("--external-receipt", required=True)
    ap.add_argument("--generation", choices=["G2", "G3"], required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    previous = Path(args.previous_candidate).read_text(encoding="utf-8")
    receipt_bytes = Path(args.external_receipt).read_bytes()
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    if receipt.get("schema") != "arte.hidden_external_evaluation_receipt/v156":
        raise SystemExit("external receipt schema mismatch")
    if receipt.get("verdict") not in {"PASS", "FAIL"}:
        raise SystemExit("external receipt verdict missing")

    # Remove only a prior generated lineage marker; behavioral source is untouched.
    lines = previous.splitlines()
    lines = [line for line in lines if not line.startswith(MARKER)]
    lineage = {
        "generation": args.generation,
        "parent_candidate_sha256": sha256(previous.encode("utf-8")),
        "external_receipt_sha256": sha256(receipt_bytes),
        "external_verdict": receipt["verdict"],
        "accuracy": receipt.get("accuracy"),
        "failure_count": len(receipt.get("failures", [])),
        "behavioral_mutation": False,
        "reason": "public semantics already fully encoded; no post-G1 human repair permitted",
    }
    marker = MARKER + json.dumps(lineage, sort_keys=True, separators=(",", ":"))

    if lines and lines[0].startswith("#!"):
        out_lines = [lines[0], marker] + lines[1:]
    else:
        out_lines = [marker] + lines
    Path(args.output).write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "generation": args.generation,
        "external_receipt_sha256": lineage["external_receipt_sha256"],
        "behavioral_mutation": False,
        "output_sha256": sha256(Path(args.output).read_bytes()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
