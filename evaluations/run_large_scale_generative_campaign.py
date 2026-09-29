from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, tempfile, time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "large-scale-generative-campaign"

SEEDED = [
    "evaluations/run_world_caused_cognition_rewrite.py",
    "evaluations/run_matched_descendant_search_acceleration.py",
    "evaluations/run_multi_episode_world_rewrite.py",
    "evaluations/run_end_to_end_world_genesis.py",
    "evaluations/run_morphology_meta_acceleration.py",
    "evaluations/run_certificate_driven_morphology_search.py",
    "evaluations/run_world_driven_primitive_genesis.py",
    "evaluations/run_compositional_causal_grammar_genesis.py",
    "evaluations/run_falsification_driven_linear_primitive_genesis.py",
    "evaluations/run_generated_morphology_rewrite_schema.py",
    "evaluations/run_parametric_morphology_macro_transfer.py",
    "evaluations/run_relation_access_plan_genesis.py",
    "evaluations/run_reflective_relation_genesis.py",
]

STATIC = [
    "evaluations/run_natural_repair_class_genesis.py",
    "evaluations/run_natural_repair_constructor_genesis.py",
    "evaluations/run_natural_failure_extractor_program_genesis.py",
    "evaluations/run_source_derived_binding_schema_genesis.py",
    "evaluations/run_source_derived_dataflow_path_genesis.py",
    "evaluations/run_selector_representation_program_genesis.py",
]

CLAIM_KEYS = {
    "AGI", "ASI", "superintelligence", "global_recursive_acceleration",
    "recursive_acceleration", "independent_organizational_custody",
    "foundation_weight_change", "physical_world",
}

def stable_seed(base: str, script: str, replicate: int) -> int:
    material = f"{base}|{script}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")

def parse_last_json(stdout: str):
    for line in reversed([x.strip() for x in stdout.splitlines() if x.strip()]):
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return None

def flatten_claims(obj, found=None, path=""):
    if found is None:
        found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if k in CLAIM_KEYS and isinstance(v, bool):
                found.append({"path": p, "value": v})
            flatten_claims(v, found, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flatten_claims(v, found, f"{path}[{i}]")
    return found

def run_one(script: str, replicate: int, base_seed: str, seeded: bool):
    started = time.time()
    seed = stable_seed(base_seed, script, replicate)
    cmd = [sys.executable, script]
    seed_path = None
    if seeded:
        fd, tmp = tempfile.mkstemp(prefix="arte_campaign_seed_", text=True)
        os.close(fd)
        seed_path = Path(tmp)
        seed_path.write_text(str(seed), encoding="utf-8")
        cmd.append(str(seed_path))
    try:
        cp = subprocess.run(
            cmd, cwd=ROOT, text=True, capture_output=True,
            timeout=240, check=False,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        payload = parse_last_json(cp.stdout)
        return {
            "script": script,
            "replicate": replicate,
            "seed": seed if seeded else None,
            "returncode": cp.returncode,
            "duration_sec": round(time.time() - started, 4),
            "status": payload.get("status") if isinstance(payload, dict) else None,
            "payload": payload,
            "claim_flags": flatten_claims(payload) if payload is not None else [],
            "stdout_tail": cp.stdout[-6000:],
            "stderr_tail": cp.stderr[-6000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "script": script, "replicate": replicate, "seed": seed if seeded else None,
            "returncode": 124, "duration_sec": round(time.time() - started, 4),
            "status": "TIMEOUT", "payload": None, "claim_flags": [],
            "stdout_tail": (exc.stdout or "")[-6000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-6000:] if isinstance(exc.stderr, str) else "",
        }
    finally:
        if seed_path is not None:
            seed_path.unlink(missing_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeded-replicates", type=int, default=4)
    ap.add_argument("--static-replicates", type=int, default=2)
    ap.add_argument("--base-seed", default="ARTE-LARGE-SCALE-20260929")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    records = []

    for script in SEEDED:
        for r in range(args.seeded_replicates):
            records.append(run_one(script, r, args.base_seed, True))

    for script in STATIC:
        for r in range(args.static_replicates):
            records.append(run_one(script, r, args.base_seed, False))

    with (OUT / "raw.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    status_counts = Counter((r["status"] or ("PROCESS_OK" if r["returncode"] == 0 else "NO_JSON_FAILURE")) for r in records)
    by_script = {}
    for script in SEEDED + STATIC:
        rr = [r for r in records if r["script"] == script]
        by_script[script] = {
            "runs": len(rr),
            "process_successes": sum(r["returncode"] == 0 for r in rr),
            "process_failures": sum(r["returncode"] != 0 for r in rr),
            "statuses": dict(Counter((r["status"] or "NO_STATUS") for r in rr)),
            "unique_payload_statuses": sorted({str(r["status"]) for r in rr}),
            "mean_duration_sec": round(sum(r["duration_sec"] for r in rr) / max(1, len(rr)), 4),
        }

    true_claim_flags = []
    for r in records:
        for flag in r["claim_flags"]:
            if flag["value"]:
                true_claim_flags.append({"script": r["script"], "replicate": r["replicate"], **flag})

    summary = {
        "schema": "arte.large_scale_generative_campaign/v1",
        "base_seed": args.base_seed,
        "seeded_replicates": args.seeded_replicates,
        "static_replicates": args.static_replicates,
        "planned_runs": len(SEEDED) * args.seeded_replicates + len(STATIC) * args.static_replicates,
        "completed_runs": len(records),
        "process_successes": sum(r["returncode"] == 0 for r in records),
        "process_failures": sum(r["returncode"] != 0 for r in records),
        "status_counts": dict(status_counts),
        "by_script": by_script,
        "true_high_claim_flags": true_claim_flags,
        "claim_boundary": {
            "campaign_execution_is_intelligence_gain": False,
            "ci_success_is_recursive_improvement": False,
            "independent_organizational_custody": False,
            "physical_world": False,
            "foundation_weight_change": False,
            "AGI": False,
            "ASI": False,
            "superintelligence": False,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# ARTE Large-Scale Generative Campaign",
        "",
        f"- planned/completed: {summary['planned_runs']}/{summary['completed_runs']}",
        f"- process successes: {summary['process_successes']}",
        f"- process failures: {summary['process_failures']}",
        "",
        "## Per-script",
        "",
        "| script | runs | ok | failed | statuses |",
        "|---|---:|---:|---:|---|",
    ]
    for script, row in by_script.items():
        lines.append(f"| {script} | {row['runs']} | {row['process_successes']} | {row['process_failures']} | {json.dumps(row['statuses'], sort_keys=True)} |")
    lines += [
        "",
        "## Claim boundary",
        "",
        "Execution volume is not counted as intelligence gain. This campaign re-runs bounded evaluators and causal controls; it does not establish independent custody, physical-world agency, foundation-model weight change, AGI, ASI, or global recursive acceleration.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))

if __name__ == "__main__":
    main()
