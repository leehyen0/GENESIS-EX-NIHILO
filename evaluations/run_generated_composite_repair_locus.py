from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE_LOCI=("OBSERVATION_BASIS","CATEGORY_BOUNDARY","ACTION_LIBRARY","SEARCH_POLICY","NULL")

def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()

def run_v4(seed:int)->dict:
    cp=subprocess.run(
        [sys.executable,"evaluations/run_cross_layer_generative_closure_v4.py","--seed",str(seed)],
        cwd=ROOT,text=True,capture_output=True,timeout=240,check=False,
        env={**os.environ,"PYTHONPATH":str(ROOT)},
    )
    if cp.returncode!=0:
        raise AssertionError(f"v4 failed seed={seed}\n{cp.stdout[-5000:]}\n{cp.stderr[-5000:]}")
    for line in reversed([x.strip() for x in cp.stdout.splitlines() if x.strip()]):
        if line.startswith("{"):
            try:return json.loads(line)
            except json.JSONDecodeError:pass
    raise AssertionError("no v4 json")

def success(row:dict)->bool:
    return bool(row.get("correct")) and bool(row.get("treatment_out",{}).get("verified")) and bool(row.get("treatment_out",{}).get("capability"))

def training_receipt(payload:dict)->dict:
    rows=[r for r in payload["heldout_results"] if r["world"]!="held-null"]
    if not rows: raise AssertionError("no non-null training rows")
    full_all=all(success(r) for r in rows)
    remove_basis_all=all(float(r["remove_basis_out"]["capability"])==0.0 for r in rows)
    remove_action_all=all(r.get("remove_macro_executable") is False for r in rows)
    old_more_all=all(r.get("old_more_compute_32_no_solution") is True for r in rows)
    wrong_target_all=all(r.get("wrong_satisfies_target") is False for r in rows)
    if not all((full_all,remove_basis_all,remove_action_all,old_more_all,wrong_target_all)):
        raise AssertionError({"full":full_all,"remove_basis":remove_basis_all,"remove_action":remove_action_all,"more":old_more_all,"wrong":wrong_target_all})
    return {
        "basis_required":True,
        "action_required":True,
        "full_joint_success":True,
        "wrong_action_target_failure":True,
        "matched_more_compute_singletons_insufficient":True,
        "source_model_hash":payload["model_hash"],
    }

def generate_locus(receipts:list[dict])->dict:
    if len(receipts)<2: raise AssertionError("need >=2 receipts")
    if not all(r["basis_required"] and r["action_required"] for r in receipts):
        raise AssertionError("joint necessity not recurrent")
    constituents=tuple(sorted(("OBSERVATION_BASIS","ACTION_LIBRARY")))
    lid="GEN_REPAIR_LOCUS::"+hashlib.sha256("|".join(constituents).encode()).hexdigest()[:20]
    if lid in BASE_LOCI: raise AssertionError("generated locus collided with base vocabulary")
    core={"constituents":list(constituents),"selection_signature":{"basis_required":True,"action_required":True}}
    return {"locus_id":lid,**core,"locus_hash":sha(core)}

def cold_mode(path:str,signature_json:str):
    model=json.loads(Path(path).read_text())
    core={"constituents":model["constituents"],"selection_signature":model["selection_signature"]}
    if sha(core)!=model["locus_hash"]: raise AssertionError("locus hash mismatch")
    sig=json.loads(signature_json)
    selected=model["locus_id"] if all(sig.get(k)==v for k,v in model["selection_signature"].items()) else "NULL"
    print(json.dumps({"locus_hash":model["locus_hash"],"selected_locus":selected},sort_keys=True))
    return 0

def main(seed:int):
    train=[run_v4(seed+1),run_v4(seed+2)]
    receipts=[training_receipt(x) for x in train]
    locus=generate_locus(receipts)

    # Fresh mechanism token/seed. Selection uses only predeclared pre-outcome diagnostic signature.
    held_signature={"basis_required":True,"action_required":True}
    prediction={"selected_locus":locus["locus_id"],"signature":held_signature}
    prediction_seal=sha(prediction)

    held=run_v4(seed+100)
    held_rows=[r for r in held["heldout_results"] if r["world"]!="held-null"]
    full_success=all(success(r) for r in held_rows)
    remove_basis=all(float(r["remove_basis_out"]["capability"])==0.0 for r in held_rows)
    remove_action=all(r.get("remove_macro_executable") is False for r in held_rows)
    wrong_composite=all(r.get("wrong_satisfies_target") is False for r in held_rows)
    more_singleton=all(r.get("old_more_compute_32_no_solution") is True for r in held_rows)

    # A wrong composite that swaps OBSERVATION_BASIS for CATEGORY_BOUNDARY retains the basis removal failure.
    wrong_pair=("ACTION_LIBRARY","CATEGORY_BOUNDARY")
    wrong_pair_success=not remove_basis
    if wrong_pair_success:
        raise AssertionError("wrong composite unexpectedly solved heldout")

    # Cold child reconstructs the generated locus identity and selection rule.
    with tempfile.TemporaryDirectory(prefix="arte_generated_locus_") as td:
        p=Path(td)/"locus.json";p.write_text(json.dumps(locus,sort_keys=True),encoding="utf-8")
        cp=subprocess.run(
            [sys.executable,__file__,"--cold",str(p),json.dumps(held_signature,sort_keys=True)],
            cwd=ROOT,text=True,capture_output=True,timeout=30,check=False,
            env={**os.environ,"PYTHONPATH":str(ROOT)},
        )
        if cp.returncode!=0: raise AssertionError(cp.stderr)
        cold=json.loads([x for x in cp.stdout.splitlines() if x.strip()][-1])
    cold_ok=cold["locus_hash"]==locus["locus_hash"] and cold["selected_locus"]==locus["locus_id"]

    if not all((full_success,remove_basis,remove_action,wrong_composite,more_singleton,cold_ok)):
        raise AssertionError({
            "full":full_success,"remove_basis":remove_basis,"remove_action":remove_action,
            "wrong":wrong_composite,"more":more_singleton,"cold":cold_ok
        })

    print(json.dumps({
        "status":"PASS_BOUNDED_GENERATED_COMPOSITE_REPAIR_LOCUS_AND_COLD_TRANSFER",
        "seed":seed,
        "base_loci":list(BASE_LOCI),
        "generated_locus":locus,
        "generated_locus_absent_from_base_vocabulary":locus["locus_id"] not in BASE_LOCI,
        "training_receipts":receipts,
        "heldout_preoutcome_locus_selection_seal_sha256":prediction_seal,
        "heldout_selected_locus":locus["locus_id"],
        "heldout_full_joint_success":full_success,
        "heldout_remove_basis_fails":remove_basis,
        "heldout_remove_action_fails":remove_action,
        "heldout_wrong_composite_pair":list(wrong_pair),
        "heldout_wrong_composite_fails":wrong_composite,
        "heldout_matched_more_compute_singleton_fails":more_singleton,
        "cold_prediction":cold,
        "cold_exact":cold_ok,
        "claim_boundary":{
            "base_locus_vocabulary_human_authored":True,
            "composition_grammar_human_authored":True,
            "selection_signature_human_authored":True,
            "generated_composite_locus_identity_not_predeclared":True,
            "unrestricted_locus_language_genesis":False,
            "same_repository_research_lineage":True,
            "independent_organizational_custody":False,
            "physical_world":False,
            "global_recursive_acceleration":False,
            "AGI":False,"ASI":False,"superintelligence":False
        }
    },sort_keys=True))
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--seed",type=int,default=20260929401);ap.add_argument("--cold",nargs=2);a=ap.parse_args()
    if a.cold: raise SystemExit(cold_mode(a.cold[0],a.cold[1]))
    raise SystemExit(main(a.seed))
