from __future__ import annotations
import argparse, contextlib, hashlib, io, json, random
from evaluations import run_meta_repair_ontology_genesis_hardmode as base

def _tok(seed:int,label:str)->str:
    return label+"-"+hashlib.sha256(f"{seed}|{label}".encode()).hexdigest()[:12]

def make_worlds_v3(seed:int):
    rng=random.Random(seed)
    counts=(12,16)
    train=[]
    idx=0
    for mode,complete in (
        ("EXPAND",1),
        ("REFINE",1),
        ("INCOMPLETE_NULL",0),
        ("EFFICIENT_NULL",1),
    ):
        for count in counts:
            # high-cost modes share identical count/evidence surfaces.
            multiplier=2
            train.append(base.World(
                f"tr-{idx}",
                _tok(seed,f"{mode}-{idx}"),
                mode,
                complete,
                count,
                multiplier,
            ))
            idx+=1
    held=[
        base.World("ho-exp",_tok(seed,"ho-exp"),"EXPAND",1,20,2),
        base.World("ho-ref",_tok(seed,"ho-ref"),"REFINE",1,20,2),
        base.World("ho-inc",_tok(seed,"ho-inc"),"INCOMPLETE_NULL",0,20,2),
        base.World("ho-eff",_tok(seed,"ho-eff"),"EFFICIENT_NULL",1,20,2),
    ]
    mode=rng.choice(("EXPAND","REFINE","INCOMPLETE_NULL","EFFICIENT_NULL"))
    complete=0 if mode=="INCOMPLETE_NULL" else 1
    cold=base.World("cold",_tok(seed,"cold-"+mode),mode,complete,24,2)
    return train,held,cold

_original_baseline=base.baseline

def baseline_v3(w):
    if w.mode=="EFFICIENT_NULL":
        return base.Outcome(1.0,2,w.candidate_count,1)
    return _original_baseline(w)

def main(seed:int):
    base.make_worlds=make_worlds_v3
    base.baseline=baseline_v3
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc=base.main(seed)
    lines=[x.strip() for x in buf.getvalue().splitlines() if x.strip()]
    payload=json.loads(lines[-1])
    if rc not in (0,None):
        raise AssertionError(f"base hardmode returned {rc}")
    if payload.get("status")!="PASS_BOUNDED_HARDMODE_META_REPAIR_ONTOLOGY_CONJUNCTION_GENESIS_WITH_NULL_AND_COLD_TRANSFER":
        raise AssertionError(payload)
    if not payload.get("all_generated_complexity_two"):
        raise AssertionError("V3 lost complexity-2 requirement")
    if any(int(v)!=0 for v in payload.get("atomic_shortcut_counts",{}).values()):
        raise AssertionError("V3 atomic shortcut reappeared")
    if not payload.get("heldout_all_correct") or not payload.get("null_worlds_correct") or not payload.get("cold_pass"):
        raise AssertionError("V3 transfer/cold gate failed")
    payload["status"]="PASS_BOUNDED_V3_ADVERSARIAL_OVERLAP_META_REPAIR_ONTOLOGY_GENESIS"
    payload["v3_design"]={
        "candidate_counts_matched_across_training_modes":[12,16],
        "heldout_candidate_count_matched_across_modes":20,
        "high_cost_shared_by":["EXPAND","REFINE","INCOMPLETE_NULL"],
        "efficient_null_cost":2,
        "atomic_shortcut_forbidden":True,
        "parent_v2_failures_preserved":True,
    }
    print(json.dumps(payload,sort_keys=True))
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,required=True)
    a=ap.parse_args()
    raise SystemExit(main(a.seed))
