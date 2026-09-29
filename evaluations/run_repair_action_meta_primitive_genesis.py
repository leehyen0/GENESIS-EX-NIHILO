from __future__ import annotations

import argparse, hashlib, json, os, random, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STATES=((0,0),(0,1),(1,0),(1,1))

def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()

def old_step(state,op):
    f,v=state
    if op=="IDENTITY": return (f,v)
    if op=="SET_FRONTIER": return (1,v)
    if op=="SET_VERIFIED": return (f,1)
    raise ValueError(op)

def old_closure(max_depth=6):
    # enumerate exact transformations over all four states
    ident=tuple(STATES)
    trans={ident:"IDENTITY"}
    frontier=tuple(old_step(s,"SET_FRONTIER") for s in STATES)
    verified=tuple(old_step(s,"SET_VERIFIED") for s in STATES)
    trans.setdefault(frontier,"SET_FRONTIER")
    trans.setdefault(verified,"SET_VERIFIED")
    changed=True
    depth=0
    while changed and depth<max_depth:
        changed=False; depth+=1
        items=list(trans.items())
        for ta,ea in items:
            for op in ("IDENTITY","SET_FRONTIER","SET_VERIFIED"):
                out=tuple(old_step(s2,op) for s2 in ta)
                if out not in trans:
                    trans[out]=f"({ea};{op})"; changed=True
    return trans

def apply_candidate(table,state):
    f,v=state
    idx=STATES.index((f,v))
    nv=int(table[idx])
    return (f,nv)

def candidate_universe():
    return tuple(tuple((n>>i)&1 for i in range(4)) for n in range(16))

def primitive_id(table):
    return "GEN_REPAIR_PRIMITIVE::"+hashlib.sha256("".join(map(str,table)).encode()).hexdigest()[:20]

def hidden_next(state):
    f,v=state
    return (f,int(bool(f)^bool(v)))

def make_rows(seed,prefix,reps):
    rng=random.Random(seed)
    rows=[]
    for r in range(reps):
        states=list(STATES); rng.shuffle(states)
        for f,v in states:
            rows.append({"id":f"{prefix}-{r}-{rng.randrange(10**9):09d}","frontier":f,"verified":v,"nonce":rng.getrandbits(64)})
    return rows

def cold_mode(path,rows_json):
    obj=json.loads(Path(path).read_text())
    table=tuple(int(x) for x in obj["truth_table"])
    core={"primitive_id":obj["primitive_id"],"truth_table":list(table)}
    if sha(core)!=obj["primitive_hash"]: raise AssertionError("hash mismatch")
    rows=json.loads(rows_json)
    preds=[list(apply_candidate(table,(r["frontier"],r["verified"]))) for r in rows]
    print(json.dumps({"primitive_hash":obj["primitive_hash"],"predictions":preds},sort_keys=True))
    return 0

def main(seed):
    old=old_closure(6)
    target=tuple(hidden_next(s) for s in STATES)
    if target in old: raise AssertionError("old action language unexpectedly expresses hidden transition")

    training=make_rows(seed,"train",4)
    for r in training: r["target"]=hidden_next((r["frontier"],r["verified"]))

    universe=candidate_universe()
    exact=[]
    for table in universe:
        acc=sum(apply_candidate(table,(r["frontier"],r["verified"]))==tuple(r["target"]) for r in training)/len(training)
        if acc==1.0: exact.append((primitive_id(table),table))
    if not exact: raise AssertionError("no generated primitive")
    exact.sort()
    pid,table=exact[0]
    transform=tuple(apply_candidate(table,s) for s in STATES)
    if transform in old: raise AssertionError("generated primitive in old closure")
    core={"primitive_id":pid,"truth_table":list(table)}
    checkpoint={**core,"primitive_hash":sha(core)}

    train_acc=sum(apply_candidate(table,(r["frontier"],r["verified"]))==tuple(r["target"]) for r in training)/len(training)

    held=make_rows(seed+100000,"held",6)
    pred_rows=[{"id":r["id"],"prediction":list(apply_candidate(table,(r["frontier"],r["verified"])))} for r in held]
    pred_seal=sha(pred_rows)
    labels=[hidden_next((r["frontier"],r["verified"])) for r in held]
    full_acc=sum(tuple(p["prediction"])==y for p,y in zip(pred_rows,labels))/len(labels)

    old_scores=[]
    for tr,expr in old.items():
        mapping={s:tr[i] for i,s in enumerate(STATES)}
        preds=[mapping[(r["frontier"],r["verified"])] for r in held]
        acc=sum(p==y for p,y in zip(preds,labels))/len(labels)
        old_scores.append((acc,expr,tr))
    old_scores.sort(key=lambda x:(-x[0],x[1]))
    remove_acc=old_scores[0][0]

    wrong=tuple(1-x for x in table)
    wrong_preds=[apply_candidate(wrong,(r["frontier"],r["verified"])) for r in held]
    wrong_acc=sum(p==y for p,y in zip(wrong_preds,labels))/len(labels)

    with tempfile.TemporaryDirectory(prefix="arte_action_meta_") as td:
        p=Path(td)/"primitive.json";p.write_text(json.dumps(checkpoint,sort_keys=True),encoding="utf-8")
        rows=[{"frontier":r["frontier"],"verified":r["verified"]} for r in held]
        cp=subprocess.run([sys.executable,__file__,"--cold",str(p),json.dumps(rows,sort_keys=True)],cwd=ROOT,text=True,capture_output=True,timeout=30,check=False,env={**os.environ,"PYTHONPATH":str(ROOT)})
        if cp.returncode!=0: raise AssertionError(cp.stderr)
        cold=json.loads([x for x in cp.stdout.splitlines() if x.strip()][-1])
    cold_exact=(cold["primitive_hash"]==checkpoint["primitive_hash"] and cold["predictions"]==[x["prediction"] for x in pred_rows])

    if not (train_acc==1.0 and full_acc==1.0 and remove_acc<1.0 and wrong_acc<1.0 and cold_exact):
        raise AssertionError({"train":train_acc,"full":full_acc,"remove":remove_acc,"wrong":wrong_acc,"cold":cold_exact})

    print(json.dumps({
      "status":"PASS_BOUNDED_REPAIR_ACTION_META_PRIMITIVE_GENESIS_AND_COLD_TRANSFER",
      "seed":seed,
      "old_action_primitives":["IDENTITY","SET_FRONTIER","SET_VERIFIED"],
      "old_action_closure_count":len(old),
      "old_language_exact_solution_count":int(target in old),
      "candidate_transition_count":len(universe),
      "candidate_universe_frozen_before_scoring":True,
      "generated_primitive_id":pid,
      "generated_truth_table":list(table),
      "generated_primitive_hash":checkpoint["primitive_hash"],
      "generated_transition_absent_from_old_language":transform not in old,
      "training_accuracy":train_acc,
      "heldout_prediction_seal_sha256":pred_seal,
      "heldout_accuracy":full_acc,
      "remove_old_language_best_accuracy":remove_acc,
      "remove_old_language_best_expression":old_scores[0][1],
      "wrong_truth_table":list(wrong),
      "wrong_accuracy":wrong_acc,
      "cold_exact":cold_exact,
      "claim_boundary":{
        "state_bit_vocabulary_human_authored":True,
        "truth_table_enumerator_human_authored":True,
        "finite_transition_universe_human_authored":True,
        "generated_action_primitive_not_predeclared":True,
        "unrestricted_action_language_genesis":False,
        "independent_organizational_custody":False,
        "physical_world":False,
        "global_recursive_acceleration":False,
        "AGI":False,"ASI":False,"superintelligence":False
      }
    },sort_keys=True))
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--seed",type=int,default=20260929801);ap.add_argument("--cold",nargs=2);a=ap.parse_args()
    if a.cold: raise SystemExit(cold_mode(a.cold[0],a.cold[1]))
    raise SystemExit(main(a.seed))
