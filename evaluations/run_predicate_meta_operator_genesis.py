from __future__ import annotations

import argparse, hashlib, json, os, random, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ORDER=((0,0),(0,1),(1,0),(1,1))

def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()

def apply_table(table,x0,x1):
    idx=ORDER.index((int(x0),int(x1)))
    return int(table[idx])

def old_closure():
    x0=(0,0,1,1)
    x1=(0,1,0,1)
    funcs={x0:"x0",x1:"x1"}
    changed=True
    while changed:
        changed=False
        items=list(funcs)
        for a in items:
            for b in items:
                for op in ("AND","OR"):
                    if op=="AND": c=tuple(int(x and y) for x,y in zip(a,b))
                    else: c=tuple(int(x or y) for x,y in zip(a,b))
                    if c not in funcs:
                        funcs[c]=f"({funcs[a]} {op} {funcs[b]})"
                        changed=True
    return funcs

def candidate_universe():
    return tuple(tuple((n>>i)&1 for i in range(4)) for n in range(16))

def op_id(table):
    return "GEN_PRED_OP::"+hashlib.sha256("".join(map(str,table)).encode()).hexdigest()[:20]

def make_rows(seed,prefix,reps):
    rng=random.Random(seed)
    rows=[]
    for r in range(reps):
        pats=list(ORDER); rng.shuffle(pats)
        for x0,x1 in pats:
            rows.append({
              "id":f"{prefix}-{r}-{rng.randrange(10**9):09d}",
              "x0":x0,"x1":x1,
              "nonce":rng.getrandbits(64),
            })
    return rows

def hidden_label(row):
    return int(bool(row["x0"]) ^ bool(row["x1"]))

def select_operator(training):
    universe=candidate_universe()
    # universe exists before labels are scored.
    scored=[]
    for t in universe:
        acc=sum(apply_table(t,r["x0"],r["x1"])==r["label"] for r in training)/len(training)
        if acc==1.0:
            scored.append((len(set(t)),op_id(t),t))
    if not scored: raise AssertionError("no exact generated operator")
    scored.sort(key=lambda x:(x[0],x[1]))
    return scored[0][2]

def cold_mode(path,rows_json):
    obj=json.loads(Path(path).read_text())
    table=tuple(int(x) for x in obj["truth_table"])
    core={"operator_id":obj["operator_id"],"truth_table":list(table)}
    if sha(core)!=obj["operator_hash"]: raise AssertionError("hash mismatch")
    rows=json.loads(rows_json)
    preds=[apply_table(table,r["x0"],r["x1"]) for r in rows]
    print(json.dumps({"operator_hash":obj["operator_hash"],"predictions":preds},sort_keys=True))
    return 0

def main(seed):
    old=old_closure()
    target=(0,1,1,0)
    if target in old: raise AssertionError("old monotone language unexpectedly expresses XOR")
    if len(old)>16: raise AssertionError("invalid old closure")

    training=make_rows(seed,"train",4)
    for r in training: r["label"]=hidden_label(r)
    universe=candidate_universe()
    universe_hash=sha([list(t) for t in universe])
    generated=select_operator(training)
    if generated in old: raise AssertionError("selected operator belongs to old language")
    oid=op_id(generated)
    core={"operator_id":oid,"truth_table":list(generated)}
    checkpoint={**core,"operator_hash":sha(core)}

    train_acc=sum(apply_table(generated,r["x0"],r["x1"])==r["label"] for r in training)/len(training)
    if train_acc!=1.0: raise AssertionError("train fit failed")

    held=make_rows(seed+100000,"held",5)
    prediction_rows=[{"id":r["id"],"prediction":apply_table(generated,r["x0"],r["x1"])} for r in held]
    pred_seal=sha(prediction_rows)

    # Labels are revealed only after the heldout prediction seal.
    labels=[hidden_label(r) for r in held]
    full_acc=sum(p["prediction"]==y for p,y in zip(prediction_rows,labels))/len(labels)

    # REMOVE: old-language best candidate under the same heldout rows.
    old_scores=[]
    for table,expr in old.items():
        preds=[apply_table(table,r["x0"],r["x1"]) for r in held]
        acc=sum(p==y for p,y in zip(preds,labels))/len(labels)
        old_scores.append((acc,expr,table))
    old_scores.sort(key=lambda x:(-x[0],x[1]))
    remove_acc=old_scores[0][0]

    wrong=tuple(1-x for x in generated)
    if wrong==generated or wrong in old and old[wrong]==old.get(generated):
        raise AssertionError("wrong control invalid")
    wrong_preds=[apply_table(wrong,r["x0"],r["x1"]) for r in held]
    wrong_acc=sum(p==y for p,y in zip(wrong_preds,labels))/len(labels)

    # Cold descendant receives only generated operator bytes.
    with tempfile.TemporaryDirectory(prefix="arte_pred_metaop_") as td:
        p=Path(td)/"operator.json";p.write_text(json.dumps(checkpoint,sort_keys=True),encoding="utf-8")
        cold_rows=[{"x0":r["x0"],"x1":r["x1"]} for r in held]
        cp=subprocess.run(
            [sys.executable,__file__,"--cold",str(p),json.dumps(cold_rows,sort_keys=True)],
            cwd=ROOT,text=True,capture_output=True,timeout=30,check=False,
            env={**os.environ,"PYTHONPATH":str(ROOT)}
        )
        if cp.returncode!=0: raise AssertionError(cp.stderr)
        cold=json.loads([x for x in cp.stdout.splitlines() if x.strip()][-1])
    cold_exact=(cold["operator_hash"]==checkpoint["operator_hash"] and cold["predictions"]==[x["prediction"] for x in prediction_rows])

    if not (full_acc==1.0 and remove_acc<1.0 and wrong_acc<1.0 and cold_exact):
        raise AssertionError({"full":full_acc,"remove":remove_acc,"wrong":wrong_acc,"cold":cold_exact})

    print(json.dumps({
      "status":"PASS_BOUNDED_PREDICATE_META_OPERATOR_GENESIS_AND_COLD_TRANSFER",
      "seed":seed,
      "old_language_connectives":["AND","OR"],
      "old_language_function_count":len(old),
      "old_language_exact_solution_count":int(target in old),
      "candidate_truth_table_count":len(universe),
      "candidate_universe_frozen_before_scoring":True,
      "candidate_universe_sha256":universe_hash,
      "generated_operator_id":oid,
      "generated_truth_table":list(generated),
      "generated_operator_hash":checkpoint["operator_hash"],
      "generated_operator_absent_from_old_language":generated not in old,
      "training_accuracy":train_acc,
      "heldout_prediction_seal_sha256":pred_seal,
      "heldout_accuracy":full_acc,
      "remove_old_language_best_accuracy":remove_acc,
      "remove_old_language_best_expression":old_scores[0][1],
      "wrong_truth_table":list(wrong),
      "wrong_accuracy":wrong_acc,
      "cold_exact":cold_exact,
      "claim_boundary":{
        "boolean_input_feature_vocabulary_human_authored":True,
        "truth_table_enumerator_human_authored":True,
        "finite_binary_function_universe_human_authored":True,
        "generated_connective_not_predeclared":True,
        "unrestricted_predicate_language_genesis":False,
        "independent_organizational_custody":False,
        "physical_world":False,
        "global_recursive_acceleration":False,
        "AGI":False,"ASI":False,"superintelligence":False
      }
    },sort_keys=True))
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--seed",type=int,default=20260929701);ap.add_argument("--cold",nargs=2);a=ap.parse_args()
    if a.cold: raise SystemExit(cold_mode(a.cold[0],a.cold[1]))
    raise SystemExit(main(a.seed))
