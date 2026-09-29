from __future__ import annotations
import argparse, hashlib, json, os, random, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ORDER2=((0,0),(0,1),(1,0),(1,1))
ORDER3=tuple((a,b,c) for a in (0,1) for b in (0,1) for c in (0,1))

def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()

def table_apply(table,order,args): return int(table[order.index(tuple(map(int,args)))])
def fid(prefix,t): return prefix+"::"+hashlib.sha256("".join(map(str,t)).encode()).hexdigest()[:20]

def parity3(a,b,c): return int(bool(a)^bool(b)^bool(c))

def binary_best(rows):
    funcs=[tuple((n>>i)&1 for i in range(4)) for n in range(16)]
    best=(-1,None,None)
    for pair in ((0,1),(0,2),(1,2)):
        for t in funcs:
            correct=0
            for r in rows:
                vals=(r["x0"],r["x1"],r["x2"])
                p=table_apply(t,ORDER2,(vals[pair[0]],vals[pair[1]]))
                correct+=int(p==r["label"])
            score=correct/len(rows)
            cand=(score,pair,t)
            if score>best[0] or (score==best[0] and canon([pair,list(t)])<canon([best[1],list(best[2])]) if best[1] is not None else True):
                best=cand
    return best

def cold_mode(path,rows_json):
    o=json.loads(Path(path).read_text())
    core={"arity":o["arity"],"operator_id":o["operator_id"],"truth_table":o["truth_table"]}
    if sha(core)!=o["operator_hash"]:raise AssertionError("hash mismatch")
    t=tuple(o["truth_table"])
    rows=json.loads(rows_json)
    preds=[table_apply(t,ORDER3,(r["x0"],r["x1"],r["x2"])) for r in rows]
    print(json.dumps({"operator_hash":o["operator_hash"],"predictions":preds},sort_keys=True))
    return 0

def main(seed):
    rng=random.Random(seed)
    train=[]
    for rep in range(4):
        pats=list(ORDER3);rng.shuffle(pats)
        for a,b,c in pats:
            train.append({"id":f"tr-{rep}-{rng.randrange(10**8)}","x0":a,"x1":b,"x2":c,"label":parity3(a,b,c)})

    bscore,bpair,btable=binary_best(train)
    if bscore>=1.0:raise AssertionError("binary one-call language unexpectedly exact")

    universe=tuple(tuple((n>>i)&1 for i in range(8)) for n in range(256))
    exact=[]
    for t in universe:
        acc=sum(table_apply(t,ORDER3,(r["x0"],r["x1"],r["x2"]))==r["label"] for r in train)/len(train)
        if acc==1.0: exact.append((fid("GEN_ARITY3_OP",t),t))
    if not exact:raise AssertionError("arity3 universe has no exact solution")
    exact.sort();oid,t=exact[0]
    core={"arity":3,"operator_id":oid,"truth_table":list(t)}
    cp={**core,"operator_hash":sha(core)}

    held=[]
    for rep in range(6):
        pats=list(ORDER3);rng.shuffle(pats)
        for a,b,c in pats:
            held.append({"id":f"ho-{rep}-{rng.randrange(10**8)}","x0":a,"x1":b,"x2":c})
    preds=[{"id":r["id"],"prediction":table_apply(t,ORDER3,(r["x0"],r["x1"],r["x2"]))} for r in held]
    seal=sha(preds)
    labels=[parity3(r["x0"],r["x1"],r["x2"]) for r in held]
    full=sum(p["prediction"]==y for p,y in zip(preds,labels))/len(labels)
    remove=binary_best([{**r,"label":y} for r,y in zip(held,labels)])[0]
    wrong=tuple(1-x for x in t)
    wrongacc=sum(table_apply(wrong,ORDER3,(r["x0"],r["x1"],r["x2"]))==y for r,y in zip(held,labels))/len(labels)

    with tempfile.TemporaryDirectory(prefix="arte_arity3_") as td:
        p=Path(td)/"op.json";p.write_text(json.dumps(cp,sort_keys=True))
        raw=[{k:r[k] for k in ("x0","x1","x2")} for r in held]
        pr=subprocess.run([sys.executable,__file__,"--cold",str(p),json.dumps(raw,sort_keys=True)],cwd=ROOT,text=True,capture_output=True,timeout=30,check=False,env={**os.environ,"PYTHONPATH":str(ROOT)})
        if pr.returncode!=0:raise AssertionError(pr.stderr)
        cold=json.loads([x for x in pr.stdout.splitlines() if x.strip()][-1])
    cold_exact=cold["operator_hash"]==cp["operator_hash"] and cold["predictions"]==[x["prediction"] for x in preds]
    if not (full==1.0 and remove<1.0 and wrongacc<1.0 and cold_exact):
        raise AssertionError({"full":full,"remove":remove,"wrong":wrongacc,"cold":cold_exact})

    print(json.dumps({
      "status":"PASS_BOUNDED_META_LANGUAGE_ARITY_EXPANSION_AND_COLD_TRANSFER",
      "seed":seed,
      "g0_binary_universe_size":16,
      "g0_best_accuracy":bscore,
      "g0_best_input_pair":list(bpair),
      "g0_exact_solution_count":0,
      "arity_expansion_from":2,
      "generated_arity":3,
      "g1_universe_size":256,
      "generated_operator_id":oid,
      "generated_truth_table":list(t),
      "training_accuracy":1.0,
      "heldout_prediction_seal_sha256":seal,
      "heldout_accuracy":full,
      "binary_remove_best_accuracy":remove,
      "wrong_accuracy":wrongacc,
      "cold_exact":cold_exact,
      "claim_boundary":{
        "boolean_feature_vocabulary_human_authored":True,
        "arity_increment_rule_human_authored":True,
        "complete_truth_table_enumerator_human_authored":True,
        "unrestricted_arity_or_language_genesis":False,
        "independent_organizational_custody":False,
        "physical_world":False,
        "global_recursive_acceleration":False,
        "AGI":False,"ASI":False,"superintelligence":False
      }
    },sort_keys=True))
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--seed",type=int,required=True);ap.add_argument("--cold",nargs=2);a=ap.parse_args()
    if a.cold:raise SystemExit(cold_mode(a.cold[0],a.cold[1]))
    raise SystemExit(main(a.seed))
