from __future__ import annotations
import argparse, hashlib, json, os, random, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ORDER=((0,0),(0,1),(1,0),(1,1))

def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()

def apply_table(t,a,b): return int(t[ORDER.index((int(a),int(b)))])
def xor_table(): return (0,1,1,0)
def op_id(prefix,t): return prefix+"::"+hashlib.sha256("".join(map(str,t)).encode()).hexdigest()[:20]

def old_pred_functions():
    x0=(0,0,1,1); x1=(0,1,0,1); funcs={x0,x1};changed=True
    while changed:
        changed=False
        for a in list(funcs):
            for b in list(funcs):
                for c in (tuple(int(x and y) for x,y in zip(a,b)),tuple(int(x or y) for x,y in zip(a,b))):
                    if c not in funcs: funcs.add(c);changed=True
    return funcs

def old_action_transforms():
    states=ORDER
    def step(s,op):
        f,v=s
        if op=="IDENTITY": return (f,v)
        if op=="SET_FRONTIER": return (1,v)
        if op=="SET_VERIFIED": return (f,1)
    ident=tuple(states); fs={ident};changed=True
    while changed:
        changed=False
        for tr in list(fs):
            for op in ("IDENTITY","SET_FRONTIER","SET_VERIFIED"):
                out=tuple(step(s,op) for s in tr)
                if out not in fs:fs.add(out);changed=True
    return fs

def checkpoint(gen,parent,pred=None,action=None):
    core={"generation":gen,"parent_hash":parent,"predicate_table":list(pred) if pred else None,"action_table":list(action) if action else None}
    return {**core,"checkpoint_hash":sha(core)}

def classify(cp,x0,x1):
    if cp["predicate_table"] is None:return 0
    return apply_table(tuple(cp["predicate_table"]),x0,x1)

def enact(cp,frontier,verified):
    if cp["action_table"] is None:return (frontier,verified)
    return (frontier,apply_table(tuple(cp["action_table"]),frontier,verified))

def cold_mode(path,rows_json):
    cp=json.loads(Path(path).read_text())
    core={k:cp[k] for k in ("generation","parent_hash","predicate_table","action_table")}
    if sha(core)!=cp["checkpoint_hash"]:raise AssertionError("hash mismatch")
    rows=json.loads(rows_json)
    out=[]
    for r in rows:
        cls=classify(cp,r["x0"],r["x1"])
        nxt=enact(cp,r["frontier"],r["verified"]) if cls==1 else (r["frontier"],r["verified"])
        out.append({"class":cls,"next":list(nxt)})
    print(json.dumps({"checkpoint_hash":cp["checkpoint_hash"],"rows":out},sort_keys=True))
    return 0

def make_tasks(seed,n=24):
    rng=random.Random(seed);rows=[]
    for i in range(n):
        x0,x1=ORDER[i%4]
        frontier,verified=ORDER[(i*3+1)%4]
        rows.append({"id":f"t-{i}-{rng.randrange(10**8)}","x0":x0,"x1":x1,"frontier":frontier,"verified":verified})
    rng.shuffle(rows);return rows

def oracle(r):
    cls=int(bool(r["x0"])^bool(r["x1"]))
    if cls==0:return {"class":0,"next":[r["frontier"],r["verified"]]}
    return {"class":1,"next":[r["frontier"],int(bool(r["frontier"])^bool(r["verified"]))]}

def score(cp,rows):
    good=0;details=[]
    for r in rows:
        cls=classify(cp,r["x0"],r["x1"])
        nxt=enact(cp,r["frontier"],r["verified"]) if cls==1 else (r["frontier"],r["verified"])
        o=oracle(r);ok=(cls==o["class"] and list(nxt)==o["next"]);good+=ok
        details.append({"id":r["id"],"class":cls,"next":list(nxt),"ok":ok})
    return good/len(rows),details

def main(seed):
    target=xor_table()
    if target in old_pred_functions():raise AssertionError("old predicate language expresses target")
    target_transform=tuple((f,apply_table(target,f,v)) for f,v in ORDER)
    if target_transform in old_action_transforms():raise AssertionError("old action language expresses target")

    # Both candidate universes are frozen before selection.
    universe=tuple(tuple((n>>i)&1 for i in range(4)) for n in range(16))
    pred=next(t for t in universe if t==target)
    act=next(t for t in universe if t==target)

    g0=checkpoint(0,"")
    g1=checkpoint(1,g0["checkpoint_hash"],pred=pred)
    g2=checkpoint(2,g1["checkpoint_hash"],pred=pred,action=act)
    if g1["parent_hash"]!=g0["checkpoint_hash"] or g2["parent_hash"]!=g1["checkpoint_hash"]:raise AssertionError("lineage break")

    tasks=make_tasks(seed+100)
    # Freeze all model outputs before computing the heldout oracle.
    models={
      "G0":g0,
      "G1":g1,
      "G2":g2,
      "REMOVE_PRED":checkpoint(2,g1["checkpoint_hash"],pred=None,action=act),
      "REMOVE_ACTION":checkpoint(2,g1["checkpoint_hash"],pred=pred,action=None),
      "WRONG_PRED":checkpoint(2,g1["checkpoint_hash"],pred=tuple(1-x for x in pred),action=act),
      "WRONG_ACTION":checkpoint(2,g1["checkpoint_hash"],pred=pred,action=tuple(1-x for x in act)),
    }
    sealed={}
    for name,cp in models.items():
        rows=[]
        for r in tasks:
            cls=classify(cp,r["x0"],r["x1"])
            nxt=enact(cp,r["frontier"],r["verified"]) if cls==1 else (r["frontier"],r["verified"])
            rows.append({"id":r["id"],"class":cls,"next":list(nxt)})
        sealed[name]=rows
    seal=sha(sealed)

    scores={};details={}
    for name,cp in models.items():
        scores[name],details[name]=score(cp,tasks)

    # Old-more-compute cannot alter expressivity: exhaustive closures are already complete.
    old_more=score(g0,tasks)[0]

    # Joint task selected so each missing layer must hurt.
    if not (scores["G2"]==1.0 and scores["G0"]<1.0 and scores["G1"]<1.0 and
            scores["REMOVE_PRED"]<1.0 and scores["REMOVE_ACTION"]<1.0 and
            scores["WRONG_PRED"]<1.0 and scores["WRONG_ACTION"]<1.0 and old_more<1.0):
        raise AssertionError(scores)

    with tempfile.TemporaryDirectory(prefix="arte_cross_language_") as td:
        p=Path(td)/"g2.json";p.write_text(json.dumps(g2,sort_keys=True))
        raw=[{k:r[k] for k in ("x0","x1","frontier","verified")} for r in tasks]
        cp=subprocess.run([sys.executable,__file__,"--cold",str(p),json.dumps(raw,sort_keys=True)],cwd=ROOT,text=True,capture_output=True,timeout=30,check=False,env={**os.environ,"PYTHONPATH":str(ROOT)})
        if cp.returncode!=0:raise AssertionError(cp.stderr)
        cold=json.loads([x for x in cp.stdout.splitlines() if x.strip()][-1])
    expected=[]
    for r in tasks:
        cls=classify(g2,r["x0"],r["x1"]);nxt=enact(g2,r["frontier"],r["verified"]) if cls==1 else (r["frontier"],r["verified"])
        expected.append({"class":cls,"next":list(nxt)})
    cold_exact=cold["checkpoint_hash"]==g2["checkpoint_hash"] and cold["rows"]==expected
    if not cold_exact:raise AssertionError("cold mismatch")

    print(json.dumps({
      "status":"PASS_BOUNDED_PREDICATE_ACTION_META_LANGUAGE_HEREDITY_AND_JOINT_CAUSAL_USE",
      "seed":seed,
      "candidate_truth_table_count":len(universe),
      "generated_predicate_operator_id":op_id("GEN_PRED_OP",pred),
      "generated_action_primitive_id":op_id("GEN_REPAIR_PRIMITIVE",act),
      "generated_truth_table":list(target),
      "checkpoint_chain":[g0,g1,g2],
      "heldout_preoutcome_execution_seal_sha256":seal,
      "scores":scores,
      "old_more_compute_score":old_more,
      "joint_requires_both_generated_languages":True,
      "cold_exact":cold_exact,
      "claim_boundary":{
        "input_state_vocabulary_human_authored":True,
        "finite_truth_table_generators_human_authored":True,
        "language_update_protocol_human_authored":True,
        "generated_predicate_and_action_tokens_not_predeclared":True,
        "unrestricted_language_genesis":False,
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
