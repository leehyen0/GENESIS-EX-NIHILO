from __future__ import annotations
import argparse, hashlib, json, os, random, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ORDER=((0,0),(0,1),(1,0),(1,1))

def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()

def token(lag:int,sign:str)->str:
    return "GEN_OBS_REL::"+hashlib.sha256(f"lag={lag}|sign={sign}".encode()).hexdigest()[:16]

R0=token(1,"POS"); R0N=token(1,"NEG"); R1=token(2,"POS"); R1N=token(2,"NEG")

def make_raw_world(seed:int,prefix:str,b0:int,b1:int,reps:int=2):
    rng=random.Random(seed)
    rows=[]
    for rep in range(reps):
        # Each channel always has exactly one source intervention and one equal-magnitude downstream effect.
        # Thus old scalar energy/event-count summaries are identical for all four latent combinations.
        mag=float(2+rep)
        names=[f"{prefix}_a_{rng.randrange(10**8)}",f"{prefix}_b_{rng.randrange(10**8)}",
               f"{prefix}_c_{rng.randrange(10**8)}",f"{prefix}_d_{rng.randrange(10**8)}"]
        rows.append({
          "source":names[0],"target":names[1],"lag":1,"effect":mag if b0 else -mag,
          "source_magnitude":mag
        })
        rows.append({
          "source":names[2],"target":names[3],"lag":2,"effect":mag if b1 else -mag,
          "source_magnitude":mag
        })
    return rows

def old_scalar(raw):
    return {
      "event_count":len(raw),
      "total_abs_effect":sum(abs(float(x["effect"])) for x in raw),
      "total_abs_source":sum(abs(float(x["source_magnitude"])) for x in raw),
    }

def discover_features(raw):
    groups={}
    for e in raw:
        sign="POS" if float(e["effect"])>0 else "NEG"
        k=(int(e["lag"]),sign)
        groups.setdefault(k,[]).append(float(e["effect"]))
    feats=set()
    for (lag,sign),vals in groups.items():
        if len(vals)>=2 and all((v>0)==(vals[0]>0) for v in vals):
            feats.add(token(lag,sign))
    return feats

def apply_table(t,a,b): return int(t[ORDER.index((int(a),int(b)))])

def old_pred_closure():
    x0=(0,0,1,1);x1=(0,1,0,1);fs={x0:"x0",x1:"x1"};changed=True
    while changed:
        changed=False
        for a,ea in list(fs.items()):
            for b,eb in list(fs.items()):
                for op in ("AND","OR"):
                    c=tuple(int(x and y) for x,y in zip(a,b)) if op=="AND" else tuple(int(x or y) for x,y in zip(a,b))
                    if c not in fs:
                        fs[c]=f"({ea} {op} {eb})";changed=True
    return fs

def old_action_closure():
    def step(s,op):
        f,v=s
        if op=="IDENTITY":return (f,v)
        if op=="SET_FRONTIER":return (1,v)
        if op=="SET_VERIFIED":return (f,1)
    ident=tuple(ORDER);fs={ident:"IDENTITY"};changed=True
    while changed:
        changed=False
        for tr,expr in list(fs.items()):
            for op in ("IDENTITY","SET_FRONTIER","SET_VERIFIED"):
                out=tuple(step(s,op) for s in tr)
                if out not in fs:
                    fs[out]=f"({expr};{op})";changed=True
    return fs

def learn_feature_pair_and_predicate(training):
    feat_ids=sorted(set().union(*(r["features"] for r in training)))
    tables=tuple(tuple((n>>i)&1 for i in range(4)) for n in range(16))
    candidates=[]
    for i,a in enumerate(feat_ids):
        for b in feat_ids[i+1:]:
            for t in tables:
                ok=True
                for r in training:
                    p=apply_table(t,int(a in r["features"]),int(b in r["features"]))
                    if p!=r["label"]:ok=False;break
                if ok:
                    candidates.append((a,b,t))
    if not candidates:raise AssertionError("no exact generated feature-predicate candidate")
    candidates.sort(key=lambda z:canon([z[0],z[1],list(z[2])]))
    return candidates[0]

def learn_action_table(training_states):
    tables=tuple(tuple((n>>i)&1 for i in range(4)) for n in range(16))
    exact=[]
    for t in tables:
        ok=all((f,apply_table(t,f,v))==target for f,v,target in training_states)
        if ok:exact.append(t)
    if not exact:raise AssertionError("no exact generated action")
    exact=sorted(exact,key=lambda t:canon(list(t)))
    return exact[0]

def classify(selected_features,pred_table,features):
    a,b=selected_features
    return apply_table(pred_table,int(a in features),int(b in features))

def enact(action_table,state):
    f,v=state
    return (f,apply_table(action_table,f,v))

def make_cases(seed:int,prefix:str,reps:int):
    rng=random.Random(seed);out=[]
    for rep in range(reps):
        pats=list(ORDER);rng.shuffle(pats)
        for b0,b1 in pats:
            raw=make_raw_world(seed+rng.randrange(10**7),f"{prefix}_{rep}_{rng.randrange(10**7)}",b0,b1,2)
            frontier,verified=ORDER[(rep*3+b0*2+b1+1)%4]
            out.append({
              "id":f"{prefix}-{rep}-{rng.randrange(10**9)}",
              "raw":raw,"b0":b0,"b1":b1,"state":[frontier,verified]
            })
    return out

def target(case):
    active=int(bool(case["b0"])^bool(case["b1"]))
    f,v=case["state"]
    nxt=[f,int(bool(f)^bool(v))] if active else [f,v]
    return {"class":active,"next":nxt}

def model_hash(m):
    core={k:m[k] for k in ("selected_features","predicate_table","action_table")}
    return sha(core)

def execute(model,case):
    feats=discover_features(case["raw"])
    cls=classify(tuple(model["selected_features"]),tuple(model["predicate_table"]),feats)
    state=tuple(case["state"])
    nxt=enact(tuple(model["action_table"]),state) if cls else state
    return {"class":cls,"next":list(nxt)}

def cold_mode(path,cases_json):
    m=json.loads(Path(path).read_text())
    if model_hash(m)!=m["model_hash"]:raise AssertionError("model hash mismatch")
    cases=json.loads(cases_json)
    rows=[execute(m,c) for c in cases]
    print(json.dumps({"model_hash":m["model_hash"],"rows":rows},sort_keys=True))
    return 0

def accuracy(model,cases):
    return sum(execute(model,c)==target(c) for c in cases)/len(cases)

def main(seed):
    train=make_cases(seed,"train",6)
    for c in train:
        c["features"]=discover_features(c["raw"])
        c["label"]=target(c)["class"]

    scalars={canon(old_scalar(c["raw"])) for c in train}
    labels={c["label"] for c in train}
    if not (len(scalars)==1 and len(labels)==2):
        raise AssertionError("old scalar representation not proven inexpressive")

    sf0,sf1,pt=learn_feature_pair_and_predicate(train)
    selected=(sf0,sf1)
    pred_target=tuple(int(bool(a)^bool(b)) for a,b in ORDER)
    old_pred=old_pred_closure()
    if pred_target in old_pred:raise AssertionError("old predicate language exact")

    action_training=[]
    for f,v in ORDER:
        action_training.append((f,v,(f,int(bool(f)^bool(v)))))
    at=learn_action_table(action_training)
    action_transform=tuple((f,apply_table(at,f,v)) for f,v in ORDER)
    old_action=old_action_closure()
    if action_transform in old_action:raise AssertionError("old action exact")

    model={"selected_features":list(selected),"predicate_table":list(pt),"action_table":list(at)}
    model["model_hash"]=model_hash(model)

    held=make_cases(seed+100000,"held",8)
    full_preds=[{"id":c["id"],**execute(model,c)} for c in held]
    seal=sha(full_preds)
    full=accuracy(model,held)

    # REMOVE_FEATURE: replace generated selected relations with absent dummy IDs.
    rmf={**model,"selected_features":["REMOVED_A","REMOVED_B"]};rmf["model_hash"]=model_hash(rmf)
    remove_feature=accuracy(rmf,held)

    # REMOVE_PREDICATE: use best old AND/OR truth function over same generated features.
    old_pred_scores=[]
    for table,expr in old_pred.items():
        mm={**model,"predicate_table":list(table)};mm["model_hash"]=model_hash(mm)
        old_pred_scores.append((accuracy(mm,held),expr,table))
    old_pred_scores.sort(key=lambda x:(-x[0],x[1]))
    rmp={**model,"predicate_table":list(old_pred_scores[0][2])};rmp["model_hash"]=model_hash(rmp)
    remove_pred=accuracy(rmp,held)

    # REMOVE_ACTION: identity.
    identity=(0,1,0,1)  # verified'=verified over ORDER
    rma={**model,"action_table":list(identity)};rma["model_hash"]=model_hash(rma)
    remove_action=accuracy(rma,held)

    # WRONG controls.
    wf={**model,"selected_features":[R0N,sf1]};wf["model_hash"]=model_hash(wf)
    wrong_feature=accuracy(wf,held)
    wp={**model,"predicate_table":[1-x for x in pt]};wp["model_hash"]=model_hash(wp)
    wrong_pred=accuracy(wp,held)
    wa={**model,"action_table":[1-x for x in at]};wa["model_hash"]=model_hash(wa)
    wrong_action=accuracy(wa,held)
    old_more=max(remove_pred,remove_action)

    with tempfile.TemporaryDirectory(prefix="arte_full_chain_") as td:
        p=Path(td)/"model.json";p.write_text(json.dumps(model,sort_keys=True))
        raw=[{k:c[k] for k in ("id","raw","b0","b1","state")} for c in held]
        cp=subprocess.run([sys.executable,__file__,"--cold",str(p),json.dumps(raw,sort_keys=True)],cwd=ROOT,text=True,capture_output=True,timeout=30,check=False,env={**os.environ,"PYTHONPATH":str(ROOT)})
        if cp.returncode!=0:raise AssertionError(cp.stderr)
        cold=json.loads([x for x in cp.stdout.splitlines() if x.strip()][-1])
    expected=[execute(model,c) for c in held]
    cold_exact=cold["model_hash"]==model["model_hash"] and cold["rows"]==expected

    controls=[remove_feature,remove_pred,remove_action,wrong_feature,wrong_pred,wrong_action,old_more]
    if not (full==1.0 and all(x<1.0 for x in controls) and cold_exact):
        raise AssertionError({"full":full,"controls":controls,"cold":cold_exact})

    print(json.dumps({
      "status":"PASS_BOUNDED_GENERATED_FEATURE_PREDICATE_ACTION_LANGUAGE_CHAIN_AND_COLD_TRANSFER",
      "seed":seed,
      "old_scalar_representation_unique_count":len(scalars),
      "old_scalar_outcome_class_count":len(labels),
      "old_scalar_inexpressive":True,
      "generated_feature_pool":sorted(set().union(*(discover_features(c["raw"]) for c in train))),
      "selected_generated_features":list(selected),
      "generated_predicate_operator_table":list(pt),
      "generated_action_primitive_table":list(at),
      "old_predicate_exact_solution_count":int(pred_target in old_pred),
      "old_action_exact_solution_count":int(action_transform in old_action),
      "model_hash":model["model_hash"],
      "heldout_preoutcome_execution_seal_sha256":seal,
      "full_end_to_end_accuracy":full,
      "remove_feature_accuracy":remove_feature,
      "remove_predicate_accuracy":remove_pred,
      "remove_action_accuracy":remove_action,
      "wrong_feature_accuracy":wrong_feature,
      "wrong_predicate_accuracy":wrong_pred,
      "wrong_action_accuracy":wrong_action,
      "old_more_compute_ceiling_accuracy":old_more,
      "cold_exact":cold_exact,
      "claim_boundary":{
        "raw_trace_schema_human_authored":True,
        "relation_discovery_rule_human_authored":True,
        "truth_table_enumerators_human_authored":True,
        "finite_boolean_universes_human_authored":True,
        "generated_relation_feature_ids_not_predeclared":True,
        "generated_predicate_operator_not_predeclared":True,
        "generated_action_primitive_not_predeclared":True,
        "unrestricted_feature_predicate_action_language_genesis":False,
        "independent_organizational_custody":False,
        "physical_world":False,
        "global_recursive_acceleration":False,
        "AGI":False,"ASI":False,"superintelligence":False
      }
    },sort_keys=True))
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--seed",type=int,default=20260929931);ap.add_argument("--cold",nargs=2);a=ap.parse_args()
    if a.cold:raise SystemExit(cold_mode(a.cold[0],a.cold[1]))
    raise SystemExit(main(a.seed))
