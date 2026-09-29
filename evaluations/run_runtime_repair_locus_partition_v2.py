from __future__ import annotations
import argparse, hashlib, json, os, random, subprocess, sys, tempfile
from pathlib import Path

FEATURES=("old_basis_expressive","current_categories_separate_outcomes","current_action_alphabet_expressive","capability_present","normalized_evidence_cost")
LOCI=("OBSERVATION_BASIS","CATEGORY_BOUNDARY","ACTION_LIBRARY","SEARCH_POLICY","NULL")

def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()

def patterns():
    return [
      ("basis-a",dict(old_basis_expressive=0,current_categories_separate_outcomes=0,current_action_alphabet_expressive=1,capability_present=0,normalized_evidence_cost=.8),"OBSERVATION_BASIS"),
      ("basis-b",dict(old_basis_expressive=0,current_categories_separate_outcomes=1,current_action_alphabet_expressive=1,capability_present=0,normalized_evidence_cost=.4),"OBSERVATION_BASIS"),
      ("cat-a",dict(old_basis_expressive=1,current_categories_separate_outcomes=0,current_action_alphabet_expressive=1,capability_present=0,normalized_evidence_cost=.8),"CATEGORY_BOUNDARY"),
      ("cat-b",dict(old_basis_expressive=1,current_categories_separate_outcomes=0,current_action_alphabet_expressive=0,capability_present=0,normalized_evidence_cost=.4),"CATEGORY_BOUNDARY"),
      ("action-a",dict(old_basis_expressive=1,current_categories_separate_outcomes=1,current_action_alphabet_expressive=0,capability_present=0,normalized_evidence_cost=.8),"ACTION_LIBRARY"),
      ("action-b",dict(old_basis_expressive=0,current_categories_separate_outcomes=1,current_action_alphabet_expressive=0,capability_present=0,normalized_evidence_cost=.4),"ACTION_LIBRARY"),
      ("search-a",dict(old_basis_expressive=1,current_categories_separate_outcomes=1,current_action_alphabet_expressive=1,capability_present=1,normalized_evidence_cost=.9),"SEARCH_POLICY"),
      ("search-b",dict(old_basis_expressive=0,current_categories_separate_outcomes=0,current_action_alphabet_expressive=1,capability_present=1,normalized_evidence_cost=.9),"SEARCH_POLICY"),
      ("null-basis-cap",dict(old_basis_expressive=0,current_categories_separate_outcomes=1,current_action_alphabet_expressive=1,capability_present=1,normalized_evidence_cost=.2),"NULL"),
      ("null-cat-cap",dict(old_basis_expressive=1,current_categories_separate_outcomes=0,current_action_alphabet_expressive=1,capability_present=1,normalized_evidence_cost=.2),"NULL"),
      ("null-action-cap",dict(old_basis_expressive=1,current_categories_separate_outcomes=1,current_action_alphabet_expressive=0,capability_present=1,normalized_evidence_cost=.2),"NULL"),
      ("null-efficient",dict(old_basis_expressive=1,current_categories_separate_outcomes=1,current_action_alphabet_expressive=1,capability_present=1,normalized_evidence_cost=.1),"NULL"),
      ("null-incomplete",dict(old_basis_expressive=0,current_categories_separate_outcomes=0,current_action_alphabet_expressive=0,capability_present=1,normalized_evidence_cost=.1),"NULL"),
    ]

def examples(seed,held=False):
    rng=random.Random(seed+(100000 if held else 0)); out=[]; reps=1 if held else 2
    for rep in range(reps):
      for name,f,l in patterns():
        x=dict(f)
        c=x["normalized_evidence_cost"]
        if c>=.8:x["normalized_evidence_cost"]=round(rng.uniform(.75,.98),6)
        elif c<=.2:x["normalized_evidence_cost"]=round(rng.uniform(.02,.25),6)
        else:x["normalized_evidence_cost"]=round(rng.uniform(.32,.55),6)
        out.append({"id":f"{'ho' if held else 'tr'}-{rep}-{name}-{rng.randrange(10**8)}","features":x,"label":l})
    return out

def candidate_splits(rows):
    out=[]
    for f in FEATURES:
      vals=sorted(set(float(r["features"][f]) for r in rows))
      for a,b in zip(vals,vals[1:]):
        t=(a+b)/2
        out.append((f,t))
    return tuple(out)

def split_eval(sp,feat): return float(feat[sp[0]])<=float(sp[1])

def tree_predict(t,feat):
    while t["kind"]=="NODE":
      t=t["left"] if split_eval((t["feature"],t["threshold"]),feat) else t["right"]
    return t["label"]

def nodes(t):
    return 1 if t["kind"]=="LEAF" else 1+nodes(t["left"])+nodes(t["right"])
def depth(t):
    return 0 if t["kind"]=="LEAF" else 1+max(depth(t["left"]),depth(t["right"]))

def learn_tree(rows,max_depth=5):
    splits=candidate_splits(rows)
    memo={}
    def rec(ids,d,used):
      key=(tuple(ids),d,tuple(sorted(used)))
      if key in memo:return memo[key]
      labs={rows[i]["label"] for i in ids}
      if len(labs)==1:
        ans={"kind":"LEAF","label":next(iter(labs))};memo[key]=ans;return ans
      if d==0:return None
      cand=[]
      for si,sp in enumerate(splits):
        if si in used:continue
        L=[i for i in ids if split_eval(sp,rows[i]["features"])]
        R=[i for i in ids if not split_eval(sp,rows[i]["features"])]
        if not L or not R:continue
        lt=rec(L,d-1,used|{si}); rt=rec(R,d-1,used|{si})
        if lt is None or rt is None:continue
        t={"kind":"NODE","feature":sp[0],"threshold":sp[1],"left":lt,"right":rt}
        cand.append((nodes(t),depth(t),canon(t),t))
      if not cand:memo[key]=None;return None
      cand.sort(key=lambda x:(x[0],x[1],x[2]))
      memo[key]=cand[0][3];return memo[key]
    allids=list(range(len(rows)))
    candidates=[]
    for d in range(1,max_depth+1):
      t=rec(allids,d,set())
      if t is not None:
        candidates.append((nodes(t),depth(t),canon(t),t))
        break
    if not candidates:raise AssertionError("no global partition tree")
    candidates.sort(key=lambda x:(x[0],x[1],x[2]));return candidates[0][3]

def utility(selected,oracle):
    if selected==oracle:return .5 if oracle=="NULL" else 1.0
    return -0.5 if oracle=="NULL" else (0.0 if selected=="NULL" else -1.0)

def model_hash(tree): return sha({"tree":tree,"loci":LOCI})

def cold_main(path,feat_json):
    o=json.loads(Path(path).read_text())
    if model_hash(o["tree"])!=o["model_hash"]:raise AssertionError("hash mismatch")
    print(json.dumps({"model_hash":o["model_hash"],"selected_locus":tree_predict(o["tree"],json.loads(feat_json))},sort_keys=True))
    return 0

def main(seed):
    tr=examples(seed,False); tree=learn_tree(tr,5)
    train_acc=sum(tree_predict(tree,r["features"])==r["label"] for r in tr)/len(tr)
    if train_acc!=1.0:raise AssertionError(train_acc)

    ho=examples(seed,True)
    preds=[{"id":r["id"],"features":r["features"],"selected_locus":tree_predict(tree,r["features"])} for r in ho]
    seal=sha(preds)
    results=[]; correct=True; wrong_degrades=True; null_ok=True
    for r,p in zip(ho,preds):
      sel=p["selected_locus"]; oracle=r["label"]
      correct &= sel==oracle
      wrong=next(x for x in LOCI if x!=oracle and x!="NULL") if oracle!="NULL" else "ACTION_LIBRARY"
      tu=utility(sel,oracle); wu=utility(wrong,oracle); nu=utility("NULL",oracle)
      wrong_degrades &= tu>wu
      if oracle!="NULL":wrong_degrades &= tu>nu
      else:null_ok &= sel=="NULL"
      results.append({**p,"oracle_after_reveal":oracle,"correct":sel==oracle,"treatment_utility":tu,"wrong_locus":wrong,"wrong_utility":wu,"null_utility":nu})
    if not correct or not wrong_degrades or not null_ok:raise AssertionError({"correct":correct,"wrong":wrong_degrades,"null":null_ok,"tree":tree})

    mh=model_hash(tree)
    cold=examples(seed+777,True)[random.Random(seed).randrange(len(patterns()))]
    with tempfile.TemporaryDirectory(prefix="arte_locus_partition_") as td:
      p=Path(td)/"model.json";p.write_text(json.dumps({"tree":tree,"model_hash":mh},sort_keys=True))
      cp=subprocess.run([sys.executable,__file__,"--cold",str(p),json.dumps(cold["features"],sort_keys=True)],text=True,capture_output=True,timeout=30,check=False,env={**os.environ,"PYTHONPATH":str(Path(__file__).resolve().parents[1])})
      if cp.returncode!=0:raise AssertionError(cp.stderr)
      cpred=json.loads([x for x in cp.stdout.splitlines() if x.strip()][-1])
    cold_ok=cpred["model_hash"]==mh and cpred["selected_locus"]==cold["label"]
    if not cold_ok:raise AssertionError({"cold":cpred,"oracle":cold["label"]})
    print(json.dumps({
      "status":"PASS_BOUNDED_GLOBAL_RUNTIME_REPAIR_LOCUS_PARTITION_WITH_NULL_AND_COLD_TRANSFER",
      "seed":seed,"tree":tree,"tree_nodes":nodes(tree),"tree_depth":depth(tree),
      "train_accuracy":train_acc,"heldout_prediction_seal_sha256":seal,
      "heldout_accuracy":sum(x["correct"] for x in results)/len(results),
      "heldout_results":results,"wrong_locus_degrades_all":wrong_degrades,"null_cases_correct":null_ok,
      "model_hash":mh,"cold_prediction":cpred,"cold_oracle_after_reveal":cold["label"],"cold_exact":cold_ok,
      "claim_boundary":{"diagnostic_feature_vocabulary_human_authored":True,"locus_vocabulary_human_authored":True,"tree_grammar_human_authored":True,"global_partition_generated_not_predeclared":True,"autonomous_new_locus_invention":False,"independent_organizational_custody":False,"physical_world":False,"global_recursive_acceleration":False,"AGI":False,"ASI":False,"superintelligence":False}
    },sort_keys=True))
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--seed",type=int,default=20260929711);ap.add_argument("--cold",nargs=2);a=ap.parse_args()
    if a.cold:raise SystemExit(cold_main(a.cold[0],a.cold[1]))
    raise SystemExit(main(a.seed))
