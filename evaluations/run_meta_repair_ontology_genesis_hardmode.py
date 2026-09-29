from __future__ import annotations
import argparse, hashlib, json, os, random, subprocess, sys, tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT=Path(__file__).resolve().parents[1]
FEATURES=("frontier_complete","best_base_capability","candidate_count","successful_base_candidates","evidence_cost")
ACTIONS=("SYNTHESIZE_MISSING_OPERATOR","COMPILE_SEARCH_SELECTOR","NULL")

@dataclass(frozen=True)
class World:
    world_id:str
    mechanism_token:str
    mode:str
    frontier_complete:int
    candidate_count:int
    evidence_multiplier:int

@dataclass(frozen=True)
class Outcome:
    capability:float
    evidence_cost:int
    candidate_count:int
    successful_candidates:int

@dataclass(frozen=True)
class Atom:
    feature:str
    op:str
    threshold:float
    def eval(self,row:Dict[str,float])->bool:
        x=float(row[self.feature])
        return x<=self.threshold if self.op=="LE" else x>=self.threshold
    def ast(self): return {"kind":"ATOM","feature":self.feature,"op":self.op,"threshold":self.threshold}

@dataclass(frozen=True)
class And:
    left:Atom
    right:Atom
    def eval(self,row:Dict[str,float])->bool: return self.left.eval(row) and self.right.eval(row)
    def ast(self): return {"kind":"AND","left":self.left.ast(),"right":self.right.ast()}

@dataclass(frozen=True)
class OClass:
    class_id:str
    predicate_ast:dict
    action:str
    complexity:int

def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()

def make_worlds(seed:int):
    rng=random.Random(seed)
    def token(prefix): return prefix+"-"+hashlib.sha256(f"{seed}|{prefix}|{rng.random()}".encode()).hexdigest()[:10]
    def c(lo,hi): return rng.randrange(lo,hi+1)
    train=[
        World("tr-exp-0",token("mx-a"),"EXPAND",1,c(5,8),2),
        World("tr-exp-1",token("mx-b"),"EXPAND",1,c(7,10),3),
        World("tr-ref-0",token("mx-c"),"REFINE",1,c(12,16),2),
        World("tr-ref-1",token("mx-d"),"REFINE",1,c(16,20),2),
        World("tr-inc-0",token("mx-e"),"INCOMPLETE_NULL",0,c(8,12),2),
        World("tr-inc-1",token("mx-f"),"INCOMPLETE_NULL",0,c(10,14),3),
        World("tr-eff-0",token("mx-g"),"EFFICIENT_NULL",1,1,2),
        World("tr-eff-1",token("mx-h"),"EFFICIENT_NULL",1,2,1),
    ]
    held=[
        World("ho-exp",token("hx-a"),"EXPAND",1,c(8,11),2),
        World("ho-ref",token("hx-b"),"REFINE",1,c(18,23),2),
        World("ho-inc",token("hx-c"),"INCOMPLETE_NULL",0,c(9,13),3),
        World("ho-eff",token("hx-d"),"EFFICIENT_NULL",1,1,2),
    ]
    modes=("EXPAND","REFINE","INCOMPLETE_NULL","EFFICIENT_NULL")
    m=rng.choice(modes)
    if m=="EXPAND": cold=World("cold",token("cx"),m,1,c(9,13),2)
    elif m=="REFINE": cold=World("cold",token("cx"),m,1,c(19,25),2)
    elif m=="INCOMPLETE_NULL": cold=World("cold",token("cx"),m,0,c(10,15),2)
    else: cold=World("cold",token("cx"),m,1,1,2)
    return train,held,cold

def baseline(w:World)->Outcome:
    if w.mode=="EXPAND": return Outcome(0.0,w.candidate_count*w.evidence_multiplier,w.candidate_count,0)
    if w.mode=="REFINE": return Outcome(1.0,w.candidate_count*w.evidence_multiplier,w.candidate_count,1)
    if w.mode=="INCOMPLETE_NULL": return Outcome(0.0,w.candidate_count*w.evidence_multiplier,w.candidate_count,0)
    if w.mode=="EFFICIENT_NULL": return Outcome(1.0,w.candidate_count*w.evidence_multiplier,w.candidate_count,1)
    raise ValueError(w.mode)

def apply(w:World,action:str)->Outcome:
    b=baseline(w)
    if action=="NULL": return b
    if w.mode=="INCOMPLETE_NULL":
        return Outcome(0.0,b.evidence_cost+2,b.candidate_count+1,0)
    if action=="SYNTHESIZE_MISSING_OPERATOR":
        if w.mode=="EXPAND": return Outcome(1.0,2,1,1)
        return Outcome(0.0,b.evidence_cost+2,b.candidate_count+1,0)
    if action=="COMPILE_SEARCH_SELECTOR":
        if w.mode=="REFINE": return Outcome(1.0,2,1,1)
        if w.mode=="EFFICIENT_NULL": return Outcome(1.0,b.evidence_cost,b.candidate_count,b.successful_candidates)
        return Outcome(0.0,b.evidence_cost+1,b.candidate_count,0)
    raise ValueError(action)

def features(w:World):
    b=baseline(w)
    return {
        "frontier_complete":float(w.frontier_complete),
        "best_base_capability":b.capability,
        "candidate_count":float(b.candidate_count),
        "successful_base_candidates":float(b.successful_candidates),
        "evidence_cost":float(b.evidence_cost),
    }

def utility(o:Outcome): return o.capability*1000.0-o.evidence_cost

def best_action(w:World):
    base=apply(w,"NULL")
    rows={a:apply(w,a) for a in ACTIONS}
    # require strict gain over NULL; otherwise abstain.
    best_nonnull=max(("SYNTHESIZE_MISSING_OPERATOR","COMPILE_SEARCH_SELECTOR"),key=lambda a:(utility(rows[a]),a))
    if utility(rows[best_nonnull])>utility(base):
        return best_nonnull,rows
    return "NULL",rows

def atoms(rows:List[dict]):
    out=[]
    for f in FEATURES:
        vals=sorted(set(float(r[f]) for r in rows))
        for a,b in zip(vals,vals[1:]):
            t=(a+b)/2.0
            out.extend((Atom(f,"LE",t),Atom(f,"GE",t)))
    return out

def learn(training:List[dict]):
    xs=[r["features"] for r in training]
    ats=atoms(xs)
    classes=[]
    atomic_shortcuts={}
    for action in ("SYNTHESIZE_MISSING_OPERATOR","COMPILE_SEARCH_SELECTOR"):
        target=tuple(r["best_action"]==action for r in training)
        atomic=[a for a in ats if tuple(a.eval(x) for x in xs)==target]
        atomic_shortcuts[action]=len(atomic)
        if atomic:
            raise AssertionError(f"atomic shortcut remained for {action}: {atomic[0]}")
        candidates=[]
        for i,a in enumerate(ats):
            for b in ats[i+1:]:
                p=And(a,b)
                if tuple(p.eval(x) for x in xs)==target:
                    ast=p.ast(); candidates.append((canon(ast),ast))
        if not candidates: raise AssertionError(f"no depth-2 predicate for {action}")
        candidates.sort()
        ast=candidates[0]
        ast=ast[1]
        cid="GEN_META_REPAIR_CLASS::"+sha(ast)[:20]
        classes.append(OClass(cid,ast,action,2))
    payload={"classes":[asdict(c) for c in sorted(classes,key=lambda c:c.class_id)],"default_action":"NULL"}
    return tuple(sorted(classes,key=lambda c:c.class_id)),sha(payload),atomic_shortcuts

def eval_ast(ast,row):
    if ast["kind"]=="ATOM":
        x=float(row[ast["feature"]]);t=float(ast["threshold"])
        return x<=t if ast["op"]=="LE" else x>=t
    return eval_ast(ast["left"],row) and eval_ast(ast["right"],row)

def choose(classes,row):
    hits=[c for c in classes if eval_ast(c.predicate_ast,row)]
    if len(hits)==0:return "DEFAULT_NULL","NULL"
    if len(hits)>1:raise AssertionError([c.class_id for c in hits])
    return hits[0].class_id,hits[0].action

def wrong(a):
    if a=="SYNTHESIZE_MISSING_OPERATOR": return "COMPILE_SEARCH_SELECTOR"
    if a=="COMPILE_SEARCH_SELECTOR": return "SYNTHESIZE_MISSING_OPERATOR"
    return "SYNTHESIZE_MISSING_OPERATOR"

def cold_mode(path,feat_json):
    obj=json.loads(Path(path).read_text())
    classes=tuple(OClass(str(c["class_id"]),dict(c["predicate_ast"]),str(c["action"]),int(c["complexity"])) for c in obj["classes"])
    payload={"classes":[asdict(c) for c in sorted(classes,key=lambda c:c.class_id)],"default_action":"NULL"}
    if sha(payload)!=obj["ontology_hash"]: raise AssertionError("hash mismatch")
    cid,a=choose(classes,json.loads(feat_json))
    print(json.dumps({"ontology_hash":obj["ontology_hash"],"class_id":cid,"selected_action":a},sort_keys=True))
    return 0

def main(seed:int):
    train,held,cold=make_worlds(seed)
    training=[]
    for w in train:
        ba,outs=best_action(w)
        training.append({
            "world_id":w.world_id,
            "mechanism_token_hash":hashlib.sha256(w.mechanism_token.encode()).hexdigest(),
            "features":features(w),
            "best_action":ba,
            "outcomes":{k:asdict(v) for k,v in outs.items()},
        })
    classes,ontology_hash,atomic_shortcuts=learn(training)
    if any(c.complexity!=2 for c in classes): raise AssertionError("non-depth2 generated class")
    train_acc=sum(choose(classes,r["features"])[1]==r["best_action"] for r in training)/len(training)
    if train_acc!=1.0: raise AssertionError("train mismatch")

    preds=[]
    for w in held:
        cid,a=choose(classes,features(w))
        preds.append({"world_id":w.world_id,"features":features(w),"class_id":cid,"selected_action":a})
    pred_seal=sha(preds)

    results=[]; held_ok=True; wrong_degrades=True; null_ok=True
    for w,p in zip(held,preds):
        oracle,_=best_action(w)
        tr=apply(w,p["selected_action"]); rm=apply(w,"NULL"); wr=apply(w,wrong(p["selected_action"])); mc=baseline(w)
        correct=p["selected_action"]==oracle
        if oracle=="SYNTHESIZE_MISSING_OPERATOR":
            perf=tr.capability==1.0 and rm.capability==0.0 and mc.capability==0.0
        elif oracle=="COMPILE_SEARCH_SELECTOR":
            perf=tr.capability==1.0 and tr.evidence_cost<rm.evidence_cost
        else:
            perf=p["selected_action"]=="NULL" and utility(tr)>=max(utility(apply(w,"SYNTHESIZE_MISSING_OPERATOR")),utility(apply(w,"COMPILE_SEARCH_SELECTOR")))
            null_ok &= perf
        held_ok &= correct and perf
        wrong_degrades &= utility(wr)<utility(tr)
        results.append({**p,"oracle_after_reveal":oracle,"correct":correct,"treatment":asdict(tr),"remove_null":asdict(rm),"wrong":asdict(wr),"fixed_more_compute":asdict(mc)})

    cf=features(cold)
    with tempfile.TemporaryDirectory(prefix="arte_ontology_hard_") as td:
        p=Path(td)/"ontology.json"
        p.write_text(json.dumps({"classes":[asdict(c) for c in classes],"default_action":"NULL","ontology_hash":ontology_hash},sort_keys=True))
        cp=subprocess.run([sys.executable,__file__,"--cold",str(p),json.dumps(cf,sort_keys=True)],cwd=ROOT,text=True,capture_output=True,timeout=30,check=False,env={**os.environ,"PYTHONPATH":str(ROOT)})
        if cp.returncode!=0: raise AssertionError(cp.stderr)
        cold_pred=None
        for line in reversed([x.strip() for x in cp.stdout.splitlines() if x.strip()]):
            if line.startswith("{"): cold_pred=json.loads(line);break
        if cold_pred is None: raise AssertionError("no cold prediction")
    cold_seal=sha(cold_pred)
    oracle,_=best_action(cold); tr=apply(cold,cold_pred["selected_action"]); wr=apply(cold,wrong(cold_pred["selected_action"]))
    cold_ok=(cold_pred["ontology_hash"]==ontology_hash and cold_pred["selected_action"]==oracle and utility(tr)>utility(wr))
    if oracle=="NULL": cold_ok &= cold_pred["selected_action"]=="NULL"
    elif oracle=="SYNTHESIZE_MISSING_OPERATOR": cold_ok &= tr.capability==1.0
    else: cold_ok &= tr.capability==1.0 and tr.evidence_cost<baseline(cold).evidence_cost

    if not (held_ok and wrong_degrades and null_ok and cold_ok and all(v==0 for v in atomic_shortcuts.values())):
        raise AssertionError({"held_ok":held_ok,"wrong_degrades":wrong_degrades,"null_ok":null_ok,"cold_ok":cold_ok,"atomic_shortcuts":atomic_shortcuts})

    print(json.dumps({
        "status":"PASS_BOUNDED_HARDMODE_META_REPAIR_ONTOLOGY_CONJUNCTION_GENESIS_WITH_NULL_AND_COLD_TRANSFER",
        "seed":seed,
        "bootstrap_named_residual_classes":[],
        "generated_nonnull_classes":[asdict(c) for c in classes],
        "all_generated_complexity_two":all(c.complexity==2 for c in classes),
        "atomic_shortcut_counts":atomic_shortcuts,
        "ontology_hash":ontology_hash,
        "training_accuracy":train_acc,
        "heldout_preoutcome_prediction_seal_sha256":pred_seal,
        "heldout_results":results,
        "heldout_all_correct":held_ok,
        "null_worlds_correct":null_ok,
        "wrong_mapping_degrades":wrong_degrades,
        "cold_prediction_seal_sha256":cold_seal,
        "cold_prediction":cold_pred,
        "cold_oracle_after_reveal":oracle,
        "cold_pass":cold_ok,
        "claim_boundary":{
            "residual_feature_vocabulary_human_authored":True,
            "predicate_grammar_human_authored":True,
            "repair_primitive_vocabulary_human_authored":True,
            "world_generator_human_authored":True,
            "generated_class_predicates_not_predeclared":True,
            "generated_thresholds_not_predeclared":True,
            "unrestricted_ontology_language_genesis":False,
            "independent_organizational_custody":False,
            "physical_world":False,
            "foundation_weight_change":False,
            "global_recursive_acceleration":False,
            "AGI":False,"ASI":False,"superintelligence":False,
        }
    },sort_keys=True))
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=20260929101)
    ap.add_argument("--cold",nargs=2)
    a=ap.parse_args()
    if a.cold: raise SystemExit(cold_mode(a.cold[0],a.cold[1]))
    raise SystemExit(main(a.seed))
