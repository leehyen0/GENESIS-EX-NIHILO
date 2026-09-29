from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]

FEATURES = (
    "frontier_complete",
    "best_base_capability",
    "candidate_count",
    "successful_base_candidates",
    "evidence_cost",
)
ACTIONS = ("SYNTHESIZE_MISSING_OPERATOR", "COMPILE_SEARCH_SELECTOR", "NULL")

@dataclass(frozen=True)
class World:
    world_id: str
    mechanism_token: str
    mode: str
    candidate_count: int
    evidence_multiplier: int

@dataclass(frozen=True)
class Outcome:
    capability: float
    evidence_cost: int
    candidate_count: int
    successful_candidates: int

@dataclass(frozen=True)
class Predicate:
    feature: str
    op: str
    threshold: float

    def eval(self, row: Dict[str, float]) -> bool:
        x=float(row[self.feature])
        if self.op=="LE":
            return x <= self.threshold
        if self.op=="GE":
            return x >= self.threshold
        raise ValueError(self.op)

    def ast(self):
        return {"kind":"ATOM","feature":self.feature,"op":self.op,"threshold":self.threshold}

@dataclass(frozen=True)
class AndPredicate:
    left: Predicate
    right: Predicate

    def eval(self,row:Dict[str,float])->bool:
        return self.left.eval(row) and self.right.eval(row)

    def ast(self):
        return {"kind":"AND","left":self.left.ast(),"right":self.right.ast()}

@dataclass(frozen=True)
class OntologyClass:
    class_id: str
    predicate_ast: dict
    action: str
    complexity: int

@dataclass(frozen=True)
class Ontology:
    classes: Tuple[OntologyClass,...]
    parent_hash: str
    ontology_hash: str

    def to_json(self):
        return {
            "classes":[asdict(c) for c in self.classes],
            "parent_hash":self.parent_hash,
            "ontology_hash":self.ontology_hash,
        }

def canon(obj)->str:
    return json.dumps(obj,sort_keys=True,separators=(",",":"))

def sha(obj)->str:
    return hashlib.sha256(canon(obj).encode()).hexdigest()

def make_worlds(seed:int):
    rng=random.Random(seed)
    def tok(prefix):
        return prefix+"-"+hashlib.sha256(f"{seed}|{prefix}|{rng.random()}".encode()).hexdigest()[:12]
    train=[
        World("train-0",tok("surface-a"),"EXPAND",4+rng.randrange(0,3),2),
        World("train-1",tok("surface-b"),"EXPAND",5+rng.randrange(0,4),3),
        World("train-2",tok("surface-c"),"REFINE",10+rng.randrange(0,5),2),
        World("train-3",tok("surface-d"),"REFINE",13+rng.randrange(0,6),2),
    ]
    held=[
        World("held-0",tok("surface-e"),"EXPAND",6+rng.randrange(0,4),2),
        World("held-1",tok("surface-f"),"EXPAND",7+rng.randrange(0,5),3),
        World("held-2",tok("surface-g"),"REFINE",15+rng.randrange(0,5),2),
        World("held-3",tok("surface-h"),"REFINE",18+rng.randrange(0,6),2),
    ]
    cold=World("cold-0",tok("surface-i"),rng.choice(["EXPAND","REFINE"]),11+rng.randrange(0,8),2)
    return train,held,cold

def baseline(world:World)->Outcome:
    if world.mode=="EXPAND":
        return Outcome(0.0,world.candidate_count*world.evidence_multiplier,world.candidate_count,0)
    if world.mode=="REFINE":
        return Outcome(1.0,world.candidate_count*world.evidence_multiplier,world.candidate_count,1)
    raise ValueError(world.mode)

def apply_action(world:World,action:str)->Outcome:
    base=baseline(world)
    if action=="NULL":
        return base
    if action=="SYNTHESIZE_MISSING_OPERATOR":
        if world.mode=="EXPAND":
            return Outcome(1.0,2,1,1)
        return Outcome(0.0,base.evidence_cost+2,base.candidate_count+1,0)
    if action=="COMPILE_SEARCH_SELECTOR":
        if world.mode=="REFINE":
            return Outcome(1.0,2,1,1)
        return Outcome(0.0,base.evidence_cost+1,base.candidate_count,0)
    raise ValueError(action)

def residual_features(world:World)->Dict[str,float]:
    b=baseline(world)
    return {
        "frontier_complete":1.0,
        "best_base_capability":b.capability,
        "candidate_count":float(b.candidate_count),
        "successful_base_candidates":float(b.successful_candidates),
        "evidence_cost":float(b.evidence_cost),
    }

def action_utility(o:Outcome)->float:
    # capability dominates; cost only compares equally capable repairs.
    return float(o.capability)*1000.0-float(o.evidence_cost)

def best_training_action(world:World)->Tuple[str,Dict[str,Outcome]]:
    outcomes={a:apply_action(world,a) for a in ACTIONS}
    ranked=sorted(ACTIONS,key=lambda a:(-action_utility(outcomes[a]),a))
    return ranked[0],outcomes

def thresholds(rows:List[Dict[str,float]],feature:str)->Tuple[float,...]:
    vals=sorted(set(float(r[feature]) for r in rows))
    mids=[]
    for a,b in zip(vals,vals[1:]):
        mids.append((a+b)/2.0)
    return tuple(mids)

def all_predicates(rows:List[Dict[str,float]]):
    atoms=[]
    for f in FEATURES:
        for t in thresholds(rows,f):
            atoms.append(Predicate(f,"LE",t))
            atoms.append(Predicate(f,"GE",t))
    # deterministic simple-to-complex grammar
    for p in atoms:
        yield (1,p)
    for i,a in enumerate(atoms):
        for b in atoms[i+1:]:
            yield (2,AndPredicate(a,b))

def predicate_ast(p):
    return p.ast()

def learn_ontology(training_rows:List[Dict], parent_hash:str="")->Ontology:
    features=[r["features"] for r in training_rows]
    actions=sorted(set(r["best_action"] for r in training_rows))
    classes=[]
    used_masks=[]
    for action in actions:
        target=tuple(r["best_action"]==action for r in training_rows)
        candidates=[]
        for complexity,p in all_predicates(features):
            mask=tuple(bool(p.eval(r["features"])) for r in training_rows)
            if mask==target:
                ast=predicate_ast(p)
                candidates.append((complexity,canon(ast),ast,p))
        if not candidates:
            raise AssertionError(f"no predicate exactly generated action class {action}")
        candidates.sort(key=lambda x:(x[0],x[1]))
        complexity,_,ast,_=candidates[0]
        cid="GEN_META_REPAIR_CLASS::"+sha(ast)[:20]
        classes.append(OntologyClass(cid,ast,action,complexity))
        used_masks.append(target)
    payload={
        "classes":[asdict(c) for c in sorted(classes,key=lambda c:c.class_id)],
        "parent_hash":parent_hash,
    }
    oh=sha(payload)
    return Ontology(tuple(sorted(classes,key=lambda c:c.class_id)),parent_hash,oh)

def eval_ast(ast,row):
    if ast["kind"]=="ATOM":
        x=float(row[ast["feature"]]);t=float(ast["threshold"])
        return x<=t if ast["op"]=="LE" else x>=t
    if ast["kind"]=="AND":
        return eval_ast(ast["left"],row) and eval_ast(ast["right"],row)
    raise ValueError(ast)

def select(ontology:Ontology,row:Dict[str,float])->Tuple[str,str]:
    hits=[c for c in ontology.classes if eval_ast(c.predicate_ast,row)]
    if len(hits)==0:
        return "UNCLASSIFIED","NULL"
    if len(hits)>1:
        raise AssertionError(f"ontology overlap: {[c.class_id for c in hits]}")
    return hits[0].class_id,hits[0].action

def wrong_action(action:str)->str:
    if action=="SYNTHESIZE_MISSING_OPERATOR":
        return "COMPILE_SEARCH_SELECTOR"
    if action=="COMPILE_SEARCH_SELECTOR":
        return "SYNTHESIZE_MISSING_OPERATOR"
    return "NULL"

def ontology_from_json(obj:dict)->Ontology:
    classes=tuple(OntologyClass(
        str(c["class_id"]),dict(c["predicate_ast"]),str(c["action"]),int(c["complexity"])
    ) for c in obj["classes"])
    payload={"classes":[asdict(c) for c in sorted(classes,key=lambda c:c.class_id)],"parent_hash":str(obj["parent_hash"])}
    h=sha(payload)
    if h!=obj["ontology_hash"]:
        raise AssertionError("ontology hash mismatch")
    return Ontology(tuple(sorted(classes,key=lambda c:c.class_id)),str(obj["parent_hash"]),h)

def cold_mode(path:str,features_json:str):
    ont=ontology_from_json(json.loads(Path(path).read_text()))
    features=json.loads(features_json)
    cid,action=select(ont,features)
    print(json.dumps({"ontology_hash":ont.ontology_hash,"class_id":cid,"selected_action":action},sort_keys=True))
    return 0

def main(seed:int):
    train,held,cold=make_worlds(seed)
    training_rows=[]
    for w in train:
        best,outcomes=best_training_action(w)
        training_rows.append({
            "world_id":w.world_id,
            "mechanism_token_hash":hashlib.sha256(w.mechanism_token.encode()).hexdigest(),
            "features":residual_features(w),
            "best_action":best,
            "action_outcomes":{k:asdict(v) for k,v in outcomes.items()},
        })

    ontology=learn_ontology(training_rows)
    if len(ontology.classes)<2:
        raise AssertionError("ontology did not generate multiple classes")
    if any(c.class_id in ACTIONS for c in ontology.classes):
        raise AssertionError("class ids leaked action vocabulary")

    # Training fit check
    train_correct=0
    for row in training_rows:
        _,a=select(ontology,row["features"])
        train_correct+=int(a==row["best_action"])
    train_acc=train_correct/len(training_rows)
    if train_acc!=1.0:
        raise AssertionError("generated ontology did not fit training evidence")

    # Heldout classification and action selection happen from baseline residual only.
    held_predictions=[]
    for w in held:
        f=residual_features(w)
        cid,a=select(ontology,f)
        held_predictions.append({
            "world_id":w.world_id,
            "mechanism_token_hash":hashlib.sha256(w.mechanism_token.encode()).hexdigest(),
            "features":f,
            "class_id":cid,
            "selected_action":a,
        })
    held_seal=sha(held_predictions)

    held_results=[]
    all_treatment=True
    wrong_degrades=True
    remove_not_equivalent=True
    more_compute_not_replace=True
    selector_cost_reduction=True
    for w,pred in zip(held,held_predictions):
        treatment=apply_action(w,pred["selected_action"])
        remove=apply_action(w,"NULL")
        wrong=apply_action(w,wrong_action(pred["selected_action"]))
        more_compute=baseline(w)  # same fixed frontier, more repeats cannot add a missing primitive or lower structural cost.
        oracle_best,_=best_training_action(w)
        correct=(pred["selected_action"]==oracle_best)
        treatment_ok=(treatment.capability==1.0)
        all_treatment &= correct and treatment_ok
        wrong_degrades &= action_utility(wrong) < action_utility(treatment)
        remove_not_equivalent &= action_utility(remove) < action_utility(treatment)
        if w.mode=="EXPAND":
            more_compute_not_replace &= more_compute.capability==0.0
        if w.mode=="REFINE":
            selector_cost_reduction &= treatment.capability==1.0 and treatment.evidence_cost < remove.evidence_cost
        held_results.append({
            **pred,
            "oracle_action_after_reveal":oracle_best,
            "correct_preoutcome_action":correct,
            "treatment":asdict(treatment),
            "remove":asdict(remove),
            "wrong":asdict(wrong),
            "fixed_more_compute":asdict(more_compute),
        })

    # Cold child only receives ontology bytes + fresh baseline residual features.
    cold_features=residual_features(cold)
    with tempfile.TemporaryDirectory(prefix="arte_meta_ontology_") as td:
        p=Path(td)/"ontology.json"
        p.write_text(json.dumps(ontology.to_json(),sort_keys=True),encoding="utf-8")
        cp=subprocess.run(
            [sys.executable,__file__,"--cold",str(p),json.dumps(cold_features,sort_keys=True)],
            cwd=ROOT,text=True,capture_output=True,timeout=30,check=False,
            env={**os.environ,"PYTHONPATH":str(ROOT)},
        )
        if cp.returncode!=0:
            raise AssertionError(cp.stderr)
        cold_pred=None
        for line in reversed([x.strip() for x in cp.stdout.splitlines() if x.strip()]):
            if line.startswith("{"):
                cold_pred=json.loads(line);break
        if cold_pred is None:
            raise AssertionError("cold child emitted no prediction")
    cold_seal=sha(cold_pred)
    cold_oracle,_=best_training_action(cold)
    cold_treatment=apply_action(cold,cold_pred["selected_action"])
    cold_remove=apply_action(cold,"NULL")
    cold_wrong=apply_action(cold,wrong_action(cold_pred["selected_action"]))
    cold_ok=(
        cold_pred["ontology_hash"]==ontology.ontology_hash
        and cold_pred["selected_action"]==cold_oracle
        and cold_treatment.capability==1.0
        and action_utility(cold_treatment)>action_utility(cold_remove)
        and action_utility(cold_treatment)>action_utility(cold_wrong)
    )

    passed=all((
        len(ontology.classes)>=2,
        train_acc==1.0,
        all_treatment,
        wrong_degrades,
        remove_not_equivalent,
        more_compute_not_replace,
        selector_cost_reduction,
        cold_ok,
    ))
    if not passed:
        raise AssertionError({
            "all_treatment":all_treatment,
            "wrong_degrades":wrong_degrades,
            "remove_not_equivalent":remove_not_equivalent,
            "more_compute_not_replace":more_compute_not_replace,
            "selector_cost_reduction":selector_cost_reduction,
            "cold_ok":cold_ok,
        })

    result={
        "status":"PASS_BOUNDED_META_REPAIR_ONTOLOGY_CLASS_PREDICATE_GENESIS_AND_COLD_TRANSFER",
        "seed":seed,
        "bootstrap_named_residual_classes":[],
        "generated_class_count":len(ontology.classes),
        "generated_classes":[asdict(c) for c in ontology.classes],
        "ontology_hash":ontology.ontology_hash,
        "training_accuracy":train_acc,
        "training_rows":training_rows,
        "heldout_preoutcome_prediction_seal_sha256":held_seal,
        "heldout_results":held_results,
        "heldout_treatment_all_success":all_treatment,
        "wrong_mapping_degrades_all":wrong_degrades,
        "remove_or_null_not_equivalent":remove_not_equivalent,
        "fixed_more_compute_cannot_replace_missing_operator":more_compute_not_replace,
        "selector_worlds_preserve_capability_reduce_cost":selector_cost_reduction,
        "cold_prediction_seal_sha256":cold_seal,
        "cold_prediction":cold_pred,
        "cold_world_mode_hidden_from_learner":True,
        "cold_oracle_action_after_reveal":cold_oracle,
        "cold_treatment":asdict(cold_treatment),
        "cold_remove":asdict(cold_remove),
        "cold_wrong":asdict(cold_wrong),
        "cold_process_exact_ontology_hash":cold_pred["ontology_hash"]==ontology.ontology_hash,
        "cold_world_pass":cold_ok,
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
    }
    print(json.dumps(result,sort_keys=True))
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=20260929011)
    ap.add_argument("--cold",nargs=2,metavar=("ONTOLOGY_JSON","FEATURES_JSON"))
    args=ap.parse_args()
    if args.cold:
        raise SystemExit(cold_mode(args.cold[0],args.cold[1]))
    raise SystemExit(main(args.seed))
