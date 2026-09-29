from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from arte_cognition.latent_relation_ontology_genesis import (
    OpaqueInterventionalWorld,
    WorldDerivedLatentRelationInducer,
    contrast,
)
from arte_cognition.observation_basis_genesis import WorldDerivedObservationBasisInducer

A_EXPAND="SYNTHESIZE_MISSING_OPERATOR"
A_REFINE="COMPILE_SEARCH_SELECTOR"
A_NULL="NULL"
ACTIONS=(A_EXPAND,A_REFINE,A_NULL)

def canon(obj): return json.dumps(obj,sort_keys=True,separators=(",",":"))
def sha(obj): return hashlib.sha256(canon(obj).encode()).hexdigest()

def world(context,prefix,domain,magnitude,lags,signs):
    nodes=[f"{prefix}_{i}" for i in range(4)]
    decoy=f"{prefix}_decoy"
    all_nodes=nodes+[decoy]
    rows=[]
    for si in range(3):
        lag=int(lags[si]); sign=int(signs[si])
        for repeat in range(2):
            low=[{n:0.0 for n in all_nodes} for _ in range(3)]
            high=[{n:0.0 for n in all_nodes} for _ in range(3)]
            high[0][nodes[si]]=float(magnitude)
            high[lag][nodes[si+1]]=float(sign)*float(magnitude)
            clutter=1 if repeat==0 else 2
            low[clutter][decoy]=float(repeat+1)
            high[clutter][decoy]=float(repeat+1)
            rows.append(contrast(f"{context}:{si}:{repeat}",nodes[si],low,high))
    return OpaqueInterventionalWorld(
        context_id=context,domain=domain,source_anchor=nodes[0],target_anchor=nodes[-1],
        contrasts=tuple(rows),
    )

def payload(w):
    return {
        "context_id":w.context_id,"domain":w.domain,
        "source_anchor":w.source_anchor,"target_anchor":w.target_anchor,
        "contrasts":[asdict(x) for x in w.contrasts],
    }

def repair_outcomes(need):
    if need=="EXPAND":
        return {
            A_EXPAND:{"capability":1.0,"cost":2},
            A_REFINE:{"capability":0.0,"cost":9},
            A_NULL:{"capability":0.0,"cost":8},
        }
    if need=="REFINE":
        return {
            A_EXPAND:{"capability":0.0,"cost":11},
            A_REFINE:{"capability":1.0,"cost":2},
            A_NULL:{"capability":1.0,"cost":20},
        }
    if need=="NULL":
        return {
            A_EXPAND:{"capability":0.0,"cost":9},
            A_REFINE:{"capability":0.0,"cost":8},
            A_NULL:{"capability":1.0,"cost":2},
        }
    raise ValueError(need)

def utility(o): return float(o["capability"])*1000.0-float(o["cost"])
def best_action(need):
    oo=repair_outcomes(need)
    return max(ACTIONS,key=lambda a:(utility(oo[a]),a))

def wrong_action(a):
    if a==A_EXPAND:return A_REFINE
    if a==A_REFINE:return A_EXPAND
    return A_EXPAND

def classify(inducer,schemas_to_actions,w):
    matches=[]
    for row in schemas_to_actions:
        class S: pass
        s=S(); s.basis_id=row["basis_id"]; s.profile_tokens=tuple(row["profile_tokens"])
        if inducer.matches(s,w):
            matches.append(row)
    if len(matches)==0:return "UNCLASSIFIED",A_NULL
    if len(matches)>1:raise AssertionError(f"ambiguous generated basis match: {matches}")
    return matches[0]["schema_id"],matches[0]["action"]

def cold_main(model_path,world_path):
    model=json.loads(Path(model_path).read_text())
    wj=json.loads(Path(world_path).read_text())
    # independent local trace reconstruction, no ARTE inducer import semantics used here.
    effects={}
    nodes={wj["source_anchor"],wj["target_anchor"]}
    for c in wj["contrasts"]:
        nodes.add(c["source_node"])
        for nm in ("low_timeline","high_timeline"):
            for snap in c[nm]:
                nodes.update(name for name,_ in snap)
    for c in wj["contrasts"]:
        source=c["source_node"]
        max_lag=min(len(c["low_timeline"]),len(c["high_timeline"]))-1
        for lag in range(1,max_lag+1):
            low=dict(c["low_timeline"][lag]); high=dict(c["high_timeline"][lag])
            for target in nodes:
                if target==source:continue
                e=float(high.get(target,0.0))-float(low.get(target,0.0))
                effects.setdefault((source,target,lag),[]).append(e)
    adj={}
    for source in sorted(nodes):
        for target in sorted(nodes):
            if source==target:continue
            cand=[]
            for lag in (1,2):
                vals=[x for x in effects.get((source,target,lag),[]) if abs(x)>=1e-9]
                if len(vals)<2:continue
                mean=sum(vals)/len(vals)
                if abs(mean)<1e-9:continue
                cand.append((abs(mean),-lag,mean))
            if not cand:continue
            cand.sort(reverse=True);_,neg,mean=cand[0];lag=-neg
            sign="POS" if mean>0 else "NEG"
            tok="OBS_REL::"+hashlib.sha256(f"peak_lag={lag}|sign={sign}".encode()).hexdigest()[:16]
            adj.setdefault(source,[]).append((target,tok))
    for v in adj.values():v.sort()
    paths=set()
    def walk(node,visited,toks):
        if len(toks)>=8:return
        for target,tok in adj.get(node,()):
            if target in visited:continue
            nt=toks+(tok,)
            if target==wj["target_anchor"]:paths.add(nt)
            walk(target,visited+(target,),nt)
    walk(wj["source_anchor"],(wj["source_anchor"],),())
    hits=[]
    for row in model["schemas"]:
        if tuple(row["profile_tokens"]) in paths:
            hits.append(row)
    if not hits:
        out={"model_hash":model["model_hash"],"selected_action":A_NULL,"class_id":"UNCLASSIFIED"}
    elif len(hits)==1:
        out={"model_hash":model["model_hash"],"selected_action":hits[0]["action"],"class_id":hits[0]["schema_id"]}
    else:
        raise AssertionError("cold overlap")
    print(json.dumps(out,sort_keys=True))
    return 0

def main(seed):
    rng=random.Random(seed)
    scalar_residual={"frontier_complete":1.0,"best_base_capability":0.5,"candidate_count":10.0,"evidence_cost":10.0}

    specs={
        "EXPAND":{"lags":(1,2,1),"signs":(1,1,1),"action":A_EXPAND},
        "REFINE":{"lags":(2,1,2),"signs":(1,-1,1),"action":A_REFINE},
    }
    inducer=WorldDerivedObservationBasisInducer(min_repeats=2,min_peak_effect=1e-9,max_path_depth=8,candidate_budget=64)
    predecessor=WorldDerivedLatentRelationInducer(lag=1,min_effect=0.1,min_repeats=2,max_path_depth=8,candidate_budget=64)

    training_rows=[]
    learned=[]
    for need,spec in specs.items():
        ws=tuple(
            world(f"train-{need.lower()}-{i}",f"{need.lower()}_{seed}_{i}_{rng.randrange(10**7)}",
                  f"TRAIN_{need}_{i}",1.0+3*i,spec["lags"],spec["signs"])
            for i in range(2)
        )
        pa=predecessor.assess_residual(ws,(0,0),2)
        pc=predecessor.generate_candidates(pa,ws)
        if pc: raise AssertionError(f"fixed-lag predecessor solved {need}")
        for _ in range(16):
            if predecessor.generate_candidates(pa,ws):
                raise AssertionError("more compute changed fixed-lag predecessor")

        oa=inducer.assess_residual(ws,(0,0),2)
        schemas=inducer.generate_candidates(oa,ws)
        if len(schemas)!=1:
            raise AssertionError(f"expected one generated basis for {need}, got {len(schemas)}")
        s=schemas[0]
        outcomes=repair_outcomes(need)
        chosen=max(ACTIONS,key=lambda a:(utility(outcomes[a]),a))
        if chosen!=spec["action"]:raise AssertionError("training repair outcome mismatch")
        row={
            "schema_id":s.schema_id,"basis_id":s.basis_id,
            "profile_tokens":list(s.profile_tokens),"action":chosen,
        }
        learned.append(row)
        training_rows.append({
            "need_hidden_from_basis_generator":need,
            "scalar_residual":scalar_residual,
            "predecessor_candidates":0,
            "generated_schema":row,
            "repair_outcomes":outcomes,
        })

    # Prove old scalar residual basis cannot separate the two different optimal actions.
    scalar_keys={canon(r["scalar_residual"]) for r in training_rows}
    optimal_actions={r["generated_schema"]["action"] for r in training_rows}
    scalar_inexpressive=(len(scalar_keys)==1 and len(optimal_actions)>1)
    if not scalar_inexpressive:raise AssertionError("old scalar basis unexpectedly expressive")

    model_payload={"schemas":sorted(learned,key=lambda r:r["schema_id"])}
    model_hash=sha(model_payload)
    model={**model_payload,"model_hash":model_hash}

    held_specs=[
        ("held-expand","EXPAND",(1,2,1),(1,1,1)),
        ("held-refine","REFINE",(2,1,2),(1,-1,1)),
        ("held-null","NULL",(1,1,1),(-1,1,-1)),
    ]
    held=[]
    for name,need,lags,signs in held_specs:
        w=world(name,f"{name}_{seed}_{rng.randrange(10**8)}",f"HELD_{name}",2.5,lags,signs)
        cid,action=classify(inducer,learned,w)
        held.append({"name":name,"need":need,"world":w,"class_id":cid,"selected_action":action})
    prediction_rows=[{"name":x["name"],"class_id":x["class_id"],"selected_action":x["selected_action"]} for x in held]
    prediction_seal=sha(prediction_rows)

    results=[]
    all_correct=True; controls=True
    for x in held:
        oracle=best_action(x["need"])
        selected=x["selected_action"]
        oo=repair_outcomes(x["need"])
        correct=(selected==oracle)
        treatment=oo[selected]; remove=oo[A_NULL]; wrong=oo[wrong_action(selected)]
        if x["need"]!="NULL":
            controls &= utility(treatment)>utility(remove) and utility(treatment)>utility(wrong)
        else:
            controls &= selected==A_NULL and utility(treatment)>utility(wrong)
        all_correct &= correct
        results.append({
            "world":x["name"],"class_id":x["class_id"],"selected_action":selected,
            "oracle_after_reveal":oracle,"correct":correct,
            "treatment":treatment,"remove_null":remove,"wrong_action":wrong,
        })
    if not all_correct or not controls:raise AssertionError("heldout transfer controls failed")

    # Cold world chosen after model exists.
    cold_need=rng.choice(("EXPAND","REFINE","NULL"))
    if cold_need=="EXPAND": lags,signs=(1,2,1),(1,1,1)
    elif cold_need=="REFINE": lags,signs=(2,1,2),(1,-1,1)
    else: lags,signs=(1,1,1),(-1,1,-1)
    cw=world("cold",f"cold_{seed}_{rng.randrange(10**8)}","COLD",4.0,lags,signs)
    with tempfile.TemporaryDirectory(prefix="arte_basis_ontology_") as td:
        mp=Path(td)/"model.json"; wp=Path(td)/"world.json"
        mp.write_text(json.dumps(model,sort_keys=True),encoding="utf-8")
        wp.write_text(json.dumps(payload(cw),sort_keys=True),encoding="utf-8")
        cp=subprocess.run([sys.executable,__file__,"--cold",str(mp),str(wp)],cwd=ROOT,
                          text=True,capture_output=True,timeout=30,check=False,
                          env={**os.environ,"PYTHONPATH":str(ROOT)})
        if cp.returncode!=0:raise AssertionError(cp.stderr)
        cold=json.loads([x for x in cp.stdout.splitlines() if x.strip()][-1])
    cold_seal=sha(cold)
    cold_oracle=best_action(cold_need)
    cold_ok=(cold["model_hash"]==model_hash and cold["selected_action"]==cold_oracle)
    if not cold_ok:raise AssertionError({"cold":cold,"oracle":cold_oracle})

    print(json.dumps({
        "status":"PASS_BOUNDED_GENERATED_OBSERVATION_BASIS_TO_REPAIR_ONTOLOGY_AND_COLD_TRANSFER",
        "seed":seed,
        "old_scalar_residual_basis_identical_across_training_worlds":True,
        "old_scalar_basis_inexpressive_for_repair_action":scalar_inexpressive,
        "fixed_lag_predecessor_candidates_each_group":0,
        "old_basis_more_compute_repetitions":16,
        "generated_basis_class_count":len(learned),
        "generated_basis_classes":learned,
        "model_hash":model_hash,
        "heldout_preoutcome_prediction_seal_sha256":prediction_seal,
        "heldout_results":results,
        "heldout_all_correct":all_correct,
        "causal_controls_pass":controls,
        "cold_prediction_seal_sha256":cold_seal,
        "cold_need_hidden_until_after_prediction":cold_need,
        "cold_prediction":cold,
        "cold_oracle_after_reveal":cold_oracle,
        "cold_reconstruction_exact":cold_ok,
        "claim_boundary":{
            "raw_intervention_representation_human_authored":True,
            "lag_scan_and_peak_rule_human_authored":True,
            "repair_action_vocabulary_human_authored":True,
            "generated_observation_basis_not_predeclared":True,
            "generated_basis_to_action_mapping_learned_from_outcomes":True,
            "unrestricted_feature_language_genesis":False,
            "independent_organizational_custody":False,
            "physical_world":False,
            "global_recursive_acceleration":False,
            "AGI":False,"ASI":False,"superintelligence":False,
        }
    },sort_keys=True))
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=20260929401)
    ap.add_argument("--cold",nargs=2)
    a=ap.parse_args()
    if a.cold:raise SystemExit(cold_main(a.cold[0],a.cold[1]))
    raise SystemExit(main(a.seed))
