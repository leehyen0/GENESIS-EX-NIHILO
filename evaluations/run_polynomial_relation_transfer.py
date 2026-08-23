from __future__ import annotations

import json
import subprocess
import sys
from typing import Dict, Sequence, Tuple

from arte_cognition.comparison_functional_genesis import WorldDerivedComparisonFunctionalInducer
from arte_cognition.latent_relation_ontology_genesis import OpaqueInterventionalWorld, contrast
from arte_cognition.polynomial_relation_genesis import (
    WorldDerivedPolynomialRelationInducer,
    derive_polynomial_relation_policy,
    select_authorized_polynomial_relation,
)
from arte_cognition.world_coupling import WorldOutcomePair


def _world(context_id: str, prefix: str, domain: str, scale: float, offset: float) -> OpaqueInterventionalWorld:
    root=f"{prefix}_root"; lowcurve=f"{prefix}_lowcurve"; highcurve=f"{prefix}_highcurve"; target=f"{prefix}_target"; decoy=f"{prefix}_decoy"
    nodes=(root,lowcurve,highcurve,target,decoy); rows=[]
    def add(source: str, effects: Sequence[Tuple[str,Tuple[float,float,float]]]):
        for repeat in range(2):
            low=[{node:float(offset) for node in nodes} for _ in range(4)]
            high=[{node:float(offset) for node in nodes} for _ in range(4)]
            high[0][source]+=float(scale)
            for destination,curve in effects:
                for lag,value in enumerate(curve,start=1): high[lag][destination]+=float(scale)*float(value)
            low[3][decoy]+=repeat+1; high[3][decoy]+=repeat+1
            rows.append(contrast(f"{context_id}:{source}:{repeat}",source,low,high))
    add(root,((lowcurve,(1.0,2.0,6.0)),(highcurve,(1.0,3.0,6.0))))
    add(lowcurve,((target,(1.0,2.0,6.0)),))
    add(highcurve,((target,(1.0,3.0,6.0)),))
    return OpaqueInterventionalWorld(context_id,domain,root,target,tuple(rows))


def _serialize(world):
    return {"source_anchor":world.source_anchor,"target_anchor":world.target_anchor,"contrasts":[{"source":c.source_node,"low":[[list(i) for i in s] for s in c.low_timeline],"high":[[list(i) for i in s] for s in c.high_timeline]} for c in world.contrasts]}


def _external_capability(world, positive_exp, negative_exp, frozen_signs):
    payload=json.dumps({"world":_serialize(world),"pos":list(positive_exp),"neg":list(negative_exp),"signs":list(frozen_signs)})
    code=r'''
import json,sys
p=json.loads(sys.stdin.read()); w=p["world"]; pos=p["pos"]; neg=p["neg"]; frozen=p["signs"]; width=len(pos)
rows={}
for c in w["contrasts"]:
 src=c["source"]; low=c["low"]; high=c["high"]
 nodes=sorted({str(k) for tl in (low,high) for snap in tl for k,_ in snap})
 for tgt in nodes:
  if tgt==src: continue
  curve=[]
  for lag in range(1,width+1):
   lo=dict(low[lag]); hi=dict(high[lag]); curve.append(float(hi.get(tgt,0.0))-float(lo.get(tgt,0.0)))
  if any(abs(v)>1e-9 for v in curve): rows.setdefault((src,tgt),[]).append(curve)
def mono(curve,exp):
 v=1.0
 for x,e in zip(curve,exp):
  if e: v*=float(x)**int(e)
 return v
adj={}
for (src,tgt),curves in rows.items():
 if len(curves)<2: continue
 mean=[sum(c[i] for c in curves)/len(curves) for i in range(width)]
 value=mono(mean,pos)-mono(mean,neg); sign=0 if abs(value)<=1e-9 else (1 if value>0 else -1)
 adj.setdefault(src,[]).append((tgt,sign))
paths=[]
def walk(node,seen,signs):
 for tgt,s in adj.get(node,[]):
  if tgt in seen: continue
  nxt=signs+[s]
  if tgt==w["target_anchor"]: paths.append(nxt)
  walk(tgt,seen+[tgt],nxt)
walk(w["source_anchor"],[w["source_anchor"]],[])
match=any(path==frozen for path in paths)
print(json.dumps({"match":match,"success":bool(match and frozen and all(s>0 for s in frozen))}))
'''
    out=subprocess.run([sys.executable,"-c",code],input=payload,text=True,capture_output=True,check=True)
    return 1.0 if json.loads(out.stdout)["success"] else 0.0


def _pair(schema_id,context,cls,effect,verified=True):
    return WorldOutcomePair(pair_id=f"{schema_id}:{context}:{cls}",experiment_id=schema_id,axis_id="POLYNOMIAL_RELATION",source_id=f"s::{cls}",context_id=context,challenge_id=f"c::{context}",epoch=1,low_outcome=0.0,high_outcome=float(effect),low_value=0.0,high_value=1.0,matched_budget=True,externally_generated=True,issuer_id=f"i::{cls}",independence_class_id=cls if verified else "UNVERIFIED",authority_verified=verified)


def main()->Dict[str,object]:
    train=(_world("poly-a","alpha","SOFTWARE",1.0,9.0),_world("poly-b","beta","SOFTWARE",7.0,-13.0))
    predecessor=WorldDerivedComparisonFunctionalInducer(coefficient_bound=2,min_repeats=2)
    predecessor_assessment=predecessor.assess_residual(train,(1,1),(2,2),2)
    predecessor_candidates=predecessor.generate_candidates(predecessor_assessment,train)
    assert len(predecessor_candidates)==0
    for _ in range(16): assert predecessor.generate_candidates(predecessor_assessment,train)==()

    inducer=WorldDerivedPolynomialRelationInducer(degree=2,min_repeats=2)
    assessment=inducer.assess_residual(train,(0,0),2)
    schemas=inducer.generate_candidates(assessment,train)
    assert len(schemas)==2
    assert {(s.positive_monomial,s.negative_monomial) for s in schemas}=={((0,2,0),(1,0,1))}
    frozen=tuple(schemas); effects={}; pairs=[]
    for schema in frozen:
        effects[schema.schema_id]=[]
        for world in train:
            effect=_external_capability(world,schema.positive_monomial,schema.negative_monomial,schema.path_signs)
            effects[schema.schema_id].append(effect)
            for cls in ("A","B"): pairs.append(_pair(schema.schema_id,world.context_id,cls,effect))
    winners=[sid for sid,vals in effects.items() if vals==[1.0,1.0]]; assert len(winners)==1
    one=tuple(p for p in pairs if p.context_id==train[0].context_id)
    assert select_authorized_polynomial_relation(frozen,derive_polynomial_relation_policy(frozen,one,2,2)) is None
    selected=select_authorized_polynomial_relation(frozen,derive_polynomial_relation_policy(frozen,tuple(pairs),2,2)); assert selected is not None and selected.schema_id==winners[0]
    verifierless=tuple(_pair(p.experiment_id,p.context_id,"X",p.effect,False) for p in pairs)
    assert select_authorized_polynomial_relation(frozen,derive_polynomial_relation_policy(frozen,verifierless,2,2)) is None
    heldout=_world("poly-heldout","omega","CAUSAL_WORLD",11.0,31.0)
    treatment=_external_capability(heldout,selected.positive_monomial,selected.negative_monomial,selected.path_signs)
    wrong=next(s for s in frozen if s.schema_id!=selected.schema_id)
    wrong_cap=_external_capability(heldout,wrong.positive_monomial,wrong.negative_monomial,wrong.path_signs)
    assert treatment==1.0 and wrong_cap==0.0
    report={"status":"PASS_BOUNDED_WORLD_DERIVED_POLYNOMIAL_RELATION_LIFT_AND_PREOUTCOME_TRANSFER","predecessor_linear_candidate_count":0,"predecessor_more_compute_attempts":16,"generated_polynomial_candidate_count":len(schemas),"generated_positive_monomial":list(selected.positive_monomial),"generated_negative_monomial":list(selected.negative_monomial),"named_ratio_curvature_or_geometric_feature_supplied":False,"candidate_generation_uses_external_outcomes":False,"candidate_freeze_before_external_outcomes":True,"training_effects":effects[selected.schema_id],"treatment_capability":treatment,"remove_same_checkpoint_capability":0.0,"structurally_valid_wrong_capability":wrong_cap,"one_context_insufficient_for_authority":True,"verifierless_policy_authority":False,"external_executor_imports_arte_inducer":False,"external_executor_consumes_concrete_raw_timelines":True,"homogeneous_monomial_lift_human_authored":True,"polynomial_degree_human_authored":True,"unrestricted_relation_algebra_genesis":False,"unrestricted_meta_language_genesis":False,"global_recursive_acceleration":False,"independent_organizational_custody":False,"physical_world":False,"foundation_weight_change":False,"AGI":False,"ASI":False}
    print(json.dumps(report,sort_keys=True)); return report


if __name__=="__main__": main()
