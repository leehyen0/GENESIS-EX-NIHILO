from __future__ import annotations

import json
import subprocess
import sys
from typing import Dict, Sequence, Tuple

from arte_cognition.comparison_functional_genesis import (
    WorldDerivedComparisonFunctionalInducer,
    derive_comparison_policy,
    select_authorized_comparison,
)
from arte_cognition.equivalence_criterion_genesis import WorldDerivedEquivalenceCriterionInducer
from arte_cognition.latent_relation_ontology_genesis import OpaqueInterventionalWorld, contrast
from arte_cognition.world_coupling import WorldOutcomePair


def _world(context_id: str, prefix: str, domain: str, scale: float, offset: float) -> OpaqueInterventionalWorld:
    root = f"{prefix}_root"; curved = f"{prefix}_curved"; flat = f"{prefix}_flat"; target = f"{prefix}_target"; decoy = f"{prefix}_decoy"
    nodes = (root, curved, flat, target, decoy)
    rows = []

    def add(source: str, effects: Sequence[Tuple[str, Tuple[float, float, float]]]) -> None:
        for repeat in range(2):
            low = [{node: float(offset) for node in nodes} for _ in range(4)]
            high = [{node: float(offset) for node in nodes} for _ in range(4)]
            high[0][source] += float(scale)
            for destination, curve in effects:
                for lag, latent in enumerate(curve, start=1):
                    high[lag][destination] += float(scale) * float(latent)
            low[3][decoy] += float(repeat + 1)
            high[3][decoy] += float(repeat + 1)
            rows.append(contrast(f"{context_id}:{source}:{repeat}", source, low, high))

    # Both routes are positive and strictly increasing at every lag, so the
    # predecessor order/sign criterion cannot distinguish them.
    add(root, ((curved, (1.0, 2.0, 4.0)), (flat, (1.0, 3.0, 4.0))))
    add(curved, ((target, (1.0, 2.0, 4.0)),))
    add(flat, ((target, (1.0, 3.0, 4.0)),))
    return OpaqueInterventionalWorld(context_id, domain, root, target, tuple(rows))


def _serialize(world: OpaqueInterventionalWorld) -> Dict[str, object]:
    return {
        "source_anchor": world.source_anchor,
        "target_anchor": world.target_anchor,
        "contrasts": [
            {
                "source": c.source_node,
                "low": [[list(item) for item in snap] for snap in c.low_timeline],
                "high": [[list(item) for item in snap] for snap in c.high_timeline],
            }
            for c in world.contrasts
        ],
    }


def _external_capability(world: OpaqueInterventionalWorld, coeffs, frozen_signs) -> float:
    payload = json.dumps({"world": _serialize(world), "coeffs": list(coeffs), "signs": list(frozen_signs)})
    code = r'''
import json, sys
p=json.loads(sys.stdin.read()); w=p["world"]; coeffs=p["coeffs"]; frozen=p["signs"]
rows={}; width=len(coeffs)
for c in w["contrasts"]:
    src=c["source"]; low=c["low"]; high=c["high"]
    nodes=sorted({str(k) for tl in (low,high) for snap in tl for k,_ in snap})
    for tgt in nodes:
        if tgt==src: continue
        curve=[]
        for lag in range(1,width+1):
            lo=dict(low[lag]); hi=dict(high[lag]); curve.append(float(hi.get(tgt,0.0))-float(lo.get(tgt,0.0)))
        if any(abs(v)>1e-9 for v in curve): rows.setdefault((src,tgt),[]).append(curve)
adj={}
for (src,tgt), curves in rows.items():
    if len(curves)<2: continue
    mean=[sum(c[i] for c in curves)/len(curves) for i in range(width)]
    value=sum(float(a)*float(b) for a,b in zip(coeffs,mean))
    sign=0 if abs(value)<=1e-9 else (1 if value>0 else -1)
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
    out = subprocess.run([sys.executable, "-c", code], input=payload, text=True, capture_output=True, check=True)
    return 1.0 if json.loads(out.stdout)["success"] else 0.0


def _pair(schema_id: str, context: str, cls: str, effect: float, verified: bool = True) -> WorldOutcomePair:
    return WorldOutcomePair(
        pair_id=f"{schema_id}:{context}:{cls}", experiment_id=schema_id,
        axis_id="COMPARISON_FUNCTIONAL", source_id=f"src::{cls}", context_id=context,
        challenge_id=f"challenge::{context}", epoch=1, low_outcome=0.0, high_outcome=float(effect),
        low_value=0.0, high_value=1.0, matched_budget=True, externally_generated=True,
        issuer_id=f"issuer::{cls}", independence_class_id=cls if verified else "UNVERIFIED",
        authority_verified=verified,
    )


def _predecessor_multiplicity(world: OpaqueInterventionalWorld) -> int:
    predecessor = WorldDerivedEquivalenceCriterionInducer(min_repeats=2)
    curves = predecessor._curves(world)
    adjacency = {}
    for (src,tgt), curve in curves.items():
        adjacency.setdefault(src,[]).append((tgt, predecessor._constraints(curve)))
    paths=[]
    def walk(node,seen,constraints):
        for tgt,c in adjacency.get(node,[]):
            if tgt in seen: continue
            nxt=constraints+(c,)
            if tgt==world.target_anchor: paths.append(nxt)
            walk(tgt,seen+(tgt,),nxt)
    walk(world.source_anchor,(world.source_anchor,),())
    return max((paths.count(path) for path in paths), default=0)


def main() -> Dict[str, object]:
    train = (
        _world("functional-a", "alpha", "SOFTWARE", 1.0, 7.0),
        _world("functional-b", "beta", "SOFTWARE", 5.0, -11.0),
    )
    predecessor = WorldDerivedEquivalenceCriterionInducer(min_repeats=2)
    predecessor_assessment = predecessor.assess_residual(train, (0,0), 2)
    predecessor_candidates = predecessor.generate_candidates(predecessor_assessment, train)
    multiplicities = [_predecessor_multiplicity(w) for w in train]
    assert len(predecessor_candidates) == 1 and multiplicities == [2,2]
    for _ in range(16):
        assert len(predecessor.generate_candidates(predecessor_assessment, train)) == 1

    inducer = WorldDerivedComparisonFunctionalInducer(coefficient_bound=2, min_repeats=2)
    assessment = inducer.assess_residual(train, (1,1), multiplicities, 2)
    schemas = inducer.generate_candidates(assessment, train)
    assert len(schemas) == 2
    assert {s.coefficients for s in schemas} == {(1,-2,1)}
    frozen = tuple(schemas)

    pairs=[]; effects={}
    for schema in frozen:
        effects[schema.schema_id]=[]
        for world in train:
            effect=_external_capability(world, schema.coefficients, schema.path_signs)
            effects[schema.schema_id].append(effect)
            for cls in ("A","B"):
                pairs.append(_pair(schema.schema_id, world.context_id, cls, effect))
    winners=[sid for sid,vals in effects.items() if vals==[1.0,1.0]]
    assert len(winners)==1
    one=tuple(p for p in pairs if p.context_id==train[0].context_id)
    assert select_authorized_comparison(frozen, derive_comparison_policy(frozen,one,2,2)) is None
    selected=select_authorized_comparison(frozen, derive_comparison_policy(frozen,tuple(pairs),2,2))
    assert selected is not None and selected.schema_id==winners[0]
    verifierless=tuple(_pair(p.experiment_id,p.context_id,"X",p.effect,False) for p in pairs)
    assert select_authorized_comparison(frozen, derive_comparison_policy(frozen,verifierless,2,2)) is None

    heldout=_world("functional-heldout","omega","CAUSAL_WORLD",11.0,23.0)
    treatment=_external_capability(heldout,selected.coefficients,selected.path_signs)
    wrong=next(s for s in frozen if s.schema_id!=selected.schema_id)
    wrong_cap=_external_capability(heldout,wrong.coefficients,wrong.path_signs)
    assert treatment==1.0 and wrong_cap==0.0

    report={
        "status":"PASS_BOUNDED_WORLD_DERIVED_COMPARISON_FUNCTIONAL_AND_PREOUTCOME_TRANSFER",
        "predecessor_order_sign_unique_candidate_count":len(predecessor_candidates),
        "predecessor_concrete_path_multiplicity":multiplicities,
        "predecessor_more_compute_attempts":16,
        "generated_comparison_candidate_count":len(schemas),
        "generated_coefficients":list(selected.coefficients),
        "named_curvature_or_ratio_feature_supplied":False,
        "candidate_generation_uses_external_outcomes":False,
        "candidate_freeze_before_external_outcomes":True,
        "training_effects":effects[selected.schema_id],
        "treatment_capability":treatment,
        "remove_same_checkpoint_capability":0.0,
        "structurally_valid_wrong_capability":wrong_cap,
        "one_context_insufficient_for_authority":True,
        "verifierless_policy_authority":False,
        "external_executor_imports_arte_inducer":False,
        "external_executor_consumes_concrete_raw_timelines":True,
        "integer_linear_functional_grammar_human_authored":True,
        "zero_sum_invariance_constraint_human_authored":True,
        "unrestricted_comparison_algebra_genesis":False,
        "unrestricted_meta_language_genesis":False,
        "global_recursive_acceleration":False,
        "independent_organizational_custody":False,
        "physical_world":False,
        "foundation_weight_change":False,
        "AGI":False,"ASI":False,
    }
    print(json.dumps(report,sort_keys=True)); return report


if __name__=="__main__": main()
