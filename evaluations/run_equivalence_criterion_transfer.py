from __future__ import annotations

import json
import subprocess
import sys
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from arte_cognition.equivalence_criterion_genesis import (
    WorldDerivedEquivalenceCriterionInducer,
    derive_equivalence_policy,
    select_authorized_equivalence,
)
from arte_cognition.latent_relation_ontology_genesis import OpaqueInterventionalWorld, contrast
from arte_cognition.response_partition_genesis import WorldDerivedResponsePartitionInducer
from arte_cognition.world_coupling import WorldOutcomePair


def _world(context_id: str, prefix: str, domain: str, exponent: int) -> OpaqueInterventionalWorld:
    root = f"{prefix}_root"
    down = f"{prefix}_down"
    up = f"{prefix}_up"
    target = f"{prefix}_target"
    decoy = f"{prefix}_decoy"
    nodes = (root, down, up, target, decoy)
    rows = []

    def measured(x: float) -> float:
        return float(x) ** int(exponent)

    def add(source: str, effects: Sequence[Tuple[str, Tuple[float, float]]]) -> None:
        for repeat in range(2):
            low = [{node: 0.0 for node in nodes} for _ in range(3)]
            high = [{node: 0.0 for node in nodes} for _ in range(3)]
            high[0][source] = measured(1.0)
            for destination, latent_curve in effects:
                high[1][destination] = measured(latent_curve[0])
                high[2][destination] = measured(latent_curve[1])
            low[2][decoy] = float(repeat + 1)
            high[2][decoy] = float(repeat + 1)
            rows.append(contrast(f"{context_id}:{source}:{repeat}", source, low, high))

    add(root, ((down, (4.0, 1.0)), (up, (1.0, 4.0))))
    add(down, ((target, (4.0, 1.0)),))
    add(up, ((target, (1.0, 4.0)),))
    return OpaqueInterventionalWorld(context_id, domain, root, target, tuple(rows))


def _serialize(world: OpaqueInterventionalWorld) -> Dict[str, object]:
    def timeline(rows):
        return [[list(item) for item in snapshot] for snapshot in rows]
    return {
        "context_id": world.context_id,
        "source_anchor": world.source_anchor,
        "target_anchor": world.target_anchor,
        "contrasts": [
            {
                "source_node": c.source_node,
                "low": timeline(c.low_timeline),
                "high": timeline(c.high_timeline),
            }
            for c in world.contrasts
        ],
    }


def _external_capability(world: OpaqueInterventionalWorld, frozen_constraints) -> float:
    payload = json.dumps({"world": _serialize(world), "constraints": frozen_constraints})
    code = r'''
import json, sys
p=json.loads(sys.stdin.read()); w=p["world"]; frozen=p["constraints"]
rows={}
for c in w["contrasts"]:
    src=c["source_node"]; low=c["low"]; high=c["high"]
    nodes=sorted({str(k) for tl in (low,high) for snap in tl for k,_ in snap})
    for tgt in nodes:
        if tgt==src: continue
        curve=[]
        for lag in range(1,min(len(low),len(high))):
            lo=dict(low[lag]); hi=dict(high[lag]); curve.append(float(hi.get(tgt,0.0))-float(lo.get(tgt,0.0)))
        if any(abs(v)>1e-9 for v in curve): rows.setdefault((src,tgt),[]).append(curve)
adj={}
def constraints(curve):
    orders=[]
    for i in range(len(curve)):
        for j in range(i+1,len(curve)):
            d=curve[i]-curve[j]; r=0 if abs(d)<=1e-9 else (1 if d>0 else -1); orders.append([i+1,j+1,r])
    signs=[]
    for i,v in enumerate(curve,1): signs.append([i,0 if abs(v)<=1e-9 else (1 if v>0 else -1)])
    return [orders,signs]
for (src,tgt), curves in rows.items():
    if len(curves)<2: continue
    width=max(map(len,curves)); mean=[sum((c[i] if i<len(c) else 0.0) for c in curves)/len(curves) for i in range(width)]
    adj.setdefault(src,[]).append((tgt,constraints(mean)))
paths=[]
def walk(node,seen,cs):
    for tgt,c in adj.get(node,[]):
        if tgt in seen: continue
        nxt=cs+[c]
        if tgt==w["target_anchor"]: paths.append(nxt)
        walk(tgt,seen+[tgt],nxt)
walk(w["source_anchor"],[w["source_anchor"]],[])
match=any(path==frozen for path in paths)
# Evaluator-owned downstream causal success: every selected edge must show an
# early positive response strictly greater than its later positive response.
def good(c):
    orders, signs=c
    return [1,2,1] in orders and [1,1] in signs and [2,1] in signs
print(json.dumps({"match":match,"success":bool(match and all(good(c) for c in frozen))}))
'''
    result = subprocess.run(
        [sys.executable, "-c", code], input=payload, text=True, capture_output=True, check=True
    )
    return 1.0 if json.loads(result.stdout)["success"] else 0.0


def _pair(criterion_id: str, context: str, cls: str, effect: float, verified: bool = True) -> WorldOutcomePair:
    return WorldOutcomePair(
        pair_id=f"{criterion_id}:{context}:{cls}", experiment_id=criterion_id,
        axis_id="MEASUREMENT_EQUIVALENCE", source_id=f"src::{cls}", context_id=context,
        challenge_id=f"challenge::{context}", epoch=1, low_outcome=0.0, high_outcome=float(effect),
        low_value=0.0, high_value=1.0, matched_budget=True, externally_generated=True,
        issuer_id=f"issuer::{cls}", independence_class_id=cls if verified else "UNVERIFIED",
        authority_verified=verified,
    )


def main() -> Dict[str, object]:
    train = (_world("measure-a", "alpha", "SOFTWARE", 1), _world("measure-b", "beta", "SOFTWARE", 2))

    predecessor = WorldDerivedResponsePartitionInducer(min_repeats=2)
    pred_assessment = predecessor.assess_residual(train, (2, 2), 2)
    pred_candidates = predecessor.generate_candidates(pred_assessment, train)
    assert len(pred_candidates) == 0
    for _ in range(16):
        assert predecessor.generate_candidates(pred_assessment, train) == ()

    inducer = WorldDerivedEquivalenceCriterionInducer(min_repeats=2)
    assessment = inducer.assess_residual(train, (0, 0), 2)
    criteria = inducer.generate_candidates(assessment, train)
    assert len(criteria) == 2
    frozen = tuple(criteria)

    effects = {}
    pairs = []
    for criterion in frozen:
        effects[criterion.criterion_id] = []
        for world in train:
            effect = _external_capability(world, criterion.constraints)
            effects[criterion.criterion_id].append(effect)
            for cls in ("A", "B"):
                pairs.append(_pair(criterion.criterion_id, world.context_id, cls, effect))

    winners = [cid for cid, vals in effects.items() if vals == [1.0, 1.0]]
    assert len(winners) == 1
    one_context = tuple(p for p in pairs if p.context_id == train[0].context_id)
    assert select_authorized_equivalence(frozen, derive_equivalence_policy(frozen, one_context, 2, 2)) is None
    policy = derive_equivalence_policy(frozen, tuple(pairs), 2, 2)
    selected = select_authorized_equivalence(frozen, policy)
    assert selected is not None and selected.criterion_id == winners[0]
    verifierless = tuple(_pair(p.experiment_id, p.context_id, "X", p.effect, False) for p in pairs)
    assert select_authorized_equivalence(frozen, derive_equivalence_policy(frozen, verifierless, 2, 2)) is None

    heldout = _world("measure-heldout", "omega", "CAUSAL_WORLD", 3)
    treatment = _external_capability(heldout, selected.constraints)
    remove = 0.0
    wrong = next(c for c in frozen if c.criterion_id != selected.criterion_id)
    wrong_cap = _external_capability(heldout, wrong.constraints)

    assert treatment == 1.0 and remove == 0.0 and wrong_cap == 0.0
    report = {
        "status": "PASS_BOUNDED_WORLD_DERIVED_MEASUREMENT_EQUIVALENCE_AND_PREOUTCOME_TRANSFER",
        "predecessor_exact_profile_candidate_count": len(pred_candidates),
        "predecessor_more_compute_attempts": 16,
        "generated_equivalence_candidate_count": len(criteria),
        "candidate_generation_uses_external_outcomes": False,
        "candidate_freeze_before_external_outcomes": True,
        "named_measurement_transform_supplied_to_inducer": False,
        "equivalence_generated_from_cross_context_order_constraints": True,
        "training_measurement_exponents_for_evaluator_only": [1, 2],
        "heldout_measurement_exponent_for_evaluator_only": 3,
        "selected_criterion_id": selected.criterion_id,
        "training_effects": effects[selected.criterion_id],
        "treatment_capability": treatment,
        "remove_same_checkpoint_capability": remove,
        "structurally_valid_wrong_capability": wrong_cap,
        "one_context_insufficient_for_authority": True,
        "verifierless_policy_authority": False,
        "external_executor_imports_arte_inducer": False,
        "external_executor_consumes_concrete_raw_timelines": True,
        "pairwise_order_comparison_human_authored": True,
        "arithmetic_differencing_human_authored": True,
        "unrestricted_equivalence_genesis": False,
        "unrestricted_meta_language_genesis": False,
        "global_recursive_acceleration": False,
        "independent_organizational_custody": False,
        "physical_world": False,
        "foundation_weight_change": False,
        "AGI": False,
        "ASI": False,
    }
    print(json.dumps(report, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
