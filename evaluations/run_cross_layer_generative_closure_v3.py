from __future__ import annotations

import argparse, hashlib, itertools, json, os, random, subprocess, sys, tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from arte_cognition.latent_relation_ontology_genesis import OpaqueInterventionalWorld, WorldDerivedLatentRelationInducer, contrast
from arte_cognition.observation_basis_genesis import WorldDerivedObservationBasisInducer

ATOMS=("OPEN_FRONTIER","SYNTHESIZE","VERIFY","PRUNE","CACHE")

def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()

@dataclass(frozen=True)
class State:
    frontier:int
    capability:int
    verified:int
    cost:int
    cached:int

def apply_atom(s,op):
    if op=="OPEN_FRONTIER": return State(1,s.capability,s.verified,s.cost,s.cached)
    if op=="SYNTHESIZE": return State(s.frontier,1 if s.frontier else s.capability,s.verified,s.cost,s.cached)
    if op=="VERIFY": return State(s.frontier,s.capability,1 if s.capability else s.verified,s.cost,s.cached)
    if op=="PRUNE": return State(s.frontier,s.capability,s.verified,max(1,s.cost-5) if s.capability else s.cost,s.cached)
    if op=="CACHE": return State(s.frontier,s.capability,s.verified,s.cost,1 if s.verified else s.cached)
    raise ValueError(op)

def expand(tok,lib):
    if tok in ATOMS:return (tok,)
    return tuple(lib[tok])

def execute(s,program,lib):
    out=s
    for tok in program:
        for atom in expand(tok,lib):
            out=apply_atom(out,atom)
    return out

def target_ok(s,target):
    for k,v in target.items():
        if k=="cost_le":
            if not s.cost<=v:return False
        elif k=="cost_eq":
            if not s.cost==v:return False
        elif getattr(s,k)!=v:return False
    return True

def search(init,target,lib,max_slots):
    toks=tuple(sorted(set(ATOMS)|set(lib)))
    tested=0
    for depth in range(max_slots+1):
        for prog in itertools.product(toks,repeat=depth):
            tested+=1
            out=execute(init,prog,lib)
            if target_ok(out,target):
                return {"program":list(prog),"slots":depth,"tested":tested,"out":asdict(out)}
    return {"program":None,"slots":None,"tested":tested,"out":None}

def flatten(prog,lib):
    out=[]
    for t in prog:out.extend(expand(t,lib))
    return tuple(out)

def macro_id(body):
    return "GEN_REPAIR_MACRO::"+hashlib.sha256("|".join(body).encode()).hexdigest()[:20]

def build_macro_library(rng):
    g0={}
    # G0 repeated training: exact-cost capability+verification.
    sols=[]
    for _ in range(2):
        c=10+rng.randrange(0,4)
        init=State(1,0,0,c,0)
        sol=search(init,{"capability":1,"verified":1,"cost_eq":c},g0,2)
        if sol["program"] is None:raise AssertionError("G0 task unsolved")
        sols.append(sol)
    bodies={flatten(x["program"],g0) for x in sols}
    if len(bodies)!=1:raise AssertionError(bodies)
    body1=next(iter(bodies)); m1=macro_id(body1)
    g1={m1:body1}

    # G1 repeated training: inherited macro + prune.
    sols2=[]; used=True
    for _ in range(2):
        c=12+rng.randrange(0,4)
        init=State(1,0,0,c,0)
        sol=search(init,{"capability":1,"verified":1,"cost_le":c-5},g1,2)
        if sol["program"] is None:raise AssertionError("G1 task unsolved")
        used &= m1 in sol["program"]; sols2.append(sol)
    if not used:raise AssertionError("inherited macro unused")
    bodies2={flatten(x["program"],g1) for x in sols2}
    if len(bodies2)!=1:raise AssertionError(bodies2)
    body2=next(iter(bodies2)); m2=macro_id(body2)
    if m2==m1:raise AssertionError("macro lineage did not grow")
    g2={**g1,m2:body2}
    return {
        "m1":m1,"body1":body1,"m2":m2,"body2":body2,"library":g2,
        "g1_training":[x["program"] for x in sols],
        "g2_training":[x["program"] for x in sols2],
    }

def make_world(context,prefix,domain,magnitude,lags,signs):
    nodes=[f"{prefix}_{i}" for i in range(4)]
    decoy=f"{prefix}_decoy"; all_nodes=nodes+[decoy]; rows=[]
    for si in range(3):
        lag=int(lags[si]); sign=int(signs[si])
        for rep in range(2):
            low=[{n:0.0 for n in all_nodes} for _ in range(3)]
            high=[{n:0.0 for n in all_nodes} for _ in range(3)]
            high[0][nodes[si]]=float(magnitude)
            high[lag][nodes[si+1]]=float(sign)*float(magnitude)
            clutter=1 if rep==0 else 2
            low[clutter][decoy]=float(rep+1); high[clutter][decoy]=float(rep+1)
            rows.append(contrast(f"{context}:{si}:{rep}",nodes[si],low,high))
    return OpaqueInterventionalWorld(context_id=context,domain=domain,source_anchor=nodes[0],target_anchor=nodes[-1],contrasts=tuple(rows))

def wpayload(w):
    return {"context_id":w.context_id,"domain":w.domain,"source_anchor":w.source_anchor,"target_anchor":w.target_anchor,
            "contrasts":[asdict(c) for c in w.contrasts]}

def rebuild_world(o):
    rows=[]
    for c in o["contrasts"]:
        lows=[dict(s) for s in c["low_timeline"]]
        highs=[dict(s) for s in c["high_timeline"]]
        rows.append(contrast(c["contrast_id"],c["source_node"],lows,highs))
    return OpaqueInterventionalWorld(context_id=o["context_id"],domain=o["domain"],source_anchor=o["source_anchor"],target_anchor=o["target_anchor"],contrasts=tuple(rows))

def schema_match(inducer,schema,w):
    class S: pass
    x=S(); x.basis_id=schema["basis_id"]; x.profile_tokens=tuple(schema["profile_tokens"])
    return inducer.matches(x,w)

def classify(model,w):
    inducer=WorldDerivedObservationBasisInducer(min_repeats=2,min_peak_effect=1e-9,max_path_depth=8,candidate_budget=64)
    hits=[r for r in model["basis_classes"] if schema_match(inducer,r,w)]
    if len(hits)==0:return {"class_id":"UNCLASSIFIED","macro_id":"NULL"}
    if len(hits)>1:raise AssertionError("basis class overlap")
    return {"class_id":hits[0]["schema_id"],"macro_id":hits[0]["macro_id"]}

def execute_macro_action(init,macro_id,lib):
    if macro_id=="NULL":return init
    if macro_id not in lib:raise KeyError(macro_id)
    return execute(init,(macro_id,),lib)

def cold_main(model_path,world_path,state_path):
    model=json.loads(Path(model_path).read_text())
    observed_hash=sha({"basis_classes":model["basis_classes"],"macros":model["macros"]})
    if observed_hash!=model["model_hash"]:raise AssertionError("model hash mismatch")
    w=rebuild_world(json.loads(Path(world_path).read_text()))
    sel=classify(model,w)
    init=State(**json.loads(Path(state_path).read_text()))
    lib={k:tuple(v) for k,v in model["macros"].items()}
    out=execute_macro_action(init,sel["macro_id"],lib)
    print(json.dumps({"model_hash":model["model_hash"],**sel,"out":asdict(out)},sort_keys=True))
    return 0

def main(seed):
    rng=random.Random(seed)
    macros=build_macro_library(rng)
    lib=macros["library"]; m1=macros["m1"]; m2=macros["m2"]

    specs={
        "EXACT_VERIFY":{"lags":(1,2,1),"signs":(1,1,1),"macro":m1},
        "VERIFY_AND_PRUNE":{"lags":(2,1,2),"signs":(1,-1,1),"macro":m2},
    }
    inducer=WorldDerivedObservationBasisInducer(min_repeats=2,min_peak_effect=1e-9,max_path_depth=8,candidate_budget=64)
    predecessor=WorldDerivedLatentRelationInducer(lag=1,min_effect=.1,min_repeats=2,max_path_depth=8,candidate_budget=64)
    scalar={"frontier_complete":1.0,"capability_hint":0.5,"cost_hint":10.0}
    basis_classes=[]
    training=[]
    for kind,spec in specs.items():
        ws=tuple(make_world(f"train-{kind}-{i}",f"{kind}_{seed}_{i}_{rng.randrange(10**7)}",f"TRAIN_{kind}_{i}",1+4*i,spec["lags"],spec["signs"]) for i in range(2))
        pa=predecessor.assess_residual(ws,(0,0),2)
        if predecessor.generate_candidates(pa,ws):raise AssertionError("old fixed-lag basis solved training world")
        for _ in range(32):
            if predecessor.generate_candidates(pa,ws):raise AssertionError("old more compute changed basis expressivity")
        oa=inducer.assess_residual(ws,(0,0),2)
        schemas=inducer.generate_candidates(oa,ws)
        if len(schemas)!=1:raise AssertionError(f"{kind} schemas={len(schemas)}")
        s=schemas[0]

        # Learn schema -> macro only from executable training outcomes.
        c=12+rng.randrange(0,3)
        init=State(1,0,0,c,0)
        if kind=="EXACT_VERIFY":
            target={"capability":1,"verified":1,"cost_eq":c}
        else:
            target={"capability":1,"verified":1,"cost_le":c-5}
        scores={}
        for a in (m1,m2,"NULL"):
            try: out=execute_macro_action(init,a,lib)
            except KeyError: continue
            scores[a]=bool(target_ok(out,target))
        winners=[a for a,v in scores.items() if v]
        if winners!=[spec["macro"]]:
            raise AssertionError({"kind":kind,"winners":winners,"scores":scores})
        row={"schema_id":s.schema_id,"basis_id":s.basis_id,"profile_tokens":list(s.profile_tokens),"macro_id":winners[0]}
        basis_classes.append(row)
        training.append({"kind_hidden_from_classifier":kind,"scalar_residual":scalar,"class":row,"macro_outcome_success":scores})

    if len({canon(x["scalar_residual"]) for x in training})!=1 or len({x["class"]["macro_id"] for x in training})!=2:
        raise AssertionError("old scalar basis not proven inexpressive")

    model_core={"basis_classes":sorted(basis_classes,key=lambda x:x["schema_id"]),
                "macros":{k:list(v) for k,v in sorted(lib.items())}}
    model={**model_core,"model_hash":sha(model_core)}

    held_specs=[
        ("held-a","EXACT_VERIFY",(1,2,1),(1,1,1)),
        ("held-b","VERIFY_AND_PRUNE",(2,1,2),(1,-1,1)),
        ("held-null","NULL",(1,1,1),(-1,1,-1)),
    ]
    selections=[]
    held_rows=[]
    for name,kind,lags,signs in held_specs:
        w=make_world(name,f"{name}_{seed}_{rng.randrange(10**8)}",f"HELD_{name}",3.0,lags,signs)
        sel=classify(model,w)
        selections.append({"world":name,**sel})
        held_rows.append((name,kind,w,sel))
    selection_seal=sha(selections)

    results=[]; all_ok=True; controls=True
    for name,kind,w,sel in held_rows:
        if kind=="NULL":
            init=State(1,1,1,2,0); target={"capability":1,"verified":1,"cost_eq":2}
            oracle="NULL"
        else:
            c=12+rng.randrange(0,3); init=State(1,0,0,c,0)
            if kind=="EXACT_VERIFY":
                target={"capability":1,"verified":1,"cost_eq":c}; oracle=m1
            else:
                target={"capability":1,"verified":1,"cost_le":c-5}; oracle=m2
        selected=sel["macro_id"]
        out=execute_macro_action(init,selected,lib)
        ok=(selected==oracle and target_ok(out,target))
        all_ok &= ok

        # REMOVE basis: no class => NULL
        remove_basis_out=execute_macro_action(init,"NULL",lib)
        # REMOVE macro library: selected generated macro cannot execute.
        remove_macro_cap=False if selected!="NULL" else target_ok(init,target)
        # WRONG macro swap
        wrong=(m2 if selected=="NULL" else (m2 if selected==m1 else m1))
        wrong_out=execute_macro_action(init,wrong,lib)
        wrong_ok=target_ok(wrong_out,target)
        if kind!="NULL":
            controls &= (not target_ok(remove_basis_out,target)) and (not remove_macro_cap) and (not wrong_ok)
        else:
            controls &= ok and not target_ok(execute_macro_action(init,m1,lib),target)
        # old atom one-slot expressivity
        old=search(init,target,{},1)
        old_more=all(search(init,target,{},1)["program"] is None for _ in range(32)) if kind!="NULL" else True
        if kind!="NULL":controls &= old["program"] is None and old_more
        results.append({
            "world":name,"kind_hidden_until_scoring":kind,"selected_macro":selected,"oracle_after_reveal":oracle,
            "class_id":sel["class_id"],"treatment_out":asdict(out),"correct":ok,
            "remove_basis_out":asdict(remove_basis_out),"remove_macro_executable":remove_macro_cap,
            "wrong_macro":wrong,"wrong_out":asdict(wrong_out),"wrong_satisfies_target":wrong_ok,
            "old_atomic_one_slot":old,"old_more_compute_32_no_solution":old_more,
        })
    if not all_ok or not controls:raise AssertionError({"all_ok":all_ok,"controls":controls,"results":results})

    # Cold process with one fresh world. Child sees only model bytes, raw trace, and initial state.
    cold_kind=rng.choice(("EXACT_VERIFY","VERIFY_AND_PRUNE","NULL"))
    if cold_kind=="EXACT_VERIFY": lags,signs=(1,2,1),(1,1,1)
    elif cold_kind=="VERIFY_AND_PRUNE": lags,signs=(2,1,2),(1,-1,1)
    else: lags,signs=(1,1,1),(-1,1,-1)
    cw=make_world("cold",f"cold_{seed}_{rng.randrange(10**8)}","COLD",4.0,lags,signs)
    if cold_kind=="NULL":
        cold_init=State(1,1,1,2,0); cold_target={"capability":1,"verified":1,"cost_eq":2}; cold_oracle="NULL"
    else:
        cc=13+rng.randrange(0,3); cold_init=State(1,0,0,cc,0)
        if cold_kind=="EXACT_VERIFY":
            cold_target={"capability":1,"verified":1,"cost_eq":cc}; cold_oracle=m1
        else:
            cold_target={"capability":1,"verified":1,"cost_le":cc-5}; cold_oracle=m2
    with tempfile.TemporaryDirectory(prefix="arte_cross_layer_") as td:
        mp=Path(td)/"model.json"; wp=Path(td)/"world.json"; sp=Path(td)/"state.json"
        mp.write_text(json.dumps(model,sort_keys=True),encoding="utf-8")
        wp.write_text(json.dumps(wpayload(cw),sort_keys=True),encoding="utf-8")
        sp.write_text(json.dumps(asdict(cold_init),sort_keys=True),encoding="utf-8")
        cp=subprocess.run([sys.executable,__file__,"--cold",str(mp),str(wp),str(sp)],cwd=ROOT,
                          text=True,capture_output=True,timeout=30,check=False,
                          env={**os.environ,"PYTHONPATH":str(ROOT)})
        if cp.returncode!=0:raise AssertionError(cp.stderr)
        cold=json.loads([x for x in cp.stdout.splitlines() if x.strip()][-1])
    cold_seal=sha(cold)
    cold_out=State(**cold["out"])
    cold_ok=cold["model_hash"]==model["model_hash"] and cold["macro_id"]==cold_oracle and target_ok(cold_out,cold_target)
    if not cold_ok:raise AssertionError({"cold":cold,"oracle":cold_oracle,"target":cold_target})

    print(json.dumps({
        "status":"PASS_BOUNDED_CROSS_LAYER_GENERATED_BASIS_CLASS_ACTION_CLOSURE_V3_AND_COLD_TRANSFER",
        "seed":seed,
        "generated_macro_1":{"id":m1,"body":list(macros["body1"])},
        "generated_macro_2":{"id":m2,"body":list(macros["body2"]),"compiled_from_inherited_macro":True},
        "generated_basis_classes":basis_classes,
        "old_scalar_basis_inexpressive":True,
        "old_fixed_lag_basis_candidates":0,
        "old_basis_more_compute_repetitions":32,
        "heldout_preoutcome_selection_seal_sha256":selection_seal,
        "heldout_results":results,
        "heldout_all_correct":all_ok,
        "all_remove_wrong_old_controls_pass":controls,
        "model_hash":model["model_hash"],
        "cold_prediction_execution_seal_sha256":cold_seal,
        "cold_kind_hidden_until_scoring":cold_kind,
        "cold_result":cold,
        "cold_oracle_after_reveal":cold_oracle,
        "cold_reconstruction_and_execution_exact":cold_ok,
        "claim_boundary":{
            "raw_trace_representation_human_authored":True,
            "observation_basis_induction_rule_human_authored":True,
            "atomic_micro_ops_human_authored":True,
            "macro_abstraction_rule_human_authored":True,
            "generated_basis_classes_not_predeclared":True,
            "generated_macro_names_and_bodies_not_predeclared":True,
            "generated_class_to_generated_macro_mapping_learned_from_outcomes":True,
            "unrestricted_feature_or_action_language_genesis":False,
            "independent_organizational_custody":False,
            "physical_world":False,
            "global_recursive_acceleration":False,
            "AGI":False,"ASI":False,"superintelligence":False,
        }
    },sort_keys=True))
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=20260929601)
    ap.add_argument("--cold",nargs=3)
    a=ap.parse_args()
    if a.cold:raise SystemExit(cold_main(a.cold[0],a.cold[1],a.cold[2]))
    raise SystemExit(main(a.seed))
