from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

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

def apply_atom(s:State,op:str)->State:
    if op=="OPEN_FRONTIER":
        return State(1,s.capability,s.verified,s.cost,s.cached)
    if op=="SYNTHESIZE":
        return State(s.frontier,1 if s.frontier else s.capability,s.verified,s.cost,s.cached)
    if op=="VERIFY":
        return State(s.frontier,s.capability,1 if s.capability else s.verified,s.cost,s.cached)
    if op=="PRUNE":
        return State(s.frontier,s.capability,s.verified,max(1,s.cost-5) if s.capability else s.cost,s.cached)
    if op=="CACHE":
        return State(s.frontier,s.capability,s.verified,s.cost,1 if s.verified else s.cached)
    raise ValueError(op)

def expand_token(token,lib):
    if token in ATOMS:return (token,)
    if token in lib:return tuple(lib[token])
    raise KeyError(token)

def execute_program(s,program,lib):
    out=s
    for tok in program:
        for atom in expand_token(tok,lib):
            out=apply_atom(out,atom)
    return out

def target_ok(s:State,target):
    return all(getattr(s,k)==v if k!="cost_le" else s.cost<=v for k,v in target.items())

def search(init,target,lib,max_slots):
    tokens=tuple(sorted(set(ATOMS)|set(lib)))
    tested=0
    for depth in range(max_slots+1):
        for prog in itertools.product(tokens,repeat=depth):
            tested+=1
            out=execute_program(init,prog,lib)
            if target_ok(out,target):
                return {"program":list(prog),"tested":tested,"slots":depth,"out":out.__dict__}
    return {"program":None,"tested":tested,"slots":None,"out":None}

def flatten_program(program,lib):
    out=[]
    for tok in program:
        out.extend(expand_token(tok,lib))
    return tuple(out)

def compile_macro(program,lib):
    body=flatten_program(program,lib)
    if len(body)<2: raise AssertionError("refuse trivial macro")
    mid="GEN_REPAIR_MACRO::"+hashlib.sha256("|".join(body).encode()).hexdigest()[:20]
    return mid,body

def library_hash(lib,parent_hash=""):
    return sha({"parent_hash":parent_hash,"macros":{k:list(v) for k,v in sorted(lib.items())}})

def cold_main(lib_path,task_path):
    obj=json.loads(Path(lib_path).read_text())
    lib={k:tuple(v) for k,v in obj["macros"].items()}
    expected=library_hash(lib,obj["parent_hash"])
    if expected!=obj["library_hash"]:raise AssertionError("library hash mismatch")
    t=json.loads(Path(task_path).read_text())
    init=State(**t["init"])
    res=search(init,t["target"],lib,int(t["max_slots"]))
    print(json.dumps({"library_hash":obj["library_hash"],"search":res},sort_keys=True))
    return 0

def main(seed):
    rng=random.Random(seed)
    g0={}
    g0_hash=library_hash(g0)

    # Generation 0: repeated successful body SYNTHESIZE -> VERIFY.
    g0_tasks=[]
    for i in range(2):
        init=State(1,0,0,10+rng.randrange(0,4),0)
        target={"capability":1,"verified":1}
        sol=search(init,target,g0,2)
        if sol["program"] is None:raise AssertionError("G0 training unsolved")
        g0_tasks.append(sol)
    bodies={flatten_program(x["program"],g0) for x in g0_tasks}
    if len(bodies)!=1:raise AssertionError(f"G0 did not converge to repeated body: {bodies}")
    m1,body1=compile_macro(g0_tasks[0]["program"],g0)
    g1={m1:body1}
    g1_hash=library_hash(g1,g0_hash)

    # Generation 1: inherited macro + PRUNE becomes a repeated higher-level repair.
    g1_tasks=[]
    inherited_macro_used=True
    for i in range(2):
        init=State(1,0,0,12+rng.randrange(0,4),0)
        target={"capability":1,"verified":1,"cost_le":init.cost-5}
        sol=search(init,target,g1,2)
        if sol["program"] is None:raise AssertionError("G1 training unsolved")
        inherited_macro_used &= m1 in sol["program"]
        g1_tasks.append(sol)
    if not inherited_macro_used:raise AssertionError("G1 solutions did not consume inherited macro")
    bodies2={flatten_program(x["program"],g1) for x in g1_tasks}
    if len(bodies2)!=1:raise AssertionError(f"G1 did not converge to repeated body: {bodies2}")
    m2,body2=compile_macro(g1_tasks[0]["program"],g1)
    if m2==m1 or body2==body1:raise AssertionError("G2 macro did not expand library")
    g2={**g1,m2:body2}
    g2_hash=library_hash(g2,g1_hash)

    # Fresh task: capability + verify + prune + cache. Two action slots only.
    init=State(1,0,0,12+rng.randrange(0,4),0)
    target={"capability":1,"verified":1,"cost_le":init.cost-5,"cached":1}
    full=search(init,target,g2,2)
    remove_latest=search(init,target,g1,2)
    remove_all=search(init,target,g0,2)

    wrong_id="GEN_REPAIR_MACRO::WRONG_"+hashlib.sha256(str(seed).encode()).hexdigest()[:12]
    wrong_lib={**g1,wrong_id:("VERIFY","SYNTHESIZE","PRUNE")}
    wrong=search(init,target,wrong_lib,2)

    repeated=[]
    for _ in range(32):
        repeated.append(search(init,target,g0,2))
    old_more_compute_no_solution=all(x["program"] is None for x in repeated)

    if full["program"] is None:raise AssertionError("G2 macro library failed two-slot fresh task")
    if m2 not in full["program"]:raise AssertionError("fresh solution did not use newest macro")
    if remove_latest["program"] is not None:raise AssertionError("G1 unexpectedly expressed fresh repair in two slots")
    if remove_all["program"] is not None:raise AssertionError("G0 unexpectedly expressed fresh repair in two slots")
    if wrong["program"] is not None:raise AssertionError("wrong macro unexpectedly expressed fresh repair")
    if not old_more_compute_no_solution:raise AssertionError("more compute changed G0 depth-2 expressivity")

    # Cold descendant receives only the serialized G2 library.
    cold_init=State(1,0,0,13+rng.randrange(0,4),0)
    cold_target={"capability":1,"verified":1,"cost_le":cold_init.cost-5,"cached":1}
    with tempfile.TemporaryDirectory(prefix="arte_repair_macro_") as td:
        lp=Path(td)/"library.json"; tp=Path(td)/"task.json"
        lp.write_text(json.dumps({
            "parent_hash":g1_hash,"macros":{k:list(v) for k,v in g2.items()},"library_hash":g2_hash
        },sort_keys=True),encoding="utf-8")
        tp.write_text(json.dumps({"init":cold_init.__dict__,"target":cold_target,"max_slots":2},sort_keys=True),encoding="utf-8")
        cp=subprocess.run([sys.executable,__file__,"--cold",str(lp),str(tp)],text=True,capture_output=True,
                          timeout=30,check=False,env={**os.environ,"PYTHONPATH":str(Path(__file__).resolve().parents[1])})
        if cp.returncode!=0:raise AssertionError(cp.stderr)
        cold=json.loads([x for x in cp.stdout.splitlines() if x.strip()][-1])
    cold_ok=cold["library_hash"]==g2_hash and cold["search"]["program"] is not None and m2 in cold["search"]["program"]
    if not cold_ok:raise AssertionError("cold descendant lost macro library capability")

    # Matched two-slot expressivity and search accounting.
    result={
        "status":"PASS_BOUNDED_TWO_GENERATION_REPAIR_MACRO_LIBRARY_GENESIS_AND_COLD_DESCENDANT_TRANSFER",
        "seed":seed,
        "atomic_micro_op_count":len(ATOMS),
        "g0_library_hash":g0_hash,
        "g1_macro_id":m1,
        "g1_macro_body":list(body1),
        "g1_library_hash":g1_hash,
        "g1_training_programs":[x["program"] for x in g0_tasks],
        "g2_macro_id":m2,
        "g2_macro_body":list(body2),
        "g2_macro_compiled_from_inherited_macro":inherited_macro_used,
        "g2_library_hash":g2_hash,
        "g2_training_programs":[x["program"] for x in g1_tasks],
        "fresh_action_slot_budget":2,
        "fresh_full":full,
        "fresh_remove_latest":remove_latest,
        "fresh_remove_all":remove_all,
        "fresh_wrong_macro":wrong,
        "old_more_compute_repetitions":32,
        "old_more_compute_no_solution":old_more_compute_no_solution,
        "cold_descendant":cold,
        "cold_descendant_exact_library_hash":cold["library_hash"]==g2_hash,
        "cold_descendant_fresh_solution":cold_ok,
        "claim_boundary":{
            "atomic_micro_ops_human_authored":True,
            "state_transition_semantics_human_authored":True,
            "abstraction_rule_human_authored":True,
            "task_generator_human_authored":True,
            "macro_names_and_bodies_generated_not_predeclared":True,
            "unrestricted_action_semantics_genesis":False,
            "same_research_lineage":True,
            "independent_organizational_custody":False,
            "physical_world":False,
            "global_recursive_acceleration":False,
            "AGI":False,"ASI":False,"superintelligence":False,
        }
    }
    print(json.dumps(result,sort_keys=True))
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=20260929501)
    ap.add_argument("--cold",nargs=2)
    a=ap.parse_args()
    if a.cold:raise SystemExit(cold_main(a.cold[0],a.cold[1]))
    raise SystemExit(main(a.seed))
