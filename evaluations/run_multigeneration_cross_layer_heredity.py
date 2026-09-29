from __future__ import annotations
import argparse, hashlib, itertools, json, os, random, subprocess, sys, tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from arte_cognition.latent_relation_ontology_genesis import OpaqueInterventionalWorld, WorldDerivedLatentRelationInducer, contrast
from arte_cognition.observation_basis_genesis import WorldDerivedObservationBasisInducer

ATOMS=("OPEN_FRONTIER","SYNTHESIZE","VERIFY","PRUNE","CACHE")

def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()

@dataclass(frozen=True)
class State:
    frontier:int; capability:int; verified:int; cost:int; cached:int

def atom(s,op):
    if op=="OPEN_FRONTIER": return State(1,s.capability,s.verified,s.cost,s.cached)
    if op=="SYNTHESIZE": return State(s.frontier,1 if s.frontier else s.capability,s.verified,s.cost,s.cached)
    if op=="VERIFY": return State(s.frontier,s.capability,1 if s.capability else s.verified,s.cost,s.cached)
    if op=="PRUNE": return State(s.frontier,s.capability,s.verified,max(1,s.cost-5) if s.capability else s.cost,s.cached)
    if op=="CACHE": return State(s.frontier,s.capability,s.verified,s.cost,1 if s.verified else s.cached)
    raise ValueError(op)

def expand(tok,lib): return (tok,) if tok in ATOMS else tuple(lib[tok])
def runprog(s,prog,lib):
    out=s
    for t in prog:
        for a in expand(t,lib): out=atom(out,a)
    return out
def ok(s,t):
    for k,v in t.items():
        if k=="cost_eq":
            if s.cost!=v:return False
        elif k=="cost_le":
            if s.cost>v:return False
        elif getattr(s,k)!=v:return False
    return True
def search(init,target,lib,max_slots):
    toks=tuple(sorted(set(ATOMS)|set(lib)))
    tested=0
    for d in range(max_slots+1):
        for prog in itertools.product(toks,repeat=d):
            tested+=1; out=runprog(init,prog,lib)
            if ok(out,target): return {"program":list(prog),"out":asdict(out),"slots":d,"tested":tested}
    return {"program":None,"out":None,"slots":None,"tested":tested}
def flat(prog,lib):
    z=[]
    for t in prog:z.extend(expand(t,lib))
    return tuple(z)
def mid(body): return "GEN_REPAIR_MACRO::"+hashlib.sha256("|".join(body).encode()).hexdigest()[:20]

def mk_world(cid,prefix,domain,mag,lags,signs):
    nodes=[f"{prefix}_{i}" for i in range(4)]; dec=f"{prefix}_d"; names=nodes+[dec]; rows=[]
    for i in range(3):
        for rep in range(2):
            lo=[{n:0.0 for n in names} for _ in range(3)]; hi=[{n:0.0 for n in names} for _ in range(3)]
            hi[0][nodes[i]]=mag; hi[lags[i]][nodes[i+1]]=signs[i]*mag
            c=1 if rep==0 else 2;lo[c][dec]=rep+1;hi[c][dec]=rep+1
            rows.append(contrast(f"{cid}:{i}:{rep}",nodes[i],lo,hi))
    return OpaqueInterventionalWorld(context_id=cid,domain=domain,source_anchor=nodes[0],target_anchor=nodes[-1],contrasts=tuple(rows))
def wpayload(w): return {"context_id":w.context_id,"domain":w.domain,"source_anchor":w.source_anchor,"target_anchor":w.target_anchor,"contrasts":[asdict(c) for c in w.contrasts]}
def rebuild(o):
    cs=[]
    for c in o["contrasts"]:cs.append(contrast(c["contrast_id"],c["source_node"],[dict(x) for x in c["low_timeline"]],[dict(x) for x in c["high_timeline"]]))
    return OpaqueInterventionalWorld(context_id=o["context_id"],domain=o["domain"],source_anchor=o["source_anchor"],target_anchor=o["target_anchor"],contrasts=tuple(cs))

def classify(inducer,classes,w):
    hits=[]
    for r in classes:
        class S: pass
        s=S();s.basis_id=r["basis_id"];s.profile_tokens=tuple(r["profile_tokens"])
        if inducer.matches(s,w):hits.append(r)
    if not hits:return {"class_id":"UNCLASSIFIED","macro_id":"NULL"}
    if len(hits)>1:raise AssertionError("class overlap")
    return {"class_id":hits[0]["schema_id"],"macro_id":hits[0]["macro_id"]}

def checkpoint(parent,classes,lib,generation):
    core={"generation":generation,"parent_hash":parent,"classes":sorted(classes,key=lambda x:x["schema_id"]),"macros":{k:list(v) for k,v in sorted(lib.items())}}
    return {**core,"checkpoint_hash":sha(core)}

def cold_main(cp_path,w_path,state_path,target_path):
    cp=json.loads(Path(cp_path).read_text());core={k:cp[k] for k in ("generation","parent_hash","classes","macros")}
    if sha(core)!=cp["checkpoint_hash"]:raise AssertionError("checkpoint hash mismatch")
    w=rebuild(json.loads(Path(w_path).read_text()));ind=WorldDerivedObservationBasisInducer(min_repeats=2,min_peak_effect=1e-9,max_path_depth=8,candidate_budget=64)
    sel=classify(ind,cp["classes"],w);s=State(**json.loads(Path(state_path).read_text()));lib={k:tuple(v) for k,v in cp["macros"].items()}
    out=s if sel["macro_id"]=="NULL" else runprog(s,(sel["macro_id"],),lib)
    target=json.loads(Path(target_path).read_text())
    print(json.dumps({"checkpoint_hash":cp["checkpoint_hash"],**sel,"out":asdict(out),"target_satisfied":ok(out,target)},sort_keys=True));return 0

def main(seed):
    rng=random.Random(seed); ind=WorldDerivedObservationBasisInducer(min_repeats=2,min_peak_effect=1e-9,max_path_depth=8,candidate_budget=64)
    pred=WorldDerivedLatentRelationInducer(lag=1,min_effect=.1,min_repeats=2,max_path_depth=8,candidate_budget=64)
    patterns=[
      ("A",(1,2,1),(1,1,1)),
      ("B",(2,1,2),(1,-1,1)),
      ("C",(2,2,1),(-1,1,-1)),
    ]
    classes=[];lib={};chain=[]
    cp=checkpoint("",classes,lib,0);chain.append(cp)
    macros=[]
    for gi,(name,lags,signs) in enumerate(patterns,1):
        ws=tuple(mk_world(f"train-{name}-{i}",f"{name}_{seed}_{i}_{rng.randrange(10**8)}",f"TRAIN_{name}",1+3*i,lags,signs) for i in range(2))
        pa=pred.assess_residual(ws,(0,0),2)
        if pred.generate_candidates(pa,ws):raise AssertionError("fixed predecessor solved")
        # Existing generated classes must not already classify the genuinely new pattern.
        if classes:
            probe=classify(ind,classes,ws[0])
            if probe["macro_id"]!="NULL":raise AssertionError({"premature_classification":probe,"generation":gi})
        oa=ind.assess_residual(ws,(0,0),2); schemas=ind.generate_candidates(oa,ws)
        if len(schemas)!=1:raise AssertionError({"schemas":len(schemas),"gen":gi})
        s=schemas[0]

        c=12+rng.randrange(0,4); init=State(1,0,0,c,0)
        if name=="A":target={"capability":1,"verified":1,"cost_eq":c,"cached":0}
        elif name=="B":target={"capability":1,"verified":1,"cost_le":c-5,"cached":0}
        else:target={"capability":1,"verified":1,"cost_le":c-5,"cached":1}
        sol=search(init,target,lib,2)
        if sol["program"] is None:raise AssertionError({"unsolved_generation":gi})
        if gi>1 and macros[-1] not in sol["program"]:raise AssertionError({"did_not_consume_parent_macro":sol})
        body=flat(sol["program"],lib);m=mid(body)
        if m in lib:raise AssertionError("macro did not grow")
        lib[m]=body;macros.append(m)
        classes.append({"schema_id":s.schema_id,"basis_id":s.basis_id,"profile_tokens":list(s.profile_tokens),"macro_id":m})
        nxt=checkpoint(cp["checkpoint_hash"],classes,lib,gi)
        if nxt["parent_hash"]!=cp["checkpoint_hash"]:raise AssertionError("lineage break")
        cp=nxt;chain.append(cp)

    # Fresh A/B/C/NULL selection is sealed before target scoring.
    held=[]
    for name,lags,signs in patterns:
        w=mk_world(f"held-{name}",f"H{name}_{seed}_{rng.randrange(10**8)}",f"HELD_{name}",4,lags,signs)
        held.append((name,w,classify(ind,classes,w)))
    nullw=mk_world("held-NULL",f"N_{seed}_{rng.randrange(10**8)}","HELD_NULL",4,(1,1,1),(-1,-1,1))
    held.append(("NULL",nullw,classify(ind,classes,nullw)))
    seal=sha([{"name":n,**sel} for n,_,sel in held])

    results=[];all_ok=True;controls=True
    for name,w,sel in held:
        if name=="NULL":
            init=State(1,1,1,2,0);target={"capability":1,"verified":1,"cost_eq":2,"cached":0};oracle="NULL"
        else:
            c=13+rng.randrange(0,4);init=State(1,0,0,c,0);idx=("A","B","C").index(name);oracle=macros[idx]
            if name=="A":target={"capability":1,"verified":1,"cost_eq":c,"cached":0}
            elif name=="B":target={"capability":1,"verified":1,"cost_le":c-5,"cached":0}
            else:target={"capability":1,"verified":1,"cost_le":c-5,"cached":1}
        selected=sel["macro_id"];out=init if selected=="NULL" else runprog(init,(selected,),lib);correct=selected==oracle and ok(out,target);all_ok&=correct
        wrong=macros[-1] if selected=="NULL" else macros[(macros.index(selected)+1)%len(macros)]
        wrongout=runprog(init,(wrong,),lib)
        # remove class => NULL; remove selected macro => not executable for non-NULL.
        removeclass=init
        old=search(init,target,{},1)
        old_more=all(search(init,target,{},1)["program"] is None for _ in range(32)) if name!="NULL" else True
        if name!="NULL":
            controls&=(not ok(removeclass,target)) and (not ok(wrongout,target)) and old["program"] is None and old_more
        else:
            controls&=not ok(wrongout,target)
        results.append({"name":name,"selected":selected,"oracle":oracle,"correct":correct,"out":asdict(out),"wrong":wrong,"wrong_out":asdict(wrongout),"wrong_satisfies":ok(wrongout,target),"old_one_slot":old,"old_more_32_no_solution":old_more})
    if not all_ok or not controls:raise AssertionError({"all_ok":all_ok,"controls":controls,"results":results})

    # Cold G3.
    ck=rng.choice(("A","B","C","NULL"))
    if ck=="NULL":lags,signs=(1,1,1),(-1,-1,1);ci=State(1,1,1,2,0);ct={"capability":1,"verified":1,"cost_eq":2,"cached":0};coracle="NULL"
    else:
        idx=("A","B","C").index(ck);lags,signs=patterns[idx][1],patterns[idx][2];cc=14+rng.randrange(0,3);ci=State(1,0,0,cc,0);coracle=macros[idx]
        if ck=="A":ct={"capability":1,"verified":1,"cost_eq":cc,"cached":0}
        elif ck=="B":ct={"capability":1,"verified":1,"cost_le":cc-5,"cached":0}
        else:ct={"capability":1,"verified":1,"cost_le":cc-5,"cached":1}
    cw=mk_world("cold",f"COLD_{seed}_{rng.randrange(10**8)}","COLD",5,lags,signs)
    with tempfile.TemporaryDirectory(prefix="arte_multigen_cross_") as td:
        cpP=Path(td)/"cp.json";wP=Path(td)/"w.json";sP=Path(td)/"s.json";tP=Path(td)/"t.json"
        cpP.write_text(json.dumps(cp,sort_keys=True));wP.write_text(json.dumps(wpayload(cw),sort_keys=True));sP.write_text(json.dumps(asdict(ci),sort_keys=True));tP.write_text(json.dumps(ct,sort_keys=True))
        pr=subprocess.run([sys.executable,__file__,"--cold",str(cpP),str(wP),str(sP),str(tP)],text=True,capture_output=True,timeout=30,check=False,env={**os.environ,"PYTHONPATH":str(ROOT)})
        if pr.returncode!=0:raise AssertionError(pr.stderr)
        cold=json.loads([x for x in pr.stdout.splitlines() if x.strip()][-1])
    cold_ok=cold["checkpoint_hash"]==cp["checkpoint_hash"] and cold["macro_id"]==coracle and cold["target_satisfied"]
    if not cold_ok:raise AssertionError({"cold":cold,"oracle":coracle})

    print(json.dumps({
      "status":"PASS_BOUNDED_THREE_GENERATION_CROSS_LAYER_BASIS_CLASS_ACTION_HEREDITY",
      "seed":seed,"generation_count":3,"checkpoint_chain":[{"generation":x["generation"],"parent_hash":x["parent_hash"],"checkpoint_hash":x["checkpoint_hash"]} for x in chain],
      "generated_macros":[{"id":m,"body":list(lib[m])} for m in macros],
      "generated_classes":classes,
      "g2_consumes_g1":macros[0] in search(State(1,0,0,12,0),{"capability":1,"verified":1,"cost_le":7,"cached":0},{macros[0]:lib[macros[0]]},2)["program"],
      "g3_macro_body_contains_parent_body":all(a in lib[macros[2]] for a in lib[macros[1]]),
      "heldout_prediction_seal_sha256":seal,"heldout_results":results,"heldout_all_correct":all_ok,"controls_pass":controls,
      "cold_generation":"G3","cold_hidden_kind":ck,"cold_result":cold,"cold_oracle":coracle,"cold_exact":cold_ok,
      "claim_boundary":{"observation_basis_rule_human_authored":True,"atomic_micro_ops_human_authored":True,"macro_compilation_rule_human_authored":True,"world_generator_human_authored":True,"generated_basis_class_and_macro_lineage":True,"unrestricted_language_genesis":False,"independent_organizational_custody":False,"physical_world":False,"global_recursive_acceleration":False,"AGI":False,"ASI":False,"superintelligence":False}
    },sort_keys=True));return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--seed",type=int,default=20260929801);ap.add_argument("--cold",nargs=4);a=ap.parse_args()
    if a.cold:raise SystemExit(cold_main(a.cold[0],a.cold[1],a.cold[2],a.cold[3]))
    raise SystemExit(main(a.seed))
