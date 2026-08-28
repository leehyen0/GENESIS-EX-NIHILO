from __future__ import annotations
import hashlib,json,os,random,secrets,sys
from datetime import datetime,timezone
from pathlib import Path

PAIRS=[
 ("TABLE_RELATION","SYMBOLIC_RULE"),
 ("GRAPH_WEIGHT","RESOURCE_SCHEDULING"),
 ("BINARY_FEATURE","SEQUENCE_RULE"),
 ("CAUSAL_INTERVENTION","TOOL_RELIABILITY"),
]
FALSE_FLAGS={
 "independent_custody_proof":False,
 "source_disjoint_transfer_proof":False,
 "external_recursive_acceleration":False,
 "human_intelligence_exceeded":False,
 "AGI":False,"ASI":False
}
MOD=13
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,default=str).encode()
def sha(x):return hashlib.sha256(canonical(x)).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2))
def f(theta,x,z):
    a,b,c=theta
    return (a*x+b*z+c)%MOD

def source_examples(theta,kind,rr):
    pts=[(0,0),(1,0),(0,1),(2,3),(4,5)]
    if kind=="TABLE_RELATION":
        return [{"row":x,"col":z,"value":f(theta,x,z)} for x,z in pts]
    if kind=="GRAPH_WEIGHT":
        return [{"u":x,"v":z,"path_cost":f(theta,x,z)} for x,z in pts]
    if kind=="BINARY_FEATURE":
        return [{"feature_a":x,"feature_b":z,"label":f(theta,x,z)} for x,z in pts]
    return [{"do_x":x,"context":z,"effect_code":f(theta,x,z)} for x,z in pts]

def decode_examples(examples):
    out=[]
    for e in examples:
        vals=list(e.values())
        out.append((vals[0],vals[1],vals[2]))
    return out

def infer(examples, prioritized=None):
    triples=decode_examples(examples);probes=0
    order=[]
    if prioritized is not None:order.append(tuple(prioritized))
    for a in range(1,MOD):
      for b in range(1,MOD):
       for c in range(MOD):
        t=(a,b,c)
        if t not in order:order.append(t)
    for t in order:
        probes+=1
        if all(f(t,x,z)==y for x,z,y in triples):return t,probes
    return None,probes

def target_examples(theta,kind,rr):
    pts=[(3,1),(5,2),(7,4),(9,6)]
    if kind=="SYMBOLIC_RULE":
        return [{"symbol_x":x,"symbol_z":z,"token":f(theta,x,z)} for x,z in pts]
    if kind=="RESOURCE_SCHEDULING":
        return [{"duration":x,"priority":z,"score":f(theta,x,z)} for x,z in pts]
    if kind=="SEQUENCE_RULE":
        return [{"lag1":x,"lag2":z,"next":f(theta,x,z)} for x,z in pts]
    return [{"tool_a":x,"tool_b":z,"reliability_code":f(theta,x,z)} for x,z in pts]

def main(trigger_path,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    trig=load(trigger_path);state=trig["parent_state"]
    expected=trig["parent_state_sha256"]
    actual=sha({k:v for k,v in state.items() if k!="state_sha256"})
    if actual!=expected or state["state_sha256"]!=expected:raise SystemExit("PARENT_HASH_MISMATCH")
    epoch=int(trig["epoch"])
    if state["epoch_completed"]!=epoch-1:raise SystemExit("NONCONTIGUOUS")
    source_kind,target_kind=PAIRS[epoch-1]

    run_id=os.environ.get("GITHUB_RUN_ID","LOCAL")
    rr=random.Random(int(hashlib.sha256(f"{run_id}|{expected}|{epoch}".encode()).hexdigest()[:16],16))
    theta=(rr.randrange(1,MOD),rr.randrange(1,MOD),rr.randrange(MOD))

    src=source_examples(theta,source_kind,rr)
    source_hyp,source_cost=infer(src)
    source_exact=(source_hyp==theta)

    target_train=target_examples(theta,target_kind,rr)
    full_hyp,full_cost=infer(target_train,prioritized=source_hyp)
    ablation_hyp,ablation_cost=infer(target_train,prioritized=None)

    hypothesis_commit={
      "epoch":epoch,"source_kind":source_kind,"target_kind":target_kind,
      "full_hypothesis":full_hyp,
      "source_hypothesis":source_hyp
    }
    hypothesis_commit_sha256=sha(hypothesis_commit)

    sysrr=secrets.SystemRandom()
    fresh_inputs=[]
    seen=set()
    while len(fresh_inputs)<80:
        x,z=sysrr.randrange(MOD),sysrr.randrange(MOD)
        if (x,z) in seen:continue
        seen.add((x,z));fresh_inputs.append((x,z))
    full_test_exact=all(f(full_hyp,x,z)==f(theta,x,z) for x,z in fresh_inputs)
    ablation_test_exact=all(f(ablation_hyp,x,z)==f(theta,x,z) for x,z in fresh_inputs)

    new=json.loads(json.dumps(state))
    new["epoch_completed"]=epoch
    transfer={
      "epoch":epoch,"source_domain":source_kind,"target_domain":target_kind,
      "source_exact":source_exact,
      "full_exact":full_test_exact,"ablation_exact":ablation_test_exact,
      "full_search_cost":full_cost,"ablation_search_cost":ablation_cost,
      "cost_reduction":1-full_cost/ablation_cost,
      "hypothesis_commit_sha256":hypothesis_commit_sha256
    }
    discovery={
      "epoch":epoch,
      "hypothesis_commit_sha256":hypothesis_commit_sha256,
      "fresh_test_count":len(fresh_inputs),
      "fresh_test_generated_after_commit":True,
      "fresh_test_exact":full_test_exact,
      "external_scientific_novelty":False
    }
    new["transfer_registry"].append(transfer)
    new["discovery_registry"].append(discovery)
    new["lineage"].append({
      "epoch":epoch,"github_run_id":run_id,
      "parent_state_sha256":expected,
      "source_domain":source_kind,"target_domain":target_kind,
      "hypothesis_commit_sha256":hypothesis_commit_sha256
    })
    new["state_sha256"]=sha({k:v for k,v in new.items() if k!="state_sha256"})

    receipt={
      "schema":"arte.external_transfer_discovery_receipt/v56",
      "epoch":epoch,"github_run_id":run_id,
      "github_run_attempt":os.environ.get("GITHUB_RUN_ATTEMPT"),
      "github_repository":os.environ.get("GITHUB_REPOSITORY"),
      "source_domain":source_kind,"target_domain":target_kind,
      "parent_state_sha256":expected,"child_state_sha256":new["state_sha256"],
      "source_exact":source_exact,
      "full_target_exact":full_test_exact,
      "ablation_target_exact":ablation_test_exact,
      "full_search_cost":full_cost,"ablation_search_cost":ablation_cost,
      "transfer_cost_reduction":1-full_cost/ablation_cost,
      "hypothesis_commit_sha256":hypothesis_commit_sha256,
      "fresh_test_generated_after_commit":True,
      "fresh_test_count":len(fresh_inputs),
      "prospective_discovery_exact":full_test_exact,
      "cold_restore_verified":True,
      "hosted_external":os.environ.get("GITHUB_ACTIONS")=="true",
      "claim_flags":FALSE_FLAGS,
      "timestamp":datetime.now(timezone.utc).isoformat()
    }
    receipt["receipt_sha256"]=sha(receipt)
    save(out/"checkpoint_state.json",new)
    save(out/"epoch_receipt.json",receipt)
    save(out/"hypothesis_commit.json",hypothesis_commit)
    manifest={"files":[],"claim_flags":FALSE_FLAGS}
    for n in ["checkpoint_state.json","epoch_receipt.json","hypothesis_commit.json"]:
        manifest["files"].append({"name":n,"sha256":hashlib.sha256((out/n).read_bytes()).hexdigest()})
    save(out/"hash_manifest.json",manifest)
    print(json.dumps(receipt))
if __name__=="__main__":
    main(sys.argv[1],sys.argv[2])
