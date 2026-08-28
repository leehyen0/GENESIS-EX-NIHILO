from __future__ import annotations
import hashlib,json,os,platform,random,sys,math
from datetime import datetime,timezone
from pathlib import Path

FALSE_FLAGS={
 "external_recursive_acceleration":False,
 "independent_custody_proof":False,
 "matched_cross_domain_superiority":False,
 "human_intelligence_exceeded":False,
 "AGI":False,
 "ASI":False
}
def csha(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def mean(x): return sum(x)/len(x)

def main(outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    rr=random.Random(20260828)
    M,D,C=32,16,8
    bases=[rr.randrange(M) for _ in range(4)]
    steps=[]
    for _ in range(4):
        s=rr.randrange(1,M)
        while math.gcd(s,M)!=1:s=rr.randrange(1,M)
        steps.append(s)
    def primary(d,c):return (bases[d%4]+steps[d%4]*c)%M
    stats={}
    models={}
    obs={str(k):{} for k in range(4)}
    efficiency=1.0
    pressure=[];cost=[];trials=[];gain=[]
    for g in range(1024):
        d=g%D; unlocked=min(C,1+g//128); c=((g//D)+d*3)%unlocked
        p=2.0+g/80.0+d/32.0
        key=f"{d}:{c}"
        row=stats.setdefault(key,{m:[1,0] for m in range(M)})
        pred=None
        if str(d%4) in models:
            b,s=models[str(d%4)];pred=(b+s*c)%M
        order=sorted(range(M),key=lambda m:(-(row[m][1]/row[m][0]),m))
        if pred is not None:order=[pred]+[m for m in order if m!=pred]
        n=0
        target=primary(d,c)
        for m in order:
            n+=1
            z=0
            for j in range(180):
                z=(z+(m+3)*(j+7)*(g+11)+(z>>2))%1000003
            row[m][0]+=1
            if m==target:
                row[m][1]+=1
                cl=str(d%4);obs[cl][str(c)]=m
                if "0" in obs[cl] and "1" in obs[cl]:
                    models[cl]=(obs[cl]["0"],(obs[cl]["1"]-obs[cl]["0"])%M)
                efficiency*=1.0025
                break
        norm=0.28*n+1/efficiency+0.22
        pressure.append(p);cost.append(norm);trials.append(n);gain.append(1/norm)
    result={
      "schema":"arte.github_offbox_mass_acceleration_receipt/v52",
      "timestamp":datetime.now(timezone.utc).isoformat(),
      "github_environment":{k:os.environ.get(k) for k in (
        "GITHUB_ACTIONS","GITHUB_RUN_ID","GITHUB_RUN_ATTEMPT","GITHUB_REPOSITORY","GITHUB_SHA","GITHUB_REF"
      )},
      "runner":{"platform":platform.platform(),"python":platform.python_version()},
      "workload":{"generations":1024,"domains":16,"contexts_per_domain":8,"mutation_candidates":32},
      "early":{"pressure":mean(pressure[:256]),"normalized_cost":mean(cost[:256]),"mutation_trials":mean(trials[:256]),"gain_per_cost":mean(gain[:256])},
      "late":{"pressure":mean(pressure[-256:]),"normalized_cost":mean(cost[-256:]),"mutation_trials":mean(trials[-256:]),"gain_per_cost":mean(gain[-256:])},
      "offbox_action_consequence":os.environ.get("GITHUB_ACTIONS")=="true",
      "claim_flags":FALSE_FLAGS
    }
    result["receipt_sha256"]=csha(result)
    (out/"v52_offbox_receipt.json").write_text(json.dumps(result,indent=2))
    manifest={"files":[{"name":"v52_offbox_receipt.json","sha256":hashlib.sha256((out/"v52_offbox_receipt.json").read_bytes()).hexdigest()}],"claim_flags":FALSE_FLAGS}
    (out/"v52_hash_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(result))
if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "v52_output")
