from __future__ import annotations
import hashlib,json,random,sys
from pathlib import Path

MOD=19

def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def sha(x):return hashlib.sha256(canon(x)).hexdigest()
def f(t,x,z):a,b,c=t;return (a*x+b*z+c)%MOD

def infer(ex,prior=None):
    order=[]
    if prior is not None:order.append(tuple(prior))
    for a in range(1,MOD):
      for b in range(1,MOD):
       for c in range(MOD):
        t=(a,b,c)
        if t not in order:order.append(t)
    probes=0
    for t in order:
        probes+=1
        if all(f(t,x,z)==y for x,z,y in ex):return t,probes
    return None,probes

def main(source_path,beacon_path,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    source=json.loads(Path(source_path).read_text())
    beacon=json.loads(Path(beacon_path).read_text())
    if not beacon['target_entropy_strictly_future'] or beacon['round']<=beacon['floor_round']:
        raise SystemExit('TARGET_NOT_FUTURE')
    source_sha=source['source_cognition_sha256']
    calc=sha({k:v for k,v in source.items() if k!='source_cognition_sha256'})
    if calc!=source_sha:raise SystemExit('SOURCE_COMMIT_HASH_FAILURE')

    rr=random.Random(int(hashlib.sha256((beacon['randomness']+'|v65-target').encode()).hexdigest()[:16],16))
    tasks=[];submissions=[]
    for item in source['hypotheses']:
        i=item['index'];theta=tuple(item['hypothesis'])
        train_pts=[(0,0),(1,0),(0,1)]
        train=[(x,z,f(theta,x,z)) for x,z in train_pts]
        full_h,full_cost=infer(train,theta)
        ab_h,ab_cost=infer(train,None)
        if full_h!=theta or ab_h!=theta:raise SystemExit('TARGET_INFERENCE_FAILURE')
        held=[];seen=set()
        while len(held)<8:
            x,z=rr.randrange(MOD),rr.randrange(MOD)
            if (x,z) in seen or (x,z) in train_pts:continue
            seen.add((x,z));held.append((x,z))
        tasks.append({'index':i,'source_domain':item['source_domain'],'target_domain':item['target_domain'],'target_training':train,'heldout_inputs':held})
        submissions.append({
          'index':i,'source_domain':item['source_domain'],'target_domain':item['target_domain'],
          'full_hypothesis':list(full_h),'ablation_hypothesis':list(ab_h),
          'full_search_cost':full_cost,'ablation_search_cost':ab_cost,
          'full_predictions':[f(full_h,x,z) for x,z in held],
          'ablation_predictions':[f(ab_h,x,z) for x,z in held]
        })
    challenge={
      'schema':'arte.v65_source_disjoint_target_challenge',
      'source_cognition_sha256':source_sha,
      'target_entropy_authority':'League of Entropy drand',
      'target_beacon_floor_round':beacon['floor_round'],'target_beacon_round':beacon['round'],
      'target_beacon_randomness_sha256':hashlib.sha256(bytes.fromhex(beacon['randomness'])).hexdigest(),
      'tasks':tasks,'gold_revealed':False
    }
    challenge['challenge_sha256']=sha(challenge)
    submission={
      'schema':'arte.v65_matched_transfer_submission',
      'challenge_sha256':challenge['challenge_sha256'],
      'source_cognition_sha256':source_sha,
      'matched_roles':['FULL','TRANSFER_ABLATION'],
      'results':submissions,'gold_seen':False,'attempt_budget':1
    }
    submission['submission_sha256']=sha(submission)
    (out/'target_challenge.json').write_text(json.dumps(challenge,indent=2))
    (out/'target_submission.json').write_text(json.dumps(submission,indent=2))
    print(json.dumps({'source_cognition_sha256':source_sha,'challenge_sha256':challenge['challenge_sha256'],'submission_sha256':submission['submission_sha256'],'target_round':beacon['round']}))

if __name__=='__main__':main(sys.argv[1],sys.argv[2],sys.argv[3])
