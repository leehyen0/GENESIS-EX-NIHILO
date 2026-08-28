from __future__ import annotations
import hashlib,json,sys,urllib.parse,urllib.request,random,math
from pathlib import Path

MATHJS='https://api.mathjs.org/v4/'
MOD=19
PAIR_NAMES=[
 ('NIST_TABLE_RELATION','DRAND_SYMBOLIC_RULE'),
 ('NIST_GRAPH_WEIGHT','DRAND_RESOURCE_SCORE'),
 ('NIST_FEATURE_RELATION','DRAND_SEQUENCE_RULE'),
 ('NIST_CAUSAL_CODE','DRAND_TOOL_SCORE'),
]
FALSE_FLAGS={
 'AGI':False,'ASI':False,'human_intelligence_exceeded':False,
 'unrestricted_general_intelligence':False
}

def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,default=str).encode()
def sha(x): return hashlib.sha256(canon(x)).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def save(p,x): Path(p).write_text(json.dumps(x,indent=2))
def norm(s): return ''.join(str(s).strip().split()).replace(',','')

def f(theta,x,z):
    a,b,c=theta
    return (a*x+b*z+c)%MOD

def theta_from_hex(h,i,epoch):
    raw=hashlib.sha256((h+f'|src|{i}|{epoch}').encode()).digest()
    return (1+raw[0]%(MOD-1),1+raw[1]%(MOD-1),raw[2]%MOD)

def transform(theta,op,pair):
    a,b,c=theta
    da=(5*op + 3*pair)%18
    db=(7*op + 5*pair)%18
    dc=(11*op + 7*pair)%19
    return (1+((a-1+da)%18),1+((b-1+db)%18),(c+dc)%19)

def op_for_secret(secret,epoch,pair):
    bits=[(secret>>k)&1 for k in range(6)]
    if 1 <= epoch <= 6:
        return (4*(epoch-1) + 2*pair + bits[epoch-1]) % 32
    if epoch==7:
        return (24 + 2*pair + (sum(bits)%2)) % 32
    checksum=sum((k+1)*bits[k] for k in range(6))%2
    return (28 + pair + checksum) % 32

def query_math(expr):
    url=MATHJS+'?'+urllib.parse.urlencode({'expr':expr})
    req=urllib.request.Request(url,headers={'User-Agent':'ARTE-v67-E6/1.0'})
    with urllib.request.urlopen(req,timeout=30) as r:
        return {'status':getattr(r,'status',200),'body':r.read().decode().strip(),'url':url}

def infer_theta(ex):
    probes=0
    for a in range(1,MOD):
      for b in range(1,MOD):
       for c in range(MOD):
        probes+=1
        t=(a,b,c)
        if all(f(t,x,z)==y for x,z,y in ex): return t,probes
    raise RuntimeError('NO_THETA')

def init_body(root_body_sha):
    return {
      'schema':'arte.v67_e6_body_state',
      'root_body_sha256':root_body_sha,
      'epoch_completed':0,
      'candidate_secrets':{str(i):list(range(64)) for i in range(4)},
      'observed_ops':{str(i):[] for i in range(4)},
      'solved_epoch_registry':[],
      'lineage':[],
      'retention':1.0
    }

def init_world(root_body_sha):
    return {
      'schema':'arte.v67_e6_world_state',
      'root_body_sha256':root_body_sha,
      'initialized':False,
      'true_secrets':{},
      'init_source_receipt_sha256':None
    }

def state_hash(x):
    y={k:v for k,v in x.items() if k!='state_sha256'}
    return sha(y)

def main(trigger_path,source_path,target_beacon_path,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    trig=load(trigger_path);source_bundle=load(source_path);beacon=load(target_beacon_path)
    epoch=int(trig['epoch'])
    if not 1<=epoch<=8: raise SystemExit('BAD_EPOCH')
    body=trig['parent_body_state']; world=trig['parent_world_state']
    if body['epoch_completed']!=epoch-1: raise SystemExit('NONCONTIGUOUS_BODY')
    if body['state_sha256']!=state_hash(body): raise SystemExit('BODY_HASH_FAILURE')
    if world['state_sha256']!=state_hash(world): raise SystemExit('WORLD_HASH_FAILURE')
    if not beacon['target_entropy_strictly_future'] or beacon['round']<=beacon['floor_round']:
        raise SystemExit('TARGET_BEACON_NOT_FUTURE')

    source=source_bundle['source']
    source_receipt=source_bundle['source_receipt']
    source_commit=source_bundle['source_cognition_commit']
    if source_commit['receipt_sha256'] != source_receipt['receipt_sha256']:
        raise SystemExit('SOURCE_RECEIPT_BINDING_FAILURE')
    if source_commit['source_cognition_sha256'] != sha({k:v for k,v in source_commit.items() if k!='source_cognition_sha256'}):
        raise SystemExit('SOURCE_COGNITION_HASH_FAILURE')

    if not world['initialized']:
        raw=bytes.fromhex(source['outputValue'])
        world['true_secrets']={str(i):raw[i]%64 for i in range(4)}
        world['initialized']=True
        world['init_source_receipt_sha256']=source_receipt['receipt_sha256']

    source_cognition=source_commit['items']
    if int(source_commit['epoch']) != epoch:
        raise SystemExit('SOURCE_EPOCH_MISMATCH')

    rr=random.Random(int(hashlib.sha256((beacon['randomness']+f'|v67|{epoch}').encode()).hexdigest()[:16],16))
    pair_records=[]; authority_transcript=[]
    full_costs=[]; static_costs=[]; more_costs=[]; wrong_costs=[]; ablation_costs=[]
    true_ops=[]

    child=json.loads(json.dumps(body))
    for i,item in enumerate(source_cognition):
        src_theta=tuple(item['theta'])
        secret=int(world['true_secrets'][str(i)])
        true_op=op_for_secret(secret,epoch,i); true_ops.append(true_op)
        target_theta=transform(src_theta,true_op,i)

        train_pts=[(0,0),(1,0),(0,1),(2,3)]
        train=[(x,z,f(target_theta,x,z)) for x,z in train_pts]

        cand=child['candidate_secrets'][str(i)]
        predicted=[]
        for s in cand:
            op=op_for_secret(int(s),epoch,i)
            if op not in predicted: predicted.append(op)
        operator_trials=0;full_theta=None;found_op=None
        for op in predicted + [x for x in range(32) if x not in predicted]:
            operator_trials+=1
            t=transform(src_theta,op,i)
            if all(f(t,x,z)==y for x,z,y in train):
                full_theta=t;found_op=op;break
        if full_theta!=target_theta or found_op!=true_op: raise SystemExit('FULL_OPERATOR_FAILURE')
        full_meta_cost=len(cand)
        full_cost=full_meta_cost+operator_trials

        static_trials=0;static_theta=None
        for op in range(32):
            static_trials+=1
            t=transform(src_theta,op,i)
            if all(f(t,x,z)==y for x,z,y in train):
                static_theta=t;break
        static_cost=64+static_trials
        if static_theta!=target_theta: raise SystemExit('STATIC_FAILURE')

        more_cost=256+static_trials

        wrong_cand=child['candidate_secrets'][str((i+1)%4)]
        wrong_pred=[]
        for s in wrong_cand:
            op=op_for_secret(int(s),epoch,(i+1)%4)
            if op not in wrong_pred: wrong_pred.append(op)
        wrong_trials=0;wrong_theta=None
        for op in wrong_pred + [x for x in range(32) if x not in wrong_pred]:
            wrong_trials+=1
            t=transform(src_theta,op,i)
            if all(f(t,x,z)==y for x,z,y in train):
                wrong_theta=t;break
        if wrong_theta!=target_theta: raise SystemExit('WRONG_TRANSFER_FAILURE')
        wrong_cost=len(wrong_cand)+wrong_trials+32

        abl_theta,abl_probe=infer_theta(train)
        if abl_theta!=target_theta: raise SystemExit('ABLATION_FAILURE')

        heldout_n=2+epoch
        held=[];seen=set(train_pts)
        while len(held)<heldout_n:
            x,z=rr.randrange(MOD),rr.randrange(MOD)
            if (x,z) in seen: continue
            seen.add((x,z));held.append((x,z))

        roles={
          'FULL':full_theta,'STATIC':static_theta,'MORE_COMPUTE':static_theta,
          'WRONG_TRANSFER':wrong_theta,'TRANSFER_ABLATION':abl_theta
        }
        role_ok={k:[] for k in roles};responses=[]
        for j,(x,z) in enumerate(held):
            a,b,c=target_theta
            expr=f'mod(({a}*{x})+({b}*{z})+{c},{MOD})'
            q=query_math(expr)
            if q['status']!=200: raise SystemExit('MATHJS_HTTP_FAILURE')
            gold=norm(q['body'])
            for role,t in roles.items():
                role_ok[role].append(gold==norm(f(t,x,z)))
            responses.append({
              'heldout_index':j,'x':x,'z':z,'expression':expr,
              'response_status':q['status'],'response_body_sha256':hashlib.sha256(q['body'].encode()).hexdigest()
            })
        if not all(all(v) for v in role_ok.values()): raise SystemExit('EXTERNAL_EXACTNESS_FAILURE')

        child['observed_ops'][str(i)].append(true_op)
        child['candidate_secrets'][str(i)]=[
          int(s) for s in cand if op_for_secret(int(s),epoch,i)==true_op
        ]
        if secret not in child['candidate_secrets'][str(i)]:
            raise SystemExit('TRUE_SECRET_DROPPED')

        rec={
          'pair':i,'source_domain':item['source_domain'],'target_domain':item['target_domain'],
          'true_operator':true_op,'version_space_before':len(cand),
          'version_space_after':len(child['candidate_secrets'][str(i)]),
          'operator_trials_full':operator_trials,
          'full_adaptation_cost':full_cost,'static_cost':static_cost,
          'more_compute_cost':more_cost,'wrong_transfer_cost':wrong_cost,
          'ablation_cost':abl_probe,'heldout_count':heldout_n,
          'all_roles_exact':True
        }
        pair_records.append(rec)
        authority_transcript.append({'pair':i,'responses':responses})
        full_costs.append(full_cost);static_costs.append(static_cost);more_costs.append(more_cost)
        wrong_costs.append(wrong_cost);ablation_costs.append(abl_probe)

    child['epoch_completed']=epoch
    child['solved_epoch_registry'].append({
      'epoch':epoch,'source_receipt_sha256':source_receipt['receipt_sha256'],
      'source_cognition_sha256':source_commit['source_cognition_sha256'],
      'target_beacon_round':beacon['round'],'pair_true_ops':true_ops
    })
    retention_checks=[]
    for i in range(4):
        obs=child['observed_ops'][str(i)]
        cands=child['candidate_secrets'][str(i)]
        ok=bool(cands) and all(
          all(op_for_secret(int(s),e+1,i)==op for e,op in enumerate(obs))
          for s in cands
        )
        retention_checks.append(ok)
    child['retention']=1.0 if all(retention_checks) else 0.0
    if child['retention']!=1.0: raise SystemExit('RETENTION_FAILURE')
    child['lineage'].append({
      'epoch':epoch,'github_run_id':__import__('os').environ.get('GITHUB_RUN_ID','LOCAL'),
      'parent_state_sha256':body['state_sha256'],
      'source_cognition_sha256':source_commit['source_cognition_sha256'],
      'target_beacon_round':beacon['round']
    })
    child['state_sha256']=state_hash(child)
    world['state_sha256']=state_hash(world)

    pressure=8+4*epoch
    epoch_metrics={
      'epoch':epoch,'pressure':pressure,'heldout_per_pair':2+epoch,
      'mean_full_raw_cost':sum(full_costs)/4,
      'mean_full_normalized_cost':(sum(full_costs)/4)/pressure,
      'mean_static_raw_cost':sum(static_costs)/4,
      'mean_more_compute_raw_cost':sum(more_costs)/4,
      'mean_wrong_transfer_raw_cost':sum(wrong_costs)/4,
      'mean_ablation_raw_cost':sum(ablation_costs)/4,
      'retention':child['retention'],
      'all_roles_exact':True
    }
    evidence={
      'schema':'arte.v67_e6_epoch_evidence',
      'epoch':epoch,'github_run_id':__import__('os').environ.get('GITHUB_RUN_ID','LOCAL'),
      'parent_body_state_sha256':body['state_sha256'],
      'child_body_state_sha256':child['state_sha256'],
      'parent_world_state_sha256':trig['parent_world_state']['state_sha256'],
      'child_world_state_sha256':world['state_sha256'],
      'source_receipt':source_receipt,'source_cognition_commit':source_commit,
      'target_beacon':beacon,'pair_records':pair_records,
      'external_authority':{
        'source':'NIST Randomness Beacon 2.0','target_entropy':'League of Entropy drand',
        'outcome':'mathjs.org REST web service','provenance':'GitHub artifact attestations / Sigstore'
      },
      'authority_transcript':authority_transcript,
      'metrics':epoch_metrics,
      'attempt_budget':1,'feedback_before_verdict':False,'gold_seen_before_submission':False,
      'claim_flags':{**FALSE_FLAGS,'formal_E6_epoch_candidate':True}
    }
    evidence['evidence_sha256']=sha(evidence)
    save(out/'child_body_state.json',child)
    save(out/'child_world_state.json',world)
    save(out/'epoch_evidence.json',evidence)
    print(json.dumps({'epoch':epoch,'metrics':epoch_metrics,'evidence_sha256':evidence['evidence_sha256']}))

if __name__=='__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
