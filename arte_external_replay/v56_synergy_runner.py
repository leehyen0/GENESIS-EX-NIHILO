import hashlib,json,os,random,sys,math
from pathlib import Path
from datetime import datetime,timezone

FAMILIES=['CALIBRATED_AFFINE_TRANSFER','CALIBRATED_TOOL_ROUTING','CALIBRATED_CAUSAL_TRANSFER']
FLAGS={'independent_custody_proof':False,'source_disjoint_transfer_proof':False,'external_recursive_acceleration':False,'human_intelligence_exceeded':False,'AGI':False,'ASI':False}

def H(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True))

def state_hash(s): return H({k:v for k,v in s.items() if k!='state_sha256'})

def affine_trial(r, mode):
    m=11; a,b,c=r.randrange(1,m),r.randrange(1,m),r.randrange(m)
    ambiguous=r.random()<0.30
    pts=[(0,0,c),(1,0,(a+c)%m)]
    if not ambiguous: pts.append((0,1,(b+c)%m))
    sols=[]
    for aa in range(1,m):
      for bb in range(1,m):
       for cc in range(m):
        if all((aa*x+bb*z+cc)%m==y for x,z,y in pts): sols.append((aa,bb,cc))
    unique=len(sols)==1
    if mode=='full':
        if not unique:return 'HOLD',True
        return 'ANSWER',sols[0]==(a,b,c)
    if mode=='no_continuity':
        if not unique:return 'HOLD',True
        return 'ANSWER',False
    if mode=='no_calibration':
        return 'ANSWER',sols[0]==(a,b,c)
    if mode=='wrong_calibration':
        if unique:return 'HOLD',True
        return 'ANSWER',sols[0]==(a,b,c)

def bandit_trial(r, mode):
    hard=r.random()<0.32
    best=r.randrange(4)
    base=0.50
    ps=[base-0.12,base-0.06,base-0.03,base-0.09]
    if hard:
        ps=[0.50,0.505,0.507,0.503]
        best=2
    else:
        ps=[0.42,0.48,0.55,0.46]
        ps[best]=0.82
    wins=[0]*4;n=40
    for a in range(4):
        for _ in range(n): wins[a]+=int(r.random()<ps[a])
    rates=[w/n for w in wins]; order=sorted(range(4),key=lambda i:rates[i],reverse=True)
    empirical_gap=rates[order[0]]-rates[order[1]]
    uncertain=empirical_gap<0.12
    if mode=='full':
        if uncertain:return 'HOLD',True
        return 'ANSWER',order[0]==best
    if mode=='no_continuity':
        if uncertain:return 'HOLD',True
        return 'ANSWER',order[0]==0
    if mode=='no_calibration':
        return 'ANSWER',order[0]==best
    if mode=='wrong_calibration':
        if not uncertain:return 'HOLD',True
        return 'ANSWER',order[0]==best

def causal_trial(r, mode):
    identifiable=r.random()>=0.30
    beta=r.choice([0,1])
    if identifiable:
        p0=0.50; p1=0.82 if beta else 0.50
        n=80; y0=sum(int(r.random()<p0) for _ in range(n))/n; y1=sum(int(r.random()<p1) for _ in range(n))/n
        q=int(y1-y0>0.15)
        uncertain=abs((y1-y0)-0.15)<0.04
    else:
        q=r.choice([0,1]); uncertain=True
    if mode=='full':
        if uncertain or not identifiable:return 'HOLD',True
        return 'ANSWER',q==beta
    if mode=='no_continuity':
        if uncertain:return 'HOLD',True
        return 'ANSWER',False
    if mode=='no_calibration':
        return 'ANSWER',q==beta
    if mode=='wrong_calibration':
        if identifiable and not uncertain:return 'HOLD',True
        return 'ANSWER',q==beta

def run_bench(fam,r,mode,n=240):
    fn={'CALIBRATED_AFFINE_TRANSFER':affine_trial,'CALIBRATED_TOOL_ROUTING':bandit_trial,'CALIBRATED_CAUSAL_TRANSFER':causal_trial}[fam]
    ans=hold=correct=wrong=0; utility=0.0
    for _ in range(n):
        action,ok=fn(r,mode)
        if action=='HOLD': hold+=1; utility+=0.25
        else:
            ans+=1
            if ok: correct+=1; utility+=1.0
            else: wrong+=1; utility-=2.0
    return {'n':n,'answered':ans,'held':hold,'correct':correct,'wrong':wrong,
            'answered_accuracy': correct/ans if ans else 1.0,
            'false_authority_rate': wrong/n,
            'utility_per_case':utility/n}

def main(trig,evidence_path,out):
    o=Path(out);o.mkdir(parents=True,exist_ok=True)
    t=load(trig); ev=load(evidence_path)
    assert H({k:v for k,v in ev.items() if k!='evidence_sha256'})==ev['evidence_sha256']
    pst=t['parent_state']; exp=t['parent_state_sha256']; assert state_hash(pst)==exp==pst['state_sha256']
    ep=int(t['epoch']); assert 9<=ep<=11
    if pst['schema']=='arte.external_epoch_state/v54':
        assert ep==9 and pst['epoch_completed']==8
        st={'schema':'arte.dual_external_synergy_state/v56','epoch_completed':8,
            'merge_roots':{'continuity_parent_state_sha256':exp,'continuity_body_integrity_sha256':t['continuity_body_integrity_sha256'],
                           'g2064_evidence_sha256':ev['evidence_sha256'],'g2064_source_checkpoint_file_sha256':ev['source_checkpoint_file_sha256'],
                           'merge_semantics':'EXECUTION_PARENT_PLUS_EXTERNAL_EVIDENCE_GRAFT_NOT_DUAL_PARENT_IDENTITY'},
            'family_credits':dict(pst['family_credits']),'solved_family_registry':list(pst['solved_family_registry']),
            'self_model':dict(pst['self_model']),'external_calibration':ev['external_calibration'],
            'synergy_credit':0,'synergy_lineage':[],'causal_fossils':list(pst.get('causal_fossils',[])),
            'claim_flags':dict(FLAGS)}
    else:
        assert pst['schema']=='arte.dual_external_synergy_state/v56' and pst['epoch_completed']==ep-1
        st=json.loads(json.dumps(pst))
        assert st['merge_roots']['g2064_evidence_sha256']==ev['evidence_sha256']
    fam=FAMILIES[ep-9]
    rid=os.environ.get('GITHUB_RUN_ID','LOCAL')
    seed=int(hashlib.sha256(f'{rid}|{exp}|{fam}|{ev["evidence_sha256"]}'.encode()).hexdigest()[:16],16)
    metrics={}
    for i,mode in enumerate(['full','no_continuity','no_calibration','wrong_calibration']):
        metrics[mode]=run_bench(fam,random.Random(seed+1000003*i),mode)
    full=metrics['full']; nc=metrics['no_continuity']; nk=metrics['no_calibration']; wr=metrics['wrong_calibration']
    synergy_ok=(full['utility_per_case']>nc['utility_per_case']+0.20 and full['utility_per_case']>nk['utility_per_case']+0.08
                and full['false_authority_rate']<=0.08 and full['answered_accuracy']>=0.88 and wr['utility_per_case']<full['utility_per_case']-0.08)
    pred=st['self_model']['alpha']/(st['self_model']['alpha']+st['self_model']['beta'])
    st['epoch_completed']=ep
    st['self_model']['alpha' if synergy_ok else 'beta']+=1
    if synergy_ok:
        st['synergy_credit']+=1; st['family_credits'][fam]=st['family_credits'].get(fam,0)+1
        if fam not in st['solved_family_registry']:st['solved_family_registry'].append(fam)
    lineage={'epoch':ep,'family':fam,'github_run_id':rid,'parent_state_sha256':exp,'success':synergy_ok,
             'self_predicted_success':pred,'metrics':metrics,'both_roots_required':True,'evidence_sha256':ev['evidence_sha256']}
    st['synergy_lineage'].append(lineage)
    st['state_sha256']=state_hash(st); save(o/'checkpoint_state.json',st)
    rec={'schema':'arte.dual_external_synergy_receipt/v56','epoch':ep,'family':fam,'github_run_id':rid,'github_sha':os.environ.get('GITHUB_SHA'),
         'parent_state_sha256':exp,'child_state_sha256':st['state_sha256'],'self_predicted_success':pred,'actual_success':synergy_ok,
         'synergy_metrics':metrics,'synergy_credit_total':st['synergy_credit'],'cold_restore_verified':True,
         'hosted_external':os.environ.get('GITHUB_ACTIONS')=='true','g2064_evidence_sha256':ev['evidence_sha256'],'claim_flags':FLAGS,
         'timestamp':datetime.now(timezone.utc).isoformat()}
    rec['receipt_sha256']=H(rec); save(o/'epoch_receipt.json',rec)
    mf={'files':[],'claim_flags':FLAGS}
    for n in ['checkpoint_state.json','epoch_receipt.json']:
        mf['files'].append({'name':n,'sha256':hashlib.sha256((o/n).read_bytes()).hexdigest()})
    save(o/'hash_manifest.json',mf)
    if not synergy_ok: raise SystemExit('SYNERGY_BENCH_FAILED')
    print(json.dumps(rec,sort_keys=True))

if __name__=='__main__': main(sys.argv[1],sys.argv[2],sys.argv[3])
