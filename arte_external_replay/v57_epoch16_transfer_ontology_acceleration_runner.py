import json, hashlib, os, re, sys
from pathlib import Path
from datetime import datetime, timezone
import urllib.request, tempfile, shutil

BB_COMMIT='092b196c1f8f14a54bbc62f24759d43bde46dd3b'
BB_BASE=f'https://raw.githubusercontent.com/google/BIG-bench/{BB_COMMIT}/bigbench/benchmark_tasks'
PARENT_OP_SHA='9acabe9915c45105caa1ee04bfa5c8d267706809119a616ed2f456a0674ae6fd'
E3_TRANSFER_RECEIPT_SHA='fc9d5febd706edef223b10d920a6374c18e7812cc2d1b2b0199153d7bb718440'
E3_DISCOVERY_RECEIPT_SHA='0331978e07671e6866c71b1f631d1336b2c578f064c5b617344d6a882c8d6303'
FAILED_RUN_ID='33138655768'
FAILURE_TYPE='STATE_SCHEMA_COLLISION::synergy_credit_int_vs_dict'
FLAGS={'independent_custody_proof':False,'source_disjoint_transfer_proof':False,'external_recursive_acceleration':False,'human_intelligence_exceeded':False,'AGI':False,'ASI':False}

def H(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(p): return json.load(open(p,encoding='utf-8'))
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True),encoding='utf-8')
def fetch_json(url):
    with urllib.request.urlopen(url,timeout=30) as r: return json.loads(r.read().decode())

def split_sentences(text): return [s.strip()+'.' for s in text.split('.') if s.strip()]
def step_count(s):
    m=re.search(r'Take (\d+) step',s); return int(m.group(1)) if m else None

def nav_distance_bag(text):
    total=sum(step_count(s) or 0 for s in split_sentences(text))
    return 'True' if total==0 else 'False'

def nav_signed_line(text):
    x=0
    for s in split_sentences(text):
        n=step_count(s)
        if n is None: continue
        if 'backward' in s or 'left' in s: x-=n
        else: x+=n
    return 'True' if x==0 else 'False'

def nav_absolute_vector(text):
    x=y=0
    for s in split_sentences(text):
        n=step_count(s)
        if n is None: continue
        if 'left' in s: x-=n
        elif 'right' in s: x+=n
        elif 'backward' in s: y-=n
        else: y+=n
    return 'True' if x==0 and y==0 else 'False'

def _rot(v,turn):
    x,y=v
    if turn=='left': return (-y,x)
    if turn=='right': return (y,-x)
    if turn=='around': return (-x,-y)
    return v

def nav_ordered_pose(text, wrong_all_turns_first=False):
    face_fixed='Always face forward' in text
    x=y=0; facing=(0,1); ss=split_sentences(text)
    if wrong_all_turns_first and not face_fixed:
        for s in ss:
            if 'Turn left' in s: facing=_rot(facing,'left')
            elif 'Turn right' in s: facing=_rot(facing,'right')
            elif 'Turn around' in s: facing=_rot(facing,'around')
        ss=[s for s in ss if not s.startswith('Turn ')]
    for s in ss:
        if s.startswith('Turn ') and not face_fixed and not wrong_all_turns_first:
            if 'left' in s: facing=_rot(facing,'left')
            elif 'right' in s: facing=_rot(facing,'right')
            elif 'around' in s: facing=_rot(facing,'around')
            continue
        n=step_count(s)
        if n is None: continue
        if face_fixed:
            if 'left' in s: dx,dy=(-1,0)
            elif 'right' in s: dx,dy=(1,0)
            elif 'backward' in s: dx,dy=(0,-1)
            else: dx,dy=(0,1)
        else:
            fx,fy=facing
            if 'backward' in s: dx,dy=(-fx,-fy)
            elif 'left' in s: dx,dy=_rot(facing,'left')
            elif 'right' in s: dx,dy=_rot(facing,'right')
            else: dx,dy=facing
        x+=dx*n; y+=dy*n
    return 'True' if x==0 and y==0 else 'False'

CANDIDATES={'DISTANCE_BAG':nav_distance_bag,'SIGNED_LINE':nav_signed_line,'ABSOLUTE_VECTOR':nav_absolute_vector,'ORDERED_POSE':lambda t:nav_ordered_pose(t,False)}
FEATURES={'DISTANCE_BAG':{'persistent_state':False,'ordered_transforms':False,'terminal_query':True},'SIGNED_LINE':{'persistent_state':True,'ordered_transforms':False,'terminal_query':True},'ABSOLUTE_VECTOR':{'persistent_state':True,'ordered_transforms':False,'terminal_query':True},'ORDERED_POSE':{'persistent_state':True,'ordered_transforms':True,'terminal_query':True}}
INHERITED_SCHEMA={'persistent_state':True,'ordered_transforms':True,'terminal_query':True}
BASE_ORDER=['DISTANCE_BAG','SIGNED_LINE','ABSOLUTE_VECTOR','ORDERED_POSE']

def target(ex): return max(ex['target_scores'],key=ex['target_scores'].get)
def accuracy(name,examples):
    f=CANDIDATES[name]; return sum(f(e['input'])==target(e) for e in examples)/len(examples)
def search(order,dev):
    trace=[]
    for i,name in enumerate(order,1):
        acc=accuracy(name,dev); trace.append({'rank':i,'candidate':name,'dev_accuracy':acc})
        if acc==1.0: return name,i,trace
    return None,len(order),trace

def full_order():
    def score(name):
        match=sum(int(FEATURES[name][k]==v) for k,v in INHERITED_SCHEMA.items())
        complexity=BASE_ORDER.index(name)
        return (-match,complexity,name)
    return sorted(BASE_ORDER,key=score)

def restore_epoch15_parent(expected_sha):
    import v57_epoch15_ontology_expansion_runner as e15
    td=Path(tempfile.mkdtemp(prefix='arte_e15_restore_')); old=os.environ.get('GITHUB_RUN_ID')
    try:
        os.environ['GITHUB_RUN_ID']='33137539741'
        e15.main('arte_external_replay/v57_epoch15_trigger.json', str(td))
        st=load(td/'checkpoint_state.json')
    finally:
        if old is None: os.environ.pop('GITHUB_RUN_ID',None)
        else: os.environ['GITHUB_RUN_ID']=old
        shutil.rmtree(td,ignore_errors=True)
    assert st['state_sha256']==expected_sha
    return st

def apply_failure_backpressure(st, base_sha):
    ns=json.loads(json.dumps(st))
    assert isinstance(ns.get('synergy_credit'),int), 'parent synergy_credit type changed'
    fossil={'epoch_attempt':16,'external_failure_run_id':FAILED_RUN_ID,'from_state_sha256':base_sha,'failure_type':FAILURE_TYPE,'failed_run_outcome_credit':False,'mutation_scope':'BOOKKEEPING_FIELD_ONLY','target_or_prediction_reused':False}
    if not any(isinstance(x,dict) and x.get('external_failure_run_id')==FAILED_RUN_ID for x in ns.setdefault('causal_fossils',[])):
        ns['causal_fossils'].append(fossil)
    event={'event_type':'DELAYED_EXTERNAL_FAILURE_BACKPRESSURE','external_failure_run_id':FAILED_RUN_ID,'from_state_sha256':base_sha,'failure_type':FAILURE_TYPE,'self_model_update':'BETA_PLUS_ONE','failed_run_outcome_credit':False}
    if not any(isinstance(x,dict) and x.get('external_failure_run_id')==FAILED_RUN_ID for x in ns.setdefault('recovery_events',[])):
        ns['recovery_events'].append(event)
        ns['self_model']['beta']+=1
    ns['state_sha256']=H({k:v for k,v in ns.items() if k!='state_sha256'})
    return ns

def main(trigger,outdir):
    o=Path(outdir); o.mkdir(parents=True,exist_ok=True)
    t=load(trigger); base_sha=t['parent_state_sha256']; base=restore_epoch15_parent(base_sha)
    assert t['epoch']==16 and base['epoch_completed']==15
    assert H({k:v for k,v in base.items() if k!='state_sha256'})==base_sha==base['state_sha256']
    assert any(x.get('operator_sha256')==PARENT_OP_SHA and x.get('target_consumed_for_generation') is False for x in base.get('generated_operator_registry',[]))
    assert t['e3_roots']['cross_domain_transfer_receipt_sha256']==E3_TRANSFER_RECEIPT_SHA
    assert t['e3_roots']['novel_discovery_receipt_sha256']==E3_DISCOVERY_RECEIPT_SHA
    assert t['failure_conditioning']['failed_run_id']==FAILED_RUN_ID
    st=apply_failure_backpressure(base,base_sha); recovery_sha=st['state_sha256']
    assert st['self_model']['alpha']==16 and st['self_model']['beta']==3
    assert st['synergy_credit']==base['synergy_credit']==3

    d=fetch_json(f'{BB_BASE}/navigate/task.json'); xs=d['examples']; rid=os.environ.get('GITHUB_RUN_ID','LOCAL')
    eligible=[i for i,e in enumerate(xs) if nav_ordered_pose(e['input'],False)!=nav_ordered_pose(e['input'],True)]
    if not eligible: raise RuntimeError('NO_INPUT_ONLY_WRONG_DISCRIMINATING_NAV_CASE')
    failed_j=int(hashlib.sha256(f'{FAILED_RUN_ID}|{base_sha}|NAV_HELDOUT'.encode()).hexdigest()[:16],16)%len(eligible)
    failed_idx=eligible[failed_j]
    fresh_eligible=[i for i in eligible if i!=failed_idx]
    if not fresh_eligible: raise RuntimeError('NO_FRESH_HELDOUT_AFTER_FAILED_RUN_EXCLUSION')
    j=int(hashlib.sha256(f'{rid}|{recovery_sha}|NAV_HELDOUT_RECOVERY'.encode()).hexdigest()[:16],16)%len(fresh_eligible)
    held_idx=fresh_eligible[j]; held=xs[held_idx]
    dev=[e for i,e in enumerate(xs) if i not in {held_idx,failed_idx}][:120]
    f_order=full_order(); a_order=list(BASE_ORDER)
    full_name,full_cost,full_trace=search(f_order,dev); abl_name,abl_cost,abl_trace=search(a_order,dev)
    if full_name is None or abl_name is None: raise RuntimeError('NO_DEV_EXACT_REPRESENTATION')
    pred=CANDIDATES[full_name](held['input']); wrong_pred=nav_ordered_pose(held['input'],True)
    generated={'kind':'ABSTRACT_ORDERED_STATE_TRANSITION_QUERY','source_operator_sha256':PARENT_OP_SHA,'components':['PERSISTENT_STATE','ORDERED_TRANSFORMS','TERMINAL_QUERY'],'selected_navigation_representation':full_name}; gen_sha=H(generated)
    pre={'selected_index':held_idx,'excluded_failed_run_index':failed_idx,'eligible_count':len(eligible),'fresh_eligible_count':len(fresh_eligible),'input_sha256':hashlib.sha256(held['input'].encode()).hexdigest(),'prediction':pred,'wrong_prediction':wrong_pred,'selected_representation':full_name,'generated_abstraction_sha256':gen_sha,'full_search_cost':full_cost,'ablation_search_cost':abl_cost,'heldout_target_accessed':False,'failed_run_outcome_reused':False,'recovery_parent_state_sha256':recovery_sha}; pre_sha=H(pre)
    gold=target(held); ok=pred==gold; wrong_ok=wrong_pred==gold
    ev={'source_owner':'google','source_repo':'BIG-bench','source_commit':BB_COMMIT,'task_path':'bigbench/benchmark_tasks/navigate/task.json','task_kind':'NAVIGATE','selected_index':held_idx,'excluded_failed_run_index':failed_idx,'eligible_count':len(eligible),'selection_used_heldout_target':False,'failed_run_outcome_reused':False,'heldout_target_hidden_until_after_prediction':True,'prediction_precommit_sha256':pre_sha,'prediction':pred,'target':gold,'success':ok,'wrong_all_turns_first_prediction':wrong_pred,'wrong_all_turns_first_success':wrong_ok,'full_order':f_order,'ablation_order':a_order,'full_selected':full_name,'ablation_selected':abl_name,'full_search_cost':full_cost,'ablation_search_cost':abl_cost,'transfer_prior_cost_reduction':(abl_cost-full_cost)/abl_cost if abl_cost else 0.0,'full_trace':full_trace,'ablation_trace':abl_trace,'generated_abstraction':generated,'generated_abstraction_sha256':gen_sha,'remove_transfer_prior_target_success':CANDIDATES[abl_name](held['input'])==gold,'claim_class':'BOUNDED_DOMAIN_DISJOINT_EXTERNAL_TRANSFER_ACCELERATION'}
    ns=json.loads(json.dumps(st)); ns['epoch_completed']=16
    ns.setdefault('generated_operator_registry',[]).append({'epoch':16,'operator':generated,'operator_sha256':gen_sha,'parent_operator_sha256':PARENT_OP_SHA,'source':'google/BIG-bench navigate','target_consumed_for_generation':False})
    ns.setdefault('synergy_lineage',[]).append({'epoch':16,'github_run_id':rid,'base_parent_state_sha256':base_sha,'recovery_parent_state_sha256':recovery_sha,'source_domain':'GENERIC_POSSESSION_EXCHANGE','target_domain':'NAVIGATE','e3_transfer_root':E3_TRANSFER_RECEIPT_SHA,'full_search_cost':full_cost,'ablation_search_cost':abl_cost,'success':ok,'wrong_control_success':wrong_ok,'generated_abstraction_sha256':gen_sha})
    details=ns.setdefault('synergy_credit_details',{})
    details['TRANSFER_TO_ONTOLOGY_SEARCH_ACCELERATION']={'status':'SUPPORTED_BOUNDED' if ok and full_cost<abl_cost and not wrong_ok else 'NOT_SUPPORTED','full_cost':full_cost,'ablation_cost':abl_cost,'wrong_control_success':wrong_ok,'independent_custody':False,'failed_run_outcome_credit':False}
    ns.setdefault('family_credits',{})['NAVIGATE']=ns.get('family_credits',{}).get('NAVIGATE',0)+(1 if ok else 0)
    if ok and 'NAVIGATE' not in ns.setdefault('solved_family_registry',[]): ns['solved_family_registry'].append('NAVIGATE')
    ns['self_model']['alpha' if ok else 'beta']+=1; ns['state_sha256']=H({k:v for k,v in ns.items() if k!='state_sha256'})
    save(o/'checkpoint_state.json',ns); save(o/'prediction_precommit.json',pre); save(o/'external_evaluation.json',ev)
    rec={'schema':'arte.transfer_to_ontology_search_acceleration_receipt/v57','epoch':16,'github_run_id':rid,'github_sha':os.environ.get('GITHUB_SHA'),'hosted_external':os.environ.get('GITHUB_ACTIONS')=='true','base_parent_state_sha256':base_sha,'recovery_parent_state_sha256':recovery_sha,'child_state_sha256':ns['state_sha256'],'failed_run_id':FAILED_RUN_ID,'failure_type':FAILURE_TYPE,'failure_conditioned':True,'failed_run_outcome_credit':False,'source_domain':'GENERIC_POSSESSION_EXCHANGE','target_domain':'NAVIGATE','source_owner':'google','source_repo':'BIG-bench','source_commit':BB_COMMIT,'excluded_failed_run_index':failed_idx,'heldout_target_hidden_until_after_prediction':True,'full_search_cost':full_cost,'ablation_search_cost':abl_cost,'wrong_control_success':wrong_ok,'actual_success':ok,'generated_abstraction_sha256':gen_sha,'e3_roots':t['e3_roots'],'claim_flags':FLAGS,'timestamp':datetime.now(timezone.utc).isoformat()}; rec['receipt_sha256']=H(rec); save(o/'epoch_receipt.json',rec)
    mf={'files':[],'claim_flags':FLAGS}
    for n in ('checkpoint_state.json','prediction_precommit.json','external_evaluation.json','epoch_receipt.json'): mf['files'].append({'name':n,'sha256':hashlib.sha256((o/n).read_bytes()).hexdigest()})
    save(o/'hash_manifest.json',mf); print(json.dumps(rec,sort_keys=True))
if __name__=='__main__': main(sys.argv[1],sys.argv[2])
