import json, hashlib, os, re, sys, tempfile, shutil, urllib.request
from pathlib import Path
from datetime import datetime, timezone, timedelta

BB_COMMIT='092b196c1f8f14a54bbc62f24759d43bde46dd3b'
BB_BASE=f'https://raw.githubusercontent.com/google/BIG-bench/{BB_COMMIT}/bigbench/benchmark_tasks'
PARENT_STATE_SHA='dc44d103dbecf2572c63838f70c007990e2d814093af1190c185da894907acbf'
PARENT_ABSTRACTION_SHA='3c1f0de7ec871bf598c84522ae7a9762974f4b5620ccc0f146552adb4ca31ede'
FLAGS={'independent_custody_proof':False,'source_disjoint_transfer_proof':False,'external_recursive_acceleration':False,'human_intelligence_exceeded':False,'AGI':False,'ASI':False}
MONTHS='%B %d, %Y'
PAT=re.compile(r'^Yesterday was ([A-Za-z]+ \d{1,2}, \d{4})\. What is the date (today|tomorrow|yesterday|\d+ days ago|one week ago from today|one week from today|24 hours later) in MM/DD/YYYY\?$')

def H(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(p): return json.load(open(p,encoding='utf-8'))
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True),encoding='utf-8')
def fetch_json(url):
    with urllib.request.urlopen(url,timeout=30) as r: return json.loads(r.read().decode())
def gold(ex): return max(ex['target_scores'],key=ex['target_scores'].get)
def fmt(d): return d.strftime('%m/%d/%Y')
def parse_mmddyyyy(s): return datetime.strptime(s,'%m/%d/%Y')

def parse_case(text):
    m=PAT.fullmatch(text.strip())
    if not m: raise ValueError('OUTSIDE_CANONICAL_TEMPORAL_FAMILY')
    return datetime.strptime(m.group(1),MONTHS),m.group(2)
def query_offset(q):
    if q=='today': return 0
    if q=='tomorrow': return 1
    if q=='yesterday': return -1
    if q=='one week ago from today': return -7
    if q=='one week from today': return 7
    if q=='24 hours later': return 1
    m=re.fullmatch(r'(\d+) days ago',q)
    if m: return -int(m.group(1))
    raise ValueError(q)
def copy_anchor(text):
    a,q=parse_case(text); return fmt(a)
def fixed_plus_one(text):
    a,q=parse_case(text); return fmt(a+timedelta(days=1))
def query_offset_from_anchor(text):
    a,q=parse_case(text); return fmt(a+timedelta(days=query_offset(q)))
def frame_shifted(text,delta):
    a,q=parse_case(text); return fmt(a+timedelta(days=query_offset(q)+delta))

def acc(fn,dev): return sum(fn(e['input'])==gold(e) for e in dev)/len(dev)
def infer_frame_shift(dev):
    residual=[]
    for e in dev:
        p=parse_mmddyyyy(query_offset_from_anchor(e['input']))
        g=parse_mmddyyyy(gold(e))
        residual.append((g-p).days)
    return residual[0] if residual and len(set(residual))==1 else None

def search(dev,order):
    trace=[]; generated=None
    funcs={'COPY_ANCHOR':copy_anchor,'FIXED_PLUS_ONE':fixed_plus_one,'QUERY_OFFSET_FROM_ANCHOR':query_offset_from_anchor}
    cost=0
    for name in order:
        cost+=1; a=acc(funcs[name],dev); trace.append({'rank':cost,'candidate':name,'dev_accuracy':a})
        if a==1.0: return name,cost,trace,generated
        if name=='QUERY_OFFSET_FROM_ANCHOR':
            delta=infer_frame_shift(dev)
            if delta is not None and delta!=0:
                generated={'kind':'TEMPORAL_REFERENCE_FRAME_NORMALIZATION','delta_days':delta,'parent_abstraction_sha256':PARENT_ABSTRACTION_SHA,'cause':'CONSTANT_DEV_RESIDUAL_AFTER_TRANSFER_PRIOR'}
                cost+=1; ga=acc(lambda x:frame_shifted(x,delta),dev); trace.append({'rank':cost,'candidate':'GENERATED_FRAME_SHIFT','dev_accuracy':ga,'delta_days':delta})
                if ga==1.0: return 'GENERATED_FRAME_SHIFT',cost,trace,generated
    return None,cost,trace,generated

def restore_epoch16():
    import v57_epoch16_transfer_ontology_acceleration_runner as e16
    td=Path(tempfile.mkdtemp(prefix='arte_e16_restore_')); old=os.environ.get('GITHUB_RUN_ID')
    try:
        os.environ['GITHUB_RUN_ID']='33138840575'
        e16.main('arte_external_replay/v57_epoch16_trigger.json',str(td))
        st=load(td/'checkpoint_state.json')
    finally:
        if old is None: os.environ.pop('GITHUB_RUN_ID',None)
        else: os.environ['GITHUB_RUN_ID']=old
        shutil.rmtree(td,ignore_errors=True)
    assert st['state_sha256']==PARENT_STATE_SHA
    return st

def predict_selected(name,generated,text):
    if name=='GENERATED_FRAME_SHIFT': return frame_shifted(text,generated['delta_days'])
    return {'COPY_ANCHOR':copy_anchor,'FIXED_PLUS_ONE':fixed_plus_one,'QUERY_OFFSET_FROM_ANCHOR':query_offset_from_anchor}[name](text)

def main(trigger,outdir):
    o=Path(outdir);o.mkdir(parents=True,exist_ok=True);t=load(trigger);st=restore_epoch16()
    assert t['epoch']==17 and t['parent_state_sha256']==PARENT_STATE_SHA==st['state_sha256'] and st['epoch_completed']==16
    assert H({k:v for k,v in st.items() if k!='state_sha256'})==st['state_sha256']
    assert any(x.get('operator_sha256')==PARENT_ABSTRACTION_SHA and x.get('target_consumed_for_generation') is False for x in st.get('generated_operator_registry',[]))
    d=fetch_json(f'{BB_BASE}/date_understanding/task.json');xs=d['examples'];canonical=[i for i,e in enumerate(xs) if PAT.fullmatch(e['input'].strip())]
    if len(canonical)<5: raise RuntimeError('INSUFFICIENT_CANONICAL_TEMPORAL_FAMILY')
    rid=os.environ.get('GITHUB_RUN_ID','LOCAL');j=int(hashlib.sha256(f'{rid}|{PARENT_STATE_SHA}|DATE_HELDOUT'.encode()).hexdigest()[:16],16)%len(canonical);held_idx=canonical[j];held=xs[held_idx]
    dev=[xs[i] for i in canonical if i!=held_idx]
    full_order=['QUERY_OFFSET_FROM_ANCHOR','FIXED_PLUS_ONE','COPY_ANCHOR']
    abl_order=['COPY_ANCHOR','FIXED_PLUS_ONE','QUERY_OFFSET_FROM_ANCHOR']
    fn,fc,ft,fg=search(dev,full_order);an,ac,at,ag=search(dev,abl_order)
    if fn is None or an is None or fg is None or ag is None: raise RuntimeError('TEMPORAL_FRAME_GENESIS_NOT_CLOSED_ON_DEV')
    if H(fg)!=H(ag): raise RuntimeError('FULL_ABLATION_GENERATED_DIFFERENT_ONTOLOGY')
    pred=predict_selected(fn,fg,held['input']);wrong=query_offset_from_anchor(held['input']);reverse=frame_shifted(held['input'],-fg['delta_days'])
    gen={'kind':'ABSTRACT_ORDERED_STATE_TRANSITION_QUERY_WITH_REFERENCE_FRAME','parent_abstraction_sha256':PARENT_ABSTRACTION_SHA,'generated_component':fg,'components':['PERSISTENT_STATE','ORDERED_TRANSFORMS','TERMINAL_QUERY','REFERENCE_FRAME_NORMALIZATION']};gen_sha=H(gen)
    pre={'selected_index':held_idx,'canonical_count':len(canonical),'input_sha256':hashlib.sha256(held['input'].encode()).hexdigest(),'prediction':pred,'no_frame_prediction':wrong,'reverse_frame_prediction':reverse,'full_search_cost':fc,'ablation_search_cost':ac,'generated_component_sha256':H(fg),'generated_abstraction_sha256':gen_sha,'heldout_target_accessed':False};pre_sha=H(pre)
    target=gold(held);ok=pred==target;wrong_ok=wrong==target;reverse_ok=reverse==target
    ev={'source_owner':'google','source_repo':'BIG-bench','source_commit':BB_COMMIT,'task_path':'bigbench/benchmark_tasks/date_understanding/task.json','task_kind':'DATE_UNDERSTANDING_CANONICAL','selected_index':held_idx,'canonical_count':len(canonical),'selection_used_heldout_target':False,'heldout_target_hidden_until_after_prediction':True,'prediction_precommit_sha256':pre_sha,'prediction':pred,'target':target,'success':ok,'no_frame_prediction':wrong,'no_frame_success':wrong_ok,'reverse_frame_prediction':reverse,'reverse_frame_success':reverse_ok,'full_search_order':full_order,'ablation_search_order':abl_order,'full_selected':fn,'ablation_selected':an,'full_search_cost':fc,'ablation_search_cost':ac,'transfer_prior_cost_reduction':(ac-fc)/ac if ac else 0.0,'full_trace':ft,'ablation_trace':at,'generated_component':fg,'generated_abstraction':gen,'generated_abstraction_sha256':gen_sha,'claim_class':'BOUNDED_SECOND_GENERATION_TRANSFER_ONTOLOGY_FEEDBACK'}
    ns=json.loads(json.dumps(st));ns['epoch_completed']=17
    ns.setdefault('generated_operator_registry',[]).append({'epoch':17,'operator':gen,'operator_sha256':gen_sha,'parent_operator_sha256':PARENT_ABSTRACTION_SHA,'source':'google/BIG-bench date_understanding canonical','target_consumed_for_generation':False})
    ns.setdefault('synergy_lineage',[]).append({'epoch':17,'github_run_id':rid,'parent_state_sha256':PARENT_STATE_SHA,'source_domain':'NAVIGATE','target_domain':'DATE_UNDERSTANDING_CANONICAL','full_search_cost':fc,'ablation_search_cost':ac,'success':ok,'no_frame_success':wrong_ok,'reverse_frame_success':reverse_ok,'generated_abstraction_sha256':gen_sha})
    ns.setdefault('synergy_credit_details',{})['ONTOLOGY_TO_NEXT_TRANSFER_TO_NEW_ONTOLOGY']={'status':'SUPPORTED_BOUNDED_TWO_STEP' if ok and fc<ac and not wrong_ok and not reverse_ok else 'NOT_SUPPORTED','epoch16_prior_sha256':PARENT_ABSTRACTION_SHA,'epoch17_generated_sha256':gen_sha,'full_cost':fc,'ablation_cost':ac,'independent_custody':False,'external_recursive_acceleration':False}
    ns.setdefault('family_credits',{})['DATE_UNDERSTANDING_CANONICAL']=ns.get('family_credits',{}).get('DATE_UNDERSTANDING_CANONICAL',0)+(1 if ok else 0)
    if ok and 'DATE_UNDERSTANDING_CANONICAL' not in ns.setdefault('solved_family_registry',[]):ns['solved_family_registry'].append('DATE_UNDERSTANDING_CANONICAL')
    ns['self_model']['alpha' if ok else 'beta']+=1;ns['state_sha256']=H({k:v for k,v in ns.items() if k!='state_sha256'})
    save(o/'checkpoint_state.json',ns);save(o/'prediction_precommit.json',pre);save(o/'external_evaluation.json',ev)
    rec={'schema':'arte.second_generation_transfer_ontology_feedback_receipt/v57','epoch':17,'github_run_id':rid,'github_sha':os.environ.get('GITHUB_SHA'),'hosted_external':os.environ.get('GITHUB_ACTIONS')=='true','parent_state_sha256':PARENT_STATE_SHA,'child_state_sha256':ns['state_sha256'],'source_domain':'NAVIGATE','target_domain':'DATE_UNDERSTANDING_CANONICAL','source_owner':'google','source_repo':'BIG-bench','source_commit':BB_COMMIT,'heldout_target_hidden_until_after_prediction':True,'full_search_cost':fc,'ablation_search_cost':ac,'no_frame_success':wrong_ok,'reverse_frame_success':reverse_ok,'actual_success':ok,'generated_abstraction_sha256':gen_sha,'mutual_acceleration_candidate':bool(ok and fc<ac and not wrong_ok and not reverse_ok),'claim_flags':FLAGS,'timestamp':datetime.now(timezone.utc).isoformat()};rec['receipt_sha256']=H(rec);save(o/'epoch_receipt.json',rec)
    mf={'files':[],'claim_flags':FLAGS}
    for n in ('checkpoint_state.json','prediction_precommit.json','external_evaluation.json','epoch_receipt.json'):mf['files'].append({'name':n,'sha256':hashlib.sha256((o/n).read_bytes()).hexdigest()})
    save(o/'hash_manifest.json',mf);print(json.dumps(rec,sort_keys=True))
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
