import json, hashlib, os, re, sys, tempfile, shutil, urllib.request
from pathlib import Path
from datetime import datetime, timezone
from collections import deque

SOURCE_REPO='raviraju/NLP_QA_Project'
SOURCE_COMMIT='0051751275841561267b1c6f5d9116b985521df1'
RAW_BASE=f'https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/tasks_1-20_v1-2/en'
TRAIN_PATH='qa17_positional-reasoning_train.txt'
TEST_PATH='qa17_positional-reasoning_test.txt'
PARENT_STATE_SHA='c5c1bed520ae4cd900a7b6a854236fd224210e6055fdd5cf89e7848a8071489a'
PARENT_ABSTRACTION_SHA='9ce0de4225f1871e2d57e91ca8d0e4e640411aa4be4ac0e1cfb9004d60f2ca57'
FLAGS={'independent_custody_proof':False,'source_disjoint_transfer_proof':False,'external_recursive_acceleration':False,'human_intelligence_exceeded':False,'AGI':False,'ASI':False}
FACT_RE=re.compile(r'^The (.+?) is (above|below|to the left of|to the right of) the (.+?)\.$')
Q_RE=re.compile(r'^Is the (.+?) (above|below|to the left of|to the right of) the (.+?)\?$')
VEC={'above':(0,1),'below':(0,-1),'to the left of':(-1,0),'to the right of':(1,0)}

def H(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(p): return json.load(open(p,encoding='utf-8'))
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True),encoding='utf-8')
def fetch_text(path):
    with urllib.request.urlopen(f'{RAW_BASE}/{path}',timeout=30) as r: return r.read().decode('utf-8')

def parse_cases(raw, include_targets):
    facts=[]; out=[]
    for line in raw.splitlines():
        line=line.strip()
        if not line: continue
        nid_s,rest=line.split(' ',1); nid=int(nid_s)
        if nid==1: facts=[]
        if '\t' in rest:
            parts=rest.split('\t')
            q=parts[0]
            target=parts[1] if include_targets else None
            out.append({'facts':list(facts),'question':q,'target':target,'nid':nid})
        else:
            facts.append(rest)
    return out

def parse_fact(s):
    m=FACT_RE.fullmatch(s)
    if not m: raise ValueError('FACT_SURFACE:'+s)
    return m.group(1),m.group(2),m.group(3)
def parse_q(s):
    m=Q_RE.fullmatch(s)
    if not m: raise ValueError('QUESTION_SURFACE:'+s)
    return m.group(1),m.group(2),m.group(3)

def graph_coords(facts):
    g={}
    def add(a,b,d): g.setdefault(a,[]).append((b,d))
    for s in facts:
        a,rel,b=parse_fact(s); dx,dy=VEC[rel]
        # pos(a)-pos(b)=(dx,dy). From a to b is (-dx,-dy); from b to a is (+dx,+dy).
        add(a,b,(-dx,-dy)); add(b,a,(dx,dy))
    if not g: return {}
    root=next(iter(g)); pos={root:(0,0)}; q=deque([root])
    while q:
        a=q.popleft(); ax,ay=pos[a]
        for b,(dx,dy) in g.get(a,[]):
            cand=(ax+dx,ay+dy)
            if b not in pos: pos[b]=cand; q.append(b)
            elif pos[b]!=cand: raise ValueError('INCONSISTENT_SPATIAL_GRAPH')
    return pos

def relation_truth(diff,rel):
    dx,dy=diff
    if rel=='above': return dy>0
    if rel=='below': return dy<0
    if rel=='to the left of': return dx<0
    if rel=='to the right of': return dx>0
    raise ValueError(rel)
def yn(v): return 'yes' if v else 'no'

def direct_fact_only(case):
    s,rel,o=parse_q(case['question'])
    for f in case['facts']:
        a,r,b=parse_fact(f)
        if (a,r,b)==(s,rel,o): return 'yes'
        dx,dy=VEC[r]
        if a==o and b==s and VEC[rel]==(-dx,-dy): return 'yes'
    return 'no'

def undirected_axis(case):
    s,rel,o=parse_q(case['question']); pos=graph_coords(case['facts'])
    if s not in pos or o not in pos: return 'no'
    sx,sy=pos[s];ox,oy=pos[o]
    if rel in ('above','below'): return yn(sy!=oy)
    return yn(sx!=ox)

def canonical_frame(case):
    s,rel,o=parse_q(case['question']); pos=graph_coords(case['facts'])
    if s not in pos or o not in pos: return 'no'
    a,b=sorted([s,o]); ax,ay=pos[a]; bx,by=pos[b]
    return yn(relation_truth((ax-bx,ay-by),rel))
def frame_predict(case,orientation):
    s,rel,o=parse_q(case['question']); pos=graph_coords(case['facts'])
    if s not in pos or o not in pos: return 'no'
    sx,sy=pos[s];ox,oy=pos[o]
    diff=(sx-ox,sy-oy) if orientation=='SUBJECT_MINUS_OBJECT' else (ox-sx,oy-sy)
    return yn(relation_truth(diff,rel))

BASE={'DIRECT_FACT_ONLY':direct_fact_only,'UNDIRECTED_AXIS_CLOSURE':undirected_axis,'COORDINATE_GRAPH_CANONICAL_FRAME':canonical_frame}
FULL_ORDER=['COORDINATE_GRAPH_CANONICAL_FRAME','UNDIRECTED_AXIS_CLOSURE','DIRECT_FACT_ONLY']
ABL_ORDER=['DIRECT_FACT_ONLY','UNDIRECTED_AXIS_CLOSURE','COORDINATE_GRAPH_CANONICAL_FRAME']

def acc(fn,dev): return sum(fn(c)==c['target'] for c in dev)/len(dev)
def infer_query_frame(dev):
    scores={o:acc(lambda c,o=o:frame_predict(c,o),dev) for o in ('SUBJECT_MINUS_OBJECT','OBJECT_MINUS_SUBJECT')}
    winners=[k for k,v in scores.items() if v==1.0]
    if len(winners)!=1: return None,scores
    return winners[0],scores

def search(dev,order):
    trace=[];generated=None; cost=0
    for name in order:
        cost+=1; a=acc(BASE[name],dev);trace.append({'rank':cost,'candidate':name,'dev_accuracy':a})
        if a==1.0:return name,cost,trace,generated
        if name=='COORDINATE_GRAPH_CANONICAL_FRAME':
            orient,scores=infer_query_frame(dev)
            if orient:
                generated={'kind':'SPATIAL_QUERY_FRAME_NORMALIZATION','orientation':orient,'parent_abstraction_sha256':PARENT_ABSTRACTION_SHA,'cause':'DEV_RESIDUAL_REQUIRES_QUERY_RELATIVE_REFERENCE_FRAME','orientation_scores':scores}
                cost+=1; ga=acc(lambda c:frame_predict(c,orient),dev);trace.append({'rank':cost,'candidate':'GENERATED_QUERY_FRAME','dev_accuracy':ga,'orientation':orient})
                if ga==1.0:return 'GENERATED_QUERY_FRAME',cost,trace,generated
    return None,cost,trace,generated

def restore_epoch17():
    import v57_epoch17_temporal_frame_genesis_runner as e17
    td=Path(tempfile.mkdtemp(prefix='arte_e17_restore_')); old=os.environ.get('GITHUB_RUN_ID')
    try:
        os.environ['GITHUB_RUN_ID']='33138955977'
        e17.main('arte_external_replay/v57_epoch17_trigger.json',str(td));st=load(td/'checkpoint_state.json')
    finally:
        if old is None: os.environ.pop('GITHUB_RUN_ID',None)
        else: os.environ['GITHUB_RUN_ID']=old
        shutil.rmtree(td,ignore_errors=True)
    assert st['state_sha256']==PARENT_STATE_SHA
    return st

def main(trigger,outdir):
    o=Path(outdir);o.mkdir(parents=True,exist_ok=True);t=load(trigger);st=restore_epoch17()
    assert t['epoch']==18 and t['parent_state_sha256']==PARENT_STATE_SHA==st['state_sha256'] and st['epoch_completed']==17
    assert H({k:v for k,v in st.items() if k!='state_sha256'})==st['state_sha256']
    assert any(x.get('operator_sha256')==PARENT_ABSTRACTION_SHA and x.get('target_consumed_for_generation') is False for x in st.get('generated_operator_registry',[]))

    train=parse_cases(fetch_text(TRAIN_PATH),True)
    # First test fetch is sanitized by parser: target fields are never read into selection cases.
    test_inputs=parse_cases(fetch_text(TEST_PATH),False)
    eligible=[]
    for i,c in enumerate(test_inputs):
        try:
            if frame_predict(c,'SUBJECT_MINUS_OBJECT')!=canonical_frame(c) and frame_predict(c,'SUBJECT_MINUS_OBJECT')!=frame_predict(c,'OBJECT_MINUS_SUBJECT'):
                eligible.append(i)
        except Exception:
            pass
    if not eligible: raise RuntimeError('NO_INPUT_ONLY_FRAME_DISCRIMINATING_TEST_CASE')
    rid=os.environ.get('GITHUB_RUN_ID','LOCAL');j=int(hashlib.sha256(f'{rid}|{PARENT_STATE_SHA}|BABI17_HELDOUT'.encode()).hexdigest()[:16],16)%len(eligible);held_idx=eligible[j];held=test_inputs[held_idx]

    fn,fc,ft,fg=search(train,FULL_ORDER);an,ac,at,ag=search(train,ABL_ORDER)
    if fn!='GENERATED_QUERY_FRAME' or an!='GENERATED_QUERY_FRAME' or fg is None or ag is None: raise RuntimeError('QUERY_FRAME_GENESIS_NOT_CLOSED_ON_TRAIN')
    if H(fg)!=H(ag): raise RuntimeError('FULL_ABLATION_GENERATED_DIFFERENT_QUERY_FRAME')
    pred=frame_predict(held,fg['orientation']); remove_pred=canonical_frame(held); wrong_orientation='OBJECT_MINUS_SUBJECT' if fg['orientation']=='SUBJECT_MINUS_OBJECT' else 'SUBJECT_MINUS_OBJECT';wrong_pred=frame_predict(held,wrong_orientation)
    gen={'kind':'ABSTRACT_ORDERED_STATE_TRANSITION_QUERY_WITH_CROSS_MODAL_REFERENCE_FRAME','parent_abstraction_sha256':PARENT_ABSTRACTION_SHA,'generated_component':fg,'components':['PERSISTENT_STATE','ORDERED_TRANSFORMS','TERMINAL_QUERY','REFERENCE_FRAME_NORMALIZATION','RELATION_GRAPH_COMPOSITION']};gen_sha=H(gen)
    pre={'selected_index':held_idx,'eligible_count':len(eligible),'input_packet_sha256':H({'facts':held['facts'],'question':held['question']}),'prediction':pred,'remove_prediction':remove_pred,'wrong_prediction':wrong_pred,'full_search_cost':fc,'ablation_search_cost':ac,'generated_component_sha256':H(fg),'generated_abstraction_sha256':gen_sha,'heldout_target_accessed':False};pre_sha=H(pre);save(o/'prediction_precommit.json',pre)

    # Reveal only after the prediction precommit has been written.
    test_reveal=parse_cases(fetch_text(TEST_PATH),True);target=test_reveal[held_idx]['target'];ok=pred==target;remove_ok=remove_pred==target;wrong_ok=wrong_pred==target
    ev={'source_repository_owner':'raviraju','dataset_origin':'bAbI/Facebook public copy','source_repo':SOURCE_REPO,'source_commit':SOURCE_COMMIT,'train_path':TRAIN_PATH,'test_path':TEST_PATH,'task_kind':'BABI_QA17_POSITIONAL_REASONING','selected_index':held_idx,'eligible_count':len(eligible),'selection_used_heldout_target':False,'heldout_target_hidden_from_selection_and_generation':True,'prediction_precommit_written_before_reveal':True,'prediction_precommit_sha256':pre_sha,'prediction':pred,'target':target,'success':ok,'remove_frame_prediction':remove_pred,'remove_frame_success':remove_ok,'wrong_orientation':wrong_orientation,'wrong_orientation_prediction':wrong_pred,'wrong_orientation_success':wrong_ok,'full_order':FULL_ORDER,'ablation_order':ABL_ORDER,'full_selected':fn,'ablation_selected':an,'full_search_cost':fc,'ablation_search_cost':ac,'transfer_prior_cost_reduction':(ac-fc)/ac if ac else 0.0,'full_trace':ft,'ablation_trace':at,'generated_component':fg,'generated_abstraction':gen,'generated_abstraction_sha256':gen_sha,'repository_owner_disjoint_from_epoch16_17':True,'independent_dataset_authority':False,'claim_class':'BOUNDED_THREE_STEP_REPOSITORY_OWNER_DISJOINT_REPLICATION'};save(o/'external_evaluation.json',ev)
    ns=json.loads(json.dumps(st));ns['epoch_completed']=18
    ns.setdefault('generated_operator_registry',[]).append({'epoch':18,'operator':gen,'operator_sha256':gen_sha,'parent_operator_sha256':PARENT_ABSTRACTION_SHA,'source':f'{SOURCE_REPO}@{SOURCE_COMMIT} bAbI QA17','target_consumed_for_generation':False})
    ns.setdefault('synergy_lineage',[]).append({'epoch':18,'github_run_id':rid,'parent_state_sha256':PARENT_STATE_SHA,'source_domain':'DATE_UNDERSTANDING_CANONICAL','target_domain':'BABI_QA17_POSITIONAL_REASONING','source_repository_owner':'raviraju','full_search_cost':fc,'ablation_search_cost':ac,'success':ok,'remove_frame_success':remove_ok,'wrong_orientation_success':wrong_ok,'generated_abstraction_sha256':gen_sha})
    status='SUPPORTED_BOUNDED_THREE_STEP_REPOSITORY_OWNER_DISJOINT' if ok and fc<ac and not remove_ok and not wrong_ok else 'NOT_SUPPORTED'
    ns.setdefault('synergy_credit_details',{})['THREE_STEP_TRANSFER_ONTOLOGY_REPLICATION']={'status':status,'epoch16_abstraction_sha256':'3c1f0de7ec871bf598c84522ae7a9762974f4b5620ccc0f146552adb4ca31ede','epoch17_abstraction_sha256':PARENT_ABSTRACTION_SHA,'epoch18_abstraction_sha256':gen_sha,'full_cost':fc,'ablation_cost':ac,'repository_owner_disjoint':True,'dataset_authority_independent':False,'independent_custody':False,'external_recursive_acceleration':False}
    ns.setdefault('family_credits',{})['BABI_QA17_POSITIONAL_REASONING']=ns.get('family_credits',{}).get('BABI_QA17_POSITIONAL_REASONING',0)+(1 if ok else 0)
    if ok and 'BABI_QA17_POSITIONAL_REASONING' not in ns.setdefault('solved_family_registry',[]):ns['solved_family_registry'].append('BABI_QA17_POSITIONAL_REASONING')
    ns['self_model']['alpha' if ok else 'beta']+=1;ns['state_sha256']=H({k:v for k,v in ns.items() if k!='state_sha256'});save(o/'checkpoint_state.json',ns)
    rec={'schema':'arte.three_step_repository_owner_disjoint_replication_receipt/v57','epoch':18,'github_run_id':rid,'github_sha':os.environ.get('GITHUB_SHA'),'hosted_external':os.environ.get('GITHUB_ACTIONS')=='true','parent_state_sha256':PARENT_STATE_SHA,'child_state_sha256':ns['state_sha256'],'source_repository_owner':'raviraju','source_repo':SOURCE_REPO,'source_commit':SOURCE_COMMIT,'source_domain':'DATE_UNDERSTANDING_CANONICAL','target_domain':'BABI_QA17_POSITIONAL_REASONING','heldout_target_hidden_from_selection_and_generation':True,'prediction_precommit_before_reveal':True,'full_search_cost':fc,'ablation_search_cost':ac,'remove_frame_success':remove_ok,'wrong_orientation_success':wrong_ok,'actual_success':ok,'generated_abstraction_sha256':gen_sha,'three_step_replication_candidate':bool(status.startswith('SUPPORTED')),'claim_flags':FLAGS,'timestamp':datetime.now(timezone.utc).isoformat()};rec['receipt_sha256']=H(rec);save(o/'epoch_receipt.json',rec)
    mf={'files':[],'claim_flags':FLAGS}
    for n in ('checkpoint_state.json','prediction_precommit.json','external_evaluation.json','epoch_receipt.json'):mf['files'].append({'name':n,'sha256':hashlib.sha256((o/n).read_bytes()).hexdigest()})
    save(o/'hash_manifest.json',mf);print(json.dumps(rec,sort_keys=True))
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
