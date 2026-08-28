import ast, copy, csv, hashlib, io, json, os, sys, tempfile, shutil, urllib.request
from pathlib import Path
from datetime import datetime, timezone
import v57_epoch18_recovery_runner as e18r

SOURCE_REPO='kliang5/CLUTRR_huggingface_dataset'
SOURCE_COMMIT='e5b496941e91abb7c319d2618a3ce96752bc4ab7'
SOURCE_CONFIG='gen_train23_test2to10'
RAW_BASE=f'https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/{SOURCE_CONFIG}'
TRAIN_PATH='train.csv'
VAL_PATH='validation.csv'
TEST_PATH='test.csv'
PARENT_STATE_SHA='41b0971dc3e79e30cc7454a656224a86ace56287294a84f8ada8f378882d5b68'
PARENT_ABSTRACTION_SHA='0f8c310b6c945dcd0e5bc5047a74cffe522fe905de1f4b2e05cf827393ad9eb0'
FLAGS={'independent_custody_proof':False,'independent_evaluator':False,'external_recursive_acceleration':False,'human_intelligence_exceeded':False,'AGI':False,'ASI':False,'natural_language_clutrr_solution':False}

def H(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(p): return json.load(open(p,encoding='utf-8'))
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True),encoding='utf-8')
def fetch_text(path):
    with urllib.request.urlopen(f'{RAW_BASE}/{path}',timeout=60) as r: return r.read().decode('utf-8')

def parse_edges(s):
    v=ast.literal_eval(s)
    if not isinstance(v,(list,tuple)) or not v: raise ValueError('BAD_EDGE_TYPES')
    return [str(x) for x in v]

def parse_rows(raw,include_targets):
    out=[]
    rd=csv.reader(io.StringIO(raw))
    for i,row in enumerate(rd):
        if i==0: continue
        if len(row)<15: continue
        try: edges=parse_edges(row[11])
        except Exception: continue
        out.append({
            'id':row[1],
            'query':row[3],
            'edges':edges,
            'query_edge':row[12],
            'task_name':row[9],
            'target':row[5] if include_targets else None
        })
    return out

def build_binary_table(train):
    table={}; conflicts=set(); counts={}
    for c in train:
        if len(c['edges'])!=2 or c['target'] is None: continue
        k=tuple(c['edges']); v=c['target']; counts[k]=counts.get(k,0)+1
        if k in table and table[k]!=v: conflicts.add(k)
        else: table[k]=v
    for k in conflicts: table.pop(k,None)
    return table,{'unique_pairs':len(table),'conflicting_pairs':len(conflicts),'observations':sum(counts.values())}

def compose(edges,table,orientation='FORWARD'):
    seq=list(edges if orientation=='FORWARD' else reversed(edges))
    if not seq: return None
    state=seq[0]
    for nxt in seq[1:]:
        k=(state,nxt)
        if k not in table: return None
        state=table[k]
    return state

def last_edge(c): return c['edges'][-1]
def first_edge(c): return c['edges'][0]
def forward(c,table): return compose(c['edges'],table,'FORWARD')
def reverse(c,table): return compose(c['edges'],table,'REVERSE')

def accuracy(fn,dev):
    scored=[]
    for c in dev:
        p=fn(c)
        if p is not None: scored.append(p==c['target'])
    return (sum(scored)/len(scored) if scored else 0.0),len(scored)

def search(dev,table,order,gate=0.999999):
    trace=[]; cost=0
    funcs={
      'ORDERED_BINARY_CLOSURE':lambda c:forward(c,table),
      'REVERSED_BINARY_CLOSURE':lambda c:reverse(c,table),
      'LAST_EDGE_ONLY':last_edge,
      'FIRST_EDGE_ONLY':first_edge
    }
    for name in order:
        cost+=1; a,n=accuracy(funcs[name],dev);trace.append({'rank':cost,'candidate':name,'dev_accuracy':a,'scored_cases':n})
        if n>0 and a>=gate:return name,cost,trace
    # If no exact candidate exists, preserve failure instead of silently relaxing the gate.
    return None,cost,trace

FULL_ORDER=['ORDERED_BINARY_CLOSURE','REVERSED_BINARY_CLOSURE','LAST_EDGE_ONLY','FIRST_EDGE_ONLY']
ABL_ORDER=['LAST_EDGE_ONLY','FIRST_EDGE_ONLY','REVERSED_BINARY_CLOSURE','ORDERED_BINARY_CLOSURE']

def restore_epoch18():
    td=Path(tempfile.mkdtemp(prefix='arte_e18_restore_'));old=os.environ.get('GITHUB_RUN_ID')
    try:
        os.environ['GITHUB_RUN_ID']='33140447809'
        e18r.main('arte_external_replay/v57_epoch18_recovery_trigger.json',str(td))
        st=load(td/'checkpoint_state.json')
    finally:
        if old is None: os.environ.pop('GITHUB_RUN_ID',None)
        else: os.environ['GITHUB_RUN_ID']=old
        shutil.rmtree(td,ignore_errors=True)
    assert st['state_sha256']==PARENT_STATE_SHA
    assert H({k:v for k,v in st.items() if k!='state_sha256'})==PARENT_STATE_SHA
    return st

def main(trigger,outdir):
    o=Path(outdir);o.mkdir(parents=True,exist_ok=True);t=load(trigger);st=restore_epoch18()
    assert t['epoch']==19 and t['parent_state_sha256']==PARENT_STATE_SHA==st['state_sha256'] and st['epoch_completed']==18
    assert t['inherited_abstraction_sha256']==PARENT_ABSTRACTION_SHA
    assert any(x.get('operator_sha256')==PARENT_ABSTRACTION_SHA and x.get('target_consumed_for_generation') is False for x in st.get('generated_operator_registry',[]))

    train=parse_rows(fetch_text(TRAIN_PATH),True)
    val=parse_rows(fetch_text(VAL_PATH),True)
    table,table_stats=build_binary_table(train)
    if not table: raise RuntimeError('NO_NONCONFLICTING_BINARY_RELATION_LAWS')

    # Validation pressure is length-3 only. Applicability is decided without looking at validation targets.
    dev=[c for c in val if len(c['edges'])==3 and forward(c,table) is not None]
    if len(dev)<10: raise RuntimeError(f'INSUFFICIENT_LENGTH3_COMPOSITION_DEV::{len(dev)}')
    fn,fc,ft=search(dev,table,FULL_ORDER);an,ac,at=search(dev,table,ABL_ORDER)
    if fn!='ORDERED_BINARY_CLOSURE' or an!='ORDERED_BINARY_CLOSURE':
        raise RuntimeError('ORDERED_BINARY_COMPOSITION_NOT_EXACT_ON_LENGTH3_DEV::'+json.dumps({'full':ft,'ablation':at},sort_keys=True))

    generated={'kind':'GENERATED_KINSHIP_BINARY_RELATION_COMPOSITION','law_source':'CLUTRR_LENGTH2_TRAIN','orientation':'FORWARD_QUERY_PATH','binary_table_sha256':H({f'{a}|{b}':v for (a,b),v in sorted(table.items())}),'binary_table_size':len(table),'parent_abstraction_sha256':PARENT_ABSTRACTION_SHA,'cause':'TRANSFERRED_RELATION_GRAPH_PRIOR_PLUS_LENGTH3_VALIDATION_RESIDUAL'}
    gen={'kind':'ABSTRACT_ORDERED_RELATION_GRAPH_QUERY_FRAME_WITH_GENERATED_BINARY_COMPOSITION','parent_abstraction_sha256':PARENT_ABSTRACTION_SHA,'generated_component':generated,'components':['RELATION_GRAPH_COMPOSITION','QUERY_FRAME_NORMALIZATION','BINARY_LAW_INDUCTION','RECURSIVE_APPLICATION']};gen_sha=H(gen)

    # First test pass strips target columns from selection objects.
    test_inputs=parse_rows(fetch_text(TEST_PATH),False)
    eligible=[]
    for i,c in enumerate(test_inputs):
        if len(c['edges'])<4: continue
        p=forward(c,table); w=reverse(c,table); rem=last_edge(c)
        if p is not None and p!=w and p!=rem: eligible.append(i)
    if not eligible: raise RuntimeError('NO_INPUT_ONLY_LENGTH4PLUS_DISCRIMINATING_CASE')
    rid=os.environ.get('GITHUB_RUN_ID','LOCAL')
    j=int(hashlib.sha256(f'{rid}|{PARENT_STATE_SHA}|CLUTRR19_HELDOUT'.encode()).hexdigest()[:16],16)%len(eligible)
    held_idx=eligible[j];held=test_inputs[held_idx]
    pred=forward(held,table);remove_pred=last_edge(held);wrong_pred=reverse(held,table)
    pre={'selected_index':held_idx,'eligible_count':len(eligible),'input_packet_sha256':H({'id':held['id'],'query':held['query'],'edges':held['edges'],'query_edge':held['query_edge'],'task_name':held['task_name']}),'prediction':pred,'remove_prediction':remove_pred,'wrong_prediction':wrong_pred,'full_search_cost':fc,'ablation_search_cost':ac,'generated_abstraction_sha256':gen_sha,'target_loaded_into_selection_case':False};pre_sha=H(pre);save(o/'prediction_precommit.json',pre)

    test_reveal=parse_rows(fetch_text(TEST_PATH),True);target=test_reveal[held_idx]['target'];ok=pred==target;remove_ok=remove_pred==target;wrong_ok=wrong_pred==target
    ev={'source_repository_owner':'kliang5','benchmark_origin':'CLUTRR / facebookresearch','source_repo':SOURCE_REPO,'source_commit':SOURCE_COMMIT,'source_config':SOURCE_CONFIG,'task_kind':'CLUTRR_STRUCTURED_KINSHIP_RELATION_COMPOSITION','selected_index':held_idx,'eligible_count':len(eligible),'heldout_relation_length':len(held['edges']),'selection_used_target':False,'target_excluded_from_selection_case_objects':True,'prediction_precommit_written_before_target_reveal_parse':True,'prediction_precommit_sha256':pre_sha,'prediction':pred,'target':target,'success':ok,'remove_prediction':remove_pred,'remove_success':remove_ok,'wrong_prediction':wrong_pred,'wrong_success':wrong_ok,'binary_table_stats':table_stats,'length3_dev_cases':len(dev),'full_order':FULL_ORDER,'ablation_order':ABL_ORDER,'full_selected':fn,'ablation_selected':an,'full_search_cost':fc,'ablation_search_cost':ac,'transfer_prior_cost_reduction':(ac-fc)/ac if ac else 0.0,'full_trace':ft,'ablation_trace':at,'generated_component':generated,'generated_abstraction':gen,'generated_abstraction_sha256':gen_sha,'repository_owner_disjoint_from_epoch16_18':True,'benchmark_origin_owner_new_vs_epoch16_18':True,'independent_evaluator':False,'natural_language_surface_used_for_solver':False,'claim_class':'BOUNDED_FOUR_STEP_THIRD_SOURCE_OWNER_STRUCTURED_RELATION_REPLICATION'};save(o/'external_evaluation.json',ev)

    ns=copy.deepcopy(st);ns['epoch_completed']=19
    ns.setdefault('generated_operator_registry',[]).append({'epoch':19,'operator':gen,'operator_sha256':gen_sha,'parent_operator_sha256':PARENT_ABSTRACTION_SHA,'source':f'{SOURCE_REPO}@{SOURCE_COMMIT} {SOURCE_CONFIG}','target_consumed_for_generation':False})
    ns.setdefault('synergy_lineage',[]).append({'epoch':19,'github_run_id':rid,'parent_state_sha256':PARENT_STATE_SHA,'source_domain':'BABI_QA17_POSITIONAL_REASONING','target_domain':'CLUTRR_STRUCTURED_KINSHIP_RELATION_COMPOSITION','source_repository_owner':'kliang5','full_search_cost':fc,'ablation_search_cost':ac,'success':ok,'remove_success':remove_ok,'wrong_success':wrong_ok,'generated_abstraction_sha256':gen_sha})
    status='SUPPORTED_BOUNDED_FOUR_STEP_THIRD_SOURCE_OWNER_STRUCTURED_RELATION' if ok and fc<ac and not remove_ok and not wrong_ok else 'NOT_SUPPORTED'
    ns.setdefault('synergy_credit_details',{})['FOUR_STEP_TRANSFER_ONTOLOGY_REPLICATION']={'status':status,'epoch18_abstraction_sha256':PARENT_ABSTRACTION_SHA,'epoch19_abstraction_sha256':gen_sha,'full_cost':fc,'ablation_cost':ac,'repository_owner_disjoint':True,'benchmark_origin_owner_new':True,'structured_relation_channel_only':True,'natural_language_solution':False,'independent_evaluator':False,'external_recursive_acceleration':False}
    ns.setdefault('family_credits',{})['CLUTRR_STRUCTURED_KINSHIP_RELATION_COMPOSITION']=ns.get('family_credits',{}).get('CLUTRR_STRUCTURED_KINSHIP_RELATION_COMPOSITION',0)+(1 if ok else 0)
    if ok and 'CLUTRR_STRUCTURED_KINSHIP_RELATION_COMPOSITION' not in ns.setdefault('solved_family_registry',[]):ns['solved_family_registry'].append('CLUTRR_STRUCTURED_KINSHIP_RELATION_COMPOSITION')
    ns['self_model']['alpha' if ok else 'beta']+=1
    ns['state_sha256']=H({k:v for k,v in ns.items() if k!='state_sha256'});save(o/'checkpoint_state.json',ns)
    rec={'schema':'arte.four_step_third_source_owner_structured_relation_receipt/v57','epoch':19,'github_run_id':rid,'github_sha':os.environ.get('GITHUB_SHA'),'hosted_external':os.environ.get('GITHUB_ACTIONS')=='true','parent_state_sha256':PARENT_STATE_SHA,'child_state_sha256':ns['state_sha256'],'source_repository_owner':'kliang5','benchmark_origin':'CLUTRR / facebookresearch','source_repo':SOURCE_REPO,'source_commit':SOURCE_COMMIT,'source_domain':'BABI_QA17_POSITIONAL_REASONING','target_domain':'CLUTRR_STRUCTURED_KINSHIP_RELATION_COMPOSITION','selection_used_target':False,'prediction_precommit_before_target_reveal_parse':True,'heldout_relation_length':len(held['edges']),'full_search_cost':fc,'ablation_search_cost':ac,'remove_success':remove_ok,'wrong_success':wrong_ok,'actual_success':ok,'generated_abstraction_sha256':gen_sha,'four_step_replication_candidate':bool(status.startswith('SUPPORTED')),'claim_flags':FLAGS,'timestamp':datetime.now(timezone.utc).isoformat()};rec['receipt_sha256']=H(rec);save(o/'epoch_receipt.json',rec)
    mf={'files':[],'claim_flags':FLAGS}
    for n in ('checkpoint_state.json','prediction_precommit.json','external_evaluation.json','epoch_receipt.json'):mf['files'].append({'name':n,'sha256':hashlib.sha256((o/n).read_bytes()).hexdigest()})
    save(o/'hash_manifest.json',mf);print(json.dumps(rec,sort_keys=True))

if __name__=='__main__': main(sys.argv[1],sys.argv[2])
