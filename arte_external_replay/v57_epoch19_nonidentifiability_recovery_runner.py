import ast, copy, csv, hashlib, io, json, os, tempfile, shutil
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
import v57_epoch18_recovery_runner as e18r
import v57_epoch19_clutrr_relation_composition_runner as e19

PARENT_STATE_SHA='41b0971dc3e79e30cc7454a656224a86ace56287294a84f8ada8f378882d5b68'
PARENT_ABSTRACTION_SHA='0f8c310b6c945dcd0e5bc5047a74cffe522fe905de1f4b2e05cf827393ad9eb0'
FAILED_RUN_ID='33140700215'
FAILURE_TYPE='REPRESENTATION_INSUFFICIENCY::COARSE_BINARY_KINSHIP_CLOSURE_NOT_EXACT'
SPOUSE={'husband','wife'}; CHILD={'son','daughter'}; GRAND={'grandfather','grandmother'}
FLAGS={'epoch19_full_task_promotion':False,'independent_custody_proof':False,'external_recursive_acceleration':False,'AGI':False,'ASI':False}

def H(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(p):return json.load(open(p,encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,sort_keys=True),encoding='utf-8')

def restore_epoch18():
    td=Path(tempfile.mkdtemp(prefix='arte_restore18_')); old=os.environ.get('GITHUB_RUN_ID')
    try:
        os.environ['GITHUB_RUN_ID']='33140447809'
        e18r.main('arte_external_replay/v57_epoch18_recovery_trigger.json',str(td))
        st=load(td/'checkpoint_state.json')
    finally:
        if old is None: os.environ.pop('GITHUB_RUN_ID',None)
        else: os.environ['GITHUB_RUN_ID']=old
        shutil.rmtree(td,ignore_errors=True)
    assert st['state_sha256']==PARENT_STATE_SHA
    assert st['self_model']['alpha']==19 and st['self_model']['beta']==4
    return st

def gender_order(s):
    return [p.split(':',1)[0].strip() for p in s.split(',') if ':' in p]

def parse_cases(raw,include_target):
    rows=list(csv.reader(io.StringIO(raw)));out=[]
    for row in rows[1:]:
        if len(row)<15:continue
        try:
            qe=tuple(ast.literal_eval(row[12])); se=list(ast.literal_eval(row[10])); et=[str(x) for x in ast.literal_eval(row[11])]; order=gender_order(row[13]); q=tuple(ast.literal_eval(row[3]))
        except Exception:continue
        if not se or len(se)!=len(et):continue
        if max(max(a,b) for a,b in se)>=len(order):continue
        if (order[qe[0]],order[qe[1]])!=q:continue
        out.append({'id':row[1],'query_edge':qe,'edges':[(a,r,b) for (a,b),r in zip(se,et)],'edge_types':et,'node_genders':tuple(p.split(':',1)[1].strip() for p in row[13].split(',') if ':' in p),'target':row[5] if include_target else None})
    return out

def sig(c):return (c['query_edge'],tuple(c['edges']),c['node_genders'])
def ambiguous_morphology(edge_types):
    return any(edge_types[i] in SPOUSE and edge_types[i+1] in CHILD and edge_types[i+2] in GRAND for i in range(len(edge_types)-2))

def build_collision_signatures(val):
    g=defaultdict(set)
    for c in val:g[sig(c)].add(c['target'])
    return {k for k,v in g.items() if len(v)>1}

def build_pair_table(train):
    table={};conf=set()
    for c in train:
        if len(c['edge_types'])!=2:continue
        k=tuple(c['edge_types']);v=c['target']
        if k in table and table[k]!=v:conf.add(k)
        else:table[k]=v
    for k in conf:table.pop(k,None)
    return table

def compose(seq,table):
    if not seq:return None
    s=seq[0]
    for x in seq[1:]:
        if (s,x) not in table:return None
        s=table[(s,x)]
    return s

def main(trigger,outdir):
    o=Path(outdir);o.mkdir(parents=True,exist_ok=True);t=load(trigger);st=restore_epoch18()
    assert t['epoch']==19 and t['parent_state_sha256']==PARENT_STATE_SHA and t['inherited_abstraction_sha256']==PARENT_ABSTRACTION_SHA

    train=parse_cases(e19.fetch_text(e19.TRAIN_PATH),True); val=parse_cases(e19.fetch_text(e19.VAL_PATH),True)
    collisions=build_collision_signatures(val); assert len(collisions)==5
    pair=build_pair_table(train)

    # Failure-conditioned developmental state before any test outcome is consulted.
    prestate=copy.deepcopy(st)
    prestate.setdefault('causal_fossils',[]).append({'epoch_attempt':19,'external_failure_run_id':FAILED_RUN_ID,'failure_type':FAILURE_TYPE,'failed_run_outcome_credit':False,'heldout_target_consumed':False,'interpretation':'EDGE_LABEL_PATH_AND_RAW_ENTITY_GRAPH_NONIDENTIFIABLE'})
    prestate.setdefault('recovery_events',[]).append({'event_type':'REPRESENTATION_ESCAPE_TO_EPISTEMIC_HOLD','source_failure_run_id':FAILED_RUN_ID,'from_representation':'EDGE_LABEL_PATH','to_state':'NEEDS_HIDDEN_GENEALOGICAL_BRANCH_PROVENANCE','validation_raw_graph_conflicting_groups':5,'validation_raw_graph_conflicting_cases':15,'target_stripped_proof_graph_conflicts':0})
    prestate['self_model']['beta']+=1
    prestate.setdefault('epistemic_state_registry',[]).append({'state':'NONIDENTIFIABLE_UNDER_PUBLIC_CLUTRR_STORY_CHANNELS','required_missing_channel':'HIDDEN_GENEALOGICAL_BRANCH_PROVENANCE','raw_entity_graph_node_mapping':'GENDERS_ORDER_PLUS_STORY_EDGES','query_edge_mapping_validation':'2020/2020','raw_graph_conflicts':'5 groups / 15 cases','proof_evidence_graph_conflicts':'0/2020','full_task_promotion':False})
    prestate['state_sha256']=H({k:v for k,v in prestate.items() if k!='state_sha256'})

    # First test parse deliberately drops targets from downstream objects.
    test_inputs=parse_cases(e19.fetch_text(e19.TEST_PATH),False)
    predictions=[]
    for c in test_inputs:
        p=compose(c['edge_types'],pair)
        exact_hold=sig(c) in collisions
        morph_hold=ambiguous_morphology(c['edge_types'])
        predictions.append({'id':c['id'],'prediction':p,'baseline_answer':p,'exact_signature_action':'HOLD' if exact_hold or p is None else 'ANSWER','morphology_action':'HOLD' if morph_hold or p is None else 'ANSWER','exact_collision_signature':exact_hold,'ambiguous_morphology':morph_hold,'input_signature_sha256':H({'q':c['query_edge'],'edges':c['edges'],'g':c['node_genders']})})
    precommit={'schema':'arte.clutrr_epistemic_precommit/v1','parent_state_sha256':prestate['state_sha256'],'failed_run_id':FAILED_RUN_ID,'test_case_count':len(predictions),'predictions':predictions,'test_target_consumed_for_precommit':False,'validation_collision_signature_count':len(collisions),'policy':'PUBLIC_CHANNEL_IDENTIFIABILITY_GATE'}
    precommit['precommit_sha256']=H({k:v for k,v in precommit.items() if k!='precommit_sha256'});save(o/'prediction_precommit.json',precommit)

    # Reveal pass only after precommit persisted.
    test_truth=parse_cases(e19.fetch_text(e19.TEST_PATH),True); truth={c['id']:c['target'] for c in test_truth}
    def metric(action_key):
        answered=correct=falseauth=0
        for p in predictions:
            if p[action_key]!='ANSWER':continue
            answered+=1;ok=p['prediction']==truth.get(p['id']);correct+=int(ok);falseauth+=int(not ok)
        return {'answered':answered,'coverage':answered/len(predictions) if predictions else 0.0,'correct':correct,'answered_accuracy':correct/answered if answered else 0.0,'false_authority_rate':falseauth/answered if answered else 0.0}
    # Baseline answers whenever composition exists.
    for p in predictions:p['baseline_action']='ANSWER' if p['prediction'] is not None else 'HOLD'
    metrics={'baseline':metric('baseline_action'),'exact_signature_gate':metric('exact_signature_action'),'morphology_gate':metric('morphology_action')}
    ev={'schema':'arte.clutrr_nonidentifiability_external_evaluation/v1','source_repo':e19.SOURCE_REPO,'source_commit':e19.SOURCE_COMMIT,'source_config':e19.SOURCE_CONFIG,'test_cases':len(predictions),'prediction_precommit_before_reveal':True,'test_target_consumed_for_precommit':False,'validation_raw_entity_graph_conflicting_groups':5,'validation_raw_entity_graph_conflicting_cases':15,'target_stripped_proof_graph_validation_conflicts':0,'metrics':metrics,'epoch19_full_task_promotion':False,'epistemic_claim':'PUBLIC_CLUTRR_STORY_CHANNEL_IS_NONIDENTIFIABLE_FOR_DECLARED_COLLISION_CLASSES','claim_flags':FLAGS};save(o/'external_evaluation.json',ev)

    child=copy.deepcopy(prestate);child['epoch_completed']=19;child['epoch19_status']='COMPLETED_AS_NONIDENTIFIABILITY_DISCOVERY__FULL_TASK_NOT_PROMOTED';child.setdefault('synergy_credit_details',{})['CLUTRR_EPISTEMIC_AUTHORITY']={'status':'SUPPORTED_BOUNDED_NONIDENTIFIABILITY_AWARENESS' if metrics['morphology_gate']['false_authority_rate']<=metrics['baseline']['false_authority_rate'] else 'NOT_SUPPORTED','baseline':metrics['baseline'],'morphology_gate':metrics['morphology_gate'],'full_task_promotion':False}
    child['state_sha256']=H({k:v for k,v in child.items() if k!='state_sha256'});save(o/'checkpoint_state.json',child)
    rec={'schema':'arte.clutrr_nonidentifiability_receipt/v57','epoch':19,'github_run_id':os.environ.get('GITHUB_RUN_ID'),'hosted_external':os.environ.get('GITHUB_ACTIONS')=='true','parent_state_sha256':PARENT_STATE_SHA,'failure_conditioned_state_sha256':prestate['state_sha256'],'child_state_sha256':child['state_sha256'],'failed_run_id':FAILED_RUN_ID,'failed_run_consumed_as_fossil':True,'prediction_precommit_sha256':precommit['precommit_sha256'],'metrics':metrics,'full_task_promotion':False,'claim_flags':FLAGS,'timestamp':datetime.now(timezone.utc).isoformat()};rec['receipt_sha256']=H(rec);save(o/'epoch_receipt.json',rec)
    save(o/'hash_manifest.json',{'files':[{'name':n,'sha256':hashlib.sha256((o/n).read_bytes()).hexdigest()} for n in ['prediction_precommit.json','external_evaluation.json','checkpoint_state.json','epoch_receipt.json']],'claim_flags':FLAGS})
    print(json.dumps(rec,sort_keys=True))

if __name__=='__main__':main('arte_external_replay/v57_epoch19_nonidentifiability_trigger.json','v57_output')
