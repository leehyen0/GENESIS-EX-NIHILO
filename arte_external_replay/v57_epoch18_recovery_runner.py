import copy, hashlib, json, sys
from pathlib import Path
import v57_epoch18_spatial_query_frame_runner as base

BASE_PARENT_SHA='c5c1bed5ce34f2fd60f819d9b05fc97f9f03277234ed2f939a82aef6e886dcb9'
ACTUAL_ABSTRACTION_SHA='e5045e0d3b894911d0c20e12a7dfb0c051f3f8fc5e185abe8c6d34521175b74a'
RECOVERY_PARENT_SHA='8fa92abf907b3c5c237077849b3b59eb530e701c53a5a61fdbbeba14bfad96d3'
FAILED_RUN_ID='33140176077'
FAILURE_TYPE='PARENT_RESTORE_EXPECTATION_DRIFT::STALE_EXPECTED_SHA'
STALE_PARENT_SHA='c5c1bed520ae4cd900a7b6a854236fd224210e6055fdd5cf89e7848a8071489a'
STALE_ABSTRACTION_SHA='9ce0de4225f1871e2d57e91ca8d0e4e640411aa4be4ac0e1cfb9004d60f2ca57'

def H(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def load(p):
    return json.load(open(p,encoding='utf-8'))

def save(p,x):
    Path(p).write_text(json.dumps(x,indent=2,sort_keys=True),encoding='utf-8')

def build_recovery_parent():
    # Reconstruct the canonical epoch17 BODY first, using its historical hosted run.
    base.PARENT_STATE_SHA=BASE_PARENT_SHA
    base.PARENT_ABSTRACTION_SHA=ACTUAL_ABSTRACTION_SHA
    st=base.restore_epoch17()
    assert st['state_sha256']==BASE_PARENT_SHA
    assert H({k:v for k,v in st.items() if k!='state_sha256'})==BASE_PARENT_SHA
    assert st['epoch_completed']==17
    assert st['self_model']['alpha']==18 and st['self_model']['beta']==3

    ns=copy.deepcopy(st)
    ns.setdefault('causal_fossils',[]).append({
        'epoch_attempt':18,
        'external_failure_run_id':FAILED_RUN_ID,
        'failed_run_outcome_credit':False,
        'failure_type':FAILURE_TYPE,
        'failure_layer':'PARENT_RESTORE',
        'from_state_sha256':BASE_PARENT_SHA,
        'mutation_scope':'EXPECTED_PARENT_AND_ABSTRACTION_BINDING_ONLY',
        'heldout_selected':False,
        'target_consumed':False,
        'target_or_prediction_reused':False,
        'stale_expected_parent_state_sha256':STALE_PARENT_SHA,
        'actual_parent_state_sha256':BASE_PARENT_SHA,
        'stale_expected_abstraction_sha256':STALE_ABSTRACTION_SHA,
        'actual_abstraction_sha256':ACTUAL_ABSTRACTION_SHA
    })
    ns.setdefault('recovery_events',[]).append({
        'event_type':'DELAYED_EXTERNAL_FAILURE_BACKPRESSURE',
        'external_failure_run_id':FAILED_RUN_ID,
        'failed_run_outcome_credit':False,
        'failure_type':FAILURE_TYPE,
        'failure_layer':'PARENT_RESTORE',
        'from_state_sha256':BASE_PARENT_SHA,
        'self_model_update':'BETA_PLUS_ONE',
        'heldout_selected':False,
        'target_consumed':False
    })
    ns['self_model']['beta']+=1
    ns['state_sha256']=H({k:v for k,v in ns.items() if k!='state_sha256'})
    assert ns['state_sha256']==RECOVERY_PARENT_SHA
    assert ns['self_model']['alpha']==18 and ns['self_model']['beta']==4
    return ns

def main(trigger,outdir):
    t=load(trigger)
    assert t['epoch']==18
    assert t['base_parent_state_sha256']==BASE_PARENT_SHA
    assert t['parent_state_sha256']==RECOVERY_PARENT_SHA
    assert t['inherited_abstraction_sha256']==ACTUAL_ABSTRACTION_SHA
    assert t['failed_run_id']==FAILED_RUN_ID
    assert t['failed_run_heldout_selected'] is False
    assert t['failed_run_target_consumed'] is False

    recovery=build_recovery_parent()

    # The new challenge is selected only from this failure-conditioned recovery state.
    base.PARENT_STATE_SHA=RECOVERY_PARENT_SHA
    base.PARENT_ABSTRACTION_SHA=ACTUAL_ABSTRACTION_SHA
    original_restore=base.restore_epoch17
    base.restore_epoch17=lambda: copy.deepcopy(recovery)
    try:
        base.main(trigger,outdir)
    finally:
        base.restore_epoch17=original_restore

    o=Path(outdir)
    ev=load(o/'external_evaluation.json')
    ev.update({
        'base_parent_state_sha256':BASE_PARENT_SHA,
        'recovery_parent_state_sha256':RECOVERY_PARENT_SHA,
        'failure_conditioned':True,
        'failed_run_id':FAILED_RUN_ID,
        'failed_run_outcome_credit':False,
        'failed_run_heldout_selected':False,
        'failed_run_target_consumed':False,
        'failure_type':FAILURE_TYPE
    })
    save(o/'external_evaluation.json',ev)

    rec=load(o/'epoch_receipt.json')
    rec.update({
        'schema':'arte.three_step_repository_owner_disjoint_recovery_receipt/v57',
        'base_parent_state_sha256':BASE_PARENT_SHA,
        'recovery_parent_state_sha256':RECOVERY_PARENT_SHA,
        'failure_conditioned':True,
        'failure_causally_consumed':True,
        'failed_run_id':FAILED_RUN_ID,
        'failed_run_outcome_credit':False,
        'failed_run_heldout_selected':False,
        'failed_run_target_consumed':False,
        'failure_type':FAILURE_TYPE
    })
    rec.pop('receipt_sha256',None)
    rec['receipt_sha256']=H(rec)
    save(o/'epoch_receipt.json',rec)

    st=load(o/'checkpoint_state.json')
    assert st['epoch_completed']==18
    assert st['self_model']['alpha']==19 and st['self_model']['beta']==4
    assert any(x.get('external_failure_run_id')==FAILED_RUN_ID and x.get('failure_type')==FAILURE_TYPE for x in st.get('causal_fossils',[]))
    assert any(x.get('external_failure_run_id')==FAILED_RUN_ID and x.get('self_model_update')=='BETA_PLUS_ONE' for x in st.get('recovery_events',[]))

    mf={'files':[],'claim_flags':base.FLAGS}
    for n in ('checkpoint_state.json','prediction_precommit.json','external_evaluation.json','epoch_receipt.json'):
        mf['files'].append({'name':n,'sha256':hashlib.sha256((o/n).read_bytes()).hexdigest()})
    save(o/'hash_manifest.json',mf)
    print(json.dumps(rec,sort_keys=True))

if __name__=='__main__':
    main(sys.argv[1],sys.argv[2])
