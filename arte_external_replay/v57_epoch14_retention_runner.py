import json,hashlib,os,sys
from pathlib import Path
from datetime import datetime,timezone
import v57_source_disjoint_runner as core

def save(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True))
def main(trig,out):
 o=Path(out);o.mkdir(parents=True,exist_ok=True);t=core.load(trig);st=t['parent_state'];exp=t['parent_state_sha256'];act=core.H({k:v for k,v in st.items() if k!='state_sha256'});assert act==exp==st['state_sha256'];assert t['epoch']==14 and st['epoch_completed']==13
 inherited_hash='94f42a81275b0d04c4ac1776651d033df887e9dc3db4ceea0a315d8ed99ec3cc'
 assert any(x.get('operator_sha256')==inherited_hash for x in st.get('generated_operator_registry',[]))
 pred=st['self_model']['alpha']/(st['self_model']['alpha']+st['self_model']['beta']);ev=core.run_bigbench(exp,'seven');ev['novel_operator_generated']=False;ev['retained_operator_reused']=True;ev['retained_operator_sha256']=inherited_hash;ok=bool(ev['success'])
 ns=json.loads(json.dumps(st));ns['epoch_completed']=14;ns.setdefault('source_disjoint_lineage',[]).append({'epoch':14,'github_run_id':os.environ.get('GITHUB_RUN_ID','LOCAL'),'parent_state_sha256':exp,'source_owner':ev['source_owner'],'source_repo':ev['source_repo'],'source_commit':ev['source_commit'],'task_kind':ev['task_kind'],'success':ok,'novel_operator_generated':False,'retained_operator_reused':True,'challenge_or_program_sha256':ev['challenge_sha256']})
 if ok:
  ns.setdefault('family_credits',{})[ev['task_kind']]=ns.get('family_credits',{}).get(ev['task_kind'],0)+1
  if ev['task_kind'] not in ns.setdefault('solved_family_registry',[]):ns['solved_family_registry'].append(ev['task_kind'])
 ns['self_model']['alpha' if ok else 'beta']+=1
 owners=sorted(set(x['source_owner'] for x in ns.get('source_disjoint_lineage',[])));ns.setdefault('evidence_candidates',{})['SOURCE_DISJOINT_EXTERNAL_TRANSFER']={'external_source_owners':owners,'successful_epochs':sum(1 for x in ns.get('source_disjoint_lineage',[]) if x['success']),'independent_custody':False}
 ns['state_sha256']=core.H({k:v for k,v in ns.items() if k!='state_sha256'});save(o/'checkpoint_state.json',ns);save(o/'external_evaluation.json',ev)
 rec={'schema':'arte.source_disjoint_external_epoch_receipt/v57','epoch':14,'github_run_id':os.environ.get('GITHUB_RUN_ID','LOCAL'),'github_sha':os.environ.get('GITHUB_SHA'),'hosted_external':os.environ.get('GITHUB_ACTIONS')=='true','parent_state_sha256':exp,'child_state_sha256':ns['state_sha256'],'self_predicted_success':pred,'actual_success':ok,'source_owner':ev['source_owner'],'source_repo':ev['source_repo'],'source_commit':ev['source_commit'],'task_kind':ev['task_kind'],'target_hidden_until_after_prediction':True,'novel_operator_generated':False,'retained_operator_reused':True,'claim_flags':core.FLAGS,'timestamp':datetime.now(timezone.utc).isoformat()};rec['receipt_sha256']=core.H(rec);save(o/'epoch_receipt.json',rec)
 mf={'files':[],'claim_flags':core.FLAGS}
 for n in ('checkpoint_state.json','external_evaluation.json','epoch_receipt.json'):mf['files'].append({'name':n,'sha256':hashlib.sha256((o/n).read_bytes()).hexdigest()})
 save(o/'hash_manifest.json',mf);print(json.dumps(rec,sort_keys=True))
if __name__=='__main__':main(sys.argv[1],sys.argv[2])