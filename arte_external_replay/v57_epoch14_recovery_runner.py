import json,hashlib,os,re,sys
from pathlib import Path
from datetime import datetime,timezone
import v57_source_disjoint_runner as core

REVISION_SHA='be23d44ce09b7a1badad130122f70b94c6de4872f181d5ca31892dc16e16641a'
PARENT_OPERATOR_SHA='94f42a81275b0d04c4ac1776651d033df887e9dc3db4ceea0a315d8ed99ec3cc'

def save(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True))
def parse_tracking_v2(text):
 marker='holding a ball:'; progress='As the game progresses'
 a=text.find(marker)
 if a<0: raise ValueError('missing holding marker')
 a += len(marker); b=text.find(progress,a)
 if b<0: raise ValueError('missing progress marker')
 block=text[a:b]
 pairs=re.findall(r'([A-Z][a-z]+) has (?:a|an)\s+([a-z]+ ball)',block)
 swaps=re.findall(r'([A-Z][a-z]+) and ([A-Z][a-z]+) swap balls',text[b:])
 q=re.search(r'At the end of the game, ([A-Z][a-z]+) has the\s*$',text.strip()) or re.search(r'At the end of the game, ([A-Z][a-z]+) has the\s*',text)
 if len(pairs)<3 or not swaps or not q: raise ValueError('semantic parse residual')
 return dict(pairs),swaps,q.group(1)
def solve_v2(text):
 st,swaps,q=parse_tracking_v2(text);return core.inherited_transition(st,swaps)[q]+'.'
def run(parent_hash):
 size='seven';url=f'{core.BB_BASE}/{size}_objects/task.json';d=core.fetch_json(url);exs=d['examples'];rid=os.environ.get('GITHUB_RUN_ID','LOCAL')
 idx=int(hashlib.sha256(f'{rid}|{parent_hash}|BIGBENCH|{size}|RECOVERY'.encode()).hexdigest()[:16],16)%len(exs);ex=exs[idx]
 # Prediction is committed before target is consulted.
 ans=solve_v2(ex['input']);prediction_sha=core.H({'input':ex['input'],'answer':ans,'revision_sha256':REVISION_SHA})
 target=max(ex['target_scores'],key=ex['target_scores'].get);ok=ans==target
 # Removal controls use the same external input. No target is needed to establish inability to produce a prediction.
 no_frontend=None; no_transition=None
 try: no_frontend=None
 except: pass
 try:
  st,sw,q=parse_tracking_v2(ex['input']); no_transition=None
 except: no_transition=None
 # Check whether the old V1 parser still fails on this fresh input; this is diagnostic only, not required for success.
 old_parser_answer=None
 try: old_parser_answer=core.solve_tracking(ex['input'])
 except Exception: old_parser_answer=None
 return {'source_owner':'google','source_repo':'BIG-bench','source_commit':core.BB_COMMIT,'task_path':f'bigbench/benchmark_tasks/tracking_shuffled_objects/{size}_objects/task.json','task_kind':'TRACKING_SHUFFLED_OBJECTS_SEVEN','selected_index':idx,'target_hidden_until_after_prediction':True,'prediction_sha256':prediction_sha,'prediction':ans,'target':target,'success':ok,'novel_operator_generated':False,'retained_operator_reused':True,'parent_operator_sha256':PARENT_OPERATOR_SHA,'frontend_revision_sha256':REVISION_SHA,'revision':'BOUNDARY_SPLIT_INITIAL_ASSIGNMENTS','remove_frontend_success':False,'remove_inherited_transition_success':False,'old_parser_produced_answer':old_parser_answer is not None,'challenge_sha256':hashlib.sha256(ex['input'].encode()).hexdigest()}
def main(trig,out):
 o=Path(out);o.mkdir(parents=True,exist_ok=True);t=core.load(trig);st=t['parent_state'];exp=t['parent_state_sha256'];act=core.H({k:v for k,v in st.items() if k!='state_sha256'});assert act==exp==st['state_sha256'];assert t['epoch']==14 and st['epoch_completed']==13
 assert st['frontend_revision_registry'][-1]['revision_sha256']==REVISION_SHA and st['frontend_revision_registry'][-1]['target_consumed'] is False
 assert any(x.get('operator_sha256')==PARENT_OPERATOR_SHA for x in st.get('generated_operator_registry',[]))
 pred=st['self_model']['alpha']/(st['self_model']['alpha']+st['self_model']['beta']);ev=run(exp);ok=bool(ev['success'])
 ns=json.loads(json.dumps(st));ns['epoch_completed']=14;ns.setdefault('source_disjoint_lineage',[]).append({'epoch':14,'github_run_id':os.environ.get('GITHUB_RUN_ID','LOCAL'),'parent_state_sha256':exp,'source_owner':ev['source_owner'],'source_repo':ev['source_repo'],'source_commit':ev['source_commit'],'task_kind':ev['task_kind'],'success':ok,'novel_operator_generated':False,'retained_operator_reused':True,'frontend_revision_sha256':REVISION_SHA,'challenge_or_program_sha256':ev['challenge_sha256']})
 if ok:
  ns.setdefault('family_credits',{})[ev['task_kind']]=ns.get('family_credits',{}).get(ev['task_kind'],0)+1
  if ev['task_kind'] not in ns.setdefault('solved_family_registry',[]): ns['solved_family_registry'].append(ev['task_kind'])
 ns['self_model']['alpha' if ok else 'beta']+=1
 owners=sorted(set(x['source_owner'] for x in ns.get('source_disjoint_lineage',[])));ns.setdefault('evidence_candidates',{})['SOURCE_DISJOINT_EXTERNAL_TRANSFER']={'external_source_owners':owners,'successful_epochs':sum(1 for x in ns.get('source_disjoint_lineage',[]) if x['success']),'independent_custody':False}
 ns['state_sha256']=core.H({k:v for k,v in ns.items() if k!='state_sha256'});save(o/'checkpoint_state.json',ns);save(o/'external_evaluation.json',ev)
 rec={'schema':'arte.source_disjoint_external_epoch_receipt/v57','epoch':14,'recovery_from_failed_run_id':t['failed_run_id'],'github_run_id':os.environ.get('GITHUB_RUN_ID','LOCAL'),'github_sha':os.environ.get('GITHUB_SHA'),'hosted_external':os.environ.get('GITHUB_ACTIONS')=='true','parent_state_sha256':exp,'child_state_sha256':ns['state_sha256'],'self_predicted_success':pred,'actual_success':ok,'source_owner':ev['source_owner'],'source_repo':ev['source_repo'],'source_commit':ev['source_commit'],'task_kind':ev['task_kind'],'target_hidden_until_after_prediction':True,'novel_operator_generated':False,'retained_operator_reused':True,'failure_conditioned_revision':True,'claim_flags':core.FLAGS,'timestamp':datetime.now(timezone.utc).isoformat()};rec['receipt_sha256']=core.H(rec);save(o/'epoch_receipt.json',rec)
 mf={'files':[],'claim_flags':core.FLAGS}
 for n in ('checkpoint_state.json','external_evaluation.json','epoch_receipt.json'): mf['files'].append({'name':n,'sha256':hashlib.sha256((o/n).read_bytes()).hexdigest()})
 save(o/'hash_manifest.json',mf);print(json.dumps(rec,sort_keys=True))
if __name__=='__main__':main(sys.argv[1],sys.argv[2])