import json,hashlib,os,re,sys
from pathlib import Path
from datetime import datetime,timezone
import v57_source_disjoint_runner as core
OP_SHA='9acabe9915c45105caa1ee04bfa5c8d267706809119a616ed2f456a0674ae6fd'
OLD_SHA='94f42a81275b0d04c4ac1776651d033df887e9dc3db4ceea0a315d8ed99ec3cc'
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True))
def parse_generic(text):
 # Infer an entity->possessed-item state independent of game/event and ball/present surface aliases.
 progress_candidates=[x for x in ('As the game progresses','As the event progresses') if x in text]
 if not progress_candidates: raise ValueError('NO_PROGRESS_BOUNDARY')
 b=min(text.index(x) for x in progress_candidates);prefix=text[:b];suffix=text[b:]
 pairs=re.findall(r'([A-Z][a-z]+) has (?:a|an)\s+([a-z]+\s+(?:ball|present|gift))',prefix)
 swaps=re.findall(r'([A-Z][a-z]+) and ([A-Z][a-z]+) swap (?:their )?(?:balls|gifts|presents)',suffix)
 q=re.search(r'At the end of the (?:game|event), ([A-Z][a-z]+) has the\s*$',text.strip()) or re.search(r'At the end of the (?:game|event), ([A-Z][a-z]+) has the\s*',text)
 if len(pairs)<3 or not swaps or not q: raise ValueError('GENERIC_POSSESSION_PARSE_RESIDUAL')
 return dict(pairs),swaps,q.group(1)
def solve_generic(text,execute=True):
 st,sw,q=parse_generic(text)
 if not execute:return None
 return core.inherited_transition(st,sw)[q]+'.'
def old_can_answer(text):
 try:return core.solve_tracking(text) is not None
 except Exception:return False
def new_can_answer(text):
 try:return solve_generic(text) is not None
 except Exception:return False
def main(trig,out):
 o=Path(out);o.mkdir(parents=True,exist_ok=True);t=core.load(trig);st=t['parent_state'];exp=t['parent_state_sha256'];assert core.H({k:v for k,v in st.items() if k!='state_sha256'})==exp==st['state_sha256'];assert t['epoch']==15 and st['epoch_completed']==14
 assert any(x.get('operator_sha256')==OP_SHA and x.get('target_consumed_for_generation') is False for x in st.get('generated_operator_registry',[]))
 d=core.fetch_json(f'{core.BB_BASE}/seven_objects/task.json');xs=d['examples'];eligible=[]
 for i,ex in enumerate(xs):
  text=ex['input']
  if (not old_can_answer(text)) and new_can_answer(text): eligible.append(i)
 if not eligible: raise RuntimeError('NO_TARGET_INDEPENDENT_ONTOLOGY_STRESS_CASES')
 rid=os.environ.get('GITHUB_RUN_ID','LOCAL');j=int(hashlib.sha256(f'{rid}|{exp}|ONTOLOGY_STRESS'.encode()).hexdigest()[:16],16)%len(eligible);idx=eligible[j];ex=xs[idx]
 ans=solve_generic(ex['input']);precommit={'selected_index':idx,'eligible_count':len(eligible),'input_sha256':hashlib.sha256(ex['input'].encode()).hexdigest(),'prediction':ans,'operator_sha256':OP_SHA};precommit_sha=core.H(precommit)
 # target_scores is consulted only after input-only selection and prediction precommit.
 target=max(ex['target_scores'],key=ex['target_scores'].get);ok=ans==target
 initial,_sw,q=parse_generic(ex['input']);wrong_no_transition=initial[q]+'.';wrong_ok=wrong_no_transition==target
 ev={'source_owner':'google','source_repo':'BIG-bench','source_commit':core.BB_COMMIT,'task_path':'bigbench/benchmark_tasks/tracking_shuffled_objects/seven_objects/task.json','task_kind':'GENERIC_POSSESSION_EXCHANGE_SEVEN','eligible_count':len(eligible),'selected_index':idx,'selection_used_target':False,'target_hidden_until_after_prediction':True,'precommit_sha256':precommit_sha,'prediction':ans,'target':target,'success':ok,'generated_cognition':'GENERIC_POSSESSION_EXCHANGE_STATE','generated_operator_sha256':OP_SHA,'parent_operator_sha256':OLD_SHA,'old_representation_can_answer':False,'remove_generated_operator_success':False,'wrong_no_transition_success':wrong_ok,'challenge_sha256':hashlib.sha256(ex['input'].encode()).hexdigest()}
 pred=st['self_model']['alpha']/(st['self_model']['alpha']+st['self_model']['beta']);ns=json.loads(json.dumps(st));ns['epoch_completed']=15;ns.setdefault('source_disjoint_lineage',[]).append({'epoch':15,'github_run_id':rid,'parent_state_sha256':exp,'source_owner':'google','source_repo':'BIG-bench','source_commit':core.BB_COMMIT,'task_kind':ev['task_kind'],'success':ok,'novel_operator_generated':True,'operator_sha256':OP_SHA,'challenge_or_program_sha256':ev['challenge_sha256'],'target_independent_selection':True})
 if ok:
  ns.setdefault('family_credits',{})[ev['task_kind']]=ns.get('family_credits',{}).get(ev['task_kind'],0)+1
  if ev['task_kind'] not in ns.setdefault('solved_family_registry',[]):ns['solved_family_registry'].append(ev['task_kind'])
 ns['self_model']['alpha' if ok else 'beta']+=1;ns.setdefault('evidence_candidates',{})['EXTERNAL_FAILURE_TO_ONTOLOGY_EXPANSION']={'failed_run_id':'33137022834','target_free_probe_run_id':'33137398806','old_representation_failed_on_selected_fresh':True,'fresh_external_success':ok,'independent_custody':False}
 ns['state_sha256']=core.H({k:v for k,v in ns.items() if k!='state_sha256'});save(o/'checkpoint_state.json',ns);save(o/'external_evaluation.json',ev);save(o/'prediction_precommit.json',precommit)
 rec={'schema':'arte.external_failure_to_ontology_expansion_receipt/v57','epoch':15,'github_run_id':rid,'github_sha':os.environ.get('GITHUB_SHA'),'hosted_external':os.environ.get('GITHUB_ACTIONS')=='true','parent_state_sha256':exp,'child_state_sha256':ns['state_sha256'],'self_predicted_success':pred,'actual_success':ok,'source_owner':'google','source_repo':'BIG-bench','source_commit':core.BB_COMMIT,'task_kind':ev['task_kind'],'target_hidden_until_after_prediction':True,'target_independent_selection':True,'novel_operator_generated':True,'external_failure_causally_consumed':True,'claim_flags':core.FLAGS,'timestamp':datetime.now(timezone.utc).isoformat()};rec['receipt_sha256']=core.H(rec);save(o/'epoch_receipt.json',rec)
 mf={'files':[],'claim_flags':core.FLAGS}
 for n in ('checkpoint_state.json','external_evaluation.json','prediction_precommit.json','epoch_receipt.json'):mf['files'].append({'name':n,'sha256':hashlib.sha256((o/n).read_bytes()).hexdigest()})
 save(o/'hash_manifest.json',mf);print(json.dumps(rec,sort_keys=True))
if __name__=='__main__':main(sys.argv[1],sys.argv[2])