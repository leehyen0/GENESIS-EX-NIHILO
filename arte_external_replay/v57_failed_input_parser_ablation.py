import hashlib,json,sys,re
from pathlib import Path
import v57_source_disjoint_runner as core
FAILED_RUN_ID='33137022834'
FAILED_PARENT='fd7726f43e0265c148a1c66eb11e69a03a37d388ff29c2768603c47d80f06e91'
def parse_v2(text):
 marker='holding a ball:'; progress='As the game progresses';a=text.find(marker)
 if a<0:return None
 a+=len(marker);b=text.find(progress,a)
 if b<0:return None
 block=text[a:b];pairs=re.findall(r'([A-Z][a-z]+) has (?:a|an)\s+([a-z]+ ball)',block);swaps=re.findall(r'([A-Z][a-z]+) and ([A-Z][a-z]+) swap balls',text[b:]);q=re.search(r'At the end of the game, ([A-Z][a-z]+) has the\s*$',text.strip()) or re.search(r'At the end of the game, ([A-Z][a-z]+) has the\s*',text)
 return {'pair_count':len(pairs),'swap_count':len(swaps),'query_present':q is not None} if len(pairs)>=3 and swaps and q else None
def main(out):
 d=core.fetch_json(f'{core.BB_BASE}/seven_objects/task.json');exs=d['examples'];idx=int(hashlib.sha256(f'{FAILED_RUN_ID}|{FAILED_PARENT}|BIGBENCH|seven'.encode()).hexdigest()[:16],16)%len(exs);text=exs[idx]['input']
 old_ok=True;old_err=None
 try: core.parse_tracking(text)
 except Exception as e: old_ok=False;old_err=f'{type(e).__name__}:{e}'
 v2=parse_v2(text)
 # Deliberately never read target_scores.
 r={'schema':'arte.target_free_parser_failure_replay/v57','failed_run_id':FAILED_RUN_ID,'failed_parent_state_sha256':FAILED_PARENT,'selected_index':idx,'input_sha256':hashlib.sha256(text.encode()).hexdigest(),'target_scores_accessed':False,'old_parser_success':old_ok,'old_parser_error':old_err,'revised_parser_success':v2 is not None,'revised_parse_summary':v2,'causal_parser_repair_supported':(not old_ok and v2 is not None)}
 r['receipt_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest();Path(out).write_text(json.dumps(r,indent=2,sort_keys=True));print(json.dumps(r,sort_keys=True))
if __name__=='__main__':main(sys.argv[1])