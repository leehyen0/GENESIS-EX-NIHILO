import hashlib,json,sys
from pathlib import Path
import v57_source_disjoint_runner as core
RUN='33137022834';PARENT='fd7726f43e0265c148a1c66eb11e69a03a37d388ff29c2768603c47d80f06e91'
def main(out):
 d=core.fetch_json(f'{core.BB_BASE}/seven_objects/task.json');xs=d['examples'];idx=int(hashlib.sha256(f'{RUN}|{PARENT}|BIGBENCH|seven'.encode()).hexdigest()[:16],16)%len(xs);text=xs[idx]['input']
 facts={'schema':'arte.target_free_input_surface_probe/v57','failed_run_id':RUN,'selected_index':idx,'target_scores_accessed':False,'input_sha256':hashlib.sha256(text.encode()).hexdigest(),'length':len(text),'contains_holding_a_ball':('holding a ball:' in text),'contains_game_progresses':('As the game progresses' in text),'contains_trade_balls':('trade balls' in text),'contains_swap_balls':('swap balls' in text),'contains_end_phrase':('At the end of the game' in text),'prefix':text[:700]}
 facts['receipt_sha256']=hashlib.sha256(json.dumps(facts,sort_keys=True,separators=(',',':')).encode()).hexdigest();Path(out).write_text(json.dumps(facts,indent=2,sort_keys=True));print(json.dumps(facts,sort_keys=True))
if __name__=='__main__':main(sys.argv[1])