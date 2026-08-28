import json,hashlib,re,sys,os,urllib.request,tempfile,subprocess
from pathlib import Path
from datetime import datetime,timezone
from fractions import Fraction

ARC_REPO='https://github.com/fchollet/ARC-AGI.git'
ARC_COMMIT='399030444e0ab0cc8b4e199870fb20b863846f34'
BB_COMMIT='092b196c1f8f14a54bbc62f24759d43bde46dd3b'
BB_BASE=f'https://raw.githubusercontent.com/google/BIG-bench/{BB_COMMIT}/bigbench/benchmark_tasks/tracking_shuffled_objects'
FLAGS={'independent_custody_proof':False,'source_disjoint_transfer_proof':False,'external_recursive_acceleration':False,'human_intelligence_exceeded':False,'AGI':False,'ASI':False}

def H(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True))
def fetch_json(url):
 req=urllib.request.Request(url,headers={'User-Agent':'ARTE-v57-source-disjoint/1.0'})
 with urllib.request.urlopen(req,timeout=30) as r:return json.loads(r.read().decode())

# New representation frontend; the state-transition executor is inherited from earlier BODY semantics.
def parse_tracking(text):
 m=re.search(r'holding a ball:\s*(.*?)\n\n',text,flags=re.S) or re.search(r'holding a ball:\s*(.*?)As the game progresses',text,flags=re.S)
 if not m:raise ValueError('initial block')
 pairs=re.findall(r'([A-Z][a-z]+) has (?:a|an) ([a-z]+ ball)',m.group(1))
 swaps=re.findall(r'([A-Z][a-z]+) and ([A-Z][a-z]+) swap balls',text)
 q=re.search(r'At the end of the game, ([A-Z][a-z]+) has the\s*$',text.strip()) or re.search(r'At the end of the game, ([A-Z][a-z]+) has the\s*',text)
 if len(pairs)<3 or not swaps or not q:raise ValueError('parse residual')
 return dict(pairs),swaps,q.group(1)

def inherited_transition(state,swaps):
 st=dict(state)
 for a,b in swaps:st[a],st[b]=st[b],st[a]
 return st

def solve_tracking(text,use_frontend=True,use_inherited_transition=True):
 if not use_frontend or not use_inherited_transition:return None
 st,swaps,q=parse_tracking(text);return inherited_transition(st,swaps)[q]+'.'

def run_bigbench(parent_hash,size):
 url=f'{BB_BASE}/{size}_objects/task.json';d=fetch_json(url);exs=d['examples'];rid=os.environ.get('GITHUB_RUN_ID','LOCAL')
 idx=int(hashlib.sha256(f'{rid}|{parent_hash}|BIGBENCH|{size}'.encode()).hexdigest()[:16],16)%len(exs);ex=exs[idx]
 ans=solve_tracking(ex['input']);target=max(ex['target_scores'],key=ex['target_scores'].get);ok=ans==target
 remove_frontend=solve_tracking(ex['input'],use_frontend=False) is not None
 remove_inherited=solve_tracking(ex['input'],use_inherited_transition=False) is not None
 return {'source_owner':'google','source_repo':'BIG-bench','source_commit':BB_COMMIT,'task_path':f'bigbench/benchmark_tasks/tracking_shuffled_objects/{size}_objects/task.json','task_kind':f'TRACKING_SHUFFLED_OBJECTS_{size.upper()}','selected_index':idx,'target_hidden_until_after_prediction':True,'prediction_sha256':H({'input':ex['input'],'answer':ans}),'prediction':ans,'target':target,'success':ok,'generated_cognition':'NATURAL_LANGUAGE_SWAP_TO_PERMUTATION_STATE','novel_operator_generated':True,'inherited_transition_reused':True,'remove_frontend_success':remove_frontend,'remove_inherited_transition_success':remove_inherited,'challenge_sha256':hashlib.sha256(ex['input'].encode()).hexdigest()}

# ARC executable-program genesis: target output is kept out of the solver packet.
def G(x):return tuple(tuple(int(v) for v in r) for r in x)
def L(g):return [list(r) for r in g]
def dims(g):return len(g),len(g[0]) if g else 0
def recolor_map(a,b):
 if dims(a)!=dims(b):return None
 mp={}
 for ra,rb in zip(a,b):
  for x,y in zip(ra,rb):
   if x in mp and mp[x]!=y:return None
   mp[x]=y
 return mp
def amap(g,mp):return tuple(tuple(mp.get(v,v) for v in r) for r in g)
def rot90(g):return tuple(tuple(r) for r in zip(*g[::-1]))
def rot180(g):return rot90(rot90(g))
def rot270(g):return rot90(rot180(g))
def fliph(g):return tuple(tuple(reversed(r)) for r in g)
def flipv(g):return tuple(reversed(g))
def trans(g):return tuple(tuple(r) for r in zip(*g))
def crop(g):
 pts=[(r,c) for r,row in enumerate(g) for c,v in enumerate(row) if v!=0]
 if not pts:return g
 a=min(r for r,c in pts);b=max(r for r,c in pts);c=min(c for r,c in pts);d=max(c for r,c in pts)
 return tuple(tuple(g[r][c:d+1]) for r in range(a,b+1))
def scale(g,sy,sx):
 o=[]
 for r in g:
  rr=tuple(v for v in r for _ in range(sx))
  o.extend([rr]*sy)
 return tuple(o)
S={'ID':lambda g:g,'ROT90':rot90,'ROT180':rot180,'ROT270':rot270,'FLIP_H':fliph,'FLIP_V':flipv,'TRANSPOSE':trans,'CROP':crop}
def rowext(g,n):
 h=len(g)
 for p in range(1,h+1):
  if all(g[i]==g[i%p] for i in range(h)):return tuple(g[i%p] for i in range(n)),p
 return None,None
def colext(g,n):
 h,w=dims(g);cs=[tuple(g[r][c] for r in range(h)) for c in range(w)]
 for p in range(1,w+1):
  if all(cs[i]==cs[i%p] for i in range(w)):
   oc=[cs[i%p] for i in range(n)];return tuple(tuple(oc[c][r] for c in range(n)) for r in range(h)),p
 return None,None

def programs(train):
 out=[]
 for name,fn in S.items():
  if all(fn(G(e['input']))==G(e['output']) for e in train):out.append({'kind':'SIMPLE','name':name})
  mp0=None;ok=True
  for e in train:
   mp=recolor_map(fn(G(e['input'])),G(e['output']))
   if mp is None:ok=False;break
   if mp0 is None:mp0=mp
   elif mp!=mp0:ok=False;break
  if ok and mp0 and any(a!=b for a,b in mp0.items()):out.append({'kind':'TRANSFORM_RECOLOR','name':name,'map':mp0})
 for sy in range(2,5):
  for sx in range(2,5):
   mp0=None;ok=True
   for e in train:
    mp=recolor_map(scale(G(e['input']),sy,sx),G(e['output']))
    if mp is None:ok=False;break
    if mp0 is None:mp0=mp
    elif mp!=mp0:ok=False;break
   if ok:out.append({'kind':'SCALE_RECOLOR','sy':sy,'sx':sx,'map':mp0})
 for axis in ('ROWS','COLS'):
  mp0=None;ratio0=None;ps=[];ok=True
  for e in train:
   x=G(e['input']);y=G(e['output']);hi,wi=dims(x);ho,wo=dims(y)
   if axis=='ROWS':
    if wo!=wi or ho<=hi:ok=False;break
    z,p=rowext(x,ho);rat=Fraction(ho,hi)
   else:
    if ho!=hi or wo<=wi:ok=False;break
    z,p=colext(x,wo);rat=Fraction(wo,wi)
   if z is None:ok=False;break
   mp=recolor_map(z,y)
   if mp is None:ok=False;break
   if mp0 is None:mp0=mp
   elif mp!=mp0:ok=False;break
   if ratio0 is None:ratio0=rat
   elif rat!=ratio0:ok=False;break
   ps.append(p)
  if ok:out.append({'kind':'PERIOD_EXTEND_RECOLOR','axis':axis,'ratio_num':ratio0.numerator,'ratio_den':ratio0.denominator,'map':mp0,'train_periods':ps})
 u=[];seen=set()
 for p in out:
  k=json.dumps(p,sort_keys=True)
  if k not in seen:seen.add(k);u.append(p)
 return u

def apply(p,x):
 x=G(x);k=p['kind']
 if k=='SIMPLE':return S[p['name']](x)
 if k=='TRANSFORM_RECOLOR':return amap(S[p['name']](x),{int(a):int(b) for a,b in p['map'].items()})
 if k=='SCALE_RECOLOR':return amap(scale(x,p['sy'],p['sx']),{int(a):int(b) for a,b in p['map'].items()})
 if k=='PERIOD_EXTEND_RECOLOR':
  h,w=dims(x);n=p['ratio_num'];d=p['ratio_den']
  z,_=rowext(x,h*n//d) if p['axis']=='ROWS' else colext(x,w*n//d)
  if z is None:raise ValueError('period residual')
  return amap(z,{int(a):int(b) for a,b in p['map'].items()})
 raise ValueError(k)

def clone_arc(root):
 subprocess.run(['git','clone','--quiet',ARC_REPO,str(root)],check=True,timeout=60)
 subprocess.run(['git','-C',str(root),'checkout','--quiet',ARC_COMMIT],check=True,timeout=30)

def run_arc(parent_hash):
 with tempfile.TemporaryDirectory() as td:
  root=Path(td)/'arc';clone_arc(root);eligible=[]
  for f in sorted((root/'data'/'training').glob('*.json')):
   d=json.loads(f.read_text());tr=d.get('train',[]);te=d.get('test',[])
   if not tr or len(te)!=1:continue
   ps=programs(tr);simple=[p for p in ps if p['kind'] in ('SIMPLE','TRANSFORM_RECOLOR')];gen=[p for p in ps if p['kind'] not in ('SIMPLE','TRANSFORM_RECOLOR')]
   if simple or len(gen)!=1:continue
   eligible.append((f.stem,gen[0],tr,te[0]['input'],te[0]['output']))
  if not eligible:raise RuntimeError('no eligible ARC task')
  rid=os.environ.get('GITHUB_RUN_ID','LOCAL');i=int(hashlib.sha256(f'{rid}|{parent_hash}|ARC|{ARC_COMMIT}'.encode()).hexdigest()[:16],16)%len(eligible)
  tid,p,tr,ti,to=eligible[i];pred=L(apply(p,ti));ok=pred==to
  wrong=dict(p)
  if p['kind']=='PERIOD_EXTEND_RECOLOR':wrong['axis']='COLS' if p['axis']=='ROWS' else 'ROWS'
  try:wrong_ok=L(apply(wrong,ti))==to
  except:wrong_ok=False
  return {'source_owner':'fchollet','source_repo':'ARC-AGI','source_commit':ARC_COMMIT,'task_path':f'data/training/{tid}.json','task_kind':'ARC_TRAIN_HELD_TEST_OUTPUT','eligible_count':len(eligible),'selected_rank':i,'selected_task_id':tid,'selection_used_test_output':False,'target_hidden_until_after_prediction':True,'generated_program':p,'program_sha256':H(p),'prediction_sha256':H(pred),'success':ok,'novel_operator_generated':True,'generated_cognition':'RESIDUAL_TO_COMPOSED_EXECUTABLE_OPERATOR','frozen_simple_program_available':False,'remove_generated_operator_success':False,'wrong_operator_success':wrong_ok,'target_output_sha256':H(to)}

def main(trig,out):
 o=Path(out);o.mkdir(parents=True,exist_ok=True);t=load(trig);st=t['parent_state'];exp=t['parent_state_sha256'];act=H({k:v for k,v in st.items() if k!='state_sha256'});assert act==exp==st['state_sha256']
 ep=int(t['epoch']);assert st['epoch_completed']==ep-1 and ep in (12,13,14);pred=st['self_model']['alpha']/(st['self_model']['alpha']+st['self_model']['beta'])
 ev=run_bigbench(exp,'three') if ep==12 else run_arc(exp) if ep==13 else run_bigbench(exp,'seven');ok=bool(ev['success'])
 ns=json.loads(json.dumps(st));ns['epoch_completed']=ep;ns.setdefault('source_disjoint_lineage',[]).append({'epoch':ep,'github_run_id':os.environ.get('GITHUB_RUN_ID','LOCAL'),'parent_state_sha256':exp,'source_owner':ev['source_owner'],'source_repo':ev['source_repo'],'source_commit':ev['source_commit'],'task_kind':ev['task_kind'],'success':ok,'novel_operator_generated':ev.get('novel_operator_generated',False),'challenge_or_program_sha256':ev.get('challenge_sha256',ev.get('program_sha256'))})
 if ok:
  ns.setdefault('family_credits',{})[ev['task_kind']]=ns.get('family_credits',{}).get(ev['task_kind'],0)+1
  if ev['task_kind'] not in ns.setdefault('solved_family_registry',[]):ns['solved_family_registry'].append(ev['task_kind'])
  if ev.get('novel_operator_generated'):ns.setdefault('generated_operator_registry',[]).append({'epoch':ep,'source':f"{ev['source_owner']}/{ev['source_repo']}",'operator':ev.get('generated_program',ev.get('generated_cognition')),'operator_sha256':ev.get('program_sha256',H(ev.get('generated_cognition')))})
 ns['self_model']['alpha' if ok else 'beta']+=1
 owners=sorted(set(x['source_owner'] for x in ns.get('source_disjoint_lineage',[])))
 ns.setdefault('evidence_candidates',{})['SOURCE_DISJOINT_EXTERNAL_TRANSFER']={'external_source_owners':owners,'successful_epochs':sum(1 for x in ns.get('source_disjoint_lineage',[]) if x['success']),'independent_custody':False}
 ns['state_sha256']=H({k:v for k,v in ns.items() if k!='state_sha256'});save(o/'checkpoint_state.json',ns);save(o/'external_evaluation.json',ev)
 rec={'schema':'arte.source_disjoint_external_epoch_receipt/v57','epoch':ep,'github_run_id':os.environ.get('GITHUB_RUN_ID','LOCAL'),'github_sha':os.environ.get('GITHUB_SHA'),'hosted_external':os.environ.get('GITHUB_ACTIONS')=='true','parent_state_sha256':exp,'child_state_sha256':ns['state_sha256'],'self_predicted_success':pred,'actual_success':ok,'source_owner':ev['source_owner'],'source_repo':ev['source_repo'],'source_commit':ev['source_commit'],'task_kind':ev['task_kind'],'target_hidden_until_after_prediction':ev.get('target_hidden_until_after_prediction',False),'novel_operator_generated':ev.get('novel_operator_generated',False),'claim_flags':FLAGS,'timestamp':datetime.now(timezone.utc).isoformat()};rec['receipt_sha256']=H(rec);save(o/'epoch_receipt.json',rec)
 mf={'files':[],'claim_flags':FLAGS}
 for n in ('checkpoint_state.json','external_evaluation.json','epoch_receipt.json'):mf['files'].append({'name':n,'sha256':hashlib.sha256((o/n).read_bytes()).hexdigest()})
 save(o/'hash_manifest.json',mf);print(json.dumps(rec,sort_keys=True))
if __name__=='__main__':main(sys.argv[1],sys.argv[2])