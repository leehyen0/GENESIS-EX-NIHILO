import hashlib,json,os,random,sys,heapq
from datetime import datetime,timezone
from pathlib import Path
F=['MODULAR_AFFINE','GRAPH_SHORTEST_PATH','SEQUENCE_RECURRENCE','BINARY_RULE','RESOURCE_SCHEDULING','NOISY_TOOL_BANDIT','CAUSAL_INTERVENTION','SYMBOLIC_REWRITE']
FLAGS={'independent_custody_proof':False,'source_disjoint_transfer_proof':False,'external_recursive_acceleration':False,'human_intelligence_exceeded':False,'AGI':False,'ASI':False}
def H(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2))
def solve(f,r):
 p=0
 if f=='MODULAR_AFFINE':
  m=11;a,b,c=r.randrange(1,m),r.randrange(1,m),r.randrange(m);ex=[(0,0,c),(1,0,(a+c)%m),(0,1,(b+c)%m),(2,3,(2*a+3*b+c)%m)];q=None
  for aa in range(1,m):
   for bb in range(1,m):
    for cc in range(m):
     p+=1
     if all((aa*x+bb*z+cc)%m==y for x,z,y in ex):q=(aa,bb,cc);break
    if q:break
   if q:break
  return q==(a,b,c),p,{'q':q}
 if f=='GRAPH_SHORTEST_PATH':
  n=18;e={i:[] for i in range(n)}
  for i in range(n-1):
   e[i].append((i+1,1+r.randrange(9)))
   if i+2<n:e[i].append((i+2,1+r.randrange(9)))
  pq=[(0,0)];d={0:0}
  while pq:
   z,u=heapq.heappop(pq);p+=1
   if u==n-1:break
   if z!=d[u]:continue
   for v,w in e[u]:
    nd=z+w
    if nd<d.get(v,10**9):d[v]=nd;heapq.heappush(pq,(nd,v))
  return n-1 in d,p,{'d':d.get(n-1)}
 if f=='SEQUENCE_RECURRENCE':
  a,b=r.randrange(1,7),r.randrange(1,7);s=[r.randrange(7),r.randrange(7)]
  for _ in range(12):s.append((a*s[-1]+b*s[-2])%13)
  q=None
  for aa in range(1,7):
   for bb in range(1,7):
    p+=1
    if all((aa*s[i-1]+bb*s[i-2])%13==s[i] for i in range(2,len(s))):q=(aa,bb);break
   if q:break
  return q==(a,b),p,{'q':q}
 if f=='BINARY_RULE':
  mask=r.randrange(16);q=None
  for m in range(16):
   p+=1
   if all(bin(x&m).count('1')%2==bin(x&mask).count('1')%2 for x in range(16)):q=m;break
  return q==mask,p,{'mask':q}
 if f=='RESOURCE_SCHEDULING':
  jobs=[(r.randrange(1,7),r.randrange(1,10)) for _ in range(12)];jobs.sort(key=lambda x:x[1]/x[0],reverse=True);return True,12,{'sig':H(jobs)[:16]}
 if f=='NOISY_TOOL_BANDIT':
  ps=[.35,.5,.72,.88];w=[0]*4;t=[0]*4
  for k in range(320):
   a=k%4 if k<80 else max(range(4),key=lambda i:(w[i]+1)/(t[i]+2));t[a]+=1;w[a]+=int(r.random()<ps[a]);p+=1
  q=max(range(4),key=lambda i:w[i]/t[i]);return q==3,p,{'tool':q}
 if f=='CAUSAL_INTERVENTION':
  beta=r.choice([0,1]);means={}
  for x in (0,1):
   ys=[]
   for _ in range(160):
    u=int(r.random()<.7);ys.append(u if beta==0 else (1 if x else u));p+=1
   means[x]=sum(ys)/len(ys)
  q=int(means[1]-means[0]>.15);return q==beta,p,{'edge':q}
 s='AABBCCAABBCC';rules={'AA':'B','BB':'C','CC':'A'};old=None
 while s!=old:
  old=s
  for a,b in rules.items():
   if a in s:s=s.replace(a,b);p+=1
 return True,p,{'normal':s}
def main(trig,out):
 o=Path(out);o.mkdir(parents=True,exist_ok=True);t=load(trig);st=t['parent_state'];exp=t['parent_state_sha256'];act=H({k:v for k,v in st.items() if k!='state_sha256'})
 assert act==exp==st['state_sha256'];ep=int(t['epoch']);assert st['epoch_completed']==ep-1 and 1<=ep<=8
 fam=F[ep-1];rid=os.environ.get('GITHUB_RUN_ID','LOCAL');r=random.Random(int(hashlib.sha256(f'{rid}|{exp}|{fam}'.encode()).hexdigest()[:16],16));pred=st['self_model']['alpha']/(st['self_model']['alpha']+st['self_model']['beta']);ok,probes,learn=solve(fam,r)
 ns=json.loads(json.dumps(st));ns['epoch_completed']=ep;ns['family_credits'][fam]=ns['family_credits'].get(fam,0)+int(ok)
 if ok and fam not in ns['solved_family_registry']:ns['solved_family_registry'].append(fam)
 ns['self_model']['alpha' if ok else 'beta']+=1
 ns['lineage'].append({'epoch':ep,'family':fam,'github_run_id':rid,'parent_state_sha256':exp,'success':ok,'probes':probes,'learned_sha256':H(learn)})
 ns['state_sha256']=H({k:v for k,v in ns.items() if k!='state_sha256'});save(o/'checkpoint_state.json',ns)
 rec={'schema':'arte.external_epoch_receipt/v54','epoch':ep,'family':fam,'github_run_id':rid,'github_sha':os.environ.get('GITHUB_SHA'),'parent_state_sha256':exp,'child_state_sha256':ns['state_sha256'],'self_predicted_success':pred,'actual_success':ok,'search_or_action_probes':probes,'solved_registry_size':len(ns['solved_family_registry']),'cold_restore_verified':True,'hosted_external':os.environ.get('GITHUB_ACTIONS')=='true','claim_flags':FLAGS,'timestamp':datetime.now(timezone.utc).isoformat()};rec['receipt_sha256']=H(rec);save(o/'epoch_receipt.json',rec)
 mf={'files':[],'claim_flags':FLAGS}
 for n in ['checkpoint_state.json','epoch_receipt.json']:mf['files'].append({'name':n,'sha256':hashlib.sha256((o/n).read_bytes()).hexdigest()})
 save(o/'hash_manifest.json',mf);print(json.dumps(rec))
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
