from __future__ import annotations
import hashlib,json,sys,urllib.request
from pathlib import Path

NIST='https://beacon.nist.gov/beacon/2.0/pulse/last'
MOD=19
PAIRS=[
 ('NIST_TABLE_RELATION','DRAND_SYMBOLIC_RULE'),
 ('NIST_GRAPH_WEIGHT','DRAND_RESOURCE_SCORE'),
 ('NIST_FEATURE_RELATION','DRAND_SEQUENCE_RULE'),
 ('NIST_CAUSAL_CODE','DRAND_TOOL_SCORE')
]

def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def sha(x):return hashlib.sha256(canon(x)).hexdigest()
def f(t,x,z):a,b,c=t;return (a*x+b*z+c)%MOD

def infer(ex):
    probes=0
    for a in range(1,MOD):
      for b in range(1,MOD):
       for c in range(MOD):
        probes+=1
        if all(f((a,b,c),x,z)==y for x,z,y in ex):return (a,b,c),probes
    return None,probes

def theta_from_hex(h,i):
    raw=bytes.fromhex(h)
    a=1+(raw[i*3]% (MOD-1));b=1+(raw[i*3+1]%(MOD-1));c=raw[i*3+2]%MOD
    return (a,b,c)

def main(outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    req=urllib.request.Request(NIST,headers={'User-Agent':'ARTE-v65-source/1.0'})
    with urllib.request.urlopen(req,timeout=30) as r:
        doc=json.loads(r.read().decode())
    p=doc['pulse'];value=p['outputValue']
    records=[];hyps=[]
    anchors=[(0,0),(1,0),(0,1),(2,3),(5,7)]
    for i,(src,tgt) in enumerate(PAIRS):
        theta=theta_from_hex(value,i)
        demos=[(x,z,f(theta,x,z)) for x,z in anchors]
        hyp,cost=infer(demos)
        if hyp!=theta:raise SystemExit('SOURCE_INFERENCE_FAILURE')
        records.append({'index':i,'source_domain':src,'target_domain':tgt,'demos':demos,'search_cost':cost})
        hyps.append({'index':i,'source_domain':src,'target_domain':tgt,'hypothesis':list(hyp)})
    source_receipt={
      'schema':'arte.v65_nist_source_receipt',
      'authority':'NIST Randomness Beacon 2.0',
      'endpoint':NIST,
      'pulse_uri':p['uri'],'chainIndex':p['chainIndex'],'pulseIndex':p['pulseIndex'],
      'timeStamp':p['timeStamp'],'outputValue_sha256':hashlib.sha256(bytes.fromhex(value)).hexdigest(),
      'signatureValue_sha256':hashlib.sha256(bytes.fromhex(p['signatureValue'])).hexdigest(),
      'statusCode':p['statusCode']
    }
    source_receipt['receipt_sha256']=sha(source_receipt)
    cognition={'schema':'arte.v65_source_cognition_commit','source_receipt_sha256':source_receipt['receipt_sha256'],'hypotheses':hyps,'records':records}
    cognition['source_cognition_sha256']=sha(cognition)
    (out/'nist_source_receipt.json').write_text(json.dumps(source_receipt,indent=2))
    (out/'source_cognition.json').write_text(json.dumps(cognition,indent=2))
    print(json.dumps({'source_cognition_sha256':cognition['source_cognition_sha256'],'pulseIndex':p['pulseIndex']}))

if __name__=='__main__':main(sys.argv[1])
