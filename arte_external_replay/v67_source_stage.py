from __future__ import annotations
import hashlib,json,sys,urllib.request
from pathlib import Path

NIST='https://beacon.nist.gov/beacon/2.0/pulse/last'
MOD=19
PAIR_NAMES=[
 ('NIST_TABLE_RELATION','DRAND_SYMBOLIC_RULE'),
 ('NIST_GRAPH_WEIGHT','DRAND_RESOURCE_SCORE'),
 ('NIST_FEATURE_RELATION','DRAND_SEQUENCE_RULE'),
 ('NIST_CAUSAL_CODE','DRAND_TOOL_SCORE'),
]
def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,default=str).encode()
def sha(x): return hashlib.sha256(canon(x)).hexdigest()
def theta_from_hex(h,i,epoch):
    raw=hashlib.sha256((h+f'|src|{i}|{epoch}').encode()).digest()
    return (1+raw[0]%(MOD-1),1+raw[1]%(MOD-1),raw[2]%MOD)

def main(trigger_path,outpath):
    trig=json.loads(Path(trigger_path).read_text())
    epoch=int(trig['epoch'])
    req=urllib.request.Request(NIST,headers={'User-Agent':'ARTE-v67-E6-source/1.0'})
    with urllib.request.urlopen(req,timeout=30) as r:
        doc=json.loads(r.read().decode())
    p=doc['pulse'];value=p['outputValue']
    source={
      'authority':'NIST Randomness Beacon 2.0','endpoint':NIST,
      'pulse_uri':p['uri'],'chainIndex':p['chainIndex'],'pulseIndex':p['pulseIndex'],
      'timeStamp':p['timeStamp'],'outputValue':value,
      'outputValue_sha256':hashlib.sha256(bytes.fromhex(value)).hexdigest(),
      'signatureValue_sha256':hashlib.sha256(bytes.fromhex(p['signatureValue'])).hexdigest(),
      'statusCode':p['statusCode']
    }
    receipt={k:v for k,v in source.items() if k!='outputValue'}
    receipt['receipt_sha256']=sha(receipt)
    items=[]
    for i,(src,tgt) in enumerate(PAIR_NAMES):
        theta=theta_from_hex(value,i,epoch)
        items.append({'pair':i,'source_domain':src,'target_domain':tgt,'theta':list(theta)})
    commit={
      'schema':'arte.v67_source_cognition_commit','epoch':epoch,
      'receipt_sha256':receipt['receipt_sha256'],'items':items
    }
    commit['source_cognition_sha256']=sha(commit)
    bundle={'schema':'arte.v67_source_bundle','source':source,'source_receipt':receipt,'source_cognition_commit':commit}
    bundle['source_bundle_sha256']=sha(bundle)
    Path(outpath).write_text(json.dumps(bundle,indent=2))
    print(json.dumps({'epoch':epoch,'pulseIndex':p['pulseIndex'],'source_cognition_sha256':commit['source_cognition_sha256']}))
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
