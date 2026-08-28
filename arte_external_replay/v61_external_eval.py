from __future__ import annotations
import hashlib,json,sys,urllib.parse,urllib.request
from pathlib import Path

BASE='https://api.wolframalpha.com/v1/result'

def canon(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()

def sha(x):
    return hashlib.sha256(canon(x)).hexdigest()

def query_wolfram(expr):
    qs=urllib.parse.urlencode({'appid':'DEMO','i':expr})
    req=urllib.request.Request(BASE+'?'+qs,headers={'User-Agent':'ARTE-v61-independent-evaluator/1.0'})
    with urllib.request.urlopen(req,timeout=30) as r:
        return r.read().decode('utf-8').strip()

def normalize(s):
    return ''.join(str(s).strip().split()).replace(',','')

def main(workdir,outdir):
    work=Path(workdir);out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    ch=json.loads((work/'challenge.json').read_text())
    sub=json.loads((work/'submission.json').read_text())
    if sub['challenge_sha256']!=ch['challenge_sha256']:
        raise SystemExit('CHALLENGE_BINDING_FAILURE')
    verdicts=[];gold_hashes=[]
    for task,pred in zip(ch['tasks'],sub['predictions']):
        raw=query_wolfram(task['expression'])
        gold_hashes.append(hashlib.sha256(raw.encode()).hexdigest())
        verdicts.append(normalize(raw)==normalize(pred))
    receipt={
      'schema':'arte.v61_wolfram_api_verdict',
      'authority':'Wolfram Short Answers API',
      'authority_endpoint':'https://api.wolframalpha.com/v1/result',
      'challenge_sha256':ch['challenge_sha256'],
      'submission_sha256':sub['submission_sha256'],
      'verdicts':verdicts,
      'all_exact':all(verdicts),
      'gold_hashes':gold_hashes,
      'gold_revealed':False,
      'external_http_evaluation':True
    }
    receipt['receipt_sha256']=sha(receipt)
    (out/'verdict.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))

if __name__=='__main__':
    main(sys.argv[1],sys.argv[2])
