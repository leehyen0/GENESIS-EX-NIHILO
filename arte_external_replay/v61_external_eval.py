from __future__ import annotations
import hashlib,json,sys,urllib.parse,urllib.request,urllib.error
from pathlib import Path

BASE='https://api.wolframalpha.com/v1/result'

def canon(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()

def sha(x):
    return hashlib.sha256(canon(x)).hexdigest()

def query_wolfram(expr):
    qs=urllib.parse.urlencode({'appid':'DEMO','i':expr})
    url=BASE+'?'+qs
    req=urllib.request.Request(url,headers={'User-Agent':'ARTE-v61-independent-evaluator/1.0'})
    try:
        with urllib.request.urlopen(req,timeout=30) as r:
            body=r.read().decode('utf-8','replace').strip()
            return {'ok':True,'status':getattr(r,'status',200),'body':body,'url':url}
    except urllib.error.HTTPError as e:
        body=e.read().decode('utf-8','replace')
        return {'ok':False,'status':e.code,'body':body,'url':url,'error':'HTTPError'}
    except Exception as e:
        return {'ok':False,'status':None,'body':'','url':url,'error':repr(e)}

def normalize(s):
    return ''.join(str(s).strip().split()).replace(',','')

def main(workdir,outdir):
    work=Path(workdir);out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    ch=json.loads((work/'challenge.json').read_text())
    sub=json.loads((work/'submission.json').read_text())
    if sub['challenge_sha256']!=ch['challenge_sha256']:
        raise SystemExit('CHALLENGE_BINDING_FAILURE')

    diagnostic={'demo_probe_2_plus_2':query_wolfram('2+2'),'task_queries':[]}
    verdicts=[];gold_hashes=[]
    for task,pred in zip(ch['tasks'],sub['predictions']):
        q=query_wolfram(task['expression'])
        diagnostic['task_queries'].append({'expression':task['expression'],'result':q})
        if not q['ok']:
            (out/'wolfram_diagnostic.json').write_text(json.dumps(diagnostic,indent=2))
            print(json.dumps(diagnostic))
            raise SystemExit('WOLFRAM_EXTERNAL_QUERY_FAILED')
        raw=q['body']
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
    (out/'wolfram_diagnostic.json').write_text(json.dumps(diagnostic,indent=2))
    (out/'verdict.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))

if __name__=='__main__':
    main(sys.argv[1],sys.argv[2])
