from __future__ import annotations
import hashlib,json,sys,urllib.parse,urllib.request,urllib.error
from pathlib import Path

BASE='https://api.mathjs.org/v4/'

def canon(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()

def sha(x):
    return hashlib.sha256(canon(x)).hexdigest()

def query_authority(expr):
    qs=urllib.parse.urlencode({'expr':expr})
    url=BASE+'?'+qs
    req=urllib.request.Request(url,headers={'User-Agent':'ARTE-v62-independent-evaluator/1.0'})
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

def main(beacon_path,workdir,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    work=Path(workdir)
    beacon=json.loads(Path(beacon_path).read_text())
    challenge=json.loads((work/'challenge.json').read_text())
    submission=json.loads((work/'submission.json').read_text())
    if submission['challenge_sha256'] != challenge['challenge_sha256']:
        raise SystemExit('CHALLENGE_BINDING_FAILURE')

    responses=[];verdicts=[];gold_hashes=[]
    for task,pred in zip(challenge['tasks'],submission['predictions']):
        q=query_authority(task['expression'])
        responses.append({'index':task['index'],'expression':task['expression'],'response':q})
        if not q['ok']:
            (out/'authority_transcript.json').write_text(json.dumps({'authority':'mathjs.org REST web service','responses':responses},indent=2))
            raise SystemExit('MATHJS_EXTERNAL_QUERY_FAILED')
        raw=q['body']
        gold_hashes.append(hashlib.sha256(raw.encode()).hexdigest())
        verdicts.append(normalize(raw)==normalize(pred))

    transcript={
      'schema':'arte.v62_external_authority_transcript',
      'authority':'mathjs.org REST web service',
      'authority_endpoint':'https://api.mathjs.org/v4/',
      'responses':responses
    }
    transcript['transcript_sha256']=sha(transcript)

    verdict={
      'schema':'arte.v62_external_verdict',
      'authority':'mathjs.org REST web service',
      'challenge_sha256':challenge['challenge_sha256'],
      'submission_sha256':submission['submission_sha256'],
      'authority_transcript_sha256':transcript['transcript_sha256'],
      'verdicts':verdicts,
      'all_exact':all(verdicts),
      'gold_hashes':gold_hashes,
      'gold_revealed_only_after_submission_commit':True,
      'external_http_evaluation':True
    }
    verdict['receipt_sha256']=sha(verdict)

    evidence={
      'schema':'arte.v62_attestable_evidence_bundle',
      'beacon':beacon,
      'challenge':challenge,
      'submission':submission,
      'authority_transcript':transcript,
      'verdict':verdict,
      'ordering':[
        'verified_drand_beacon',
        'challenge_generation',
        'submission_commit',
        'external_mathjs_gold_fetch',
        'deterministic_comparison',
        'github_sigstore_attestation'
      ]
    }
    evidence['evidence_bundle_sha256']=sha(evidence)

    (out/'authority_transcript.json').write_text(json.dumps(transcript,indent=2))
    (out/'verdict.json').write_text(json.dumps(verdict,indent=2))
    (out/'evidence_bundle.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps({'all_exact':verdict['all_exact'],'receipt_sha256':verdict['receipt_sha256'],'evidence_bundle_sha256':evidence['evidence_bundle_sha256']}))

if __name__=='__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3])
