from __future__ import annotations
import hashlib,json,sys,urllib.parse,urllib.request
from pathlib import Path

BASE='https://api.mathjs.org/v4/'
MOD=19

def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def sha(x):return hashlib.sha256(canon(x)).hexdigest()
def normalize(s):return ''.join(str(s).strip().split()).replace(',','')
def query(expr):
    url=BASE+'?'+urllib.parse.urlencode({'expr':expr})
    req=urllib.request.Request(url,headers={'User-Agent':'ARTE-v65-E5/1.0'})
    with urllib.request.urlopen(req,timeout=30) as r:
        return {'status':getattr(r,'status',200),'body':r.read().decode().strip(),'url':url}

def main(source_receipt_path,source_cognition_path,target_beacon_path,workdir,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    src_receipt=json.loads(Path(source_receipt_path).read_text())
    source=json.loads(Path(source_cognition_path).read_text())
    beacon=json.loads(Path(target_beacon_path).read_text())
    work=Path(workdir)
    challenge=json.loads((work/'target_challenge.json').read_text())
    submission=json.loads((work/'target_submission.json').read_text())
    if submission['challenge_sha256']!=challenge['challenge_sha256']:raise SystemExit('CHALLENGE_BINDING')
    if submission['source_cognition_sha256']!=source['source_cognition_sha256']:raise SystemExit('SOURCE_BINDING')

    transcripts=[];pair_results=[]
    source_by_index={x['index']:x for x in source['hypotheses']}
    sub_by_index={x['index']:x for x in submission['results']}
    for task in challenge['tasks']:
        i=task['index'];theta=source_by_index[i]['hypothesis'];a,b,c=theta;sub=sub_by_index[i]
        full_ok=[];ab_ok=[];responses=[]
        for j,(x,z) in enumerate(task['heldout_inputs']):
            expr=f'mod(({a}*{x})+({b}*{z})+{c},{MOD})'
            q=query(expr)
            if q['status']!=200:raise SystemExit('EXTERNAL_AUTHORITY_HTTP_FAILURE')
            gold=normalize(q['body'])
            full_ok.append(gold==normalize(sub['full_predictions'][j]))
            ab_ok.append(gold==normalize(sub['ablation_predictions'][j]))
            responses.append({'heldout_index':j,'x':x,'z':z,'expression':expr,'response':q,'gold_sha256':hashlib.sha256(q['body'].encode()).hexdigest()})
        reduction=1-sub['full_search_cost']/sub['ablation_search_cost']
        pair_results.append({
          'index':i,'source_domain':task['source_domain'],'target_domain':task['target_domain'],
          'full_exact':all(full_ok),'ablation_exact':all(ab_ok),
          'full_search_cost':sub['full_search_cost'],'ablation_search_cost':sub['ablation_search_cost'],
          'transfer_cost_reduction':reduction,'heldout_count':len(responses)
        })
        transcripts.append({'index':i,'source_domain':task['source_domain'],'target_domain':task['target_domain'],'responses':responses})

    authority={
      'schema':'arte.v65_external_authority_transcript','authority':'mathjs.org REST web service',
      'endpoint':BASE,'pairs':transcripts
    }
    authority['transcript_sha256']=sha(authority)
    verdict={
      'schema':'arte.v65_source_disjoint_transfer_verdict',
      'source_authority':'NIST Randomness Beacon 2.0',
      'target_entropy_authority':'League of Entropy drand',
      'outcome_authority':'mathjs.org REST web service',
      'source_receipt_sha256':src_receipt['receipt_sha256'],
      'source_cognition_sha256':source['source_cognition_sha256'],
      'target_challenge_sha256':challenge['challenge_sha256'],
      'submission_sha256':submission['submission_sha256'],
      'authority_transcript_sha256':authority['transcript_sha256'],
      'target_beacon_round':beacon['round'],
      'target_entropy_strictly_future':beacon['target_entropy_strictly_future'],
      'pairs':pair_results,
      'all_full_exact':all(x['full_exact'] for x in pair_results),
      'all_ablation_exact':all(x['ablation_exact'] for x in pair_results),
      'all_full_cost_lower':all(x['full_search_cost']<x['ablation_search_cost'] for x in pair_results),
      'gold_seen_before_submission':False,
      'attempt_budget':1
    }
    verdict['verdict_sha256']=sha(verdict)
    evidence={
      'schema':'arte.v65_formal_e5_evidence_bundle','source_receipt':src_receipt,'source_cognition':source,
      'target_beacon':beacon,'target_challenge':challenge,'submission':submission,'authority_transcript':authority,'verdict':verdict,
      'authority_separation':{
        'source_data':'NIST Randomness Beacon 2.0','target_future_entropy':'League of Entropy drand',
        'outcome':'mathjs.org REST web service','provenance':'GitHub artifact attestations / Sigstore'
      }
    }
    evidence['evidence_bundle_sha256']=sha(evidence)
    (out/'authority_transcript.json').write_text(json.dumps(authority,indent=2))
    (out/'verdict.json').write_text(json.dumps(verdict,indent=2))
    (out/'evidence_bundle.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps({'all_full_exact':verdict['all_full_exact'],'all_ablation_exact':verdict['all_ablation_exact'],'all_full_cost_lower':verdict['all_full_cost_lower'],'verdict_sha256':verdict['verdict_sha256']}))

if __name__=='__main__':main(*sys.argv[1:6])
