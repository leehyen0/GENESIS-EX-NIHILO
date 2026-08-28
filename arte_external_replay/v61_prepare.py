from __future__ import annotations
import hashlib,json,random,sys
from pathlib import Path


def canon(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()

def sha(x):
    return hashlib.sha256(canon(x)).hexdigest()

def main(beacon_path,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    b=json.loads(Path(beacon_path).read_text())
    seed=int(hashlib.sha256((b['randomness']+'|v61').encode()).hexdigest()[:16],16)
    rr=random.Random(seed)
    tasks=[];pred=[]
    for i in range(6):
        a=rr.randrange(11,80);bb=rr.randrange(3,20);c=rr.randrange(2,25);d=rr.randrange(1,40)
        expr=f"({a}^2)+({bb}*{c})-{d}"
        ans=a*a+bb*c-d
        tasks.append({'index':i,'expression':expr})
        pred.append(str(ans))
    challenge={
      'schema':'arte.v61_external_challenge',
      'beacon_round':b['round'],
      'beacon_randomness_sha256':hashlib.sha256(bytes.fromhex(b['randomness'])).hexdigest(),
      'tasks':tasks,
      'gold_revealed':False,
      'gold_source':'Wolfram Short Answers API, fetched only after submission commit'
    }
    challenge['challenge_sha256']=sha(challenge)
    submission={
      'schema':'arte.v61_submission_commit',
      'challenge_sha256':challenge['challenge_sha256'],
      'predictions':pred,
      'gold_seen':False,
      'solver':'frozen deterministic arithmetic solver'
    }
    submission['submission_sha256']=sha(submission)
    (out/'challenge.json').write_text(json.dumps(challenge,indent=2))
    (out/'submission.json').write_text(json.dumps(submission,indent=2))
    print(json.dumps({'challenge_sha256':challenge['challenge_sha256'],'submission_sha256':submission['submission_sha256']}))

if __name__=='__main__':
    main(sys.argv[1],sys.argv[2])
