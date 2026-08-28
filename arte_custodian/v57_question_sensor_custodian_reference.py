#!/usr/bin/env python3
"""Reference generator for an EXTERNAL information custodian.

Do NOT run this inside the user/ChatGPT/ARTE trust boundary and then claim independent custody.
The external custodian must generate all G1/G2/G3 hidden packages before G1 candidate freeze,
retain all OTP keys privately, and return only the commitment batch.
"""
import argparse, base64, hashlib, json, secrets
from pathlib import Path

HIDDEN_SCHEMA='arte.hidden_question_sensor_challenge/v57'
COMMIT_SCHEMA='arte.question_sensor_commitment/v57'
BATCH_SCHEMA='arte.question_sensor_commitment_batch/v57'
REVEAL_SCHEMA='arte.question_sensor_reveal/v57'


def sha(b): return hashlib.sha256(b).hexdigest()
def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')

def make_one(path,custodian_id):
    plaintext=Path(path).read_bytes()
    h=json.loads(plaintext.decode('utf-8'))
    if h.get('schema')!=HIDDEN_SCHEMA: raise ValueError(f'{path}: bad hidden schema')
    cid=h.get('challenge_id'); gen=h.get('generation'); batch=h.get('batch_id')
    if not cid or gen not in ('G1','G2','G3') or not batch: raise ValueError(f'{path}: missing challenge_id/generation/batch_id')
    if not isinstance(h.get('public_packet'),dict): raise ValueError(f'{path}: public_packet required')
    if h.get('claim_boundary')!={'AGI':False,'ASI':False}: raise ValueError(f'{path}: hidden claim boundary must be false')
    key=secrets.token_bytes(len(plaintext)); ct=bytes(a^b for a,b in zip(plaintext,key))
    c={
      'schema':COMMIT_SCHEMA,'batch_id':batch,'challenge_id':cid,'generation':gen,'custodian_id':custodian_id,
      'byte_length':len(plaintext),'plaintext_sha256':sha(plaintext),'ciphertext_sha256':sha(ct),
      'ciphertext_b64':base64.b64encode(ct).decode('ascii'),'public_packet_sha256':sha(canon(h['public_packet'])),
      'key_withheld':True,'all_generations_fixed_before_G1':True,
      'claim_boundary':{'AGI':False,'ASI':False,'independent_organization_custody':False}
    }
    r={
      'schema':REVEAL_SCHEMA,'batch_id':batch,'challenge_id':cid,'generation':gen,'custodian_id':custodian_id,
      'key_sha256':sha(key),'key_b64':base64.b64encode(key).decode('ascii'),
      'reveal_only_after_prediction_freeze':True
    }
    return c,r

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--g1',required=True);ap.add_argument('--g2',required=True);ap.add_argument('--g3',required=True)
    ap.add_argument('--custodian-id',required=True);ap.add_argument('--commitment-batch-out',required=True);ap.add_argument('--reveal-dir',required=True)
    a=ap.parse_args()
    pairs=[make_one(a.g1,a.custodian_id),make_one(a.g2,a.custodian_id),make_one(a.g3,a.custodian_id)]
    commits=[x[0] for x in pairs]; reveals=[x[1] for x in pairs]
    if [x['generation'] for x in commits]!=['G1','G2','G3']: raise SystemExit('inputs must be G1,G2,G3 respectively')
    batches={x['batch_id'] for x in commits}
    if len(batches)!=1: raise SystemExit('all hidden packages must share one batch_id')
    ids=[x['challenge_id'] for x in commits]
    if len(set(ids))!=3: raise SystemExit('challenge IDs must be unique')
    batch={
      'schema':BATCH_SCHEMA,'batch_id':commits[0]['batch_id'],'custodian_id':a.custodian_id,
      'all_commitments_fixed_before_G1':True,'commitments':commits,
      'keys_and_plaintexts_withheld':True,
      'claim_boundary':{'AGI':False,'ASI':False,'independent_organization_custody':False}
    }
    Path(a.commitment_batch_out).write_text(json.dumps(batch,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    rd=Path(a.reveal_dir);rd.mkdir(parents=True,exist_ok=True)
    for r in reveals:
        (rd/f"{r['generation']}_REVEAL_KEEP_PRIVATE.json").write_text(json.dumps(r,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'batch_id':batch['batch_id'],'batch_sha256':sha(canon(batch)),'commitment_file':a.commitment_batch_out,'IMPORTANT':'Return only commitment batch now; keep reveal directory private until the exact protocol stage.'},sort_keys=True))

if __name__=='__main__': main()
