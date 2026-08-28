#!/usr/bin/env python3
"""External-custodian reference protocol with explicit stage-hash binding.

This utility is for an EXTERNAL custodian. Running it inside the user/ChatGPT/ARTE
trust boundary does not establish independent custody.

Stages:
  commit  -> fix G1/G2/G3 OTP commitments and private keys before G1
  public  -> release exact precommitted public packet only after formal freeze
  sensor  -> bind sensor response to exact frozen question-output hash
  reveal  -> bind OTP-key reveal to exact frozen prediction-output hash
"""
import argparse, base64, hashlib, json, secrets
from pathlib import Path

HIDDEN_SCHEMA='arte.hidden_question_sensor_challenge/v57'
COMMIT_SCHEMA='arte.question_sensor_commitment/v57'
BATCH_SCHEMA='arte.question_sensor_commitment_batch/v57'
QUESTION_OUTPUT_SCHEMA='arte.independent_question_output/v57'
SENSOR_SCHEMA='arte.independent_sensor_input/v57'
PREDICTION_SCHEMA='arte.independent_prediction_output/v57'
FREEZE_SCHEMA='arte.question_sensor_candidate_freeze/v57-v3'
REVEAL_SCHEMA='arte.question_sensor_reveal/v57-v3'
PRIVATE_SCHEMA='arte.question_sensor_private_key_state/v57-v3'


def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
def sha(b): return hashlib.sha256(b).hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')

def freeze_hash_valid(f):
    if f.get('schema')!=FREEZE_SCHEMA or f.get('formal_candidate_freeze') is not True: return False
    d=dict(f); declared=d.pop('freeze_sha256',None)
    return bool(declared and declared==sha(canon(d)))

def hidden_validate(h,path='hidden'):
    if h.get('schema')!=HIDDEN_SCHEMA: raise ValueError(f'{path}: BAD_HIDDEN_SCHEMA')
    if h.get('generation') not in ('G1','G2','G3'): raise ValueError(f'{path}: BAD_GENERATION')
    if not h.get('challenge_id') or not h.get('batch_id'): raise ValueError(f'{path}: MISSING_IDS')
    if not isinstance(h.get('public_packet'),dict): raise ValueError(f'{path}: PUBLIC_PACKET_REQUIRED')
    if h.get('claim_boundary')!={'AGI':False,'ASI':False}: raise ValueError(f'{path}: BAD_CLAIM_BOUNDARY')
    pub=h['public_packet']
    if pub.get('challenge_id')!=h['challenge_id'] or pub.get('generation')!=h['generation']:
        raise ValueError(f'{path}: PUBLIC_PACKET_BINDING_MISMATCH')
    pub_ids=[x.get('case_id') for x in pub.get('cases',[])]
    hid_ids=[x.get('case_id') for x in h.get('cases',[])]
    if len(pub_ids)!=len(set(pub_ids)) or set(pub_ids)!=set(hid_ids):
        raise ValueError(f'{path}: CASE_ID_SET_MISMATCH_OR_DUPLICATE')


def make_commitment(hidden_path,custodian_id):
    plaintext=Path(hidden_path).read_bytes(); h=json.loads(plaintext.decode('utf-8')); hidden_validate(h,hidden_path)
    key=secrets.token_bytes(len(plaintext)); ct=bytes(a^b for a,b in zip(plaintext,key))
    c={
      'schema':COMMIT_SCHEMA,'batch_id':h['batch_id'],'challenge_id':h['challenge_id'],'generation':h['generation'],'custodian_id':custodian_id,
      'byte_length':len(plaintext),'plaintext_sha256':sha(plaintext),'ciphertext_sha256':sha(ct),'ciphertext_b64':base64.b64encode(ct).decode('ascii'),
      'public_packet_sha256':sha(canon(h['public_packet'])),'key_withheld':True,'all_generations_fixed_before_G1':True,
      'claim_boundary':{'AGI':False,'ASI':False,'independent_organization_custody':False}
    }
    priv={
      'schema':PRIVATE_SCHEMA,'batch_id':h['batch_id'],'challenge_id':h['challenge_id'],'generation':h['generation'],'custodian_id':custodian_id,
      'plaintext_sha256':c['plaintext_sha256'],'ciphertext_sha256':c['ciphertext_sha256'],'key_sha256':sha(key),'key_b64':base64.b64encode(key).decode('ascii'),
      'created_with_commitment_before_G1':True
    }
    return c,priv

def cmd_commit(a):
    triples=[('G1',a.g1),('G2',a.g2),('G3',a.g3)]; commits=[]; privs=[]
    for expected,p in triples:
        c,s=make_commitment(p,a.custodian_id)
        if c['generation']!=expected: raise SystemExit('INPUT_GENERATION_ORDER_MISMATCH:'+expected)
        commits.append(c); privs.append(s)
    if len({c['batch_id'] for c in commits})!=1: raise SystemExit('ALL_GENERATIONS_MUST_SHARE_BATCH_ID')
    if len({c['challenge_id'] for c in commits})!=3: raise SystemExit('CHALLENGE_IDS_MUST_BE_UNIQUE')
    batch={
      'schema':BATCH_SCHEMA,'batch_id':commits[0]['batch_id'],'custodian_id':a.custodian_id,
      'all_commitments_fixed_before_G1':True,'commitments':commits,'keys_and_plaintexts_withheld':True,
      'claim_boundary':{'AGI':False,'ASI':False,'independent_organization_custody':False}
    }
    save(a.commitment_batch_out,batch); rd=Path(a.private_state_dir); rd.mkdir(parents=True,exist_ok=True)
    for s in privs: save(rd/f"{s['generation']}_PRIVATE_KEEP_SECRET.json",s)
    print(json.dumps({'stage':'commit','batch_id':batch['batch_id'],'commitment_batch_sha256':sha(canon(batch)),'private_state_dir':a.private_state_dir,'IMPORTANT':'Return only commitment batch. Keep hidden packages and private state secret.'},sort_keys=True))

def cmd_public(a):
    h=load(a.hidden); hidden_validate(h,a.hidden); f=load(a.formal_freeze)
    if not freeze_hash_valid(f): raise SystemExit('INVALID_FORMAL_FREEZE')
    if f.get('commitment_batch_id')!=h['batch_id']: raise SystemExit('HIDDEN_BATCH_NOT_BOUND_TO_FREEZE')
    save(a.out,h['public_packet'])
    print(json.dumps({'stage':'public','generation':h['generation'],'challenge_id':h['challenge_id'],'formal_freeze_sha256':f['freeze_sha256'],'public_packet_sha256':sha(canon(h['public_packet']))},sort_keys=True))

def cmd_sensor(a):
    h=load(a.hidden); hidden_validate(h,a.hidden); q=load(a.questions); f=load(a.formal_freeze)
    if not freeze_hash_valid(f): raise SystemExit('INVALID_FORMAL_FREEZE')
    if f.get('commitment_batch_id')!=h['batch_id']: raise SystemExit('HIDDEN_BATCH_NOT_BOUND_TO_FREEZE')
    if q.get('schema')!=QUESTION_OUTPUT_SCHEMA or q.get('challenge_id')!=h['challenge_id'] or q.get('generation')!=h['generation']:
        raise SystemExit('QUESTION_OUTPUT_BINDING_MISMATCH')
    qmap={x.get('case_id'):x for x in q.get('questions',[])}
    hmap={x.get('case_id'):x for x in h.get('cases',[])}
    out=[]
    for pc in h['public_packet'].get('cases',[]):
        cid=pc.get('case_id'); hc=hmap[cid]; qc=qmap.get(cid,{})
        expected=hc.get('expected_action','QUESTION')
        row={'case_id':cid,'public_observation':pc.get('public_observation')}
        if expected=='HOLD':
            if qc.get('action')!='HOLD': raise SystemExit('CANDIDATE_FAILED_EXPECTED_HOLD:'+str(cid))
        else:
            eq=hc.get('expected_question')
            if qc.get('action')!='QUESTION' or qc.get('question')!=eq or qc.get('question_sha256')!=sha(canon(eq)):
                raise SystemExit('QUESTION_MISMATCH__DO_NOT_RELEASE_SENSOR:'+str(cid))
            bit=hc.get('sensor_bit')
            if bit not in (0,1,False,True): raise SystemExit('BAD_HIDDEN_SENSOR_BIT:'+str(cid))
            row.update({'question':eq,'question_sha256':sha(canon(eq)),'sensor_bit':int(bit)})
        out.append(row)
    sensor={
      'schema':SENSOR_SCHEMA,'batch_id':h['batch_id'],'challenge_id':h['challenge_id'],'generation':h['generation'],'custodian_id':f.get('custodian_id'),
      'formal_freeze_sha256':f['freeze_sha256'],'question_output_sha256':sha(canon(q)),
      'sensor_bits_released_only_after_exact_question_output_hash_received':True,'cases':out,
      'claim_boundary':{'AGI':False,'ASI':False,'independent_custody':False}
    }
    save(a.out,sensor)
    print(json.dumps({'stage':'sensor','generation':h['generation'],'challenge_id':h['challenge_id'],'question_output_sha256':sensor['question_output_sha256'],'sensor_input_sha256':sha(canon(sensor))},sort_keys=True))

def cmd_reveal(a):
    s=load(a.private_state); sensor=load(a.sensor_input); pred=load(a.predictions); f=load(a.formal_freeze)
    if s.get('schema')!=PRIVATE_SCHEMA: raise SystemExit('BAD_PRIVATE_STATE')
    if not freeze_hash_valid(f): raise SystemExit('INVALID_FORMAL_FREEZE')
    for x in (sensor,pred):
        if x.get('challenge_id')!=s['challenge_id'] or x.get('generation')!=s['generation']: raise SystemExit('STAGE_CHALLENGE_BINDING_MISMATCH')
    if sensor.get('schema')!=SENSOR_SCHEMA or pred.get('schema')!=PREDICTION_SCHEMA: raise SystemExit('BAD_STAGE_SCHEMA')
    if sensor.get('formal_freeze_sha256')!=f['freeze_sha256']: raise SystemExit('SENSOR_FREEZE_BINDING_MISMATCH')
    reveal={
      'schema':REVEAL_SCHEMA,'batch_id':s['batch_id'],'challenge_id':s['challenge_id'],'generation':s['generation'],'custodian_id':s['custodian_id'],
      'formal_freeze_sha256':f['freeze_sha256'],'question_output_sha256':sensor.get('question_output_sha256'),'sensor_input_sha256':sha(canon(sensor)),
      'prediction_output_sha256':sha(canon(pred)),'key_sha256':s['key_sha256'],'key_b64':s['key_b64'],
      'reveal_only_after_exact_prediction_output_hash_received':True,
      'claim_boundary':{'AGI':False,'ASI':False,'independent_custody':False}
    }
    save(a.out,reveal)
    print(json.dumps({'stage':'reveal','generation':s['generation'],'challenge_id':s['challenge_id'],'prediction_output_sha256':reveal['prediction_output_sha256'],'reveal_sha256':sha(canon(reveal))},sort_keys=True))

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    c=sub.add_parser('commit'); c.add_argument('--g1',required=True);c.add_argument('--g2',required=True);c.add_argument('--g3',required=True);c.add_argument('--custodian-id',required=True);c.add_argument('--commitment-batch-out',required=True);c.add_argument('--private-state-dir',required=True);c.set_defaults(fn=cmd_commit)
    p=sub.add_parser('public'); p.add_argument('--hidden',required=True);p.add_argument('--formal-freeze',required=True);p.add_argument('--out',required=True);p.set_defaults(fn=cmd_public)
    s=sub.add_parser('sensor'); s.add_argument('--hidden',required=True);s.add_argument('--questions',required=True);s.add_argument('--formal-freeze',required=True);s.add_argument('--out',required=True);s.set_defaults(fn=cmd_sensor)
    r=sub.add_parser('reveal'); r.add_argument('--private-state',required=True);r.add_argument('--sensor-input',required=True);r.add_argument('--predictions',required=True);r.add_argument('--formal-freeze',required=True);r.add_argument('--out',required=True);r.set_defaults(fn=cmd_reveal)
    a=ap.parse_args(); a.fn(a)

if __name__=='__main__': main()
