#!/usr/bin/env python3
import argparse, base64, hashlib, json
from pathlib import Path

HIDDEN_SCHEMA='arte.hidden_question_sensor_challenge/v57'
COMMIT_SCHEMA='arte.question_sensor_commitment/v57'
BATCH_SCHEMA='arte.question_sensor_commitment_batch/v57'
REVEAL_SCHEMA='arte.question_sensor_reveal/v57'
QUESTION_OUTPUT_SCHEMA='arte.independent_question_output/v57'
SENSOR_INPUT_SCHEMA='arte.independent_sensor_input/v57'
PREDICTION_OUTPUT_SCHEMA='arte.independent_prediction_output/v57'


def sha(b): return hashlib.sha256(b).hexdigest()
def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def fail(errors,msg): errors.append(msg)

def verify(batch_path,freeze_path,public_input_path,questions_path,sensor_input_path,predictions_path,reveal_path):
    errors=[]
    batch=load(batch_path); freeze=load(freeze_path); public=load(public_input_path); questions=load(questions_path); sensor=load(sensor_input_path); preds=load(predictions_path); reveal=load(reveal_path)
    if batch.get('schema')!=BATCH_SCHEMA: fail(errors,'BAD_BATCH_SCHEMA')
    if freeze.get('formal_candidate_freeze') is not True: fail(errors,'FORMAL_FREEZE_REQUIRED')
    if freeze.get('commitment_batch_sha256')!=sha(canon(batch)): fail(errors,'FREEZE_BATCH_SHA_MISMATCH')
    cid=public.get('challenge_id'); gen=public.get('generation')
    matches=[c for c in batch.get('commitments',[]) if c.get('challenge_id')==cid and c.get('generation')==gen]
    if len(matches)!=1: fail(errors,'COMMITMENT_NOT_UNIQUE_FOR_PUBLIC_INPUT'); c=matches[0] if matches else {}
    else: c=matches[0]
    if c.get('schema')!=COMMIT_SCHEMA: fail(errors,'BAD_COMMITMENT_SCHEMA')
    if reveal.get('schema')!=REVEAL_SCHEMA: fail(errors,'BAD_REVEAL_SCHEMA')
    for f in ('batch_id','challenge_id','generation','custodian_id'):
        if c.get(f)!=reveal.get(f): fail(errors,'REVEAL_BINDING_MISMATCH:'+f)
    try: ct=base64.b64decode(c.get('ciphertext_b64',''),validate=True)
    except Exception: ct=b''; fail(errors,'INVALID_CIPHERTEXT_BASE64')
    try: key=base64.b64decode(reveal.get('key_b64',''),validate=True)
    except Exception: key=b''; fail(errors,'INVALID_KEY_BASE64')
    if len(ct)!=c.get('byte_length'): fail(errors,'CIPHERTEXT_LENGTH_MISMATCH')
    if len(key)!=len(ct): fail(errors,'OTP_KEY_LENGTH_MISMATCH')
    if sha(ct)!=c.get('ciphertext_sha256'): fail(errors,'CIPHERTEXT_SHA_MISMATCH')
    if sha(key)!=reveal.get('key_sha256'): fail(errors,'KEY_SHA_MISMATCH')
    plain=bytes(a^b for a,b in zip(ct,key)) if len(ct)==len(key) else b''
    if sha(plain)!=c.get('plaintext_sha256'): fail(errors,'PLAINTEXT_SHA_MISMATCH')
    try: hidden=json.loads(plain.decode('utf-8'))
    except Exception: hidden={}; fail(errors,'DECRYPTED_PLAINTEXT_NOT_VALID_UTF8_JSON')
    if hidden.get('schema')!=HIDDEN_SCHEMA: fail(errors,'BAD_HIDDEN_SCHEMA')
    for f in ('batch_id','challenge_id','generation'):
        if hidden.get(f)!=c.get(f): fail(errors,'HIDDEN_COMMITMENT_BINDING_MISMATCH:'+f)
    if hidden.get('claim_boundary')!={'AGI':False,'ASI':False}: fail(errors,'BAD_HIDDEN_CLAIM_BOUNDARY')
    if sha(canon(public))!=c.get('public_packet_sha256'): fail(errors,'PUBLIC_PACKET_COMMITMENT_MISMATCH')
    if hidden.get('public_packet')!=public: fail(errors,'PUBLIC_PACKET_HIDDEN_PACKAGE_MISMATCH')
    if questions.get('schema')!=QUESTION_OUTPUT_SCHEMA or questions.get('challenge_id')!=cid or questions.get('generation')!=gen: fail(errors,'BAD_QUESTION_OUTPUT_BINDING')
    if sensor.get('schema')!=SENSOR_INPUT_SCHEMA or sensor.get('challenge_id')!=cid or sensor.get('generation')!=gen: fail(errors,'BAD_SENSOR_INPUT_BINDING')
    if preds.get('schema')!=PREDICTION_OUTPUT_SCHEMA or preds.get('challenge_id')!=cid or preds.get('generation')!=gen: fail(errors,'BAD_PREDICTION_OUTPUT_BINDING')
    pub_cases={x.get('case_id'):x for x in public.get('cases',[])}
    q_cases={x.get('case_id'):x for x in questions.get('questions',[])}
    s_cases={x.get('case_id'):x for x in sensor.get('cases',[])}
    p_cases={x.get('case_id'):x for x in preds.get('predictions',[])}
    h_cases={x.get('case_id'):x for x in hidden.get('cases',[])}
    ids=set(pub_cases)
    if ids!=set(h_cases): fail(errors,'PUBLIC_HIDDEN_CASE_ID_SET_MISMATCH')
    case_receipts=[]
    for case_id in sorted(ids):
        pc=pub_cases[case_id]; hc=h_cases.get(case_id,{}); qc=q_cases.get(case_id); sc=s_cases.get(case_id); pr=p_cases.get(case_id)
        ce=[]
        expected_action=hc.get('expected_action','QUESTION')
        if expected_action=='HOLD':
            if not qc or qc.get('action')!='HOLD': ce.append('EXPECTED_QUESTION_PHASE_HOLD')
            if not pr or pr.get('action')!='HOLD': ce.append('EXPECTED_PREDICTION_PHASE_HOLD')
        else:
            eq=hc.get('expected_question')
            if not qc or qc.get('action')!='QUESTION' or qc.get('question')!=eq or qc.get('question_sha256')!=sha(canon(eq)): ce.append('QUESTION_MISMATCH')
            if not sc: ce.append('MISSING_SENSOR_CASE')
            else:
                if sc.get('public_observation')!=pc.get('public_observation'): ce.append('SENSOR_PUBLIC_OBSERVATION_MISMATCH')
                if sc.get('question')!=eq or sc.get('question_sha256')!=sha(canon(eq)): ce.append('SENSOR_QUESTION_BINDING_MISMATCH')
                if int(sc.get('sensor_bit'))!=int(hc.get('sensor_bit')): ce.append('SENSOR_BIT_MISMATCH')
            if not pr or pr.get('action')!='PREDICT' or pr.get('prediction')!=hc.get('target'): ce.append('PREDICTION_TARGET_MISMATCH')
        case_receipts.append({'case_id':case_id,'expected_action':expected_action,'passed':not ce,'errors':ce})
        errors.extend(f'{case_id}:{x}' for x in ce)
    receipt={
      'schema':'arte.question_sensor_independent_verification_receipt/v57','batch_id':c.get('batch_id'),'challenge_id':cid,'generation':gen,'custodian_id':c.get('custodian_id'),
      'formal_freeze_sha256':freeze.get('freeze_sha256'),'commitment_batch_sha256':sha(canon(batch)),
      'public_input_sha256':sha(canon(public)),'questions_sha256':sha(canon(questions)),'sensor_input_sha256':sha(canon(sensor)),'predictions_sha256':sha(canon(preds)),
      'hidden_plaintext_sha256':sha(plain),'case_count':len(case_receipts),'passed_cases':sum(x['passed'] for x in case_receipts),
      'all_cases_passed':not errors,'errors':errors,'cases':case_receipts,
      'claim_boundary':{'this_receipt_alone_proves_independent_custody':False,'AGI':False,'ASI':False,'external_recursive_acceleration':False}
    }
    receipt['receipt_sha256']=sha(canon(receipt))
    return receipt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--commitment-batch',required=True);ap.add_argument('--freeze',required=True);ap.add_argument('--public-input',required=True);ap.add_argument('--questions',required=True);ap.add_argument('--sensor-input',required=True);ap.add_argument('--predictions',required=True);ap.add_argument('--reveal',required=True);ap.add_argument('--receipt-out',required=True)
    a=ap.parse_args();r=verify(a.commitment_batch,a.freeze,a.public_input,a.questions,a.sensor_input,a.predictions,a.reveal)
    Path(a.receipt_out).write_text(json.dumps(r,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(r,sort_keys=True)); raise SystemExit(0 if r['all_cases_passed'] else 2)

if __name__=='__main__':main()
