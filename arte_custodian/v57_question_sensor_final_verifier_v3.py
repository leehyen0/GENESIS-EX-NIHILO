#!/usr/bin/env python3
import argparse, base64, hashlib, json
from pathlib import Path

HIDDEN_SCHEMA='arte.hidden_question_sensor_challenge/v57'
COMMIT_SCHEMA='arte.question_sensor_commitment/v57'
BATCH_SCHEMA='arte.question_sensor_commitment_batch/v57'
FREEZE_SCHEMA='arte.question_sensor_candidate_freeze/v57-v3'
SNAPSHOT_SCHEMA='arte.question_sensor_candidate_prefreeze_snapshot/v57-v3'
REVEAL_SCHEMA='arte.question_sensor_reveal/v57-v3'
QUESTION_INPUT_SCHEMA='arte.independent_question_input/v57'
QUESTION_OUTPUT_SCHEMA='arte.independent_question_output/v57'
SENSOR_INPUT_SCHEMA='arte.independent_sensor_input/v57'
PREDICTION_OUTPUT_SCHEMA='arte.independent_prediction_output/v57'
RECEIPT_SCHEMA='arte.question_sensor_independent_verification_receipt/v57-v3'
GENERATIONS=('G1','G2','G3')
POLICY_PATH='arte_custodian/v57_question_sensor_independent_authority_policy.json'


def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
def sha(b): return hashlib.sha256(b).hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def self_hash_valid(x,field):
    d=dict(x); declared=d.pop(field,None)
    return bool(declared and declared==sha(canon(d)))

def verify(batch_path,freeze_path,snapshot_path,public_input_path,questions_path,sensor_input_path,predictions_path,reveal_path):
    errors=[]
    def fail(x): errors.append(x)
    batch=load(batch_path); freeze=load(freeze_path); snap=load(snapshot_path); public=load(public_input_path); questions=load(questions_path); sensor=load(sensor_input_path); preds=load(predictions_path); reveal=load(reveal_path)

    # Root/freeze integrity.
    if batch.get('schema')!=BATCH_SCHEMA: fail('BAD_BATCH_SCHEMA')
    if snap.get('schema')!=SNAPSHOT_SCHEMA: fail('BAD_PREFREEZE_SNAPSHOT_SCHEMA')
    snapshot_hash_ok=self_hash_valid(snap,'snapshot_sha256')
    if not snapshot_hash_ok: fail('PREFREEZE_SNAPSHOT_SELF_HASH_MISMATCH')
    if freeze.get('schema')!=FREEZE_SCHEMA or freeze.get('formal_candidate_freeze') is not True: fail('FORMAL_V3_FREEZE_REQUIRED')
    freeze_hash_ok=self_hash_valid(freeze,'freeze_sha256')
    if not freeze_hash_ok: fail('FORMAL_FREEZE_SELF_HASH_MISMATCH')
    if freeze.get('prefreeze_snapshot_sha256')!=snap.get('snapshot_sha256'): fail('FREEZE_PREFREEZE_SNAPSHOT_BINDING_MISMATCH')
    if freeze.get('files')!=snap.get('files'): fail('FREEZE_FILE_HASH_MAP_MISMATCH')
    if freeze.get('authority_policy_sha256')!=snap.get('files',{}).get(POLICY_PATH): fail('FREEZE_AUTHORITY_POLICY_BINDING_MISMATCH')
    batch_sha=sha(canon(batch))
    if freeze.get('commitment_batch_sha256')!=batch_sha: fail('FREEZE_BATCH_SHA_MISMATCH')
    if freeze.get('commitment_batch_id')!=batch.get('batch_id'): fail('FREEZE_BATCH_ID_MISMATCH')
    if freeze.get('custodian_id')!=batch.get('custodian_id'): fail('FREEZE_CUSTODIAN_MISMATCH')
    if batch.get('all_commitments_fixed_before_G1') is not True or batch.get('keys_and_plaintexts_withheld') is not True: fail('BATCH_PRECOMMIT_BOUNDARY_NOT_ASSERTED')
    commits=batch.get('commitments',[])
    if len(commits)!=3 or {x.get('generation') for x in commits}!=set(GENERATIONS): fail('BATCH_MUST_HAVE_EXACT_G1_G2_G3')
    if len({x.get('challenge_id') for x in commits})!=3: fail('BATCH_CHALLENGE_IDS_NOT_UNIQUE')

    # Select exact generation commitment.
    if public.get('schema')!=QUESTION_INPUT_SCHEMA: fail('BAD_PUBLIC_INPUT_SCHEMA')
    cid=public.get('challenge_id'); gen=public.get('generation')
    matches=[c for c in commits if c.get('challenge_id')==cid and c.get('generation')==gen]
    c=matches[0] if len(matches)==1 else {}
    if len(matches)!=1: fail('COMMITMENT_NOT_UNIQUE_FOR_PUBLIC_INPUT')
    if c.get('schema')!=COMMIT_SCHEMA: fail('BAD_COMMITMENT_SCHEMA')
    for k in ('batch_id','challenge_id','generation','custodian_id'):
        if k=='batch_id' and c.get(k)!=batch.get(k): fail('COMMITMENT_BATCH_BINDING_MISMATCH')
        if k=='custodian_id' and c.get(k)!=batch.get(k): fail('COMMITMENT_CUSTODIAN_BINDING_MISMATCH')
    if c.get('all_generations_fixed_before_G1') is not True or c.get('key_withheld') is not True: fail('COMMITMENT_PRECOMMIT_BOUNDARY_NOT_ASSERTED')

    # Stage schema/binding.
    if questions.get('schema')!=QUESTION_OUTPUT_SCHEMA or questions.get('challenge_id')!=cid or questions.get('generation')!=gen: fail('BAD_QUESTION_OUTPUT_BINDING')
    if sensor.get('schema')!=SENSOR_INPUT_SCHEMA or sensor.get('challenge_id')!=cid or sensor.get('generation')!=gen: fail('BAD_SENSOR_INPUT_BINDING')
    if preds.get('schema')!=PREDICTION_OUTPUT_SCHEMA or preds.get('challenge_id')!=cid or preds.get('generation')!=gen: fail('BAD_PREDICTION_OUTPUT_BINDING')
    if reveal.get('schema')!=REVEAL_SCHEMA: fail('BAD_V3_REVEAL_SCHEMA')
    for f in ('batch_id','challenge_id','generation','custodian_id'):
        expected=c.get(f)
        if reveal.get(f)!=expected: fail('REVEAL_BINDING_MISMATCH:'+f)
    expected_bank=snap.get('learned_question_bank_canonical_sha256')
    if questions.get('question_bank_sha256')!=expected_bank: fail('QUESTION_OUTPUT_BANK_SHA_MISMATCH')
    if preds.get('question_bank_sha256')!=expected_bank: fail('PREDICTION_OUTPUT_BANK_SHA_MISMATCH')

    q_sha=sha(canon(questions)); s_sha=sha(canon(sensor)); p_sha=sha(canon(preds))
    q_bound=(sensor.get('formal_freeze_sha256')==freeze.get('freeze_sha256') and sensor.get('question_output_sha256')==q_sha and sensor.get('sensor_bits_released_only_after_exact_question_output_hash_received') is True)
    if not q_bound: fail('SENSOR_RESPONSE_NOT_BOUND_TO_EXACT_QUESTION_OUTPUT')
    if sensor.get('batch_id')!=batch.get('batch_id') or sensor.get('custodian_id')!=batch.get('custodian_id'): fail('SENSOR_CUSTODIAN_OR_BATCH_BINDING_MISMATCH')
    p_bound=(reveal.get('formal_freeze_sha256')==freeze.get('freeze_sha256') and reveal.get('question_output_sha256')==q_sha and reveal.get('sensor_input_sha256')==s_sha and reveal.get('prediction_output_sha256')==p_sha and reveal.get('reveal_only_after_exact_prediction_output_hash_received') is True)
    if not p_bound: fail('REVEAL_NOT_BOUND_TO_EXACT_PREDICTION_OUTPUT')

    # OTP reveal and original hidden bytes.
    try: ct=base64.b64decode(c.get('ciphertext_b64',''),validate=True)
    except Exception: ct=b''; fail('INVALID_CIPHERTEXT_BASE64')
    try: key=base64.b64decode(reveal.get('key_b64',''),validate=True)
    except Exception: key=b''; fail('INVALID_KEY_BASE64')
    if len(ct)!=c.get('byte_length'): fail('CIPHERTEXT_LENGTH_MISMATCH')
    if len(key)!=len(ct): fail('OTP_KEY_LENGTH_MISMATCH')
    if sha(ct)!=c.get('ciphertext_sha256'): fail('CIPHERTEXT_SHA_MISMATCH')
    if sha(key)!=reveal.get('key_sha256'): fail('KEY_SHA_MISMATCH')
    plain=bytes(a^b for a,b in zip(ct,key)) if len(ct)==len(key) else b''
    if sha(plain)!=c.get('plaintext_sha256'): fail('PLAINTEXT_SHA_MISMATCH')
    try: hidden=json.loads(plain.decode('utf-8'))
    except Exception: hidden={}; fail('DECRYPTED_PLAINTEXT_NOT_VALID_UTF8_JSON')
    if hidden.get('schema')!=HIDDEN_SCHEMA: fail('BAD_HIDDEN_SCHEMA')
    for f in ('batch_id','challenge_id','generation'):
        if hidden.get(f)!=c.get(f): fail('HIDDEN_COMMITMENT_BINDING_MISMATCH:'+f)
    if hidden.get('claim_boundary')!={'AGI':False,'ASI':False}: fail('BAD_HIDDEN_CLAIM_BOUNDARY')
    if sha(canon(public))!=c.get('public_packet_sha256'): fail('PUBLIC_PACKET_COMMITMENT_MISMATCH')
    if hidden.get('public_packet')!=public: fail('PUBLIC_PACKET_HIDDEN_PACKAGE_MISMATCH')

    # Case-level semantic verification.
    pub_cases={x.get('case_id'):x for x in public.get('cases',[])}
    q_cases={x.get('case_id'):x for x in questions.get('questions',[])}
    s_cases={x.get('case_id'):x for x in sensor.get('cases',[])}
    p_cases={x.get('case_id'):x for x in preds.get('predictions',[])}
    h_cases={x.get('case_id'):x for x in hidden.get('cases',[])}
    ids=set(pub_cases)
    if None in ids or len(ids)!=len(public.get('cases',[])): fail('PUBLIC_CASE_IDS_MISSING_OR_DUPLICATE')
    if ids!=set(h_cases) or ids!=set(q_cases) or ids!=set(s_cases) or ids!=set(p_cases): fail('STAGE_CASE_ID_SET_MISMATCH')
    case_receipts=[]
    for case_id in sorted(ids,key=str):
        pc=pub_cases[case_id]; hc=h_cases.get(case_id,{}); qc=q_cases.get(case_id,{}); sc=s_cases.get(case_id,{}); pr=p_cases.get(case_id,{})
        ce=[]; expected_action=hc.get('expected_action','QUESTION')
        if sc.get('public_observation')!=pc.get('public_observation'): ce.append('SENSOR_PUBLIC_OBSERVATION_MISMATCH')
        if expected_action=='HOLD':
            if qc.get('action')!='HOLD': ce.append('EXPECTED_QUESTION_PHASE_HOLD')
            if 'sensor_bit' in sc: ce.append('SENSOR_BIT_RELEASED_FOR_HOLD_CASE')
            if pr.get('action')!='HOLD': ce.append('EXPECTED_PREDICTION_PHASE_HOLD')
        else:
            eq=hc.get('expected_question')
            eqsha=sha(canon(eq)) if isinstance(eq,dict) else None
            if qc.get('action')!='QUESTION' or qc.get('question')!=eq or qc.get('question_sha256')!=eqsha: ce.append('QUESTION_MISMATCH')
            if sc.get('question')!=eq or sc.get('question_sha256')!=eqsha: ce.append('SENSOR_QUESTION_BINDING_MISMATCH')
            try: bit=int(sc.get('sensor_bit'))
            except Exception: bit=-1
            if bit!=int(hc.get('sensor_bit')): ce.append('SENSOR_BIT_MISMATCH')
            if pr.get('action')!='PREDICT' or pr.get('prediction')!=hc.get('target'): ce.append('PREDICTION_TARGET_MISMATCH')
        case_receipts.append({'case_id':case_id,'expected_action':expected_action,'passed':not ce,'errors':ce})
        errors.extend(f'{case_id}:{x}' for x in ce)

    receipt={
      'schema':RECEIPT_SCHEMA,'batch_id':c.get('batch_id'),'challenge_id':cid,'generation':gen,'custodian_id':c.get('custodian_id'),
      'formal_freeze_sha256':freeze.get('freeze_sha256'),'prefreeze_snapshot_sha256':snap.get('snapshot_sha256'),'commitment_batch_sha256':batch_sha,
      'public_input_sha256':sha(canon(public)),'questions_sha256':q_sha,'sensor_input_sha256':s_sha,'predictions_sha256':p_sha,
      'hidden_plaintext_sha256':sha(plain),'formal_freeze_self_hash_valid':freeze_hash_ok,'prefreeze_snapshot_self_hash_valid':snapshot_hash_ok,
      'question_output_hash_custodian_bound':q_bound,'prediction_output_hash_custodian_bound':p_bound,'stage_hash_chain_complete':bool(q_bound and p_bound),
      'case_count':len(case_receipts),'passed_cases':sum(x['passed'] for x in case_receipts),'all_cases_passed':not errors,'errors':errors,'cases':case_receipts,
      'claim_boundary':{'this_receipt_alone_proves_independent_custody':False,'AGI':False,'ASI':False,'external_recursive_acceleration':False}
    }
    receipt['receipt_sha256']=sha(canon(receipt))
    return receipt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--commitment-batch',required=True);ap.add_argument('--freeze',required=True);ap.add_argument('--prefreeze-snapshot',required=True);ap.add_argument('--public-input',required=True);ap.add_argument('--questions',required=True);ap.add_argument('--sensor-input',required=True);ap.add_argument('--predictions',required=True);ap.add_argument('--reveal',required=True);ap.add_argument('--receipt-out',required=True)
    a=ap.parse_args(); r=verify(a.commitment_batch,a.freeze,a.prefreeze_snapshot,a.public_input,a.questions,a.sensor_input,a.predictions,a.reveal)
    Path(a.receipt_out).write_text(json.dumps(r,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(r,sort_keys=True)); raise SystemExit(0 if r['all_cases_passed'] else 2)

if __name__=='__main__': main()
