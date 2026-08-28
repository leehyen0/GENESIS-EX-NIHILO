#!/usr/bin/env python3
import argparse, base64, hashlib, json
from pathlib import Path

BATCH_SCHEMA='arte.question_sensor_commitment_batch/v57'
COMMIT_SCHEMA='arte.question_sensor_commitment/v57'
REQ={'schema','batch_id','challenge_id','generation','custodian_id','byte_length','plaintext_sha256','ciphertext_sha256','ciphertext_b64','public_packet_sha256','key_withheld','all_generations_fixed_before_G1','claim_boundary'}
HEX64=set('0123456789abcdef')

def is_sha(s): return isinstance(s,str) and len(s)==64 and set(s)<=HEX64
def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
def sha(b): return hashlib.sha256(b).hexdigest()

def validate(path):
    b=json.loads(Path(path).read_text(encoding='utf-8')); errors=[]
    if b.get('schema')!=BATCH_SCHEMA: errors.append('BAD_BATCH_SCHEMA')
    if b.get('all_commitments_fixed_before_G1') is not True: errors.append('BATCH_NOT_DECLARED_FIXED_BEFORE_G1')
    if b.get('keys_and_plaintexts_withheld') is not True: errors.append('KEYS_OR_PLAINTEXTS_NOT_DECLARED_WITHHELD')
    if b.get('claim_boundary')!={'AGI':False,'ASI':False,'independent_organization_custody':False}: errors.append('BAD_BATCH_CLAIM_BOUNDARY')
    cs=b.get('commitments')
    if not isinstance(cs,list) or len(cs)!=3: errors.append('EXACTLY_THREE_COMMITMENTS_REQUIRED'); cs=cs if isinstance(cs,list) else []
    gens=[]; ids=[]; custs=[]; batches=[]; details=[]
    for i,c in enumerate(cs):
        ce=[]
        if not isinstance(c,dict): ce.append('COMMITMENT_NOT_OBJECT'); details.append({'index':i,'errors':ce}); continue
        missing=sorted(REQ-set(c)); extra=sorted(set(c)-REQ)
        if missing: ce.append('MISSING_FIELDS:'+','.join(missing))
        if extra: ce.append('UNEXPECTED_FIELDS:'+','.join(extra))
        if c.get('schema')!=COMMIT_SCHEMA: ce.append('BAD_COMMITMENT_SCHEMA')
        gen=c.get('generation'); gens.append(gen); ids.append(c.get('challenge_id')); custs.append(c.get('custodian_id')); batches.append(c.get('batch_id'))
        if gen not in ('G1','G2','G3'): ce.append('BAD_GENERATION')
        if not c.get('challenge_id'): ce.append('MISSING_CHALLENGE_ID')
        if not c.get('custodian_id'): ce.append('MISSING_CUSTODIAN_ID')
        if c.get('key_withheld') is not True: ce.append('KEY_NOT_WITHHELD')
        if c.get('all_generations_fixed_before_G1') is not True: ce.append('COMMITMENT_NOT_FIXED_BEFORE_G1')
        if c.get('claim_boundary')!={'AGI':False,'ASI':False,'independent_organization_custody':False}: ce.append('BAD_COMMITMENT_CLAIM_BOUNDARY')
        for f in ('plaintext_sha256','ciphertext_sha256','public_packet_sha256'):
            if not is_sha(c.get(f)): ce.append('BAD_SHA256:'+f)
        try: ct=base64.b64decode(c.get('ciphertext_b64',''),validate=True)
        except Exception: ct=b''; ce.append('INVALID_CIPHERTEXT_BASE64')
        n=c.get('byte_length')
        if not isinstance(n,int) or n<=0: ce.append('BAD_BYTE_LENGTH')
        elif len(ct)!=n: ce.append(f'CIPHERTEXT_LENGTH_MISMATCH:{len(ct)}!={n}')
        actual=sha(ct)
        if is_sha(c.get('ciphertext_sha256')) and actual!=c['ciphertext_sha256']: ce.append('CIPHERTEXT_SHA256_MISMATCH')
        details.append({'generation':gen,'challenge_id':c.get('challenge_id'),'decoded_ciphertext_bytes':len(ct),'declared_byte_length':n,'actual_ciphertext_sha256':actual,'errors':ce})
        errors.extend(f'{gen or i}:{x}' for x in ce)
    if sorted(gens)!=['G1','G2','G3']: errors.append('GENERATION_SET_MUST_BE_G1_G2_G3')
    if len(ids)!=len(set(ids)): errors.append('CHALLENGE_IDS_NOT_UNIQUE')
    if len({x for x in custs if x})!=1: errors.append('CUSTODIAN_ID_NOT_UNIFORM')
    if len({x for x in batches if x})!=1: errors.append('BATCH_ID_NOT_UNIFORM')
    if batches and b.get('batch_id')!=batches[0]: errors.append('TOPLEVEL_BATCH_ID_MISMATCH')
    if custs and b.get('custodian_id')!=custs[0]: errors.append('TOPLEVEL_CUSTODIAN_ID_MISMATCH')
    report={
      'schema':'arte.question_sensor_commitment_intake_audit/v57','status':'ACCEPT_STRUCTURAL_COMMITMENTS__READY_FOR_CANDIDATE_FREEZE' if not errors else 'REJECT_INVALID_COMMITMENTS__DO_NOT_FREEZE_G1',
      'structurally_valid':not errors,'commitment_batch_sha256':sha(canon(b)),'errors':errors,'details':details,
      'authority_boundary':{
        'structural_validity_does_not_prove_external_independence':True,
        'candidate_freeze_allowed':not errors,
        'independent_custody_proven':False,'AGI':False,'ASI':False
      }
    }
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('commitment_batch');ap.add_argument('--audit-out');a=ap.parse_args()
    r=validate(a.commitment_batch); txt=json.dumps(r,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
    if a.audit_out: Path(a.audit_out).write_text(txt,encoding='utf-8')
    print(txt,end='')
    raise SystemExit(0 if r['structurally_valid'] else 2)

if __name__=='__main__':main()
