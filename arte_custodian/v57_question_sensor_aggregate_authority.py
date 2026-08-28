#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

BATCH_SCHEMA='arte.question_sensor_commitment_batch/v57'
FREEZE_SCHEMA='arte.question_sensor_candidate_freeze/v57-v2'
RECEIPT_SCHEMA='arte.question_sensor_independent_verification_receipt/v57'
PROVENANCE_SCHEMA='arte.question_sensor_external_custody_provenance/v57'
AGG_SCHEMA='arte.question_sensor_aggregate_authority_receipt/v57'
GENERATIONS=('G1','G2','G3')


def canon(x):
    return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')

def sha_obj(x):
    return hashlib.sha256(canon(x)).hexdigest()

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def receipt_hash_ok(r):
    declared=r.get('receipt_sha256')
    base=dict(r); base.pop('receipt_sha256',None)
    return declared==sha_obj(base)

def verify(batch_path, freeze_path, receipt_paths, provenance_path=None):
    errors=[]; warnings=[]
    batch=load(batch_path); freeze=load(freeze_path)
    receipts=[load(p) for p in receipt_paths]

    if batch.get('schema')!=BATCH_SCHEMA: errors.append('BAD_BATCH_SCHEMA')
    if freeze.get('schema')!=FREEZE_SCHEMA: errors.append('BAD_FREEZE_SCHEMA')
    if freeze.get('formal_candidate_freeze') is not True: errors.append('FORMAL_FREEZE_REQUIRED')
    batch_sha=sha_obj(batch)
    if freeze.get('commitment_batch_sha256')!=batch_sha: errors.append('FREEZE_BATCH_SHA_MISMATCH')
    if freeze.get('all_G1_G2_G3_commitments_present') is not True: errors.append('FREEZE_DID_NOT_ASSERT_ALL_GENERATIONS')

    if len(receipts)!=3: errors.append('EXACTLY_THREE_GENERATION_RECEIPTS_REQUIRED')
    bygen={}
    for r in receipts:
        g=r.get('generation')
        if g in bygen: errors.append('DUPLICATE_GENERATION_RECEIPT:'+str(g))
        bygen[g]=r
        if r.get('schema')!=RECEIPT_SCHEMA: errors.append('BAD_RECEIPT_SCHEMA:'+str(g))
        if not receipt_hash_ok(r): errors.append('RECEIPT_SELF_HASH_MISMATCH:'+str(g))
        if r.get('all_cases_passed') is not True: errors.append('GENERATION_NOT_ALL_CASES_PASSED:'+str(g))
        if r.get('errors') not in ([],None): errors.append('GENERATION_RECEIPT_HAS_ERRORS:'+str(g))
        if not isinstance(r.get('case_count'),int) or r.get('case_count',0)<=0: errors.append('EMPTY_GENERATION:'+str(g))
        if r.get('passed_cases')!=r.get('case_count'): errors.append('PASSED_CASE_COUNT_MISMATCH:'+str(g))
        if r.get('commitment_batch_sha256')!=batch_sha: errors.append('GENERATION_BATCH_SHA_MISMATCH:'+str(g))
        if r.get('formal_freeze_sha256')!=freeze.get('freeze_sha256'): errors.append('GENERATION_FREEZE_SHA_MISMATCH:'+str(g))
        cb=r.get('claim_boundary',{})
        if cb.get('this_receipt_alone_proves_independent_custody') is not False: errors.append('UNSAFE_RECEIPT_AUTHORITY_CLAIM:'+str(g))
        if cb.get('AGI') is not False or cb.get('ASI') is not False or cb.get('external_recursive_acceleration') is not False: errors.append('UNSAFE_RECEIPT_CLAIM_BOUNDARY:'+str(g))

    if set(bygen)!=set(GENERATIONS): errors.append('GENERATION_SET_MUST_BE_G1_G2_G3')

    commitments={c.get('generation'):c for c in batch.get('commitments',[])}
    if set(commitments)!=set(GENERATIONS): errors.append('BATCH_GENERATION_SET_MUST_BE_G1_G2_G3')
    challenge_ids=[]
    for g in GENERATIONS:
        r=bygen.get(g,{}) ; c=commitments.get(g,{})
        if r.get('batch_id')!=batch.get('batch_id'): errors.append('RECEIPT_BATCH_ID_MISMATCH:'+g)
        if r.get('custodian_id')!=batch.get('custodian_id'): errors.append('RECEIPT_CUSTODIAN_MISMATCH:'+g)
        if r.get('challenge_id')!=c.get('challenge_id'): errors.append('RECEIPT_CHALLENGE_ID_MISMATCH:'+g)
        challenge_ids.append(r.get('challenge_id'))
    if None in challenge_ids or len(set(challenge_ids))!=3: errors.append('CHALLENGE_IDS_MUST_BE_THREE_UNIQUE_VALUES')
    if freeze.get('commitment_batch_id')!=batch.get('batch_id'): errors.append('FREEZE_BATCH_ID_MISMATCH')
    if freeze.get('custodian_id')!=batch.get('custodian_id'): errors.append('FREEZE_CUSTODIAN_ID_MISMATCH')

    provenance=None
    external_account_distinct=False
    independent_authority_verified=False
    if provenance_path:
        provenance=load(provenance_path)
        if provenance.get('schema')!=PROVENANCE_SCHEMA: errors.append('BAD_EXTERNAL_PROVENANCE_SCHEMA')
        if provenance.get('custodian_id')!=batch.get('custodian_id'): errors.append('PROVENANCE_CUSTODIAN_MISMATCH')
        if provenance.get('commitment_batch_sha256')!=batch_sha: errors.append('PROVENANCE_BATCH_SHA_MISMATCH')
        if provenance.get('formal_freeze_sha256')!=freeze.get('freeze_sha256'): errors.append('PROVENANCE_FREEZE_SHA_MISMATCH')
        owner=provenance.get('repository_owner_login')
        actor=provenance.get('submission_actor_login')
        external_account_distinct=bool(actor and owner and actor!=owner and provenance.get('actor_distinct_from_repository_owner') is True)
        if not external_account_distinct: warnings.append('NO_DISTINCT_EXTERNAL_SUBMISSION_ACTOR_ESTABLISHED')
        # A distinct account is useful provenance but is not, by itself, proof that the account is independently controlled.
        iv=provenance.get('independence_verification',{})
        verifier=iv.get('verifier_actor_login')
        independent_authority_verified=bool(
            external_account_distinct and
            iv.get('verified') is True and
            iv.get('verification_basis') in ('INDEPENDENT_THIRD_PARTY','INDEPENDENT_ORGANIZATION') and
            verifier and verifier not in {owner,actor} and
            iv.get('public_evidence_url')
        )
        if not independent_authority_verified: warnings.append('INDEPENDENT_AUTHORITY_NOT_VERIFIED')
    else:
        warnings.append('EXTERNAL_CUSTODY_PROVENANCE_MISSING')
        warnings.append('INDEPENDENT_AUTHORITY_NOT_VERIFIED')

    crypto_complete=not errors
    total_cases=sum(bygen.get(g,{}).get('case_count',0) or 0 for g in GENERATIONS)
    total_passed=sum(bygen.get(g,{}).get('passed_cases',0) or 0 for g in GENERATIONS)
    e4_authorized=bool(crypto_complete and independent_authority_verified)

    out={
      'schema':AGG_SCHEMA,
      'batch_id':batch.get('batch_id'),
      'custodian_id':batch.get('custodian_id'),
      'formal_freeze_sha256':freeze.get('freeze_sha256'),
      'commitment_batch_sha256':batch_sha,
      'generation_receipt_sha256':{g:bygen.get(g,{}).get('receipt_sha256') for g in GENERATIONS},
      'generation_challenge_ids':{g:bygen.get(g,{}).get('challenge_id') for g in GENERATIONS},
      'generation_pass':{g:bool(bygen.get(g,{}).get('all_cases_passed')) for g in GENERATIONS},
      'total_cases':total_cases,
      'total_passed_cases':total_passed,
      'three_generation_cryptographic_chain_complete':crypto_complete,
      'external_submission_account_distinct':external_account_distinct,
      'independent_authority_verified':independent_authority_verified,
      'E4_independent_custody_authorized':e4_authorized,
      'errors':errors,
      'warnings':warnings,
      'claim_boundary':{
        'AGI':False,
        'ASI':False,
        'external_recursive_acceleration':False,
        'global_recursive_self_improvement':False,
        'independent_custody':e4_authorized
      }
    }
    out['aggregate_receipt_sha256']=sha_obj(out)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--commitment-batch',required=True)
    ap.add_argument('--freeze',required=True)
    ap.add_argument('--receipts',nargs=3,required=True)
    ap.add_argument('--external-provenance')
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    r=verify(a.commitment_batch,a.freeze,a.receipts,a.external_provenance)
    Path(a.out).write_text(json.dumps(r,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(r,sort_keys=True))
    raise SystemExit(0 if r['three_generation_cryptographic_chain_complete'] else 2)

if __name__=='__main__':
    main()
