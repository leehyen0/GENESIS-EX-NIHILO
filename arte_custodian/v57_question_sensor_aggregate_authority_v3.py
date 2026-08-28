#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

BATCH_SCHEMA='arte.question_sensor_commitment_batch/v57'
FREEZE_SCHEMA='arte.question_sensor_candidate_freeze/v57-v3'
SNAPSHOT_SCHEMA='arte.question_sensor_candidate_prefreeze_snapshot/v57-v3'
RECEIPT_SCHEMA='arte.question_sensor_independent_verification_receipt/v57-v3'
PROVENANCE_SCHEMA='arte.question_sensor_external_custody_provenance/v57-v3'
POLICY_SCHEMA='arte.question_sensor_independent_authority_policy/v57'
AGG_SCHEMA='arte.question_sensor_aggregate_authority_receipt/v57-v3'
GENERATIONS=('G1','G2','G3')
POLICY_PATH='arte_custodian/v57_question_sensor_independent_authority_policy.json'


def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
def sha(b): return hashlib.sha256(b).hexdigest()
def sha_obj(x): return sha(canon(x))
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def self_hash_valid(x,field):
    d=dict(x); declared=d.pop(field,None)
    return bool(declared and declared==sha_obj(d))
def receipt_hash_ok(r): return self_hash_valid(r,'receipt_sha256')

def verify(batch_path,freeze_path,snapshot_path,policy_path,receipt_paths,provenance_path=None):
    errors=[]; warnings=[]
    batch=load(batch_path); freeze=load(freeze_path); snap=load(snapshot_path); policy=load(policy_path); receipts=[load(x) for x in receipt_paths]
    batch_sha=sha_obj(batch); policy_raw_sha=sha(Path(policy_path).read_bytes())

    if batch.get('schema')!=BATCH_SCHEMA: errors.append('BAD_BATCH_SCHEMA')
    if snap.get('schema')!=SNAPSHOT_SCHEMA or not self_hash_valid(snap,'snapshot_sha256'): errors.append('BAD_OR_TAMPERED_PREFREEZE_SNAPSHOT')
    if freeze.get('schema')!=FREEZE_SCHEMA or freeze.get('formal_candidate_freeze') is not True or not self_hash_valid(freeze,'freeze_sha256'): errors.append('BAD_OR_TAMPERED_FORMAL_FREEZE')
    if policy.get('schema')!=POLICY_SCHEMA: errors.append('BAD_AUTHORITY_POLICY_SCHEMA')
    if snap.get('files',{}).get(POLICY_PATH)!=policy_raw_sha: errors.append('AUTHORITY_POLICY_NOT_BOUND_TO_PREFREEZE_SNAPSHOT')
    if freeze.get('authority_policy_sha256')!=policy_raw_sha: errors.append('AUTHORITY_POLICY_NOT_BOUND_TO_FORMAL_FREEZE')
    if freeze.get('prefreeze_snapshot_sha256')!=snap.get('snapshot_sha256'): errors.append('FREEZE_SNAPSHOT_BINDING_MISMATCH')
    if freeze.get('files')!=snap.get('files'): errors.append('FREEZE_FILE_HASH_MAP_MISMATCH')
    if freeze.get('commitment_batch_sha256')!=batch_sha: errors.append('FREEZE_BATCH_SHA_MISMATCH')
    if freeze.get('commitment_batch_id')!=batch.get('batch_id'): errors.append('FREEZE_BATCH_ID_MISMATCH')
    if freeze.get('custodian_id')!=batch.get('custodian_id'): errors.append('FREEZE_CUSTODIAN_MISMATCH')
    if freeze.get('all_G1_G2_G3_commitments_present') is not True: errors.append('FREEZE_MISSING_THREE_GENERATION_ASSERTION')
    if batch.get('all_commitments_fixed_before_G1') is not True or batch.get('keys_and_plaintexts_withheld') is not True: errors.append('BATCH_PRECOMMIT_BOUNDARY_NOT_ASSERTED')

    commits={x.get('generation'):x for x in batch.get('commitments',[])}
    if len(batch.get('commitments',[]))!=3 or set(commits)!=set(GENERATIONS): errors.append('BATCH_GENERATION_SET_MUST_BE_EXACT_G1_G2_G3')
    if len({x.get('challenge_id') for x in batch.get('commitments',[])})!=3: errors.append('BATCH_CHALLENGE_IDS_MUST_BE_UNIQUE')

    if len(receipts)!=3: errors.append('EXACTLY_THREE_GENERATION_RECEIPTS_REQUIRED')
    bygen={}
    for r in receipts:
        g=r.get('generation')
        if g in bygen: errors.append('DUPLICATE_GENERATION_RECEIPT:'+str(g))
        bygen[g]=r
        if r.get('schema')!=RECEIPT_SCHEMA: errors.append('BAD_RECEIPT_SCHEMA:'+str(g))
        if not receipt_hash_ok(r): errors.append('RECEIPT_SELF_HASH_MISMATCH:'+str(g))
        if r.get('all_cases_passed') is not True or r.get('errors') not in ([],None): errors.append('GENERATION_FAILED:'+str(g))
        if not isinstance(r.get('case_count'),int) or r.get('case_count',0)<=0 or r.get('passed_cases')!=r.get('case_count'): errors.append('GENERATION_CASE_ACCOUNTING_INVALID:'+str(g))
        if r.get('formal_freeze_sha256')!=freeze.get('freeze_sha256'): errors.append('GENERATION_FREEZE_SHA_MISMATCH:'+str(g))
        if r.get('prefreeze_snapshot_sha256')!=snap.get('snapshot_sha256'): errors.append('GENERATION_SNAPSHOT_SHA_MISMATCH:'+str(g))
        if r.get('commitment_batch_sha256')!=batch_sha: errors.append('GENERATION_BATCH_SHA_MISMATCH:'+str(g))
        for field in ('formal_freeze_self_hash_valid','prefreeze_snapshot_self_hash_valid','question_output_hash_custodian_bound','prediction_output_hash_custodian_bound','stage_hash_chain_complete'):
            if r.get(field) is not True: errors.append('GENERATION_MISSING_STAGE_AUTHORITY:'+str(g)+':'+field)
        cb=r.get('claim_boundary',{})
        if cb.get('this_receipt_alone_proves_independent_custody') is not False or cb.get('AGI') is not False or cb.get('ASI') is not False or cb.get('external_recursive_acceleration') is not False:
            errors.append('UNSAFE_GENERATION_CLAIM_BOUNDARY:'+str(g))
    if set(bygen)!=set(GENERATIONS): errors.append('RECEIPT_GENERATION_SET_MUST_BE_G1_G2_G3')

    for g in GENERATIONS:
        r=bygen.get(g,{}); c=commits.get(g,{})
        if r.get('batch_id')!=batch.get('batch_id'): errors.append('RECEIPT_BATCH_ID_MISMATCH:'+g)
        if r.get('custodian_id')!=batch.get('custodian_id'): errors.append('RECEIPT_CUSTODIAN_MISMATCH:'+g)
        if r.get('challenge_id')!=c.get('challenge_id'): errors.append('RECEIPT_CHALLENGE_ID_MISMATCH:'+g)

    crypto_and_stage_complete=not errors
    provenance=None; external_account_distinct=False; independent_authority_verified=False
    if provenance_path:
        provenance=load(provenance_path)
        if provenance.get('schema')!=PROVENANCE_SCHEMA: errors.append('BAD_EXTERNAL_PROVENANCE_SCHEMA')
        if provenance.get('custodian_id')!=batch.get('custodian_id'): errors.append('PROVENANCE_CUSTODIAN_MISMATCH')
        if provenance.get('commitment_batch_sha256')!=batch_sha: errors.append('PROVENANCE_BATCH_SHA_MISMATCH')
        if provenance.get('formal_freeze_sha256')!=freeze.get('freeze_sha256'): errors.append('PROVENANCE_FREEZE_SHA_MISMATCH')
        owner=provenance.get('repository_owner_login'); actor=provenance.get('submission_actor_login')
        external_account_distinct=bool(actor and owner and actor!=owner and provenance.get('actor_distinct_from_repository_owner') is True)
        if not external_account_distinct: warnings.append('DISTINCT_EXTERNAL_SUBMISSION_ACCOUNT_NOT_ESTABLISHED')
        iv=provenance.get('independence_verification',{})
        verifier=iv.get('verifier_actor_login')
        independent_authority_verified=bool(
          external_account_distinct and iv.get('verified') is True and
          iv.get('verification_basis') in ('INDEPENDENT_THIRD_PARTY','INDEPENDENT_ORGANIZATION') and
          verifier and verifier not in {owner,actor} and iv.get('public_evidence_url') and
          iv.get('hidden_material_never_entered_candidate_boundary') is True
        )
        if not independent_authority_verified: warnings.append('INDEPENDENT_CONTROL_AUTHORITY_NOT_VERIFIED')
    else:
        warnings.extend(['EXTERNAL_CUSTODY_PROVENANCE_MISSING','INDEPENDENT_CONTROL_AUTHORITY_NOT_VERIFIED'])

    # Recompute after provenance structural errors may have been added.
    three_generation_chain_complete=not [e for e in errors if not e.startswith('BAD_EXTERNAL_PROVENANCE') and not e.startswith('PROVENANCE_')]
    stage_order_authority_candidate=bool(three_generation_chain_complete and external_account_distinct)
    e4=bool(not errors and independent_authority_verified)
    total_cases=sum(bygen.get(g,{}).get('case_count',0) or 0 for g in GENERATIONS)
    total_passed=sum(bygen.get(g,{}).get('passed_cases',0) or 0 for g in GENERATIONS)
    out={
      'schema':AGG_SCHEMA,'batch_id':batch.get('batch_id'),'custodian_id':batch.get('custodian_id'),
      'formal_freeze_sha256':freeze.get('freeze_sha256'),'prefreeze_snapshot_sha256':snap.get('snapshot_sha256'),'authority_policy_sha256':policy_raw_sha,
      'commitment_batch_sha256':batch_sha,'generation_receipt_sha256':{g:bygen.get(g,{}).get('receipt_sha256') for g in GENERATIONS},
      'generation_challenge_ids':{g:bygen.get(g,{}).get('challenge_id') for g in GENERATIONS},
      'total_cases':total_cases,'total_passed_cases':total_passed,
      'three_generation_crypto_and_stage_hash_chain_complete':three_generation_chain_complete,
      'external_submission_account_distinct':external_account_distinct,
      'stage_order_authority_candidate':stage_order_authority_candidate,
      'independent_control_authority_verified':independent_authority_verified,
      'E4_independent_custody_authorized':e4,
      'errors':errors,'warnings':warnings,
      'claim_boundary':{'independent_custody':e4,'AGI':False,'ASI':False,'external_recursive_acceleration':False,'global_recursive_self_improvement':False}
    }
    out['aggregate_receipt_sha256']=sha_obj(out)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--commitment-batch',required=True);ap.add_argument('--freeze',required=True);ap.add_argument('--prefreeze-snapshot',required=True);ap.add_argument('--authority-policy',required=True);ap.add_argument('--receipts',nargs=3,required=True);ap.add_argument('--external-provenance');ap.add_argument('--out',required=True)
    a=ap.parse_args();r=verify(a.commitment_batch,a.freeze,a.prefreeze_snapshot,a.authority_policy,a.receipts,a.external_provenance)
    Path(a.out).write_text(json.dumps(r,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(r,sort_keys=True))
    raise SystemExit(0 if r['three_generation_crypto_and_stage_hash_chain_complete'] else 2)

if __name__=='__main__': main()
