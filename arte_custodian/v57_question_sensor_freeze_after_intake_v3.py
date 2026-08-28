#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
import v57_question_sensor_commitment_intake as intake

SNAPSHOT_PATH=Path(__file__).with_name('v57_question_sensor_candidate_prefreeze_snapshot_v3_r2.json')
SNAPSHOT_SCHEMA='arte.question_sensor_candidate_prefreeze_snapshot/v57-v3'
FREEZE_SCHEMA='arte.question_sensor_candidate_freeze/v57-v3'
POLICY_PATH='arte_custodian/v57_question_sensor_independent_authority_policy.json'
FINAL_VERIFIER_PATH='arte_custodian/v57_question_sensor_final_verifier_v3.py'
SEMANTICS_VERIFIER_PATH='arte_custodian/v57_question_sensor_challenge_semantics_v3.py'
AGGREGATE_VERIFIER_PATH='arte_custodian/v57_question_sensor_aggregate_authority_v3.py'
CUSTODIAN_PROTOCOL_PATH='arte_custodian/v57_question_sensor_custodian_protocol_v3.py'
PUBLIC_SPEC_PATH='arte_custodian/v57_epoch20_independent_question_sensor_public_spec_v3.json'


def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
def sha(b): return hashlib.sha256(b).hexdigest()
def self_hash_valid(x,field):
    d=dict(x); declared=d.pop(field,None)
    return bool(declared and declared==sha(canon(d)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('commitment_batch');ap.add_argument('--freeze-out',required=True);a=ap.parse_args()
    snap=json.loads(SNAPSHOT_PATH.read_text(encoding='utf-8'))
    if snap.get('schema')!=SNAPSHOT_SCHEMA or not self_hash_valid(snap,'snapshot_sha256'):
        raise SystemExit('VALID_V3_PREFREEZE_SNAPSHOT_REQUIRED')
    audit=intake.validate(a.commitment_batch)
    if not audit['structurally_valid']:
        raise SystemExit('COMMITMENT_INTAKE_REJECTED__DO_NOT_FREEZE_G1\n'+json.dumps(audit,sort_keys=True))
    repo=Path(__file__).resolve().parents[1]; mismatches=[]
    for rel,expected in snap['files'].items():
        p=repo/rel
        if not p.exists(): mismatches.append({'path':rel,'error':'MISSING'})
        else:
            actual=sha(p.read_bytes())
            if actual!=expected: mismatches.append({'path':rel,'expected':expected,'actual':actual})
    if mismatches:
        raise SystemExit('V3_FROZEN_FILE_CHANGED__REFUSE_FORMAL_FREEZE\n'+json.dumps(mismatches,sort_keys=True))
    batch=json.loads(Path(a.commitment_batch).read_text(encoding='utf-8'))
    freeze={
      'schema':FREEZE_SCHEMA,'formal_candidate_freeze':True,
      'candidate_content_anchor_commit':snap['candidate_content_anchor_commit'],'prefreeze_snapshot_sha256':snap['snapshot_sha256'],
      'body_state_sha256':snap['body_state_sha256'],'generated_operator_sha256':snap['generated_operator_sha256'],
      'learned_question_bank_canonical_sha256':snap['learned_question_bank_canonical_sha256'],'files':snap['files'],
      'authority_policy_sha256':snap['files'][POLICY_PATH],'per_generation_verifier_sha256':snap['files'][FINAL_VERIFIER_PATH],
      'challenge_semantics_verifier_sha256':snap['files'][SEMANTICS_VERIFIER_PATH],
      'aggregate_authority_verifier_sha256':snap['files'][AGGREGATE_VERIFIER_PATH],'custodian_protocol_sha256':snap['files'][CUSTODIAN_PROTOCOL_PATH],
      'public_spec_sha256':snap['files'][PUBLIC_SPEC_PATH],
      'commitment_batch_sha256':audit['commitment_batch_sha256'],'commitment_batch_id':batch['batch_id'],'custodian_id':batch['custodian_id'],
      'intake_status':audit['status'],'all_G1_G2_G3_commitments_present':True,
      'stage_hash_binding_protocol_required':True,'independent_challenge_semantics_required':True,
      'repository_local_E4_self_certification_forbidden':True,
      'public_packet_revealed_before_freeze':False,'sensor_bits_revealed_before_freeze':False,'targets_or_otp_keys_revealed_before_freeze':False,
      'independent_custody_proven':False,
      'claim_boundary':{'AGI':False,'ASI':False,'external_recursive_acceleration':False,'global_recursive_self_improvement':False}
    }
    freeze['freeze_sha256']=sha(canon(freeze))
    Path(a.freeze_out).write_text(json.dumps(freeze,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(freeze,sort_keys=True))

if __name__=='__main__': main()
