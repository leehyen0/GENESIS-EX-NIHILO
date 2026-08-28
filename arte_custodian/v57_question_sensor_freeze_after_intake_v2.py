#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
import v57_question_sensor_commitment_intake as intake

SNAPSHOT_PATH=Path(__file__).with_name('v57_question_sensor_candidate_prefreeze_snapshot_v2.json')

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('commitment_batch');ap.add_argument('--freeze-out',required=True);a=ap.parse_args()
    snap=json.loads(SNAPSHOT_PATH.read_text(encoding='utf-8'))
    if snap.get('schema')!='arte.question_sensor_candidate_prefreeze_snapshot/v57-v2': raise SystemExit('V2_PREFREEZE_SNAPSHOT_REQUIRED')
    audit=intake.validate(a.commitment_batch)
    if not audit['structurally_valid']:
        raise SystemExit('COMMITMENT_INTAKE_REJECTED__DO_NOT_FREEZE_G1\n'+json.dumps(audit,sort_keys=True))
    repo=Path(__file__).resolve().parents[1]; mismatches=[]
    for rel,expected in snap['files'].items():
        p=repo/rel
        if not p.exists(): mismatches.append({'path':rel,'error':'MISSING'})
        else:
            actual=sha_bytes(p.read_bytes())
            if actual!=expected:mismatches.append({'path':rel,'expected':expected,'actual':actual})
    if mismatches:
        raise SystemExit('CANDIDATE_OR_VERIFIER_CHANGED_SINCE_V2_PREFREEZE__REFUSE_FORMAL_FREEZE\n'+json.dumps(mismatches,sort_keys=True))
    batch=json.loads(Path(a.commitment_batch).read_text(encoding='utf-8'))
    freeze={
      'schema':'arte.question_sensor_candidate_freeze/v57-v2','formal_candidate_freeze':True,
      'candidate_content_anchor_commit':snap['candidate_content_anchor_commit'],'prefreeze_snapshot_sha256':snap['snapshot_sha256'],
      'body_state_sha256':snap['body_state_sha256'],'generated_operator_sha256':snap['generated_operator_sha256'],
      'learned_question_bank_canonical_sha256':snap['learned_question_bank_canonical_sha256'],'files':snap['files'],
      'commitment_batch_sha256':audit['commitment_batch_sha256'],'commitment_batch_id':batch['batch_id'],'custodian_id':batch['custodian_id'],
      'intake_status':audit['status'],'all_G1_G2_G3_commitments_present':True,
      'public_packet_revealed_before_freeze':False,'sensor_bits_revealed_before_freeze':False,'targets_revealed_before_freeze':False,
      'independent_custody_proven':False,
      'claim_boundary':{'AGI':False,'ASI':False,'external_recursive_acceleration':False}
    }
    freeze['freeze_sha256']=sha_bytes(canon(freeze))
    Path(a.freeze_out).write_text(json.dumps(freeze,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(freeze,sort_keys=True))

if __name__=='__main__':main()
