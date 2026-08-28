#!/usr/bin/env python3
import json, pathlib, subprocess, sys

root=pathlib.Path('.').resolve(); td=root/'v3_selftest'; td.mkdir(exist_ok=True)
CUST='arte_custodian/v57_question_sensor_custodian_protocol_v3.py'
CAND='arte_candidate/run_v57_question_sensor_candidate.py'
FREEZE='arte_custodian/v57_question_sensor_freeze_after_intake_v3.py'
FINAL='arte_custodian/v57_question_sensor_final_verifier_v3.py'
AGG='arte_custodian/v57_question_sensor_aggregate_authority_v3.py'
SNAP='arte_custodian/v57_question_sensor_candidate_prefreeze_snapshot_v3_r1.json'
BANK='arte_candidate/v57_epoch20_question_bank.json'
POLICY='arte_custodian/v57_question_sensor_independent_authority_policy.json'

def save(p,x): pathlib.Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def run(*args,expect=0):
    r=subprocess.run(list(args),text=True,capture_output=True)
    if r.returncode!=expect:
        print('STDOUT',r.stdout); print('STDERR',r.stderr,file=sys.stderr)
        raise SystemExit(f'command return {r.returncode} expected {expect}: {args}')
    return r

def obs(nodes,rels,genders):
    return {'query_nodes':[nodes[0],nodes[3]],'path_edges':[[nodes[0],rels[0],nodes[1]],[nodes[1],rels[1],nodes[2]],[nodes[2],rels[2],nodes[3]]],'node_genders':{nodes[i]:genders[i] for i in range(4)}}

entries=[
  (['husband','daughter','grandfather'],['female','male','female','male'],{'kind':'CANONICAL_PROVENANCE_EDGE_MEMBERSHIP','edge':[0,'daughter',2]},['father-in-law','father']),
  (['husband','daughter','grandmother'],['female','male','female','female'],{'kind':'CANONICAL_PROVENANCE_EDGE_MEMBERSHIP','edge':[0,'daughter',2]},['mother-in-law','mother']),
  (['husband','son','grandmother'],['female','male','male','female'],{'kind':'CANONICAL_PROVENANCE_EDGE_MEMBERSHIP','edge':[0,'husband',1]},['mother','mother-in-law']),
  (['wife','daughter','grandfather'],['male','female','female','male'],{'kind':'CANONICAL_PROVENANCE_EDGE_MEMBERSHIP','edge':[0,'daughter',2]},['father-in-law','father']),
  (['wife','son','grandfather'],['male','female','male','male'],{'kind':'CANONICAL_PROVENANCE_EDGE_MEMBERSHIP','edge':[0,'son',2]},['father-in-law','father'])
]

batch_id='V3-SELFTEST-BATCH-R1'; hidden_paths=[]
for gi,g in enumerate(('G1','G2','G3'),1):
    cases=[]; pub=[]
    for j in range(3):
        rels,genders,q,answers=entries[(gi+j-1)%len(entries)]
        nodes=[f'{g}-n{j}-{k}' for k in range(4)]; bit=(gi+j)%2; cid=f'{g}-P{j+1}'
        po=obs(nodes,rels,genders); pub.append({'case_id':cid,'public_observation':po})
        cases.append({'case_id':cid,'expected_action':'QUESTION','expected_question':q,'sensor_bit':bit,'target':answers[bit],'tags':['positive','selftest']})
    nodes=[f'{g}-hold-{k}' for k in range(4)]
    po=obs(nodes,['husband','son','grandfather'],['female','male','male','male'])
    pub.append({'case_id':f'{g}-H1','public_observation':po})
    cases.append({'case_id':f'{g}-H1','expected_action':'HOLD','expected_question':None,'sensor_bit':None,'target':None,'tags':['negative-hold','selftest']})
    packet={'schema':'arte.independent_question_input/v57','challenge_id':f'V3-R1-SELFTEST-{g}','generation':g,'cases':pub}
    hidden={'schema':'arte.hidden_question_sensor_challenge/v57','batch_id':batch_id,'challenge_id':packet['challenge_id'],'generation':g,'public_packet':packet,'cases':cases,'claim_boundary':{'AGI':False,'ASI':False}}
    hp=td/f'{g}_hidden.json'; save(hp,hidden); hidden_paths.append(hp)

batch=td/'commitment_batch.json'; priv=td/'private'
run('python',CUST,'commit','--g1',str(hidden_paths[0]),'--g2',str(hidden_paths[1]),'--g3',str(hidden_paths[2]),'--custodian-id','SELFTEST-NOT-INDEPENDENT','--commitment-batch-out',str(batch),'--private-state-dir',str(priv))
freeze=td/'formal_freeze.json'; run('python',FREEZE,str(batch),'--freeze-out',str(freeze))
receipts=[]; stage_files={}
for g,hp in zip(('G1','G2','G3'),hidden_paths):
    public=td/f'{g}_public.json'; qout=td/f'{g}_questions.json'; sensor=td/f'{g}_sensor.json'; pred=td/f'{g}_predictions.json'; reveal=td/f'{g}_reveal.json'; rec=td/f'{g}_receipt.json'
    run('python',CUST,'public','--hidden',str(hp),'--formal-freeze',str(freeze),'--out',str(public))
    run('python',CAND,'--phase','question','--input',str(public),'--output',str(qout))
    run('python',CUST,'sensor','--hidden',str(hp),'--questions',str(qout),'--formal-freeze',str(freeze),'--out',str(sensor))
    run('python',CAND,'--phase','predict','--input',str(sensor),'--output',str(pred))
    run('python',CUST,'reveal','--private-state',str(priv/f'{g}_PRIVATE_KEEP_SECRET.json'),'--sensor-input',str(sensor),'--predictions',str(pred),'--formal-freeze',str(freeze),'--out',str(reveal))
    run('python',FINAL,'--commitment-batch',str(batch),'--freeze',str(freeze),'--prefreeze-snapshot',SNAP,'--frozen-question-bank',BANK,'--public-input',str(public),'--questions',str(qout),'--sensor-input',str(sensor),'--predictions',str(pred),'--reveal',str(reveal),'--receipt-out',str(rec))
    rr=json.load(open(rec)); assert rr['all_cases_passed'] is True and rr['hidden_challenge_semantics_valid'] is True and rr['stage_hash_chain_complete'] is True
    receipts.append(rec); stage_files[g]=(public,qout,sensor,pred,reveal,hp)

agg=td/'aggregate.json'
run('python',AGG,'--commitment-batch',str(batch),'--freeze',str(freeze),'--prefreeze-snapshot',SNAP,'--authority-policy',POLICY,'--receipts',*[str(x) for x in receipts],'--out',str(agg))
aa=json.load(open(agg)); assert aa['three_generation_crypto_and_stage_hash_chain_complete'] is True and aa['E4_independent_custody_authorized'] is False

public,qout,sensor,pred,reveal,hp=stage_files['G1']
qbad=td/'G1_questions_tampered.json'; x=json.load(open(qout)); x['questions'][0]['question']['edge'][0]=99; save(qbad,x)
badrec=td/'attack_question_receipt.json'
run('python',FINAL,'--commitment-batch',str(batch),'--freeze',str(freeze),'--prefreeze-snapshot',SNAP,'--frozen-question-bank',BANK,'--public-input',str(public),'--questions',str(qbad),'--sensor-input',str(sensor),'--predictions',str(pred),'--reveal',str(reveal),'--receipt-out',str(badrec),expect=2)
qr=json.load(open(badrec)); assert 'SENSOR_RESPONSE_NOT_BOUND_TO_EXACT_QUESTION_OUTPUT' in qr['errors']

pbad=td/'G1_predictions_tampered.json'; x=json.load(open(pred)); x['predictions'][0]['prediction']='tampered'; save(pbad,x)
badrec2=td/'attack_prediction_receipt.json'
run('python',FINAL,'--commitment-batch',str(batch),'--freeze',str(freeze),'--prefreeze-snapshot',SNAP,'--frozen-question-bank',BANK,'--public-input',str(public),'--questions',str(qout),'--sensor-input',str(sensor),'--predictions',str(pbad),'--reveal',str(reveal),'--receipt-out',str(badrec2),expect=2)
pr=json.load(open(badrec2)); assert 'REVEAL_NOT_BOUND_TO_EXACT_PREDICTION_OUTPUT' in pr['errors']

fbad=td/'formal_freeze_tampered.json'; x=json.load(open(freeze)); x['body_state_sha256']='00'*32; save(fbad,x)
badrec3=td/'attack_freeze_receipt.json'
run('python',FINAL,'--commitment-batch',str(batch),'--freeze',str(fbad),'--prefreeze-snapshot',SNAP,'--frozen-question-bank',BANK,'--public-input',str(public),'--questions',str(qout),'--sensor-input',str(sensor),'--predictions',str(pred),'--reveal',str(reveal),'--receipt-out',str(badrec3),expect=2)
fr=json.load(open(badrec3)); assert 'FORMAL_FREEZE_SELF_HASH_MISMATCH' in fr['errors']

wrong=td/'wrong_hidden.json'; wh=json.load(open(hidden_paths[0])); wh['batch_id']='WRONG-SEM-BATCH'; wh['challenge_id']='WRONG-SEM-G1'; wh['public_packet']['challenge_id']='WRONG-SEM-G1'; wh['cases'][0]['target']='deliberately-wrong'; save(wrong,wh)
semout=subprocess.run(['python','-c',"import json,sys;sys.path.insert(0,'arte_custodian');import v57_question_sensor_challenge_semantics_v3 as s;h=json.load(open(sys.argv[1]));r=s.validate_hidden_package(h,sys.argv[2]);print(json.dumps(r));raise SystemExit(0 if not r['valid'] else 3)",str(wrong),BANK],text=True,capture_output=True)
if semout.returncode!=0: print(semout.stdout,semout.stderr); raise SystemExit('wrong hidden target was not rejected')
sr=json.loads(semout.stdout.strip()); assert any('DECLARED_TARGET_NOT_DERIVED' in e for e in sr['errors'])

dupagg=td/'aggregate_duplicate_generation.json'
rr=subprocess.run(['python',AGG,'--commitment-batch',str(batch),'--freeze',str(freeze),'--prefreeze-snapshot',SNAP,'--authority-policy',POLICY,'--receipts',str(receipts[0]),str(receipts[1]),str(receipts[1]),'--out',str(dupagg)],text=True,capture_output=True)
assert rr.returncode==2
da=json.load(open(dupagg)); assert da['three_generation_crypto_and_stage_hash_chain_complete'] is False

result={
 'self_owned_three_generation_chain_complete':aa['three_generation_crypto_and_stage_hash_chain_complete'],
 'self_owned_E4':aa['E4_independent_custody_authorized'],
 'question_post_sensor_tamper_rejected':True,
 'prediction_post_reveal_tamper_rejected':True,
 'formal_freeze_tamper_rejected':True,
 'wrong_hidden_target_rejected_by_independent_semantics':True,
 'duplicate_generation_rejected':True,
 'prior_failed_run_preserved':'33166669843',
 'claim_boundary':{'AGI':False,'ASI':False,'external_recursive_acceleration':False,'independent_custody':False}
}
save(td/'SELFTEST_RESULT.json',result); print(json.dumps(result,sort_keys=True))
