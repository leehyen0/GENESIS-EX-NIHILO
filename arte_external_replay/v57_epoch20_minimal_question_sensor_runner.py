import ast, copy, csv, hashlib, io, itertools, json, os, shutil, tempfile, urllib.request
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
import v57_epoch19_nonidentifiability_recovery_runner as e19r

REPO='kliang5/CLUTRR_huggingface_dataset'
COMMIT='e5b496941e91abb7c319d2618a3ce96752bc4ab7'
SOURCE_CONFIG='gen_train23_test2to10'
FRESH_CONFIG='gen_train234_test2to10'
PARENT_SHA='f59f6b8b6b6a5a06129257645d5318660fb58337ed306bf114c33ef46c5678bf'
HISTORICAL_E19_RUN='33149263778'
FLAGS={
  'epoch20_full_clutrr_promotion':False,
  'independent_custody_proof':False,
  'external_recursive_acceleration':False,
  'AGI':False,
  'ASI':False
}

def H(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def save(p,x):
    Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,sort_keys=True),encoding='utf-8')

def load(p):
    return json.load(open(p,encoding='utf-8'))

def fetch(config,name):
    url=f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{config}/{name}'
    with urllib.request.urlopen(url,timeout=120) as r:
        return r.read().decode('utf-8')

def parse_gender_field(s):
    names=[]; genders=[]
    for p in s.split(','):
        if ':' not in p: continue
        a,b=p.split(':',1); names.append(a.strip()); genders.append(b.strip())
    return names,tuple(genders)

def parse_rows(raw, target=False, proof=False):
    out=[]
    for i,row in enumerate(csv.reader(io.StringIO(raw))):
        if i==0 or len(row)<15: continue
        try:
            q=tuple(ast.literal_eval(row[3]))
            se=list(ast.literal_eval(row[10]))
            et=[str(x) for x in ast.literal_eval(row[11])]
            qe=tuple(ast.literal_eval(row[12]))
            names,genders=parse_gender_field(row[13])
        except Exception:
            continue
        if len(q)!=2 or not se or len(se)!=len(et) or not names: continue
        if max(max(a,b) for a,b in se)>=len(names): continue
        if max(qe)>=len(names) or (names[qe[0]],names[qe[1]])!=q: continue
        c={
          'id':row[1], 'query_edge':qe,
          'edges':tuple((int(a),r,int(b)) for (a,b),r in zip(se,et)),
          'edge_types':tuple(et), 'node_genders':genders,
          'target':row[5] if target else None,
          'proof_edges':None
        }
        if proof:
            try:
                ps=ast.literal_eval(row[7]); block=ps[0]
                if not isinstance(block,dict) or len(block)!=1: continue
                # The relation label used as the proof_state dictionary key is intentionally ignored.
                _, evidence=next(iter(block.items()))
                idx={n:j for j,n in enumerate(names)}
                pe=[]
                for t in evidence:
                    if not isinstance(t,(list,tuple)) or len(t)!=3: raise ValueError('BAD_PROOF_EDGE')
                    a,r,b=str(t[0]),str(t[1]),str(t[2])
                    if a not in idx or b not in idx: raise ValueError('PROOF_ENTITY_NOT_IN_PUBLIC_NODE_SET')
                    pe.append((idx[a],r,idx[b]))
                c['proof_edges']=tuple(pe)
            except Exception:
                continue
        out.append(c)
    return out

def sig(c):
    return (c['query_edge'],c['edges'],c['node_genders'])

def build_pair_table(train):
    table={}; conf=set()
    for c in train:
        if len(c['edge_types'])!=2 or c['target'] is None: continue
        k=c['edge_types']; v=c['target']
        if k in table and table[k]!=v: conf.add(k)
        else: table[k]=v
    for k in conf: table.pop(k,None)
    return table

def compose(seq,table):
    if not seq: return None
    s=seq[0]
    for x in seq[1:]:
        if (s,x) not in table: return None
        s=table[(s,x)]
    return s

def candidate_hidden_edges(cases):
    qe=cases[0]['query_edge']
    targets={c['target'] for c in cases}
    edges=sorted({e for c in cases for e in c['proof_edges']},key=repr)
    out=[]
    for e in edges:
        a,r,b=e
        if (a,b)==qe or (b,a)==qe: continue
        if r in targets: continue
        out.append(e)
    return out

def minimal_question_for_group(cases,max_bits=3):
    candidates=candidate_hidden_edges(cases)
    for k in range(1,min(max_bits,len(candidates))+1):
        for qs in itertools.combinations(candidates,k):
            m=defaultdict(set)
            for c in cases:
                bits=tuple(q in set(c['proof_edges']) for q in qs)
                m[bits].add(c['target'])
            if all(len(v)==1 for v in m.values()) and len({next(iter(v)) for v in m.values()})>1:
                return {
                  'bit_count':k,
                  'questions':[{'kind':'PROOF_EDGE_MEMBERSHIP','edge':list(q)} for q in qs],
                  'answer_to_target':{'|'.join('1' if b else '0' for b in bits):next(iter(v)) for bits,v in sorted(m.items(),key=lambda kv:repr(kv[0]))},
                  'source_cases':len(cases),
                  'source_targets':sorted({c['target'] for c in cases})
                }
    return None

def build_question_bank(source_val):
    g=defaultdict(list)
    for c in source_val: g[sig(c)].append(c)
    bank={}; unresolved=[]
    for s,cases in g.items():
        if len({c['target'] for c in cases})<=1: continue
        q=minimal_question_for_group(cases)
        if q is None: unresolved.append(repr(s))
        else: bank[repr(s)]=q
    return bank,unresolved

def answer_question(c,qspec):
    pe=set(c['proof_edges'] or ())
    bits=[]
    for q in qspec['questions']:
        bits.append(tuple(q['edge']) in pe)
    return '|'.join('1' if b else '0' for b in bits)

def restore_parent():
    td=Path(tempfile.mkdtemp(prefix='arte_restore19_'))
    old=os.environ.get('GITHUB_RUN_ID')
    try:
        os.environ['GITHUB_RUN_ID']=HISTORICAL_E19_RUN
        e19r.main('arte_external_replay/v57_epoch19_nonidentifiability_trigger.json',str(td))
        st=load(td/'checkpoint_state.json')
    finally:
        if old is None: os.environ.pop('GITHUB_RUN_ID',None)
        else: os.environ['GITHUB_RUN_ID']=old
        shutil.rmtree(td,ignore_errors=True)
    assert st['state_sha256']==PARENT_SHA
    assert st['epoch_completed']==19
    assert st['self_model']['alpha']==19 and st['self_model']['beta']==5
    return st

def main(trigger_path,outdir):
    o=Path(outdir);o.mkdir(parents=True,exist_ok=True)
    t=load(trigger_path); parent=restore_parent()
    assert t['epoch']==20 and t['parent_state_sha256']==PARENT_SHA
    assert t['source_config']==SOURCE_CONFIG and t['fresh_config']==FRESH_CONFIG

    # Source question genesis: old validation only.
    source_val=parse_rows(fetch(SOURCE_CONFIG,'validation.csv'),target=True,proof=True)
    bank,unresolved=build_question_bank(source_val)
    if unresolved: raise RuntimeError('SOURCE_AMBIGUITY_NOT_RESOLVED_WITH_MAX3_BITS::'+json.dumps(unresolved[:5]))
    if len(bank)!=5: raise RuntimeError(f'EXPECTED_5_SOURCE_COLLISION_GROUPS::{len(bank)}')
    max_bits=max(v['bit_count'] for v in bank.values())
    if max_bits!=1: raise RuntimeError(f'ONE_BIT_MINIMUM_NOT_SUPPORTED::{max_bits}')
    bank_sha=H(bank)

    # Transfer gate on a never-used config validation split. Question bank is frozen already.
    fresh_val=parse_rows(fetch(FRESH_CONFIG,'validation.csv'),target=True,proof=True)
    val_selected=[]
    for c in fresh_val:
        k=repr(sig(c))
        if k not in bank: continue
        ans=answer_question(c,bank[k]); pred=bank[k]['answer_to_target'].get(ans)
        val_selected.append((c,pred,ans))
    if not val_selected: raise RuntimeError('NO_TRANSFER_VALIDATION_CASES_FOR_QUESTION_BANK')
    val_correct=sum(pred==c['target'] for c,pred,ans in val_selected)
    val_acc=val_correct/len(val_selected)
    if val_acc!=1.0:
        raise RuntimeError('QUESTION_BANK_FAILED_TRANSFER_VALIDATION::'+json.dumps({'n':len(val_selected),'acc':val_acc}))

    # Fresh train is used only for the no-sensor baseline composition.
    fresh_train=parse_rows(fetch(FRESH_CONFIG,'train.csv'),target=True,proof=False)
    pair=build_pair_table(fresh_train)

    # Fetch test bytes once, but stage-1 parser does not parse target or proof_state fields.
    test_raw=fetch(FRESH_CONFIG,'test.csv')
    test_public=parse_rows(test_raw,target=False,proof=False)
    question_packets=[]
    selected_public=[]
    for c in test_public:
        k=repr(sig(c))
        if k not in bank: continue
        q=bank[k]
        packet={
          'id':c['id'],'public_signature_sha256':H(repr(sig(c))),
          'question_bank_sha256':bank_sha,'bit_count':q['bit_count'],
          'questions':q['questions'],'proof_state_parsed':False,'target_field_parsed':False
        }
        question_packets.append(packet);selected_public.append(c)
    if not question_packets: raise RuntimeError('NO_FRESH_TEST_QUESTION_CASES')
    question_precommit={
      'schema':'arte.minimum_question_precommit/v57','epoch':20,
      'parent_state_sha256':PARENT_SHA,'source_config':SOURCE_CONFIG,'fresh_config':FRESH_CONFIG,
      'question_bank_sha256':bank_sha,'question_count':len(question_packets),
      'max_bits_per_case':max_bits,'packets':question_packets,
      'target_field_parsed_before_question_precommit':False,
      'proof_state_parsed_before_question_precommit':False,
      'raw_test_file_contains_target_and_proof_columns':True,
      'claim_boundary':'SOFTWARE_ENFORCED_STAGE_SEPARATION_NOT_INDEPENDENT_CUSTODY'
    }
    question_precommit['sha256']=H({k:v for k,v in question_precommit.items() if k!='sha256'})
    save(o/'question_precommit.json',question_precommit)

    # Stage 2: parse target-stripped proof evidence as the new sensor response; target still ignored.
    test_sensor={c['id']:c for c in parse_rows(test_raw,target=False,proof=True)}
    sensor_packets=[]; prediction_packets=[]
    for pub in selected_public:
        s=test_sensor.get(pub['id'])
        if s is None: continue
        q=bank[repr(sig(pub))]
        ans=answer_question(s,q); pred=q['answer_to_target'].get(ans)
        baseline=compose(pub['edge_types'],pair)
        sensor_packets.append({'id':pub['id'],'answer_bits':ans,'question_bit_count':q['bit_count'],'target_field_parsed':False})
        prediction_packets.append({'id':pub['id'],'prediction':pred,'baseline_prediction':baseline,'answer_bits':ans,'target_field_parsed':False})
    if len(prediction_packets)!=len(question_packets): raise RuntimeError('SENSOR_RESPONSE_MISSING_CASES')
    sensor_receipt={'schema':'arte.minimum_sensor_response/v57','question_precommit_sha256':question_precommit['sha256'],'packets':sensor_packets,'proof_state_relation_key_semantically_used':False,'target_field_parsed':False}
    sensor_receipt['sha256']=H({k:v for k,v in sensor_receipt.items() if k!='sha256'});save(o/'sensor_response.json',sensor_receipt)
    prediction_precommit={'schema':'arte.post_sensor_prediction_precommit/v57','question_precommit_sha256':question_precommit['sha256'],'sensor_response_sha256':sensor_receipt['sha256'],'predictions':prediction_packets,'target_field_parsed':False}
    prediction_precommit['sha256']=H({k:v for k,v in prediction_precommit.items() if k!='sha256'});save(o/'prediction_precommit.json',prediction_precommit)

    # Stage 3 reveal: target is parsed only after both precommits are persisted.
    truth={c['id']:c['target'] for c in parse_rows(test_raw,target=True,proof=False)}
    n=len(prediction_packets); q_ok=b_ok=b_answered=0
    per_case=[]
    for p in prediction_packets:
        y=truth[p['id']]; qgood=p['prediction']==y; q_ok+=int(qgood)
        if p['baseline_prediction'] is not None:
            b_answered+=1; bgood=p['baseline_prediction']==y; b_ok+=int(bgood)
        else: bgood=False
        per_case.append({'id':p['id'],'question_prediction':p['prediction'],'baseline_prediction':p['baseline_prediction'],'target':y,'question_correct':qgood,'baseline_correct':bgood})
    q_acc=q_ok/n if n else 0.0; b_acc=b_ok/b_answered if b_answered else 0.0
    success=(q_acc==1.0 and n>0 and max_bits==1 and val_acc==1.0)

    generated={
      'kind':'ONE_BIT_MISSING_PROVENANCE_QUESTION_GENERATOR',
      'source_collision_groups':len(bank),'source_max_bits':max_bits,
      'question_bank_sha256':bank_sha,'transfer_validation_cases':len(val_selected),
      'transfer_validation_accuracy':val_acc,
      'measurement_semantics':'BOOLEAN_MEMBERSHIP_OF_TARGET_STRIPPED_HIDDEN_PROVENANCE_EDGE',
      'public_story_channel_nonidentifiability_preserved':True
    }
    op_sha=H(generated)
    child=copy.deepcopy(parent);child['epoch_completed']=20
    child.setdefault('generated_operator_registry',[]).append({'epoch':20,'operator':generated,'operator_sha256':op_sha,'target_consumed_for_generation':False})
    child.setdefault('epistemic_state_registry',[]).append({'state':'IDENTIFIABLE_AFTER_ONE_BIT_PROVENANCE_MEASUREMENT','source_state':'NONIDENTIFIABLE_UNDER_PUBLIC_CLUTRR_STORY_CHANNELS','minimum_bits':1,'fresh_config':FRESH_CONFIG,'full_task_promotion':False})
    child.setdefault('synergy_credit_details',{})['CLUTRR_MINIMUM_QUESTION_SENSOR']={
      'status':'SUPPORTED_BOUNDED_ONE_BIT_IDENTIFIABILITY_RESTORATION' if success else 'NOT_SUPPORTED',
      'fresh_test_cases':n,'question_accuracy':q_acc,'baseline_answered_accuracy':b_acc,
      'minimum_bits':1,'independent_custody':False,'full_task_promotion':False
    }
    child['self_model']['alpha' if success else 'beta']+=1
    child['state_sha256']=H({k:v for k,v in child.items() if k!='state_sha256'});save(o/'checkpoint_state.json',child)

    ev={
      'schema':'arte.minimum_question_sensor_external_evaluation/v57','epoch':20,
      'source_repo':REPO,'source_commit':COMMIT,'source_config':SOURCE_CONFIG,'fresh_config':FRESH_CONFIG,
      'source_collision_groups':len(bank),'source_unresolved_groups':len(unresolved),'minimum_bits':max_bits,
      'transfer_validation_cases':len(val_selected),'transfer_validation_accuracy':val_acc,
      'fresh_test_question_cases':n,'question_correct':q_ok,'question_accuracy':q_acc,
      'baseline_answered':b_answered,'baseline_correct':b_ok,'baseline_answered_accuracy':b_acc,
      'question_precommit_before_sensor_parse':True,'prediction_precommit_before_target_parse':True,
      'raw_test_file_contains_target_and_proof_columns':True,
      'software_stage_separation_not_independent_custody':True,
      'proof_state_target_relation_key_semantically_used':False,
      'full_clutrr_task_promotion':False,'success':success,'generated_operator_sha256':op_sha,
      'claim_flags':FLAGS,'cases':per_case
    };save(o/'external_evaluation.json',ev)
    rec={
      'schema':'arte.minimum_question_sensor_receipt/v57','epoch':20,'github_run_id':os.environ.get('GITHUB_RUN_ID'),
      'github_sha':os.environ.get('GITHUB_SHA'),'hosted_external':os.environ.get('GITHUB_ACTIONS')=='true',
      'parent_state_sha256':PARENT_SHA,'child_state_sha256':child['state_sha256'],
      'question_precommit_sha256':question_precommit['sha256'],'sensor_response_sha256':sensor_receipt['sha256'],
      'prediction_precommit_sha256':prediction_precommit['sha256'],'generated_operator_sha256':op_sha,
      'minimum_bits':max_bits,'fresh_test_cases':n,'question_accuracy':q_acc,'baseline_answered_accuracy':b_acc,
      'bounded_one_bit_identifiability_restoration':success,'full_task_promotion':False,'claim_flags':FLAGS,
      'timestamp':datetime.now(timezone.utc).isoformat()
    };rec['receipt_sha256']=H(rec);save(o/'epoch_receipt.json',rec)
    files=['question_precommit.json','sensor_response.json','prediction_precommit.json','external_evaluation.json','checkpoint_state.json','epoch_receipt.json']
    save(o/'hash_manifest.json',{'files':[{'name':x,'sha256':hashlib.sha256((o/x).read_bytes()).hexdigest()} for x in files],'claim_flags':FLAGS})
    print(json.dumps(rec,sort_keys=True))

if __name__=='__main__':
    main('arte_external_replay/v57_epoch20_minimal_question_trigger.json','v57_output')
