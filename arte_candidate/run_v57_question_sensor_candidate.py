#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

BANK_PATH=Path(__file__).with_name('v57_epoch20_question_bank.json')
QUESTION_SCHEMA='arte.independent_question_input/v57'
QUESTION_OUTPUT_SCHEMA='arte.independent_question_output/v57'
SENSOR_SCHEMA='arte.independent_sensor_input/v57'
PREDICTION_OUTPUT_SCHEMA='arte.independent_prediction_output/v57'


def H(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def load_bank():
    b=json.loads(BANK_PATH.read_text(encoding='utf-8'))
    assert b['schema']=='arte.frozen_question_bank/v57'
    idx={}
    for e in b['entries']:
        k=(tuple(e['path_relations']),tuple(e['path_genders']))
        if k in idx: raise ValueError('DUPLICATE_BANK_SIGNATURE')
        idx[k]=e
    return b,idx


def canonicalize(obs):
    if not isinstance(obs,dict): raise ValueError('PUBLIC_OBSERVATION_NOT_OBJECT')
    q=obs.get('query_nodes'); edges=obs.get('path_edges'); genders=obs.get('node_genders')
    if not isinstance(q,list) or len(q)!=2 or q[0]==q[1]: raise ValueError('BAD_QUERY_NODES')
    if not isinstance(edges,list) or len(edges)!=3: raise ValueError('EXPECTED_EXACTLY_3_PATH_EDGES')
    if not isinstance(genders,dict): raise ValueError('BAD_NODE_GENDERS')
    by_from={}
    for x in edges:
        if not isinstance(x,list) or len(x)!=3: raise ValueError('BAD_PATH_EDGE')
        a,r,b=x
        if a in by_from: raise ValueError('NONUNIQUE_PATH_SUCCESSOR')
        if not isinstance(r,str): raise ValueError('BAD_RELATION')
        by_from[a]=(r,b)
    cur=q[0]; nodes=[cur]; rels=[]
    seen={cur}
    for _ in range(3):
        if cur not in by_from: raise ValueError('PATH_DOES_NOT_REACH_QUERY_TARGET')
        r,nxt=by_from[cur]
        if nxt in seen: raise ValueError('PATH_CYCLE')
        rels.append(r); nodes.append(nxt); seen.add(nxt); cur=nxt
    if cur!=q[1]: raise ValueError('PATH_TARGET_MISMATCH')
    if len(seen)!=4: raise ValueError('EXPECTED_4_PATH_NODES')
    gs=[]
    for n in nodes:
        g=genders.get(str(n),genders.get(n))
        if g not in ('male','female'): raise ValueError('MISSING_OR_BAD_GENDER')
        gs.append(g)
    return {'path_relations':rels,'path_genders':gs,'canonical_nodes':nodes}


def select_entry(obs,idx):
    try:
        c=canonicalize(obs)
    except Exception as e:
        return None,{'reason':'MALFORMED_OR_OUT_OF_LANGUAGE','detail':str(e)}
    e=idx.get((tuple(c['path_relations']),tuple(c['path_genders'])))
    if e is None: return None,{'reason':'OUT_OF_FROZEN_QUESTION_BANK','canonical':c}
    return e,{'canonical':c}


def question_phase(inp):
    bank,idx=load_bank()
    if inp.get('schema')!=QUESTION_SCHEMA: raise ValueError('BAD_QUESTION_INPUT_SCHEMA')
    cid=inp.get('challenge_id'); gen=inp.get('generation')
    if not isinstance(cid,str) or not cid: raise ValueError('MISSING_CHALLENGE_ID')
    if gen not in ('G1','G2','G3'): raise ValueError('BAD_GENERATION')
    out=[]
    for c in inp.get('cases',[]):
        case_id=c.get('case_id')
        e,meta=select_entry(c.get('public_observation'),idx)
        if e is None:
            out.append({'case_id':case_id,'action':'HOLD','reason':meta['reason']})
            continue
        q=e['question']
        out.append({'case_id':case_id,'action':'QUESTION','question':q,'question_sha256':H(q)})
    return {
      'schema':QUESTION_OUTPUT_SCHEMA,'challenge_id':cid,'generation':gen,
      'question_bank_sha256':bank['canonical_bank_sha256'],'questions':out,
      'claim_boundary':{'independent_custody':False,'AGI':False,'ASI':False}
    }


def prediction_phase(inp):
    bank,idx=load_bank()
    if inp.get('schema')!=SENSOR_SCHEMA: raise ValueError('BAD_SENSOR_INPUT_SCHEMA')
    cid=inp.get('challenge_id'); gen=inp.get('generation')
    if not isinstance(cid,str) or not cid: raise ValueError('MISSING_CHALLENGE_ID')
    if gen not in ('G1','G2','G3'): raise ValueError('BAD_GENERATION')
    out=[]
    for c in inp.get('cases',[]):
        case_id=c.get('case_id'); e,meta=select_entry(c.get('public_observation'),idx)
        if e is None:
            out.append({'case_id':case_id,'action':'HOLD','reason':meta['reason']}); continue
        expected_q=e['question']; got_q=c.get('question')
        if got_q!=expected_q or c.get('question_sha256')!=H(expected_q):
            out.append({'case_id':case_id,'action':'HOLD','reason':'QUESTION_BINDING_MISMATCH'}); continue
        bit=c.get('sensor_bit')
        if bit not in (0,1,False,True):
            out.append({'case_id':case_id,'action':'HOLD','reason':'INVALID_SENSOR_BIT'}); continue
        y=e['answer_to_target'][str(int(bit))]
        out.append({'case_id':case_id,'action':'PREDICT','prediction':y})
    return {
      'schema':PREDICTION_OUTPUT_SCHEMA,'challenge_id':cid,'generation':gen,
      'question_bank_sha256':bank['canonical_bank_sha256'],'predictions':out,
      'claim_boundary':{'independent_custody':False,'AGI':False,'ASI':False}
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=['question','predict'],required=True)
    ap.add_argument('--input',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    inp=json.loads(Path(a.input).read_text(encoding='utf-8'))
    out=question_phase(inp) if a.phase=='question' else prediction_phase(inp)
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')

if __name__=='__main__': main()
