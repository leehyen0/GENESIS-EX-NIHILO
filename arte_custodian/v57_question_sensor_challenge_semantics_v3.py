#!/usr/bin/env python3
"""Independent semantic validator for v57 question-sensor hidden challenges.

It does not trust hidden expected_question/target fields. It derives the required
QUESTION/HOLD behavior and target from the public observation plus the frozen bank.
"""
import json
from pathlib import Path

BANK_SCHEMA='arte.frozen_question_bank/v57'


def load_bank(path):
    b=json.loads(Path(path).read_text(encoding='utf-8'))
    if b.get('schema')!=BANK_SCHEMA: raise ValueError('BAD_FROZEN_BANK_SCHEMA')
    idx={}
    for e in b.get('entries',[]):
        k=(tuple(e.get('path_relations',[])),tuple(e.get('path_genders',[])))
        if len(k[0])!=3 or len(k[1])!=4 or k in idx: raise ValueError('BAD_OR_DUPLICATE_BANK_ENTRY')
        if e.get('bit_count')!=1: raise ValueError('EXPECTED_ONE_BIT_BANK_ENTRY')
        if set(e.get('answer_to_target',{}))!={'0','1'}: raise ValueError('BANK_ANSWER_MAP_MUST_HAVE_BITS_0_1')
        idx[k]=e
    return b,idx

def canonical_signature(obs):
    if not isinstance(obs,dict): raise ValueError('PUBLIC_OBSERVATION_NOT_OBJECT')
    q=obs.get('query_nodes'); edges=obs.get('path_edges'); genders=obs.get('node_genders')
    if not isinstance(q,list) or len(q)!=2 or q[0]==q[1]: raise ValueError('BAD_QUERY_NODES')
    if not isinstance(edges,list) or len(edges)!=3: raise ValueError('EXPECTED_EXACTLY_3_PATH_EDGES')
    if not isinstance(genders,dict): raise ValueError('BAD_NODE_GENDERS')
    succ={}
    for edge in edges:
        if not isinstance(edge,list) or len(edge)!=3: raise ValueError('BAD_PATH_EDGE')
        a,r,b=edge
        if a in succ: raise ValueError('NONUNIQUE_PATH_SUCCESSOR')
        if not isinstance(r,str): raise ValueError('BAD_RELATION')
        succ[a]=(r,b)
    nodes=[q[0]]; rels=[]; seen={q[0]}; cur=q[0]
    for _ in range(3):
        if cur not in succ: raise ValueError('PATH_TOO_SHORT_OR_DISCONNECTED')
        r,nxt=succ[cur]
        if nxt in seen: raise ValueError('PATH_CYCLE')
        rels.append(r); nodes.append(nxt); seen.add(nxt); cur=nxt
    if cur!=q[1]: raise ValueError('PATH_TARGET_MISMATCH')
    if len(seen)!=4: raise ValueError('EXPECTED_FOUR_DISTINCT_PATH_NODES')
    gs=[]
    for n in nodes:
        g=genders.get(str(n),genders.get(n))
        if g not in ('male','female'): raise ValueError('MISSING_OR_BAD_GENDER')
        gs.append(g)
    return tuple(rels),tuple(gs)

def derive(public_observation,bank_index,sensor_bit=None):
    try: sig=canonical_signature(public_observation)
    except Exception as e: return {'action':'HOLD','reason':'MALFORMED_OR_OUT_OF_LANGUAGE','detail':str(e)}
    entry=bank_index.get(sig)
    if entry is None: return {'action':'HOLD','reason':'OUT_OF_FROZEN_QUESTION_BANK','signature':[list(sig[0]),list(sig[1])]}
    out={'action':'QUESTION','question':entry['question'],'signature':[list(sig[0]),list(sig[1])]}
    if sensor_bit is not None:
        if sensor_bit not in (0,1,False,True): raise ValueError('INVALID_SENSOR_BIT')
        out['target']=entry['answer_to_target'][str(int(sensor_bit))]
    return out

def validate_hidden_case(public_case,hidden_case,bank_index):
    errors=[]
    if public_case.get('case_id')!=hidden_case.get('case_id'): errors.append('CASE_ID_MISMATCH')
    bit=hidden_case.get('sensor_bit')
    try: derived=derive(public_case.get('public_observation'),bank_index,bit if bit is not None else None)
    except Exception as e: return {'valid':False,'errors':['DERIVATION_ERROR:'+str(e)],'derived':{}}
    declared_action=hidden_case.get('expected_action','QUESTION')
    if derived['action']=='HOLD':
        if declared_action!='HOLD': errors.append('HIDDEN_EXPECTS_QUESTION_BUT_PUBLIC_OBSERVATION_REQUIRES_HOLD')
        if hidden_case.get('expected_question') is not None: errors.append('HOLD_CASE_MUST_NOT_DECLARE_EXPECTED_QUESTION')
        if hidden_case.get('sensor_bit') is not None: errors.append('HOLD_CASE_MUST_NOT_DECLARE_SENSOR_BIT')
        if hidden_case.get('target') is not None: errors.append('HOLD_CASE_MUST_NOT_DECLARE_TARGET')
    else:
        if declared_action!='QUESTION': errors.append('HIDDEN_EXPECTS_HOLD_BUT_FROZEN_BANK_REQUIRES_QUESTION')
        if hidden_case.get('expected_question')!=derived.get('question'): errors.append('DECLARED_EXPECTED_QUESTION_NOT_DERIVED_FROM_FROZEN_BANK')
        if bit not in (0,1,False,True): errors.append('QUESTION_CASE_REQUIRES_ONE_SENSOR_BIT')
        else:
            expected_target=derive(public_case.get('public_observation'),bank_index,bit).get('target')
            derived['target']=expected_target
            if hidden_case.get('target')!=expected_target: errors.append('DECLARED_TARGET_NOT_DERIVED_FROM_FROZEN_BANK_AND_SENSOR_BIT')
    return {'valid':not errors,'errors':errors,'derived':derived}

def validate_hidden_package(hidden,bank):
    _,idx=bank if isinstance(bank,tuple) else load_bank(bank)
    pub=hidden.get('public_packet',{}); pub_cases=pub.get('cases',[]); hid_cases=hidden.get('cases',[])
    pmap={x.get('case_id'):x for x in pub_cases}; hmap={x.get('case_id'):x for x in hid_cases}; errors=[]; cases=[]
    if None in pmap or None in hmap or len(pmap)!=len(pub_cases) or len(hmap)!=len(hid_cases): errors.append('MISSING_OR_DUPLICATE_CASE_IDS')
    if set(pmap)!=set(hmap): errors.append('PUBLIC_HIDDEN_CASE_ID_SET_MISMATCH')
    for cid in sorted(set(pmap)&set(hmap),key=str):
        r=validate_hidden_case(pmap[cid],hmap[cid],idx); cases.append({'case_id':cid,**r}); errors.extend(f'{cid}:{e}' for e in r['errors'])
    return {'valid':not errors,'errors':errors,'cases':cases}
