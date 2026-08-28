import ast,csv,io,json
from collections import defaultdict,Counter
from pathlib import Path
import v57_epoch19_clutrr_relation_composition_runner as e19

def gender_order(s):
    out=[]
    for p in s.split(','):
        if ':' in p:
            n=p.split(':',1)[0].strip()
            if n and n not in out:out.append(n)
    return out

def build_case(row):
    q=tuple(ast.literal_eval(row[3])); qe=tuple(ast.literal_eval(row[12])); se=list(ast.literal_eval(row[10])); et=list(ast.literal_eval(row[11])); order=gender_order(row[13])
    assert (order[qe[0]],order[qe[1]])==q
    edges=[(a,str(r),b) for (a,b),r in zip(se,et)]
    return {'id':row[1],'query_idx':qe,'query_names':q,'target':row[5],'edges':edges,'node_genders':tuple(p.split(':',1)[1].strip() for p in row[13].split(',') if ':' in p),'story':row[2]}

def key(c, include_genders=True):
    # node IDs are already canonical dataset graph indices; names are deliberately omitted.
    return (c['query_idx'],tuple(c['edges']),c['node_genders'] if include_genders else None)

def collide(cases,include_genders):
    g=defaultdict(list)
    for c in cases:g[repr(key(c,include_genders))].append(c)
    bad={k:v for k,v in g.items() if len({x['target'] for x in v})>1}
    ex=[]
    for k,v in list(bad.items())[:20]:ex.append({'key':k,'targets':dict(Counter(x['target'] for x in v)),'cases':[{'id':x['id'],'query_names':x['query_names'],'target':x['target'],'story':x['story']} for x in v[:10]]})
    return {'groups':len(g),'conflicting_groups':len(bad),'conflicting_cases':sum(len(v) for v in bad.values()),'exact_identifiable':not bad,'examples':ex}

def main(out='diag_raw_entity_graph_collision'):
    o=Path(out);o.mkdir(parents=True,exist_ok=True)
    rows=list(csv.reader(io.StringIO(e19.fetch_text(e19.VAL_PATH))))
    cases=[]
    for r in rows[1:]:
        if len(r)<15:continue
        try:cases.append(build_case(r))
        except Exception:continue
    report={'validation_cases':len(cases),'raw_entity_directed_graph_with_genders':collide(cases,True),'raw_entity_directed_graph_without_genders':collide(cases,False),'node_identity_mapping_source':'genders_order validated by query_edge','raw_names_not_used_in_key':True,'proof_state_used':False,'test_fetched':False}
    (o/'raw_entity_graph_collision.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()
