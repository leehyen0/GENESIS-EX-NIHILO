import ast,csv,io,json
from collections import defaultdict,Counter
from pathlib import Path
import v57_epoch19_clutrr_relation_composition_runner as e19

def parse_rows(raw):
    out=[]
    for i,row in enumerate(csv.reader(io.StringIO(raw))):
        if i==0 or len(row)<15: continue
        try:
            q=ast.literal_eval(row[3]); ps=ast.literal_eval(row[7])
        except Exception:
            continue
        if not isinstance(q,(list,tuple)) or len(q)!=2 or not ps: continue
        # IMPORTANT: ignore the relation label in proof_state key completely.
        block=ps[0]
        if not isinstance(block,dict) or len(block)!=1: continue
        _, evidence = next(iter(block.items()))
        triples=[]
        ok=True
        for t in evidence:
            if not isinstance(t,(list,tuple)) or len(t)!=3: ok=False; break
            triples.append((str(t[0]),str(t[1]),str(t[2])))
        if not ok: continue
        out.append({'id':row[1],'story':row[2],'query':(str(q[0]),str(q[1])),'target':row[5],'edge_types':ast.literal_eval(row[11]),'evidence':triples,'genders':row[13]})
    return out

def canonical_graph(c, include_direction=True, include_query_roles=True):
    # Canonicalize entity identities structurally so raw names cannot make cases trivially unique.
    q0,q1=c['query']
    order=[]
    if include_query_roles:
        order=[q0,q1]
    for a,r,b in c['evidence']:
        if a not in order: order.append(a)
        if b not in order: order.append(b)
    idx={x:i for i,x in enumerate(order)}
    edges=[]
    for a,r,b in c['evidence']:
        if include_direction: edges.append((idx[a],r,idx[b]))
        else: edges.append((min(idx[a],idx[b]),r,max(idx[a],idx[b])))
    return {'q':(idx[q0],idx[q1]) if include_query_roles else None,'edges':tuple(edges),'node_count':len(order)}

def collision(rows,keyfn):
    g=defaultdict(list)
    for c in rows:g[repr(keyfn(c))].append(c)
    bad={k:v for k,v in g.items() if len({x['target'] for x in v})>1}
    examples=[]
    for k,v in list(bad.items())[:20]:
        examples.append({'key':k,'targets':dict(Counter(x['target'] for x in v)),'cases':[{'id':x['id'],'query':x['query'],'evidence':x['evidence'],'edge_types':x['edge_types'],'story':x['story'],'target':x['target']} for x in v[:8]]})
    return {'groups':len(g),'conflicting_groups':len(bad),'conflicting_cases':sum(len(v) for v in bad.values()),'exact_identifiable':len(bad)==0,'examples':examples}

def main(out='diag_entity_graph'):
    o=Path(out);o.mkdir(parents=True,exist_ok=True)
    rows=parse_rows(e19.fetch_text(e19.VAL_PATH))
    report={
      'validation_cases':len(rows),
      'edge_types_only':collision(rows,lambda c:tuple(c['edge_types'])),
      'canonical_entity_directed_graph':collision(rows,lambda c:canonical_graph(c,True,True)),
      'canonical_entity_undirected_graph':collision(rows,lambda c:canonical_graph(c,False,True)),
      'raw_entity_names_not_used_as_identity_key':True,
      'proof_state_target_relation_key_consumed':False,
      'validation_target_used_only_for_collision_measurement':True,
      'test_fetched':False
    }
    (o/'entity_graph_identifiability.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()
