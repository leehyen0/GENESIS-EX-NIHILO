import ast,csv,io,json,re
from pathlib import Path
from collections import Counter
import v57_epoch19_clutrr_relation_composition_runner as e19

def names_first_story(story):
    out=[]
    for x in re.findall(r'\[([^\]]+)\]',story):
        if x not in out: out.append(x)
    return out

def parse_gender_order(s):
    out=[]
    for p in s.split(','):
        if ':' in p:
            n=p.split(':',1)[0].strip()
            if n and n not in out: out.append(n)
    return out

def oracle_evidence(ps):
    z=ast.literal_eval(ps)
    if not z:return []
    _,ev=next(iter(z[0].items()))
    return [(str(a),str(r),str(b)) for a,r,b in ev]

def build_raw(order, story_edges, edge_types):
    try: se=ast.literal_eval(story_edges); et=ast.literal_eval(edge_types)
    except Exception:return None
    if len(se)!=len(et):return None
    mx=max(max(a,b) for a,b in se) if se else -1
    if len(order)<=mx:return None
    return [(order[a],str(r),order[b]) for (a,b),r in zip(se,et)]

def edge_overlap(raw,oracle):
    if raw is None:return (0,0,0)
    rs=set(raw); os=set(oracle)
    return (len(rs&os),len(os),len(rs))

def main(out='diag_node_identity'):
    o=Path(out);o.mkdir(parents=True,exist_ok=True)
    rows=list(csv.reader(io.StringIO(e19.fetch_text(e19.VAL_PATH))))
    stats={k:Counter() for k in ['story_first','gender_order']}; examples=[]
    total=0
    for i,row in enumerate(rows[1:]):
        if len(row)<15:continue
        try: oracle=oracle_evidence(row[7])
        except Exception:continue
        total+=1
        orders={'story_first':names_first_story(row[2]),'gender_order':parse_gender_order(row[13])}
        one={'id':row[1],'query':row[3],'story':row[2],'oracle':oracle,'story_edges':row[10],'edge_types':row[11]}
        for k,ordr in orders.items():
            raw=build_raw(ordr,row[10],row[11]); inter,osz,rsz=edge_overlap(raw,oracle)
            stats[k]['cases']+=1
            stats[k]['oracle_edges']+=osz;stats[k]['raw_edges']+=rsz;stats[k]['exact_raw_eq_oracle']+=int(raw is not None and set(raw)==set(oracle))
            stats[k]['oracle_edge_recall_num']+=inter;stats[k]['oracle_edge_recall_den']+=osz
            if raw is None:stats[k]['unmappable']+=1
            one[k]={'order':ordr,'raw':raw,'intersection':inter,'oracle_size':osz}
        if len(examples)<20 and (one['story_first']['intersection']<len(oracle) or one['gender_order']['intersection']<len(oracle)):examples.append(one)
    report={'validation_cases':total,'strategies':{},'examples':examples,'proof_state_target_relation_key_used':False,'test_fetched':False}
    for k,c in stats.items():
        report['strategies'][k]=dict(c)
        report['strategies'][k]['exact_raw_eq_oracle_rate']=c['exact_raw_eq_oracle']/c['cases'] if c['cases'] else 0
        report['strategies'][k]['oracle_edge_recall']=c['oracle_edge_recall_num']/c['oracle_edge_recall_den'] if c['oracle_edge_recall_den'] else 0
    (o/'node_identity_reconstruction.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='examples'},ensure_ascii=False,sort_keys=True));print(json.dumps(examples[:8],ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()
