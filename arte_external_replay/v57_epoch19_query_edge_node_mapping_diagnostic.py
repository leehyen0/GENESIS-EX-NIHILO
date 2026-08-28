import ast,csv,io,json,re
from pathlib import Path
from collections import Counter
import v57_epoch19_clutrr_relation_composition_runner as e19

def names_story(story):
    out=[]
    for x in re.findall(r'\[([^\]]+)\]',story):
        if x not in out: out.append(x)
    return out

def gender_order(s):
    out=[]
    for p in s.split(','):
        if ':' in p:
            n=p.split(':',1)[0].strip()
            if n and n not in out: out.append(n)
    return out

def eval_order(order,q,qe):
    try:i,j=qe
    except:return {'mappable':False,'exact_oriented':False,'exact_unoriented':False}
    if i>=len(order) or j>=len(order):return {'mappable':False,'exact_oriented':False,'exact_unoriented':False}
    pair=(order[i],order[j])
    return {'mappable':True,'pair':pair,'exact_oriented':pair==q,'exact_unoriented':set(pair)==set(q)}

def main(out='diag_query_edge_mapping'):
    o=Path(out);o.mkdir(parents=True,exist_ok=True)
    rows=list(csv.reader(io.StringIO(e19.fetch_text(e19.VAL_PATH))))
    stats={k:Counter() for k in ('gender_order','story_first')};examples=[]
    for row in rows[1:]:
        if len(row)<15:continue
        try:q=tuple(ast.literal_eval(row[3]));qe=tuple(ast.literal_eval(row[12]))
        except:continue
        orders={'gender_order':gender_order(row[13]),'story_first':names_story(row[2])}
        rec={'id':row[1],'query':q,'query_edge':qe,'story':row[2]}
        for k,order in orders.items():
            z=eval_order(order,q,qe);rec[k]={'order':order,**z};stats[k]['cases']+=1
            for f in ('mappable','exact_oriented','exact_unoriented'):stats[k][f]+=int(z[f])
        if len(examples)<20 and not rec['gender_order']['exact_oriented']:examples.append(rec)
    report={'strategies':{},'examples':examples,'test_fetched':False}
    for k,c in stats.items():
        report['strategies'][k]={**dict(c),'oriented_rate':c['exact_oriented']/c['cases'],'unoriented_rate':c['exact_unoriented']/c['cases']}
    (o/'query_edge_node_mapping.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='examples'},ensure_ascii=False,sort_keys=True));print(json.dumps(examples[:10],ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()
