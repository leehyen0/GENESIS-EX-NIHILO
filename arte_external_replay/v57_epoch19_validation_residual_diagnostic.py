import ast,csv,io,json
from collections import Counter
from pathlib import Path
import v57_epoch19_clutrr_relation_composition_runner as e19

def genders_map(s):
    out={}
    for p in s.split(','):
        if ':' in p:
            a,b=p.split(':',1);out[a.strip()]=b.strip()
    return out

def parse_full(raw):
    out=[]
    for i,row in enumerate(csv.reader(io.StringIO(raw))):
        if i==0 or len(row)<15: continue
        try: edges=e19.parse_edges(row[11])
        except Exception: continue
        try: q=ast.literal_eval(row[3])
        except Exception: q=None
        out.append({'id':row[1],'query':q,'target':row[5],'f_comb':row[8],'task_name':row[9],'edges':edges,'query_edge':row[12],'genders':genders_map(row[13])})
    return out

def right_compose(edges,table):
    if not edges:return None
    state=edges[-1]
    for prev in reversed(edges[:-1]):
        k=(prev,state)
        if k not in table:return None
        state=table[k]
    return state

def build_n_table(rows,n):
    tab={};conf=set()
    for c in rows:
        if len(c['edges'])!=n: continue
        k=tuple(c['edges']);v=c['target']
        if k in tab and tab[k]!=v:conf.add(k)
        else:tab[k]=v
    for k in conf:tab.pop(k,None)
    return tab,conf

def main(out='diag_output'):
    o=Path(out);o.mkdir(parents=True,exist_ok=True)
    train=parse_full(e19.fetch_text(e19.TRAIN_PATH));val=parse_full(e19.fetch_text(e19.VAL_PATH))
    pair,_=e19.build_binary_table([{'edges':c['edges'],'target':c['target']} for c in train])
    tri,tri_conf=build_n_table(train,3)
    dev=[c for c in val if len(c['edges'])==3 and e19.compose(c['edges'],pair,'FORWARD') is not None]
    errs=[];right_scored=right_ok=tri_scored=tri_ok=left_ok=0
    for c in dev:
        lp=e19.compose(c['edges'],pair,'FORWARD');rp=right_compose(c['edges'],pair);tp=tri.get(tuple(c['edges']))
        left_ok+=lp==c['target']
        if rp is not None:right_scored+=1;right_ok+=rp==c['target']
        if tp is not None:tri_scored+=1;tri_ok+=tp==c['target']
        if lp!=c['target']:
            qg=None
            if isinstance(c['query'],(list,tuple)) and len(c['query'])==2:qg=[c['genders'].get(c['query'][0]),c['genders'].get(c['query'][1])]
            errs.append({'id':c['id'],'query':c['query'],'query_genders':qg,'edges':c['edges'],'left_prediction':lp,'right_prediction':rp,'ternary_macro_prediction':tp,'target':c['target'],'f_comb':c['f_comb'],'query_edge':c['query_edge']})
    report={
      'dev_cases':len(dev),'left_accuracy':left_ok/len(dev),'left_errors':len(errs),
      'right_scored':right_scored,'right_accuracy':right_ok/right_scored if right_scored else 0,
      'ternary_table_size':len(tri),'ternary_conflicts':len(tri_conf),'ternary_scored':tri_scored,'ternary_accuracy':tri_ok/tri_scored if tri_scored else 0,
      'error_target_counts':Counter(x['target'] for x in errs),
      'error_edge_triplet_counts':Counter('|'.join(x['edges']) for x in errs),
      'errors':errs,
      'test_fetched':False
    }
    report['error_target_counts']=dict(report['error_target_counts']);report['error_edge_triplet_counts']=dict(report['error_edge_triplet_counts'])
    (o/'epoch19_validation_residual.json').write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='errors'},sort_keys=True));print(json.dumps(errs[:20],sort_keys=True))
if __name__=='__main__':main()
