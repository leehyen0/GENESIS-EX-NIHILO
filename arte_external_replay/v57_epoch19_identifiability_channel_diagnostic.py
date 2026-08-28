import ast,csv,io,json
from collections import defaultdict,Counter
from pathlib import Path
import v57_epoch19_clutrr_relation_composition_runner as e19
import v57_epoch19_validation_residual_diagnostic as d

def parse_extended(raw):
    out=[]
    for i,row in enumerate(csv.reader(io.StringIO(raw))):
        if i==0 or len(row)<15:continue
        try: edges=e19.parse_edges(row[11]); q=ast.literal_eval(row[3])
        except Exception:continue
        genders=d.genders_map(row[13])
        try: story_edges=ast.literal_eval(row[10])
        except Exception: story_edges=row[10]
        out.append({'id':row[1],'query':q,'target':row[5],'f_comb':row[8],'task_name':row[9],'story_edges':story_edges,'edges':edges,'query_edge':row[12],'genders':genders,'query_genders':tuple(genders.get(x) for x in q) if isinstance(q,(list,tuple)) and len(q)==2 else None})
    return out

def purity(rows,keyfn):
    groups=defaultdict(list)
    for c in rows: groups[repr(keyfn(c))].append(c['target'])
    conflicts={k:dict(Counter(v)) for k,v in groups.items() if len(set(v))>1}
    return {'groups':len(groups),'conflicting_groups':len(conflicts),'conflicting_cases':sum(sum(x.values()) for x in conflicts.values()),'exact_identifiable':len(conflicts)==0,'examples':dict(list(conflicts.items())[:20])}

def main(out='diag_output7'):
    o=Path(out);o.mkdir(parents=True,exist_ok=True)
    val=parse_extended(e19.fetch_text(e19.VAL_PATH));train=parse_extended(e19.fetch_text(e19.TRAIN_PATH))
    pair,_=e19.build_binary_table([{'edges':c['edges'],'target':c['target']} for c in train])
    dev=[c for c in val if len(c['edges'])==3 and e19.compose(c['edges'],pair,'FORWARD') is not None]
    candidates={
      'EDGE_TYPES':lambda c:tuple(c['edges']),
      'EDGE_TYPES_QUERY_GENDERS':lambda c:(tuple(c['edges']),c['query_genders']),
      'EDGE_TYPES_QUERY_EDGE':lambda c:(tuple(c['edges']),c['query_edge']),
      'EDGE_TYPES_QUERY_GENDERS_QUERY_EDGE':lambda c:(tuple(c['edges']),c['query_genders'],c['query_edge']),
      'EDGE_TYPES_F_COMB':lambda c:(tuple(c['edges']),c['f_comb']),
      'EDGE_TYPES_STORY_EDGES':lambda c:(tuple(c['edges']),repr(c['story_edges'])),
      'EDGE_TYPES_STORY_EDGES_QUERY_EDGE':lambda c:(tuple(c['edges']),repr(c['story_edges']),c['query_edge']),
      'EDGE_TYPES_STORY_EDGES_QUERY_GENDERS_QUERY_EDGE':lambda c:(tuple(c['edges']),repr(c['story_edges']),c['query_genders'],c['query_edge']),
    }
    report={'dev_cases':len(dev),'channel_identifiability':{k:purity(dev,f) for k,f in candidates.items()},'test_fetched':False,'diagnostic_uses_validation_targets':True}
    (o/'epoch19_identifiability_channels.json').write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
