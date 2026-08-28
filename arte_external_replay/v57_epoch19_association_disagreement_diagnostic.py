import json
from collections import Counter
from pathlib import Path
import v57_epoch19_clutrr_relation_composition_runner as e19
import v57_epoch19_validation_residual_diagnostic as d

def main(out='diag_output6'):
    o=Path(out);o.mkdir(parents=True,exist_ok=True)
    train=d.parse_full(e19.fetch_text(e19.TRAIN_PATH));val=d.parse_full(e19.fetch_text(e19.VAL_PATH))
    pair,_=e19.build_binary_table([{'edges':c['edges'],'target':c['target']} for c in train])
    dev=[c for c in val if len(c['edges'])==3 and e19.compose(c['edges'],pair,'FORWARD') is not None]
    rows=[]
    for c in dev:
        l=e19.compose(c['edges'],pair,'FORWARD');r=d.right_compose(c['edges'],pair)
        if c['edges'][0] not in {'husband','wife'} or r is None or l==r:continue
        rows.append({'edges':c['edges'],'left':l,'right':r,'target':c['target'],'right_is_correct':r==c['target'],'left_is_correct':l==c['target']})
    good=[x for x in rows if x['right_is_correct'] and not x['left_is_correct']]
    bad=[x for x in rows if x['left_is_correct'] and not x['right_is_correct']]
    def counts(xs,pos):return dict(Counter(x['edges'][pos] for x in xs))
    report={'disagreement_cases':len(rows),'right_repairs':len(good),'right_damages':len(bad),'repair_edges':[x['edges'] for x in good],'damage_edges':[x['edges'] for x in bad],'repair_pos1_counts':counts(good,1),'repair_pos2_counts':counts(good,2),'damage_pos1_counts':counts(bad,1),'damage_pos2_counts':counts(bad,2),'repair_left_right_pairs':dict(Counter(f"{x['left']}|{x['right']}" for x in good)),'damage_left_right_pairs':dict(Counter(f"{x['left']}|{x['right']}" for x in bad)),'test_fetched':False}
    (o/'epoch19_association_disagreement.json').write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
