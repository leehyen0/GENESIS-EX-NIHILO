import json
from pathlib import Path
import v57_epoch19_clutrr_relation_composition_runner as e19
import v57_epoch19_validation_residual_diagnostic as d


def main(out='diag_output2'):
    o=Path(out);o.mkdir(parents=True,exist_ok=True)
    train=d.parse_full(e19.fetch_text(e19.TRAIN_PATH)); val=d.parse_full(e19.fetch_text(e19.VAL_PATH))
    pair,_=e19.build_binary_table([{'edges':c['edges'],'target':c['target']} for c in train])
    tri,tri_conf=d.build_n_table(train,3)
    dev=[c for c in val if len(c['edges'])==3 and e19.compose(c['edges'],pair,'FORWARD') is not None]

    def left(c): return e19.compose(c['edges'],pair,'FORWARD')
    def right(c): return d.right_compose(c['edges'],pair)
    def macro(c): return tri.get(tuple(c['edges']))
    def hybrid(c,fallback):
        m=macro(c)
        return m if m is not None else fallback(c)

    candidates={
      'LEFT_ONLY':left,
      'RIGHT_ONLY':right,
      'TERNARY_MACRO_ELSE_LEFT':lambda c:hybrid(c,left),
      'TERNARY_MACRO_ELSE_RIGHT':lambda c:hybrid(c,right),
    }
    scores={}
    for name,fn in candidates.items():
        ok=scored=0;miss=[]
        for c in dev:
            p=fn(c)
            if p is None:
                miss.append(c['id']);continue
            scored+=1;ok+=p==c['target']
        scores[name]={'accuracy':ok/scored if scored else 0.0,'scored':scored,'missing':len(miss),'exact_all':scored==len(dev) and ok==len(dev)}

    # Residual feature discovery: identify which raw edge-token positions are enriched among left-closure errors.
    errors=[c for c in dev if left(c)!=c['target']]
    correct=[c for c in dev if left(c)==c['target']]
    token_stats={}
    vocab=sorted({x for c in dev for x in c['edges']})
    for pos in range(3):
        for tok in vocab:
            e=sum(c['edges'][pos]==tok for c in errors); q=sum(c['edges'][pos]==tok for c in correct)
            if e:
                token_stats[f'POS{pos}={tok}']={'errors_with_feature':e,'error_rate_feature':e/(e+q),'error_coverage':e/len(errors),'support':e+q}
    ranked=sorted(token_stats.items(), key=lambda kv:(kv[1]['error_coverage'],kv[1]['error_rate_feature'],kv[1]['support']), reverse=True)

    report={'dev_cases':len(dev),'left_errors':len(errors),'ternary_table_size':len(tri),'ternary_conflicts':len(tri_conf),'scores':scores,'top_residual_features':ranked[:20],'test_fetched':False}
    (o/'epoch19_residual_representation_search.json').write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(report,sort_keys=True))

if __name__=='__main__':main()
