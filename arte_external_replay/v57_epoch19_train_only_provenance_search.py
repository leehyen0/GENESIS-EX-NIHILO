import json
from pathlib import Path
import v57_epoch19_clutrr_relation_composition_runner as e19
import v57_epoch19_validation_residual_diagnostic as d


def build_projection(train3,pair,keyfn):
    table={}; conflicts=set(); obs=0
    for c in train3:
        coarse=e19.compose(c['edges'],pair,'FORWARD')
        if coarse is None: continue
        k=keyfn(c,coarse);v=c['target'];obs+=1
        if k in table and table[k]!=v: conflicts.add(k)
        else: table[k]=v
    for k in conflicts:table.pop(k,None)
    return table,conflicts,obs

def main(out='diag_output3'):
    o=Path(out);o.mkdir(parents=True,exist_ok=True)
    train=d.parse_full(e19.fetch_text(e19.TRAIN_PATH));val=d.parse_full(e19.fetch_text(e19.VAL_PATH))
    pair,_=e19.build_binary_table([{'edges':c['edges'],'target':c['target']} for c in train])
    train3=[c for c in train if len(c['edges'])==3]
    dev=[c for c in val if len(c['edges'])==3 and e19.compose(c['edges'],pair,'FORWARD') is not None]
    keyfns={
      'PREFIX1_COARSE':lambda c,z:(c['edges'][0],z),
      'PREFIX1_LAST_COARSE':lambda c,z:(c['edges'][0],c['edges'][-1],z),
      'PREFIX2_COARSE':lambda c,z:(c['edges'][0],c['edges'][1],z),
      'PREFIX1_MID_LAST_COARSE':lambda c,z:(c['edges'][0],c['edges'][1],c['edges'][-1],z),
    }
    results={}
    for name,keyfn in keyfns.items():
        tab,conf,obs=build_projection(train3,pair,keyfn)
        ok=covered=changed=0;miss=0
        for c in dev:
            coarse=e19.compose(c['edges'],pair,'FORWARD');k=keyfn(c,coarse)
            p=tab.get(k,coarse)
            covered+=k in tab;changed+=p!=coarse;ok+=p==c['target'];miss+=k not in tab
        results[name]={'accuracy':ok/len(dev),'exact_all':ok==len(dev),'table_size':len(tab),'conflicts':len(conf),'train3_observations':obs,'validation_key_covered':covered,'validation_key_missing':miss,'validation_predictions_changed':changed}
    report={'train3_cases':len(train3),'dev_cases':len(dev),'base_accuracy':sum(e19.compose(c['edges'],pair,'FORWARD')==c['target'] for c in dev)/len(dev),'results':results,'test_fetched':False,'projection_laws_learned_from_train_only':True,'signature_family_selected_from_validation_residual':True}
    (o/'epoch19_train_only_provenance_search.json').write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
