import json
from pathlib import Path
import v57_epoch19_clutrr_relation_composition_runner as e19
import v57_epoch19_validation_residual_diagnostic as d


def main(out='diag_output5'):
    o=Path(out);o.mkdir(parents=True,exist_ok=True)
    train=d.parse_full(e19.fetch_text(e19.TRAIN_PATH));val=d.parse_full(e19.fetch_text(e19.VAL_PATH))
    pair,_=e19.build_binary_table([{'edges':c['edges'],'target':c['target']} for c in train])
    dev=[c for c in val if len(c['edges'])==3 and e19.compose(c['edges'],pair,'FORWARD') is not None]
    left=lambda c:e19.compose(c['edges'],pair,'FORWARD')
    right=lambda c:d.right_compose(c['edges'],pair)
    errors=[c for c in dev if left(c)!=c['target']]
    # The latent class is generated from residual position, not authored relation names.
    residual_prefix_class=sorted({c['edges'][0] for c in errors})
    def generated(c):
        l=left(c)
        if c['edges'][0] in residual_prefix_class:
            r=right(c)
            return r if r is not None else l
        return l
    ok=sum(generated(c)==c['target'] for c in dev)
    changed=[c for c in dev if generated(c)!=left(c)]
    wrong_class=sorted({c['edges'][1] for c in errors})
    def wrong(c):
        l=left(c)
        if c['edges'][0] in wrong_class:
            r=right(c);return r if r is not None else l
        return l
    wok=sum(wrong(c)==c['target'] for c in dev)
    report={'dev_cases':len(dev),'base_left_accuracy':sum(left(c)==c['target'] for c in dev)/len(dev),'residual_prefix_class':residual_prefix_class,'generated_accuracy':ok/len(dev),'generated_exact_all':ok==len(dev),'generated_changed_cases':len(changed),'generated_changed_correct':sum(generated(c)==c['target'] for c in changed),'wrong_class':wrong_class,'wrong_accuracy':wok/len(dev),'wrong_exact_all':wok==len(dev),'test_fetched':False,'class_generated_from_validation_residual':True,'composition_laws_from_train_only':True}
    (o/'epoch19_generated_association_policy.json').write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
