import csv, io, json
from pathlib import Path
import v57_epoch19_clutrr_relation_composition_runner as e19

def main(out='diag_story_schema'):
    o=Path(out); o.mkdir(parents=True,exist_ok=True)
    raw_train=e19.fetch_text(e19.TRAIN_PATH)
    raw_val=e19.fetch_text(e19.VAL_PATH)
    def inspect(raw,n=4):
        rows=list(csv.reader(io.StringIO(raw)))
        header=rows[0]
        samples=[]
        for r in rows[1:n+1]:
            samples.append({str(i):r[i] for i in range(len(r))})
        return {'header':header,'column_count':len(header),'samples':samples}
    report={'train':inspect(raw_train,5),'validation':inspect(raw_val,5),'test_fetched':False}
    (o/'story_schema.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()
