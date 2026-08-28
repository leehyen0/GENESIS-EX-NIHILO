import json
from collections import Counter,defaultdict
from pathlib import Path
import v57_epoch19_clutrr_relation_composition_runner as e19
import v57_epoch19_validation_residual_diagnostic as d

def main(out='diag_output4'):
    o=Path(out);o.mkdir(parents=True,exist_ok=True)
    train=d.parse_full(e19.fetch_text(e19.TRAIN_PATH))
    rels=sorted({x for c in train for x in c['edges']} | {c['target'] for c in train})
    pair_examples=[]; target_paths=defaultdict(list); prefix_cont=defaultdict(Counter)
    for c in train:
        target_paths[c['target']].append(c['edges'])
        if len(c['edges'])==2:
            pair_examples.append((tuple(c['edges']),c['target']))
        if len(c['edges'])==3:
            prefix_cont[tuple(c['edges'][:2])][(c['edges'][2],c['target'])]+=1
    pair_target=defaultdict(set)
    for k,v in pair_examples: pair_target[k].add(v)
    pair_det={k:next(iter(v)) for k,v in pair_target.items() if len(v)==1}
    interesting_targets=['father-in-law','mother-in-law','father','mother','son','daughter','husband','wife']
    samples={t:[list(x) for x in target_paths.get(t,[])[:40]] for t in interesting_targets}
    spouse_pairs={f'{a}|{b}':v for (a,b),v in sorted(pair_det.items()) if a in ('husband','wife') or b in ('husband','wife')}
    # Behavioral signature: for each relation r used first, deterministic pair outputs across second relation.
    signatures={}
    for r in rels:
        sig={b:pair_det[(r,b)] for b in rels if (r,b) in pair_det}
        if sig: signatures[r]=sig
    def sim(a,b):
        A=signatures.get(a,{});B=signatures.get(b,{})
        common=set(A)&set(B)
        return {'common':len(common),'agree':sum(A[k]==B[k] for k in common),'agreement':sum(A[k]==B[k] for k in common)/len(common) if common else 0.0}
    sim_hw=sim('husband','wife')
    # Find relations with signatures closest to husband/wife behavior, purely behaviorally.
    nearest={}
    for anchor in ('husband','wife'):
        arr=[]
        for r in signatures:
            if r==anchor:continue
            q=sim(anchor,r);arr.append((q['agreement'],q['common'],r))
        arr.sort(reverse=True);nearest[anchor]=[{'relation':r,'agreement':a,'common':c} for a,c,r in arr[:12]]
    report={'relations':rels,'relation_count':len(rels),'deterministic_pair_count':len(pair_det),'spouse_involving_pair_laws':spouse_pairs,'interesting_target_path_samples':samples,'husband_wife_behavioral_similarity':sim_hw,'nearest_behavioral_relations':nearest,'test_fetched':False}
    (o/'epoch19_latent_relation_profile.json').write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
