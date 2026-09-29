from __future__ import annotations

import argparse, hashlib, itertools, json, os, random, subprocess, sys, tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

FEATURES=("old_basis_expressive","current_categories_separate_outcomes","current_action_alphabet_expressive","capability_present","normalized_evidence_cost")
LOCI=("OBSERVATION_BASIS","CATEGORY_BOUNDARY","ACTION_LIBRARY","SEARCH_POLICY","NULL")

def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()

@dataclass(frozen=True)
class Example:
    example_id:str
    features:dict
    oracle_locus:str

def utility(locus,oracle):
    if locus==oracle:
        return 1.0 if oracle!="NULL" else 0.5
    if oracle=="NULL":
        return -0.5
    if locus=="NULL":
        return 0.0
    return -1.0

def base_patterns():
    # Each positive class needs a conjunction. NULL counterexamples intentionally
    # share every individual feature value used by non-NULL classes.
    return [
        ("basis-a",dict(old_basis_expressive=0,current_categories_separate_outcomes=0,current_action_alphabet_expressive=1,capability_present=0,normalized_evidence_cost=.8),"OBSERVATION_BASIS"),
        ("basis-b",dict(old_basis_expressive=0,current_categories_separate_outcomes=1,current_action_alphabet_expressive=1,capability_present=0,normalized_evidence_cost=.4),"OBSERVATION_BASIS"),

        ("cat-a",dict(old_basis_expressive=1,current_categories_separate_outcomes=0,current_action_alphabet_expressive=1,capability_present=0,normalized_evidence_cost=.8),"CATEGORY_BOUNDARY"),
        ("cat-b",dict(old_basis_expressive=1,current_categories_separate_outcomes=0,current_action_alphabet_expressive=0,capability_present=0,normalized_evidence_cost=.4),"CATEGORY_BOUNDARY"),

        ("action-a",dict(old_basis_expressive=1,current_categories_separate_outcomes=1,current_action_alphabet_expressive=0,capability_present=0,normalized_evidence_cost=.8),"ACTION_LIBRARY"),
        ("action-b",dict(old_basis_expressive=0,current_categories_separate_outcomes=1,current_action_alphabet_expressive=0,capability_present=0,normalized_evidence_cost=.4),"ACTION_LIBRARY"),

        ("search-a",dict(old_basis_expressive=1,current_categories_separate_outcomes=1,current_action_alphabet_expressive=1,capability_present=1,normalized_evidence_cost=.9),"SEARCH_POLICY"),
        ("search-b",dict(old_basis_expressive=0,current_categories_separate_outcomes=0,current_action_alphabet_expressive=1,capability_present=1,normalized_evidence_cost=.9),"SEARCH_POLICY"),

        # NULL anti-shortcut counterexamples.
        ("null-basis-cap",dict(old_basis_expressive=0,current_categories_separate_outcomes=1,current_action_alphabet_expressive=1,capability_present=1,normalized_evidence_cost=.2),"NULL"),
        ("null-cat-cap",dict(old_basis_expressive=1,current_categories_separate_outcomes=0,current_action_alphabet_expressive=1,capability_present=1,normalized_evidence_cost=.2),"NULL"),
        ("null-action-cap",dict(old_basis_expressive=1,current_categories_separate_outcomes=1,current_action_alphabet_expressive=0,capability_present=1,normalized_evidence_cost=.2),"NULL"),
        ("null-efficient",dict(old_basis_expressive=1,current_categories_separate_outcomes=1,current_action_alphabet_expressive=1,capability_present=1,normalized_evidence_cost=.1),"NULL"),
        ("null-incomplete-no-pressure",dict(old_basis_expressive=0,current_categories_separate_outcomes=0,current_action_alphabet_expressive=0,capability_present=1,normalized_evidence_cost=.1),"NULL"),
    ]

def make_examples(seed:int, heldout=False):
    rng=random.Random(seed + (100000 if heldout else 0))
    rows=[]
    reps=1 if heldout else 2
    for r in range(reps):
        for name,feat,label in base_patterns():
            f=dict(feat)
            # Only cost is jittered within regions that preserve the hidden logic.
            if f["normalized_evidence_cost"]>=.8:
                f["normalized_evidence_cost"]=round(rng.uniform(.75,.98),6)
            elif f["normalized_evidence_cost"]<=.2:
                f["normalized_evidence_cost"]=round(rng.uniform(.02,.25),6)
            else:
                f["normalized_evidence_cost"]=round(rng.uniform(.32,.55),6)
            rows.append(Example(f"{'ho' if heldout else 'tr'}-{r}-{name}-{rng.randrange(10**8)}",f,label))
    return rows

def thresholds(rows,feature):
    vals=sorted(set(float(x.features[feature]) for x in rows))
    return [(a+b)/2 for a,b in zip(vals,vals[1:])]

def atoms(rows):
    out=[]
    for f in FEATURES:
        for t in thresholds(rows,f):
            out.append((f,"LE",t))
            out.append((f,"GE",t))
    return out

def atom_eval(atom,feat):
    f,op,t=atom
    return float(feat[f])<=t if op=="LE" else float(feat[f])>=t

def rule_eval(rule,feat):
    return all(atom_eval(a,feat) for a in rule)

def rule_ast(rule):
    return [{"feature":a[0],"op":a[1],"threshold":a[2]} for a in rule]

def learn_rules(rows):
    ats=atoms(rows)
    rules={}
    shortcut_counts={}
    for locus in LOCI[:-1]:
        target=tuple(x.oracle_locus==locus for x in rows)
        atom_hits=[a for a in ats if tuple(atom_eval(a,x.features) for x in rows)==target]
        shortcut_counts[locus]=len(atom_hits)
        if atom_hits:
            raise AssertionError(f"atomic shortcut for {locus}: {atom_hits[0]}")
        candidates=[]
        for depth in (2,3):
            for combo in itertools.combinations(ats,depth):
                # no repeated feature: require relational conjunction across diagnostics
                if len({a[0] for a in combo})<depth:
                    continue
                mask=tuple(rule_eval(combo,x.features) for x in rows)
                if mask==target:
                    ast=rule_ast(combo)
                    candidates.append((depth,canon(ast),combo))
            if candidates:
                break
        if not candidates:
            raise AssertionError(f"no generated conjunction rule for {locus}")
        candidates.sort(key=lambda x:(x[0],x[1]))
        depth,_,combo=candidates[0]
        rules[locus]={"rule_id":"GEN_LOCUS_RULE::"+sha(rule_ast(combo))[:20],"atoms":rule_ast(combo),"complexity":depth}
    return rules,shortcut_counts

def eval_rule_ast(ast,feat):
    return all(float(feat[a["feature"]])<=float(a["threshold"]) if a["op"]=="LE" else float(feat[a["feature"]])>=float(a["threshold"]) for a in ast)

def select(rules,feat):
    hits=[locus for locus,row in rules.items() if eval_rule_ast(row["atoms"],feat)]
    if not hits:return "NULL"
    if len(hits)>1:raise AssertionError(f"locus rule overlap: {hits}")
    return hits[0]

def model_hash(rules):
    return sha({"rules":rules,"default":"NULL"})

def cold_main(model_path,feature_json):
    obj=json.loads(Path(model_path).read_text())
    if model_hash(obj["rules"])!=obj["model_hash"]:raise AssertionError("model hash mismatch")
    feat=json.loads(feature_json)
    print(json.dumps({"model_hash":obj["model_hash"],"selected_locus":select(obj["rules"],feat)},sort_keys=True))
    return 0

def main(seed:int):
    train=make_examples(seed,False)
    rules,shortcuts=learn_rules(train)
    train_acc=sum(select(rules,x.features)==x.oracle_locus for x in train)/len(train)
    if train_acc!=1.0:raise AssertionError(f"train accuracy {train_acc}")
    if len(rules)<4 or any(v for v in shortcuts.values()):raise AssertionError("rule generation gate")

    held=make_examples(seed,True)
    predictions=[{"id":x.example_id,"features":x.features,"selected_locus":select(rules,x.features)} for x in held]
    seal=sha(predictions)

    results=[]; correct=True; wrong_degrades=True; null_ok=True
    for x,p in zip(held,predictions):
        selected=p["selected_locus"]; oracle=x.oracle_locus
        correct &= selected==oracle
        if oracle=="NULL":
            null_ok &= selected=="NULL"
            wrong="ACTION_LIBRARY"
        else:
            wrong=next(l for l in LOCI[:-1] if l!=oracle)
        tu=utility(selected,oracle); wu=utility(wrong,oracle); ru=utility("NULL",oracle)
        if oracle!="NULL":wrong_degrades &= tu>wu and tu>ru
        else:wrong_degrades &= tu>wu
        results.append({**p,"oracle_after_reveal":oracle,"correct":selected==oracle,"treatment_utility":tu,"wrong_locus":wrong,"wrong_utility":wu,"null_utility":ru})

    if not correct or not wrong_degrades or not null_ok:
        raise AssertionError({"correct":correct,"wrong_degrades":wrong_degrades,"null_ok":null_ok})

    mh=model_hash(rules)
    cold=make_examples(seed+777,True)[random.Random(seed).randrange(len(base_patterns()))]
    with tempfile.TemporaryDirectory(prefix="arte_locus_selector_") as td:
        mp=Path(td)/"model.json"
        mp.write_text(json.dumps({"rules":rules,"default":"NULL","model_hash":mh},sort_keys=True),encoding="utf-8")
        cp=subprocess.run([sys.executable,__file__,"--cold",str(mp),json.dumps(cold.features,sort_keys=True)],text=True,capture_output=True,timeout=30,check=False,env={**os.environ,"PYTHONPATH":str(Path(__file__).resolve().parents[1])})
        if cp.returncode!=0:raise AssertionError(cp.stderr)
        cold_pred=json.loads([z for z in cp.stdout.splitlines() if z.strip()][-1])
    cold_ok=cold_pred["model_hash"]==mh and cold_pred["selected_locus"]==cold.oracle_locus
    if not cold_ok:raise AssertionError({"cold":cold_pred,"oracle":cold.oracle_locus})

    print(json.dumps({
        "status":"PASS_BOUNDED_RUNTIME_REPAIR_LOCUS_SELECTOR_WITH_NULL_AND_COLD_TRANSFER",
        "seed":seed,
        "generated_rules":rules,
        "atomic_shortcut_counts":shortcuts,
        "train_accuracy":train_acc,
        "heldout_prediction_seal_sha256":seal,
        "heldout_accuracy":sum(x["correct"] for x in results)/len(results),
        "heldout_results":results,
        "wrong_locus_degrades_all":wrong_degrades,
        "null_cases_correct":null_ok,
        "model_hash":mh,
        "cold_prediction":cold_pred,
        "cold_oracle_after_reveal":cold.oracle_locus,
        "cold_exact":cold_ok,
        "claim_boundary":{
            "diagnostic_feature_vocabulary_human_authored":True,
            "locus_vocabulary_human_authored":True,
            "classifier_grammar_human_authored":True,
            "downstream_binding_human_authored":True,
            "generated_locus_rules_not_predeclared":True,
            "autonomous_new_locus_invention":False,
            "independent_organizational_custody":False,
            "physical_world":False,
            "global_recursive_acceleration":False,
            "AGI":False,"ASI":False,"superintelligence":False
        }
    },sort_keys=True))
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=20260929701)
    ap.add_argument("--cold",nargs=2)
    a=ap.parse_args()
    if a.cold:raise SystemExit(cold_main(a.cold[0],a.cold[1]))
    raise SystemExit(main(a.seed))
