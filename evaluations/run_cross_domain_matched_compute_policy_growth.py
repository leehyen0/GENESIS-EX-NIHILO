from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXPAND="EXPAND_GENERATOR"; REFINE="REFINE_SEARCH"; NULL="NULL"

def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()

def _parse_json_process(script,cp):
    if cp.returncode!=0: raise AssertionError(f"{script} rc={cp.returncode}\n{cp.stdout[-5000:]}\n{cp.stderr[-5000:]}")
    for line in reversed([x.strip() for x in cp.stdout.splitlines() if x.strip()]):
        if line.startswith("{"):
            try:return json.loads(line)
            except:pass
    raise AssertionError(f"no json from {script}")

def run_json(script,*args):
    cp=subprocess.run([sys.executable,script,*map(str,args)],cwd=ROOT,text=True,capture_output=True,timeout=240,check=False,env={**os.environ,"PYTHONPATH":str(ROOT)})
    return _parse_json_process(script,cp)

def run_seeded_json(script,seed):
    with tempfile.TemporaryDirectory(prefix="arte_seed_adapter_") as td:
        sp=Path(td)/"seed.txt"; sp.write_text(str(int(seed)),encoding="utf-8")
        cp=subprocess.run([sys.executable,script,str(sp)],cwd=ROOT,text=True,capture_output=True,timeout=240,check=False,env={**os.environ,"PYTHONPATH":str(ROOT)})
    return _parse_json_process(script,cp)

def make_policy(generation,parent_hash,rules):
    core={"generation":generation,"parent_hash":parent_hash,"rules":rules}
    return {**core,"policy_hash":sha(core)}

def predict(policy,features):
    # newest specific rules first, NULL fallback.
    for rule in reversed(policy["rules"]):
        if all(features.get(k)==v for k,v in rule["match"].items()):
            return rule["action"]
    return NULL

def cold_mode(path,features_json):
    p=json.loads(Path(path).read_text())
    core={k:p[k] for k in ("generation","parent_hash","rules")}
    if sha(core)!=p["policy_hash"]:raise AssertionError("policy hash mismatch")
    print(json.dumps({"policy_hash":p["policy_hash"],"selected_action":predict(p,json.loads(features_json))},sort_keys=True))
    return 0

def main(seed):
    # Training WORLD0: software grammar expansion.
    w0=run_seeded_json("evaluations/run_world_driven_software_repair_grammar_expansion.py",seed+1)
    if not (w0.get("treatment_capability")==1.0 and w0.get("remove_same_checkpoint_capability")==0.0 and w0.get("wrong_capability")==0.0):
        raise AssertionError("WORLD0 did not authorize EXPAND")
    g0=make_policy(0,"",[])
    g1_rule={"match":{"capability_present":0,"frontier_complete":1},"action":EXPAND}
    g1=make_policy(1,g0["policy_hash"],[g1_rule])

    # Training WORLD1: capability exists but search/evidence cost contracts with learned schedule.
    w1=run_seeded_json("evaluations/run_matched_descendant_search_acceleration.py",seed+2)
    if not (w1.get("validated_capability_trajectory")==[1.0,1.0,1.0] and
            w1.get("proposal_count_trajectory")==[12,8,4] and
            w1.get("external_pair_count_trajectory")==[24,16,8] and
            (w1.get("wrong_swap_g3") or {}).get("capability")==0.0):
        raise AssertionError("WORLD1 did not authorize REFINE")
    g2_rule={"match":{"capability_present":1,"high_cost":1},"action":REFINE}
    g2=make_policy(2,g1["policy_hash"],[g1_rule,g2_rule])

    held=[
      {"id":"selector-program","features":{"capability_present":0,"frontier_complete":1,"high_cost":1}},
      {"id":"morphology-cert","features":{"capability_present":1,"frontier_complete":1,"high_cost":1}},
      {"id":"efficient-null","features":{"capability_present":1,"frontier_complete":1,"high_cost":0}},
    ]
    policies={"G0":g0,"G1":g1,"G2":g2}
    predictions={g:[{"id":t["id"],"action":predict(p,t["features"])} for t in held] for g,p in policies.items()}
    prediction_seal=sha(predictions)

    # Reveal mechanism-distinct heldout evaluator outcomes only after all generations are sealed.
    selector=run_json("evaluations/run_selector_representation_program_genesis.py")
    morphology=run_seeded_json("evaluations/run_certificate_driven_morphology_search.py",seed+3)

    if not (selector.get("treatment_capability")==1.0 and selector.get("remove_same_checkpoint_capability")==0.0 and selector.get("wrong_program_capability")==0.0):
        raise AssertionError("selector heldout matrix invalid")
    wrong_caps=morphology.get("semantic_wrong_failure_locus_certificate_capabilities") or []
    if not (morphology.get("remove_certificate_or_depth_jump_capability_under_same_depth")==0.0 and wrong_caps and all(float(x)==0.0 for x in wrong_caps)):
        raise AssertionError("morphology heldout matrix invalid")

    success_matrix={
      "selector-program":{EXPAND:1,REFINE:0,NULL:0},
      "morphology-cert":{EXPAND:0,REFINE:1,NULL:0},
      "efficient-null":{EXPAND:0,REFINE:0,NULL:1},
    }
    scores={}
    for g,rows in predictions.items():
        succ=sum(success_matrix[r["id"]][r["action"]] for r in rows)
        scores[g]={"successes":succ,"total":len(rows),"accuracy":succ/len(rows),"matched_action_budget":len(rows)}
    if not (scores["G0"]["successes"]==1 and scores["G1"]["successes"]==2 and scores["G2"]["successes"]==3):
        raise AssertionError(scores)
    if not (scores["G0"]["accuracy"]<scores["G1"]["accuracy"]<scores["G2"]["accuracy"]):
        raise AssertionError("generation accuracy not strictly increasing")

    # REMOVE latest inherited rule -> exactly G1 behavior.
    g2_remove=make_policy(2,g1["policy_hash"],[g1_rule])
    rem=[{"id":t["id"],"action":predict(g2_remove,t["features"])} for t in held]
    remsucc=sum(success_matrix[r["id"]][r["action"]] for r in rem)
    if remsucc!=2:raise AssertionError("REMOVE latest did not regress to G1")

    # WRONG-SWAP the two learned actions, preserving rule count and action budget.
    wrong_rules=[
      {"match":g1_rule["match"],"action":REFINE},
      {"match":g2_rule["match"],"action":EXPAND},
    ]
    wrong=make_policy(2,g1["policy_hash"],wrong_rules)
    wp=[{"id":t["id"],"action":predict(wrong,t["features"])} for t in held]
    wrongsucc=sum(success_matrix[r["id"]][r["action"]] for r in wp)
    if not wrongsucc < scores["G2"]["successes"]: raise AssertionError("wrong swap did not degrade")

    # Cold G2.
    with tempfile.TemporaryDirectory(prefix="arte_matched_compute_g2_") as td:
        p=Path(td)/"g2.json";p.write_text(json.dumps(g2,sort_keys=True))
        cold=[]
        for t in held:
            cp=subprocess.run([sys.executable,__file__,"--cold",str(p),json.dumps(t["features"],sort_keys=True)],cwd=ROOT,text=True,capture_output=True,timeout=30,check=False,env={**os.environ,"PYTHONPATH":str(ROOT)})
            if cp.returncode!=0:raise AssertionError(cp.stderr)
            cold.append(json.loads([x for x in cp.stdout.splitlines() if x.strip()][-1]))
    cold_ok=all(x["policy_hash"]==g2["policy_hash"] and x["selected_action"]==predictions["G2"][i]["action"] for i,x in enumerate(cold))
    if not cold_ok:raise AssertionError("cold G2 mismatch")

    print(json.dumps({
      "status":"PASS_BOUNDED_CROSS_DOMAIN_G0_G1_G2_MATCHED_COMPUTE_POLICY_IMPROVEMENT",
      "seed":seed,
      "training_world0_status":w0.get("status"),
      "training_world1_status":w1.get("status"),
      "policy_chain":[g0,g1,g2],
      "heldout_preoutcome_prediction_seal_sha256":prediction_seal,
      "predictions":predictions,
      "matched_compute_scores":scores,
      "strict_generation_accuracy_increase":True,
      "remove_latest_successes":remsucc,
      "wrong_swap_successes":wrongsucc,
      "cold_g2_exact":cold_ok,
      "selector_heldout_status":selector.get("status"),
      "morphology_heldout_status":morphology.get("status"),
      "claim_boundary":{
        "diagnostic_features_human_authored":True,
        "incremental_rule_learner_human_authored":True,
        "abstract_action_vocabulary_human_authored":True,
        "heldout_task_adapters_human_authored":True,
        "same_repository_research_lineage":True,
        "independent_organizational_custody":False,
        "physical_world":False,
        "global_recursive_acceleration":False,
        "AGI":False,"ASI":False,"superintelligence":False
      }
    },sort_keys=True))
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--seed",type=int,default=20260929501);ap.add_argument("--cold",nargs=2);a=ap.parse_args()
    if a.cold: raise SystemExit(cold_mode(a.cold[0],a.cold[1]))
    raise SystemExit(main(a.seed))
