from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPAND = "EXPAND_GENERATOR"
REFINE = "REFINE_SEARCH"
NULL = "NULL"

@dataclass(frozen=True)
class PolicyGenome:
    generation: int
    parent_hash: str
    mapping: tuple[tuple[str, str], ...]
    evidence: tuple[tuple[str, str, str], ...]
    genome_hash: str

    @staticmethod
    def genesis() -> "PolicyGenome":
        payload={"generation":0,"parent_hash":"","mapping":[],"evidence":[]}
        h=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        return PolicyGenome(0,"",(),(),h)

    def select(self,residual_class: str) -> str:
        return dict(self.mapping).get(str(residual_class), NULL)

    def update(self,residual_class: str,action: str,world_id: str) -> "PolicyGenome":
        m=dict(self.mapping)
        m[str(residual_class)]=str(action)
        ev=list(self.evidence)
        ev.append((str(world_id),str(residual_class),str(action)))
        payload={
            "generation":self.generation+1,
            "parent_hash":self.genome_hash,
            "mapping":sorted(m.items()),
            "evidence":ev,
        }
        h=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        return PolicyGenome(
            self.generation+1,self.genome_hash,tuple(sorted(m.items())),tuple(ev),h
        )

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(obj: dict) -> "PolicyGenome":
        g=PolicyGenome(
            int(obj["generation"]),
            str(obj["parent_hash"]),
            tuple(tuple(x) for x in obj["mapping"]),
            tuple(tuple(x) for x in obj["evidence"]),
            str(obj["genome_hash"]),
        )
        payload={
            "generation":g.generation,
            "parent_hash":g.parent_hash,
            "mapping":[list(x) for x in g.mapping],
            "evidence":[list(x) for x in g.evidence],
        }
        chk=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        if chk != g.genome_hash:
            raise AssertionError("policy genome hash mismatch")
        return g

def run_json(script: str, seed: int | None=None) -> dict:
    cmd=[sys.executable,script]
    tmp=None
    if seed is not None:
        fd,path=tempfile.mkstemp(prefix="arte_closed_chain_seed_",text=True)
        os.close(fd)
        tmp=Path(path)
        tmp.write_text(str(int(seed)),encoding="utf-8")
        cmd.append(str(tmp))
    try:
        cp=subprocess.run(
            cmd,cwd=ROOT,text=True,capture_output=True,timeout=240,check=False,
            env={**os.environ,"PYTHONPATH":str(ROOT)},
        )
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    if cp.returncode != 0:
        raise AssertionError(f"{script} failed rc={cp.returncode}\nSTDOUT={cp.stdout[-4000:]}\nSTDERR={cp.stderr[-4000:]}")
    for line in reversed([x.strip() for x in cp.stdout.splitlines() if x.strip()]):
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    raise AssertionError(f"{script} produced no JSON payload")

def seal(obj) -> str:
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def strict_decrease(xs):
    return len(xs)>=2 and all(b<a for a,b in zip(xs,xs[1:]))

def learn_world0(policy: PolicyGenome, row: dict) -> PolicyGenome:
    required=(
        row.get("status","").startswith("PASS_BOUNDED_WORLD_FALSIFICATION_DRIVEN_EXECUTABLE_SOFTWARE_REPAIR_GRAMMAR_EXPANSION"),
        row.get("treatment_capability")==1.0,
        row.get("remove_same_checkpoint_capability")==0.0,
        row.get("wrong_capability")==0.0,
        row.get("reset_capability")==0.0,
        int(row.get("old_complete_falsified_contexts",0))>=2,
    )
    if not all(required):
        raise AssertionError(f"WORLD0 did not causally authorize EXPAND: {required}")
    return policy.update("COMPLETE_FRONTIER_FAILURE",EXPAND,"WORLD0_SOFTWARE_GRAMMAR")

def learn_world1(policy: PolicyGenome, row: dict) -> PolicyGenome:
    capability=row.get("validated_capability_trajectory",[])
    proposals=row.get("proposal_count_trajectory",[])
    pairs=row.get("external_pair_count_trajectory",[])
    wrong=(row.get("wrong_swap_g3") or {}).get("capability")
    required=(
        row.get("status")=="PASS_BOUNDED_MATCHED_DESCENDANT_SEARCH_ACCELERATION",
        capability==[1.0,1.0,1.0],
        strict_decrease(proposals),
        strict_decrease(pairs),
        wrong==0.0,
        row.get("strict_efficiency_increase") is True,
    )
    if not all(required):
        raise AssertionError(f"WORLD1 did not causally authorize REFINE: {required}")
    return policy.update("VALID_BUT_HIGH_COST",REFINE,"WORLD1_DESCENDANT_SEARCH")

def score_selector(row: dict) -> bool:
    return all((
        row.get("status")=="PASS_BOUNDED_COMPOSITIONAL_SELECTOR_REPRESENTATION_PROGRAM_GENESIS_AND_PRE_OUTCOME_TRANSFER",
        row.get("treatment_capability")==1.0,
        row.get("remove_same_checkpoint_capability")==0.0,
        row.get("wrong_program_capability")==0.0,
        row.get("old_more_compute_selected_candidate_count")==0,
        row.get("heldout_candidate_outcomes_exposed_before_selection") is False,
    ))

def score_morphology(row: dict) -> bool:
    speeds=row.get("certificate_vs_exhaustive_high_level_speedups",[])
    wrong=row.get("semantic_wrong_failure_locus_certificate_capabilities",[])
    return all((
        row.get("status")=="PASS_BOUNDED_CERTIFICATE_DRIVEN_MORPHOLOGY_SEARCH_ACCELERATION_WITHOUT_GLOBAL_RECURSIVE_ACCELERATION",
        row.get("remove_certificate_or_depth_jump_capability_under_same_depth")==0.0,
        bool(wrong) and all(float(x)==0.0 for x in wrong),
        bool(speeds) and max(float(x) for x in speeds)>1.0,
        row.get("candidate_generation_uses_hidden_outcomes") is False,
    ))

def score_dataflow(row: dict) -> bool:
    return all((
        row.get("status")=="PASS_BOUNDED_SOURCE_DERIVED_SELECTOR_DATAFLOW_PATH_GENESIS_AND_PRE_OUTCOME_TRANSFER",
        row.get("treatment_capability")==1.0,
        row.get("remove_same_checkpoint_capability")==0.0,
        row.get("wrong_capability")==0.0,
        row.get("old_more_compute_selected_count")==0,
        row.get("heldout_candidate_outcomes_exposed_before_selection") is False,
    ))

def cold_predict(policy_path: str,residual_class: str) -> int:
    obj=json.loads(Path(policy_path).read_text(encoding="utf-8"))
    policy=PolicyGenome.from_json(obj)
    print(json.dumps({
        "cold_process":True,
        "policy_hash":policy.genome_hash,
        "generation":policy.generation,
        "residual_class":residual_class,
        "selected_action":policy.select(residual_class),
    },sort_keys=True))
    return 0

def main(seed: int) -> int:
    # G0 starts without a repair mapping.
    g0=PolicyGenome.genesis()

    # WORLD0: complete old software-repair frontier failure. World evidence chooses expansion.
    w0=run_json("evaluations/run_world_driven_software_repair_grammar_expansion.py",seed+11)
    g1=learn_world0(g0,w0)
    if g1.select("COMPLETE_FRONTIER_FAILURE") != EXPAND:
        raise AssertionError("G1 did not inherit expansion policy")

    # WORLD1: capability exists but evidence/search cost is unnecessarily high.
    w1=run_json("evaluations/run_matched_descendant_search_acceleration.py",seed+23)
    g2=learn_world1(g1,w1)
    if g2.select("VALID_BUT_HIGH_COST") != REFINE:
        raise AssertionError("G2 did not inherit refinement policy")

    # Fresh G2 predictions are sealed before mechanism-distinct evaluators execute.
    predictions={
        "WORLD2_SELECTOR_PROGRAM":{
            "residual_class":"COMPLETE_FRONTIER_FAILURE",
            "selected_action":g2.select("COMPLETE_FRONTIER_FAILURE"),
        },
        "WORLD2_MORPHOLOGY_CERT":{
            "residual_class":"VALID_BUT_HIGH_COST",
            "selected_action":g2.select("VALID_BUT_HIGH_COST"),
        },
    }
    if predictions["WORLD2_SELECTOR_PROGRAM"]["selected_action"] != EXPAND:
        raise AssertionError("wrong preoutcome selector-program action")
    if predictions["WORLD2_MORPHOLOGY_CERT"]["selected_action"] != REFINE:
        raise AssertionError("wrong preoutcome morphology action")
    transfer_prediction_seal=seal(predictions)

    w2a=run_json("evaluations/run_selector_representation_program_genesis.py")
    w2b=run_json("evaluations/run_certificate_driven_morphology_search.py",seed+37)
    selector_pass=score_selector(w2a)
    morphology_pass=score_morphology(w2b)

    # Cold descendant: policy bytes only, a new Python process must reconstruct exact action.
    with tempfile.TemporaryDirectory(prefix="arte_closed_chain_policy_") as td:
        policy_path=Path(td)/"policy.json"
        policy_path.write_text(json.dumps(g2.to_json(),sort_keys=True),encoding="utf-8")
        cp=subprocess.run(
            [sys.executable,__file__,"--cold-predict",str(policy_path),"COMPLETE_FRONTIER_FAILURE"],
            cwd=ROOT,text=True,capture_output=True,timeout=30,check=False,
            env={**os.environ,"PYTHONPATH":str(ROOT)},
        )
        if cp.returncode!=0:
            raise AssertionError(f"cold child failed: {cp.stderr}")
        cold=None
        for line in reversed([x.strip() for x in cp.stdout.splitlines() if x.strip()]):
            if line.startswith("{"):
                cold=json.loads(line); break
        if cold is None:
            raise AssertionError("cold child returned no prediction")
        cold_prediction={
            "policy_hash":cold["policy_hash"],
            "generation":cold["generation"],
            "residual_class":cold["residual_class"],
            "selected_action":cold["selected_action"],
        }
        cold_prediction_seal=seal(cold_prediction)

    if cold_prediction["policy_hash"]!=g2.genome_hash or cold_prediction["selected_action"]!=EXPAND:
        raise AssertionError("cold reconstructed policy mismatch")

    # Only after cold prediction seal, execute WORLD3.
    w3=run_json("evaluations/run_source_derived_dataflow_path_genesis.py")
    dataflow_pass=score_dataflow(w3)

    all_pass=selector_pass and morphology_pass and dataflow_pass
    if not all_pass:
        raise AssertionError(f"transfer scoring failed selector={selector_pass} morphology={morphology_pass} dataflow={dataflow_pass}")

    result={
        "status":"PASS_BOUNDED_CROSS_DOMAIN_META_REPAIR_POLICY_HEREDITY_AND_COLD_RECONSTRUCTION",
        "precommit":"research/ARTE_CLOSED_CAUSAL_META_POLICY_PRECOMMIT_20260929.json",
        "policy_generations":[g0.to_json(),g1.to_json(),g2.to_json()],
        "world0":{
            "residual_class":"COMPLETE_FRONTIER_FAILURE",
            "selected_update":EXPAND,
            "status":w0.get("status"),
            "treatment_capability":w0.get("treatment_capability"),
            "remove_capability":w0.get("remove_same_checkpoint_capability"),
            "wrong_capability":w0.get("wrong_capability"),
            "reset_capability":w0.get("reset_capability"),
        },
        "world1":{
            "residual_class":"VALID_BUT_HIGH_COST",
            "selected_update":REFINE,
            "status":w1.get("status"),
            "proposal_count_trajectory":w1.get("proposal_count_trajectory"),
            "external_pair_count_trajectory":w1.get("external_pair_count_trajectory"),
            "capability_trajectory":w1.get("validated_capability_trajectory"),
            "wrong_swap_capability":(w1.get("wrong_swap_g3") or {}).get("capability"),
        },
        "world2_preoutcome_prediction_seal_sha256":transfer_prediction_seal,
        "world2_predictions":predictions,
        "world2_selector_program":{
            "pass":selector_pass,
            "status":w2a.get("status"),
            "treatment_capability":w2a.get("treatment_capability"),
            "remove_capability":w2a.get("remove_same_checkpoint_capability"),
            "wrong_capability":w2a.get("wrong_program_capability"),
            "old_more_compute_selected_count":w2a.get("old_more_compute_selected_candidate_count"),
        },
        "world2_morphology_certificate":{
            "pass":morphology_pass,
            "status":w2b.get("status"),
            "speedups":w2b.get("certificate_vs_exhaustive_high_level_speedups"),
            "remove_capability":w2b.get("remove_certificate_or_depth_jump_capability_under_same_depth"),
            "wrong_capabilities":w2b.get("semantic_wrong_failure_locus_certificate_capabilities"),
            "strict_recursive_meta_productivity_acceleration":w2b.get("strict_recursive_meta_productivity_acceleration"),
        },
        "world3_cold_prediction_seal_sha256":cold_prediction_seal,
        "world3_cold_prediction":cold_prediction,
        "world3_dataflow":{
            "pass":dataflow_pass,
            "status":w3.get("status"),
            "treatment_capability":w3.get("treatment_capability"),
            "remove_capability":w3.get("remove_same_checkpoint_capability"),
            "wrong_capability":w3.get("wrong_capability"),
            "old_more_compute_selected_count":w3.get("old_more_compute_selected_count"),
        },
        "mechanism_distinct_transfer_count":3,
        "post_outcome_policy_edits":0,
        "policy_hash_chain_valid":g1.parent_hash==g0.genome_hash and g2.parent_hash==g1.genome_hash,
        "cold_process_reconstruction_exact":True,
        "matched_search_efficiency_gain_reproduced":True,
        "claim_boundary":{
            "residual_class_schema_human_authored":True,
            "action_vocabulary_human_authored":True,
            "action_implementation_binding_human_authored":True,
            "same_repository_research_lineage":True,
            "independent_organizational_custody":False,
            "physical_world":False,
            "foundation_weight_change":False,
            "autonomous_open_ended_meta_policy_invention":False,
            "global_recursive_acceleration":False,
            "AGI":False,"ASI":False,"superintelligence":False,
        }
    }
    print(json.dumps(result,sort_keys=True))
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--cold-predict",nargs=2,metavar=("POLICY_JSON","RESIDUAL_CLASS"))
    ap.add_argument("--seed",type=int,default=20260929001)
    args=ap.parse_args()
    if args.cold_predict:
        raise SystemExit(cold_predict(args.cold_predict[0],args.cold_predict[1]))
    raise SystemExit(main(args.seed))
