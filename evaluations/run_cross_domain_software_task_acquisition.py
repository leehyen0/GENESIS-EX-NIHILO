from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import operator
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.software_task_acquisition import (
    SoftwarePatchCandidate,
    SoftwareTaskAcquisitionOrgan,
    SubprocessPythonRepairExecutor,
)
from arte_cognition.world_coupling import HMACWorldReceiptSigner, HMACWorldReceiptVerifier


REPAIR_FAMILIES = {
    "COMPARE::GtE->Gt": (">=", operator.gt),
    "COMPARE::Gt->GtE": (">", operator.ge),
    "COMPARE::Lt->LtE": ("<", operator.le),
    "COMPARE::LtE->Lt": ("<=", operator.lt),
}


@dataclass(frozen=True)
class HiddenSoftwareTask:
    task_id: str
    domain: str
    source: str
    function_name: str
    cases: tuple[dict, ...]

    @property
    def source_hash(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


def _function_name(rng, prefix):
    return f"{prefix}_{rng.randrange(10000000, 99999999)}"


def _expected_filter(values, pivot, predicate):
    return [value for value in values if predicate(value, pivot)]


def make_numeric_task(rng, repair_operator, label):
    buggy_op, predicate = REPAIR_FAMILIES[repair_operator]
    fn = _function_name(rng, "numeric_gate")
    value_name = f"v_{rng.randrange(1000,9999)}"
    source = f'''def {fn}(values, pivot):
    if len(values) == -999:
        return []
    if len(values) != len(values):
        return []
    out = []
    for {value_name} in values:
        if {value_name} {buggy_op} pivot:
            out.append({value_name})
    return out
'''
    cases = []
    for values, pivot in [([-3,-1,0,2,5],0), ([1,2,3,4],2), ([7,7,8,9],7)]:
        cases.append({"args":[values,pivot], "expected":_expected_filter(values,pivot,predicate)})
    return HiddenSoftwareTask(f"{label}-{rng.randrange(10**7,10**8)}", "numeric-sequence", source, fn, tuple(cases))


def make_string_task(rng, repair_operator, label):
    buggy_op, predicate = REPAIR_FAMILIES[repair_operator]
    fn = _function_name(rng, "lexical_gate")
    source = f'''def {fn}(tokens, pivot):
    if len(tokens) == -999:
        return []
    if len(tokens) != len(tokens):
        return []
    return [token for token in tokens if token {buggy_op} pivot]
'''
    cases = []
    for tokens, pivot in [(["aa","bb","bc","zz"],"bb"), (["ant","bee","cat"],"bee"), (["k","k","m"],"k")]:
        cases.append({"args":[tokens,pivot], "expected":_expected_filter(tokens,pivot,predicate)})
    return HiddenSoftwareTask(f"{label}-{rng.randrange(10**7,10**8)}", "lexical-string", source, fn, tuple(cases))


def make_record_task(rng, repair_operator, label):
    buggy_op, predicate = REPAIR_FAMILIES[repair_operator]
    fn = _function_name(rng, "record_gate")
    score_key = f"score_{rng.randrange(100,999)}"
    id_key = f"id_{rng.randrange(100,999)}"
    source = f'''def {fn}(rows, pivot):
    if len(rows) == -999:
        return []
    if len(rows) != len(rows):
        return []
    selected = []
    for row in rows:
        if row[{score_key!r}] {buggy_op} pivot:
            selected.append(row[{id_key!r}])
    return selected
'''
    raw_cases = [
        ([{score_key:1,id_key:"a"},{score_key:2,id_key:"b"},{score_key:4,id_key:"c"}],2),
        ([{score_key:5,id_key:"q"},{score_key:5,id_key:"r"},{score_key:8,id_key:"s"}],5),
        ([{score_key:-1,id_key:"x"},{score_key:0,id_key:"y"},{score_key:1,id_key:"z"}],0),
    ]
    cases = []
    for rows,pivot in raw_cases:
        expected=[row[id_key] for row in rows if predicate(row[score_key],pivot)]
        cases.append({"args":[rows,pivot],"expected":expected})
    return HiddenSoftwareTask(f"{label}-{rng.randrange(10**7,10**8)}", "record-structure", source, fn, tuple(cases))


def execute_candidate(body, task, candidate, signers, verifier, epoch_base):
    effects=[]
    for issuer_index,(issuer,signer) in enumerate(signers.items()):
        executor=SubprocessPythonRepairExecutor(
            baseline_source=task.source,
            patched_source=candidate.patched_source,
            function_name=task.function_name,
            hidden_cases=task.cases,
            signer=signer,
            source_id=f"{task.task_id}-source-{candidate.site_index}-{issuer}",
            context_id=task.task_id,
            challenge_id=f"{task.task_id}-hidden-suite-{candidate.site_index}-{issuer}",
            epoch=epoch_base+issuer_index,
        )
        pair=body.execute_world_intervention(candidate.proposal,executor,verifier=verifier)
        if not pair.authority_verified:
            raise AssertionError("software hidden-test receipt lost external authority")
        effects.append(float(pair.effect))
    return tuple(effects)


def train_task(body, task, signers, verifier, epoch_base):
    organ=SoftwareTaskAcquisitionOrgan(body)
    candidates=organ.propose(task.task_id,task.source)
    if len(candidates)<3:
        raise AssertionError("software task did not expose a nontrivial source-derived repair search")
    strong=[]
    for index,candidate in enumerate(candidates):
        effects=execute_candidate(body,task,candidate,signers,verifier,epoch_base+index*10)
        if min(effects)>=0.9:
            strong.append(candidate)
    if len(strong)!=1:
        raise AssertionError(f"software task must have exactly one externally successful patch: {strong}")
    return candidates,strong[0]


def execute_selection(body, task, selection, signers, verifier, epoch_base):
    if len(selection.candidates)!=1:
        raise AssertionError("matched fresh repair arm requires exactly one patch candidate")
    return execute_candidate(body,task,selection.candidates[0],signers,verifier,epoch_base)


def exhaustive_capability(body, task, signers, verifier, epoch_base):
    organ=SoftwareTaskAcquisitionOrgan(body)
    candidates=organ.propose(task.task_id,task.source)
    success=False
    for index,candidate in enumerate(candidates):
        effects=execute_candidate(body,task,candidate,signers,verifier,epoch_base+index*10)
        success=success or min(effects)>=0.9
    return float(success),len(candidates),2*len(candidates)


def main(seed_path):
    seed=int(Path(seed_path).read_text().strip())
    rng=random.Random(seed)
    hidden_operator=rng.choice(tuple(sorted(REPAIR_FAMILIES)))
    wrong_operator=rng.choice(tuple(op for op in sorted(REPAIR_FAMILIES) if op!=hidden_operator))

    issuer_a=f"software-lab-{rng.randrange(10**7,10**8)}"
    issuer_b=f"software-lab-{rng.randrange(10**7,10**8)}"
    key_a=hashlib.sha256(f"{seed}:software:a".encode()).digest()
    key_b=hashlib.sha256(f"{seed}:software:b".encode()).digest()
    signers={issuer_a:HMACWorldReceiptSigner(issuer_a,key_a),issuer_b:HMACWorldReceiptSigner(issuer_b,key_b)}
    verifier=HMACWorldReceiptVerifier({issuer_a:key_a,issuer_b:key_b},independence_classes={issuer_a:"SOFTWARE_LAB_A",issuer_b:"SOFTWARE_LAB_B"})

    parent=PersistentCognitiveRuntime()
    train_numeric=make_numeric_task(rng,hidden_operator,"train-numeric")
    train_string=make_string_task(rng,hidden_operator,"train-string")
    numeric_candidates,numeric_patch=train_task(parent,train_numeric,signers,verifier,10000)
    string_candidates,string_patch=train_task(parent,train_string,signers,verifier,20000)
    policy=SoftwareTaskAcquisitionOrgan(parent).policy()
    if policy.operator_id!=hidden_operator or len(policy.supporting_contexts)!=2:
        raise AssertionError(f"source-disjoint software repair policy was not learned: {policy}")
    if numeric_patch.operator_id!=hidden_operator or string_patch.operator_id!=hidden_operator:
        raise AssertionError("external tests selected an unexpected patch operator")

    checkpoint=checkpoint_dict(parent)
    verifierless=restore_runtime(checkpoint)
    if SoftwareTaskAcquisitionOrgan(verifierless).policy().operator_id is not None:
        raise AssertionError("software repair authority restored without external verifier")

    heldout=make_record_task(rng,hidden_operator,"heldout-record")
    treatment=restore_runtime(checkpoint,world_verifier=verifier)
    remove=restore_runtime(checkpoint,world_verifier=verifier)
    treatment_organ=SoftwareTaskAcquisitionOrgan(treatment)
    remove_organ=SoftwareTaskAcquisitionOrgan(remove)
    treatment_candidates=treatment_organ.propose(heldout.task_id,heldout.source)
    remove_candidates=remove_organ.propose(heldout.task_id,heldout.source)
    if tuple(c.proposal.experiment_id for c in treatment_candidates)!=tuple(c.proposal.experiment_id for c in remove_candidates):
        raise AssertionError("same-checkpoint repair candidate generation diverged")
    treatment_selection=treatment_organ.select(treatment_candidates,max_candidates=1,apply_learned_policy=True)
    remove_selection=remove_organ.select(remove_candidates,max_candidates=1,apply_learned_policy=False)
    treatment_effects=execute_selection(treatment,heldout,treatment_selection,signers,verifier,40000)
    remove_effects=execute_selection(remove,heldout,remove_selection,signers,verifier,40000)
    treatment_cap=float(min(treatment_effects)>=0.9)
    remove_cap=float(min(remove_effects)>=0.9)

    reset=PersistentCognitiveRuntime()
    reset_organ=SoftwareTaskAcquisitionOrgan(reset)
    reset_candidates=reset_organ.propose(heldout.task_id,heldout.source)
    reset_selection=reset_organ.select(reset_candidates,max_candidates=1,apply_learned_policy=True)
    reset_effects=execute_selection(reset,heldout,reset_selection,signers,verifier,50000)
    reset_cap=float(min(reset_effects)>=0.9)

    full=PersistentCognitiveRuntime()
    full_cap,full_candidates,full_pairs=exhaustive_capability(full,heldout,signers,verifier,60000)

    wrong_parent=PersistentCognitiveRuntime()
    wrong_numeric=make_numeric_task(rng,wrong_operator,"wrong-train-numeric")
    wrong_string=make_string_task(rng,wrong_operator,"wrong-train-string")
    train_task(wrong_parent,wrong_numeric,signers,verifier,70000)
    train_task(wrong_parent,wrong_string,signers,verifier,80000)
    wrong_policy=SoftwareTaskAcquisitionOrgan(wrong_parent).policy()
    if wrong_policy.operator_id!=wrong_operator:
        raise AssertionError("wrong-control BODY failed to genuinely learn its own repair operator")
    wrong=restore_runtime(checkpoint_dict(wrong_parent),world_verifier=verifier)
    wrong_organ=SoftwareTaskAcquisitionOrgan(wrong)
    wrong_candidates=wrong_organ.propose(heldout.task_id,heldout.source)
    wrong_selection=wrong_organ.select(wrong_candidates,max_candidates=1,apply_learned_policy=True)
    wrong_effects=execute_selection(wrong,heldout,wrong_selection,signers,verifier,90000)
    wrong_cap=float(min(wrong_effects)>=0.9)

    if treatment_cap!=1.0 or remove_cap!=0.0 or reset_cap!=0.0 or wrong_cap!=0.0 or full_cap!=1.0:
        raise AssertionError("software task acquisition causal controls failed")
    if treatment_selection.candidates[0].operator_id!=hidden_operator:
        raise AssertionError("learned software policy did not prioritize the externally reproduced repair operator")
    if remove_selection.candidates[0].operator_id==hidden_operator:
        raise AssertionError("REMOVE accidentally chose the hidden repair under the one-candidate budget")
    source_hashes={train_numeric.source_hash,train_string.source_hash,heldout.source_hash}
    if len(source_hashes)!=3:
        raise AssertionError("training and heldout software sources were not source-disjoint")

    result={
        "status":"PASS_BOUNDED_SOURCE_DISJOINT_EXECUTABLE_SOFTWARE_TASK_ACQUISITION_AND_DESCENDANT_CAUSAL_TRANSFER",
        "hidden_repair_operator":hidden_operator,
        "wrong_learned_operator":wrong_operator,
        "training_domains":[train_numeric.domain,train_string.domain],
        "heldout_domain":heldout.domain,
        "source_hashes_disjoint":True,
        "training_candidate_counts":[len(numeric_candidates),len(string_candidates)],
        "heldout_full_candidate_count":full_candidates,
        "treatment_candidate_count":1,
        "remove_candidate_count":1,
        "wrong_candidate_count":1,
        "full_external_pair_count":full_pairs,
        "treatment_external_pair_count":2,
        "external_pair_reduction_vs_full":1.0-(2.0/full_pairs),
        "treatment_capability":treatment_cap,
        "remove_same_checkpoint_capability":remove_cap,
        "reset_capability":reset_cap,
        "wrong_learned_policy_capability":wrong_cap,
        "full_exhaustive_capability":full_cap,
        "verifierless_policy_authority":False,
        "external_execution":"separate_python_subprocess",
        "hidden_tests_exposed_to_body_before_execution":False,
        "patch_candidates_generated_from_source_ast_without_hidden_outcomes":True,
        "repair_mutation_alphabet_human_authored":True,
        "post_hidden_human_structural_repairs":0,
        "foundation_weight_change":False,
        "physical_world":False,
        "independent_organizational_custody":False,
        "global_recursive_acceleration":False,
        "AGI":False,
        "ASI":False,
    }
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    if len(sys.argv)!=2:
        raise SystemExit("usage: run_cross_domain_software_task_acquisition.py <seed_path>")
    main(sys.argv[1])
