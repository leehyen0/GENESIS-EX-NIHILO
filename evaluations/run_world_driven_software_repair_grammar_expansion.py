from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import operator
import random
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.software_task_acquisition import SoftwareTaskAcquisitionOrgan, SubprocessPythonRepairExecutor
from arte_cognition.software_repair_grammar_expansion import SoftwareRepairGrammarExpansionOrgan
from arte_cognition.world_coupling import HMACWorldReceiptSigner,HMACWorldReceiptVerifier


ARITHMETIC_FAMILIES={
    "BINOP::Add->Sub":("+",operator.sub),
    "BINOP::Sub->Add":("-",operator.add),
    "BINOP::Mult->Add":("*",operator.add),
}


@dataclass(frozen=True)
class Task:
    task_id:str
    domain:str
    source:str
    function_name:str
    cases:tuple[dict,...]
    @property
    def source_hash(self):
        return hashlib.sha256(self.source.encode()).hexdigest()


def fn_name(rng,prefix): return f"{prefix}_{rng.randrange(10**7,10**8)}"


def numeric_task(rng,repair,label):
    buggy,correct=ARITHMETIC_FAMILIES[repair]; fn=fn_name(rng,"scalar_transform")
    source=f'''def {fn}(value, delta):
    if delta == -999:
        return 0
    if delta != delta:
        return 0
    return value {buggy} delta
'''
    pairs=[(7,2),(-3,5),(0,4)]
    cases=tuple({"args":[a,b],"expected":correct(a,b)} for a,b in pairs)
    return Task(f"{label}-{rng.randrange(10**7,10**8)}","numeric-scalar",source,fn,cases)


def list_task(rng,repair,label):
    buggy,correct=ARITHMETIC_FAMILIES[repair]; fn=fn_name(rng,"sequence_transform")
    source=f'''def {fn}(values, delta):
    if len(values) == -999:
        return []
    if len(values) != len(values):
        return []
    return [value {buggy} delta for value in values]
'''
    raw=[([1,2,3],2),([-2,0,5],3),([7,7],1)]
    cases=tuple({"args":[values,d],"expected":[correct(v,d) for v in values]} for values,d in raw)
    return Task(f"{label}-{rng.randrange(10**7,10**8)}","numeric-sequence",source,fn,cases)


def record_task(rng,repair,label):
    buggy,correct=ARITHMETIC_FAMILIES[repair]; fn=fn_name(rng,"record_transform")
    key=f"score_{rng.randrange(100,999)}"
    source=f'''def {fn}(rows, delta):
    if len(rows) == -999:
        return []
    if len(rows) != len(rows):
        return []
    output = []
    for row in rows:
        output.append(row[{key!r}] {buggy} delta)
    return output
'''
    raw=[([{key:1},{key:4},{key:9}],2),([{key:-3},{key:0}],5),([{key:8}],3)]
    cases=tuple({"args":[rows,d],"expected":[correct(row[key],d) for row in rows]} for rows,d in raw)
    return Task(f"{label}-{rng.randrange(10**7,10**8)}","record-structure",source,fn,cases)


def execute_candidate(body,task,candidate,signers,verifier,epoch):
    effects=[]
    for i,(issuer,signer) in enumerate(signers.items()):
        executor=SubprocessPythonRepairExecutor(
            task.source,candidate.patched_source,task.function_name,task.cases,signer,
            f"{task.task_id}-src-{candidate.site_index}-{issuer}",task.task_id,
            f"{task.task_id}-suite-{candidate.site_index}-{issuer}",epoch+i,
        )
        pair=body.execute_world_intervention(candidate.proposal,executor,verifier=verifier)
        if not pair.authority_verified: raise AssertionError("repair grammar world receipt lost authority")
        effects.append(float(pair.effect))
    return tuple(effects)


def execute_all_old(body,task,signers,verifier,epoch):
    organ=SoftwareTaskAcquisitionOrgan(body); candidates=organ.propose(task.task_id,task.source)
    if len(candidates)<2: raise AssertionError("old software repair alphabet did not expose complete decoy search")
    for idx,c in enumerate(candidates):
        effects=execute_candidate(body,task,c,signers,verifier,epoch+idx*10)
        if max(effects)>=0.9: raise AssertionError("old repair alphabet unexpectedly solved arithmetic hidden task")
    return candidates


def execute_generated_training(body,task,old_map,signers,verifier,epoch):
    organ=SoftwareRepairGrammarExpansionOrgan(body)
    candidates=organ.propose(task.task_id,task.source,old_map)
    if not candidates: raise AssertionError("expanded arithmetic repair grammar failed to open")
    strong=[]
    for idx,c in enumerate(candidates):
        effects=execute_candidate(body,task,c,signers,verifier,epoch+idx*10)
        if min(effects)>=0.9: strong.append(c)
    if len(strong)!=1: raise AssertionError(f"expanded repair task must have one strong patch: {strong}")
    return candidates,strong[0]


def main(seed_path):
    seed=int(Path(seed_path).read_text().strip()); rng=random.Random(seed)
    hidden=rng.choice(tuple(sorted(ARITHMETIC_FAMILIES)))
    wrong_hidden=rng.choice(tuple(x for x in sorted(ARITHMETIC_FAMILIES) if x!=hidden))
    ia=f"repair-lab-{rng.randrange(10**7,10**8)}"; ib=f"repair-lab-{rng.randrange(10**7,10**8)}"
    ka=hashlib.sha256(f"{seed}:repair:a".encode()).digest(); kb=hashlib.sha256(f"{seed}:repair:b".encode()).digest()
    signers={ia:HMACWorldReceiptSigner(ia,ka),ib:HMACWorldReceiptSigner(ib,kb)}
    verifier=HMACWorldReceiptVerifier({ia:ka,ib:kb},independence_classes={ia:"REPAIR_LAB_A",ib:"REPAIR_LAB_B"})

    parent=PersistentCognitiveRuntime(); t1=numeric_task(rng,hidden,"grammar-numeric"); t2=list_task(rng,hidden,"grammar-list")
    old1=execute_all_old(parent,t1,signers,verifier,10000); old2=execute_all_old(parent,t2,signers,verifier,20000)
    old_map={t1.task_id:old1,t2.task_id:old2}
    expansion=SoftwareRepairGrammarExpansionOrgan(parent); assessment=expansion.assess_old_alphabet(old_map)
    if assessment.status!="SOFTWARE_REPAIR_ALPHABET_FALSIFIED_OPEN_NEXT" or assessment.missing_experiment_ids:
        raise AssertionError(f"old executable repair grammar was not completely falsified: {assessment}")
    gen1,strong1=execute_generated_training(parent,t1,old_map,signers,verifier,30000)
    gen2,strong2=execute_generated_training(parent,t2,old_map,signers,verifier,40000)
    policy=expansion.policy()
    if policy.operator_id!=hidden or len(policy.supporting_contexts)!=2:
        raise AssertionError(f"expanded arithmetic repair operator not learned: {policy}")
    if strong1.operator_id!=hidden or strong2.operator_id!=hidden: raise AssertionError("wrong expanded repair selected")

    checkpoint=checkpoint_dict(parent); verifierless=restore_runtime(checkpoint)
    if SoftwareRepairGrammarExpansionOrgan(verifierless).policy().operator_id is not None:
        raise AssertionError("expanded software repair authority restored without verifier")

    heldout=record_task(rng,hidden,"grammar-heldout"); treatment=restore_runtime(checkpoint,world_verifier=verifier); remove=restore_runtime(checkpoint,world_verifier=verifier)
    # The old candidate manifest is public/source-derived developmental provenance, not hidden outcome data.
    # Both arms receive the identical manifest; authority still comes only from reverified receipts.
    treatment_old=SoftwareTaskAcquisitionOrgan(treatment).propose(heldout.task_id,heldout.source)
    remove_old=SoftwareTaskAcquisitionOrgan(remove).propose(heldout.task_id,heldout.source)
    treatment_new=SoftwareRepairGrammarExpansionOrgan(treatment).propose(heldout.task_id,heldout.source,old_map)
    remove_new=SoftwareRepairGrammarExpansionOrgan(remove).propose(heldout.task_id,heldout.source,old_map)
    treatment_all=tuple(treatment_old)+tuple(treatment_new); remove_all=tuple(remove_old)+tuple(remove_new)
    tpolicy=SoftwareRepairGrammarExpansionOrgan(treatment).policy()
    tsel=SoftwareRepairGrammarExpansionOrgan(treatment).select(treatment_all,max_candidates=1,apply_learned_policy=True)
    rsel=SoftwareRepairGrammarExpansionOrgan(remove).select(remove_all,max_candidates=1,apply_learned_policy=False)
    te=execute_candidate(treatment,heldout,tsel.candidates[0],signers,verifier,60000); re=execute_candidate(remove,heldout,rsel.candidates[0],signers,verifier,60000)
    tcap=float(min(te)>=0.9); rcap=float(min(re)>=0.9)

    full=restore_runtime(checkpoint,world_verifier=verifier); full_old=SoftwareTaskAcquisitionOrgan(full).propose(heldout.task_id,heldout.source); full_new=SoftwareRepairGrammarExpansionOrgan(full).propose(heldout.task_id,heldout.source,old_map); full_all=tuple(full_old)+tuple(full_new)
    fullcap=0.0
    for idx,c in enumerate(full_all):
        effects=execute_candidate(full,heldout,c,signers,verifier,70000+idx*10); fullcap=max(fullcap,float(min(effects)>=0.9))

    wrong=PersistentCognitiveRuntime(); w1=numeric_task(rng,wrong_hidden,"wrong-grammar-numeric"); w2=list_task(rng,wrong_hidden,"wrong-grammar-list")
    wold1=execute_all_old(wrong,w1,signers,verifier,90000); wold2=execute_all_old(wrong,w2,signers,verifier,100000); wmap={w1.task_id:wold1,w2.task_id:wold2}
    execute_generated_training(wrong,w1,wmap,signers,verifier,110000); execute_generated_training(wrong,w2,wmap,signers,verifier,120000)
    wpolicy=SoftwareRepairGrammarExpansionOrgan(wrong).policy()
    if wpolicy.operator_id!=wrong_hidden: raise AssertionError("wrong-control BODY did not learn its own expanded repair")
    wrong=restore_runtime(checkpoint_dict(wrong),world_verifier=verifier); wold=SoftwareTaskAcquisitionOrgan(wrong).propose(heldout.task_id,heldout.source); wnew=SoftwareRepairGrammarExpansionOrgan(wrong).propose(heldout.task_id,heldout.source,wmap); wall=tuple(wold)+tuple(wnew)
    wsel=SoftwareRepairGrammarExpansionOrgan(wrong).select(wall,max_candidates=1,apply_learned_policy=True); we=execute_candidate(wrong,heldout,wsel.candidates[0],signers,verifier,130000); wcap=float(min(we)>=0.9)

    reset=PersistentCognitiveRuntime(); rold=SoftwareTaskAcquisitionOrgan(reset).propose(heldout.task_id,heldout.source)
    # RESET has not earned the grammar expansion gate, so it cannot generate arithmetic patches.
    rassessment=SoftwareRepairGrammarExpansionOrgan(reset).assess_old_alphabet({heldout.task_id:rold})
    rnew=SoftwareRepairGrammarExpansionOrgan(reset).propose(heldout.task_id,heldout.source,{heldout.task_id:rold})
    reset_selection=(tuple(rold)+tuple(rnew))[:1]; reset_effect=execute_candidate(reset,heldout,reset_selection[0],signers,verifier,140000); resetcap=float(min(reset_effect)>=0.9)

    if not (tcap==1.0 and rcap==0.0 and wcap==0.0 and resetcap==0.0 and fullcap==1.0):
        raise AssertionError("software repair grammar expansion causal controls failed")
    if tsel.candidates[0].operator_id!=hidden or rsel.candidates[0].operator_id==hidden: raise AssertionError("expanded policy selection control failed")
    if len({t1.source_hash,t2.source_hash,heldout.source_hash})!=3: raise AssertionError("expanded repair tasks were not source-disjoint")

    result={
        "status":"PASS_BOUNDED_WORLD_FALSIFICATION_DRIVEN_EXECUTABLE_SOFTWARE_REPAIR_GRAMMAR_EXPANSION_AND_DESCENDANT_TRANSFER",
        "old_repair_alphabet":"COMPARE_AND_BOOL_ONE_NODE_MUTATIONS",
        "old_complete_falsified_contexts":len(assessment.falsified_contexts),
        "old_missing_candidate_count":len(assessment.missing_experiment_ids),
        "old_evaluated_candidate_count":assessment.evaluated_candidate_count,
        "expanded_repair_operator":hidden,
        "expanded_training_domains":[t1.domain,t2.domain],
        "heldout_domain":heldout.domain,
        "source_hashes_disjoint":True,
        "generated_candidate_counts":[len(gen1),len(gen2)],
        "heldout_full_candidate_count":len(full_all),
        "treatment_candidate_count":1,
        "remove_candidate_count":1,
        "wrong_candidate_count":1,
        "treatment_capability":tcap,
        "remove_same_checkpoint_capability":rcap,
        "wrong_genuinely_learned_operator":wrong_hidden,
        "wrong_capability":wcap,
        "reset_capability":resetcap,
        "full_exhaustive_capability":fullcap,
        "treatment_external_pairs":2,
        "full_external_pairs":2*len(full_all),
        "external_pair_reduction_vs_full":1.0-(1.0/len(full_all)),
        "verifierless_expanded_policy_authority":False,
        "reset_expansion_status":rassessment.status,
        "external_execution":"separate_python_subprocess",
        "hidden_tests_exposed_to_body_before_execution":False,
        "arithmetic_repair_grammar_generation_uses_hidden_outcomes":False,
        "arithmetic_mutation_schema_human_authored":True,
        "unrestricted_software_operator_invention":False,
        "post_hidden_human_structural_repairs":0,
        "foundation_weight_change":False,"physical_world":False,"independent_organizational_custody":False,
        "global_recursive_acceleration":False,"AGI":False,"ASI":False,
    }
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    if len(sys.argv)!=2: raise SystemExit("usage: run_world_driven_software_repair_grammar_expansion.py <seed_path>")
    main(sys.argv[1])
