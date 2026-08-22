from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
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
from arte_cognition.repository_task_acquisition import (
    RepositoryTaskAcquisitionOrgan,
    derive_repository_file_roles,
    repository_hash,
)
from arte_cognition.repository_patch_cardinality import (
    RepositoryPatchCardinalityOrgan,
    RepositoryPatchSetGenerator,
    SubprocessRepositoryPatchSetExecutor,
    canonical_patch_set_signature,
)
from arte_cognition.world_coupling import HMACWorldReceiptSigner, HMACWorldReceiptVerifier


REPAIR_FAMILIES = {
    "COMPARE::GtE->Gt": (">=", operator.gt, "GREATER"),
    "COMPARE::Gt->GtE": (">", operator.ge, "GREATER"),
    "COMPARE::Lt->LtE": ("<", operator.le, "LESS"),
    "COMPARE::LtE->Lt": ("<=", operator.lt, "LESS"),
}

TARGET_ROLES = ("IMPORTED_LEAF", "INTERMEDIATE", "ROOT_IMPORTER")
WRONG_PAIR_ROLES = ("IMPORTED_LEAF", "ROOT_IMPORTER")


@dataclass(frozen=True)
class HiddenCardinalityTask:
    task_id: str
    domain: str
    files: dict[str, str]
    entry_module: str
    function_name: str
    cases: tuple[dict, ...]
    required_roles: tuple[str, ...]

    @property
    def repo_hash(self) -> str:
        return repository_hash(self.files)


def _name(rng, prefix):
    return f"{prefix}_{rng.randrange(10000000, 99999999)}"


def _module_chain(rng):
    return (
        _name(rng, "a_iso"),
        _name(rng, "b_root"),
        _name(rng, "c_mid"),
        _name(rng, "d_leaf"),
    )


def _domain_payload(domain, direction, rng):
    if domain in {"numeric-triples", "record-structure"}:
        pivot = 0
        strict = 1 if direction == "GREATER" else -1
        opposite = -1 if direction == "GREATER" else 1
    elif domain == "lexical-triples":
        pivot = "m"
        strict = "z" if direction == "GREATER" else "a"
        opposite = "a" if direction == "GREATER" else "z"
    else:
        raise ValueError(domain)

    raw = [
        (strict, strict, strict, "strict"),
        (pivot, strict, strict, "eq-0"),
        (strict, pivot, strict, "eq-1"),
        (strict, strict, pivot, "eq-2"),
        (pivot, pivot, pivot, "eq-all"),
        (opposite, strict, strict, "fail-0"),
        (strict, opposite, strict, "fail-1"),
        (strict, strict, opposite, "fail-2"),
    ]
    if domain == "record-structure":
        keys = [_name(rng, "axis0"), _name(rng, "axis1"), _name(rng, "axis2")]
        id_key = _name(rng, "id")
        items = [
            {keys[0]:a, keys[1]:b, keys[2]:c, id_key:label}
            for a,b,c,label in raw
        ]
        exprs = tuple(f"item[{key!r}]" for key in keys)
        extractors = tuple((lambda key: (lambda item: item[key]))(key) for key in keys)
    else:
        items = [(a,b,c) for a,b,c,_ in raw]
        exprs = ("item[0]", "item[1]", "item[2]")
        extractors = (
            lambda item: item[0],
            lambda item: item[1],
            lambda item: item[2],
        )
    return items, (pivot,pivot,pivot), exprs, extractors


def _make_task(rng, repair_operator, domain, label, required_roles):
    buggy_op, predicate, direction = REPAIR_FAMILIES[repair_operator]
    isolated_module, root_module, mid_module, leaf_module = _module_chain(rng)
    isolated_fn = _name(rng, "isolated")
    root_fn = _name(rng, "public")
    mid_fn = _name(rng, "middle")
    leaf_fn = _name(rng, "leaf")
    items, pivots, exprs, extractors = _domain_payload(domain, direction, rng)
    required = set(required_roles)

    if "IMPORTED_LEAF" in required:
        leaf_source = f'''def {leaf_fn}(items, pivot0):
    return [item for item in items if {exprs[0]} {buggy_op} pivot0]
'''
    else:
        leaf_source = f'''def {leaf_fn}(items, pivot0):
    _unused = pivot0 {buggy_op} pivot0
    return list(items)
'''

    if "INTERMEDIATE" in required:
        mid_source = f'''from {leaf_module} import {leaf_fn}

def {mid_fn}(items, pivot0, pivot1):
    items = {leaf_fn}(items, pivot0)
    return [item for item in items if {exprs[1]} {buggy_op} pivot1]
'''
    else:
        mid_source = f'''from {leaf_module} import {leaf_fn}

def {mid_fn}(items, pivot0, pivot1):
    _unused = pivot1 {buggy_op} pivot1
    return {leaf_fn}(items, pivot0)
'''

    if "ROOT_IMPORTER" in required:
        root_source = f'''from {mid_module} import {mid_fn}

def {root_fn}(items, pivot0, pivot1, pivot2):
    items = {mid_fn}(items, pivot0, pivot1)
    return [item for item in items if {exprs[2]} {buggy_op} pivot2]
'''
    else:
        root_source = f'''from {mid_module} import {mid_fn}

def {root_fn}(items, pivot0, pivot1, pivot2):
    _unused = pivot2 {buggy_op} pivot2
    return {mid_fn}(items, pivot0, pivot1)
'''

    isolated_source = f'''def {isolated_fn}(value, pivot):
    return value if value {buggy_op} pivot else pivot
'''
    files = {
        f"{isolated_module}.py": isolated_source,
        f"{root_module}.py": root_source,
        f"{mid_module}.py": mid_source,
        f"{leaf_module}.py": leaf_source,
    }
    roles = derive_repository_file_roles(files)
    required_counts = {"ISOLATED":1, "ROOT_IMPORTER":1, "INTERMEDIATE":1, "IMPORTED_LEAF":1}
    actual_counts = {role:list(roles.values()).count(role) for role in required_counts}
    if actual_counts != required_counts:
        raise AssertionError(f"cardinality task lost unique import roles: {roles}")

    role_to_dimension = {"IMPORTED_LEAF":0, "INTERMEDIATE":1, "ROOT_IMPORTER":2}
    expected = []
    for item in items:
        valid = True
        for role in required:
            dimension = role_to_dimension[role]
            if not predicate(extractors[dimension](item), pivots[dimension]):
                valid = False
                break
        if valid:
            expected.append(item)
    cases = ({"args":[items,pivots[0],pivots[1],pivots[2]], "expected":expected},)
    return HiddenCardinalityTask(
        task_id=f"{label}-{rng.randrange(10**7,10**8)}",
        domain=domain,
        files=files,
        entry_module=root_module,
        function_name=root_fn,
        cases=tuple(cases),
        required_roles=tuple(sorted(required_roles)),
    )


def execute_set(body, task, candidate, signers, verifier, epoch_base):
    effects=[]
    for issuer_index,(issuer,signer) in enumerate(signers.items()):
        executor=SubprocessRepositoryPatchSetExecutor(
            baseline_files=task.files,
            candidate=candidate,
            entry_module=task.entry_module,
            function_name=task.function_name,
            hidden_cases=task.cases,
            signer=signer,
            source_id=f"{task.task_id}-k{candidate.cardinality}-{hashlib.sha256(candidate.signature.encode()).hexdigest()[:8]}-{issuer}",
            context_id=task.task_id,
            challenge_id=f"{task.task_id}-k{candidate.cardinality}-{candidate.proposal.experiment_id[-12:]}-{issuer}",
            epoch=epoch_base+issuer_index,
        )
        pair=body.execute_world_intervention(candidate.proposal,executor,verifier=verifier)
        if not pair.authority_verified:
            raise AssertionError("patch-cardinality evidence lost authority")
        effects.append(float(pair.effect))
    return tuple(effects)


def train_to_cardinality(body, tasks, target_cardinality, signers, verifier, epoch_base):
    singles_by_task={}
    for task in tasks:
        singles=RepositoryTaskAcquisitionOrgan(body).propose(task.task_id,task.files)
        if len(singles)!=4:
            raise AssertionError(f"expected four source-derived single sites, got {len(singles)}")
        singles_by_task[task.task_id]=singles

    organ=RepositoryPatchCardinalityOrgan(body)
    assessments=[]
    counts={}
    strong_final=[]
    candidate_maps={}
    for cardinality in range(1,target_cardinality+1):
        by_context={}
        strong_this=[]
        for task_index,task in enumerate(tasks):
            candidates=organ.propose(
                task.task_id,task.files,singles_by_task[task.task_id],
                cardinality=cardinality,
                prerequisite_assessments=tuple(assessments),
            )
            expected_count={1:4,2:6,3:4}.get(cardinality)
            if expected_count is not None and len(candidates)!=expected_count:
                raise AssertionError(
                    f"unexpected k={cardinality} candidate count {len(candidates)} != {expected_count}"
                )
            by_context[task.task_id]=candidates
            for candidate_index,candidate in enumerate(candidates):
                effects=execute_set(
                    body,task,candidate,signers,verifier,
                    epoch_base+cardinality*10000+task_index*2000+candidate_index*10,
                )
                if min(effects)>=0.9:
                    strong_this.append((task.task_id,candidate))
        counts[cardinality]=tuple(len(by_context[task.task_id]) for task in tasks)
        candidate_maps[cardinality]=by_context
        if cardinality < target_cardinality:
            if strong_this:
                raise AssertionError(f"lower cardinality {cardinality} unexpectedly solved task: {strong_this}")
            assessment=organ.assess(by_context)
            if assessment.status!="PATCH_SET_SPACE_FALSIFIED_OPEN_NEXT_CARDINALITY":
                raise AssertionError(f"k={cardinality} complete failure did not open next cardinality: {assessment}")
            if assessment.missing_experiment_ids:
                raise AssertionError("cardinality expansion admitted missing evidence")
            assessments.append(assessment)
        else:
            by_task={task.task_id:[] for task in tasks}
            for context_id,candidate in strong_this:
                by_task[context_id].append(candidate)
            for task in tasks:
                if len(by_task[task.task_id])!=1:
                    raise AssertionError(
                        f"target cardinality must have exactly one strong set in {task.task_id}: {by_task[task.task_id]}"
                    )
                roles=tuple(sorted(member.file_role for member in by_task[task.task_id][0].members))
                if roles!=task.required_roles:
                    raise AssertionError(f"strong patch set localized wrong roles: {roles} != {task.required_roles}")
                strong_final.append(by_task[task.task_id][0])
    return singles_by_task,tuple(assessments),tuple(strong_final),counts,candidate_maps


def main(seed_path):
    seed=int(Path(seed_path).read_text().strip())
    rng=random.Random(seed)
    hidden_operator=rng.choice(tuple(sorted(REPAIR_FAMILIES)))

    issuer_a=f"cardinality-lab-{rng.randrange(10**7,10**8)}"
    issuer_b=f"cardinality-lab-{rng.randrange(10**7,10**8)}"
    key_a=hashlib.sha256(f"{seed}:cardinality:a".encode()).digest()
    key_b=hashlib.sha256(f"{seed}:cardinality:b".encode()).digest()
    signers={issuer_a:HMACWorldReceiptSigner(issuer_a,key_a),issuer_b:HMACWorldReceiptSigner(issuer_b,key_b)}
    verifier=HMACWorldReceiptVerifier(
        {issuer_a:key_a,issuer_b:key_b},
        independence_classes={issuer_a:"CARDINALITY_LAB_A",issuer_b:"CARDINALITY_LAB_B"},
    )

    parent=PersistentCognitiveRuntime()
    train_numeric=_make_task(rng,hidden_operator,"numeric-triples","k3-train-numeric",TARGET_ROLES)
    train_string=_make_task(rng,hidden_operator,"lexical-triples","k3-train-string",TARGET_ROLES)
    _,assessments,strong_sets,counts,_=train_to_cardinality(
        parent,(train_numeric,train_string),3,signers,verifier,10000,
    )
    target_signature=canonical_patch_set_signature(strong_sets[0].members)
    if canonical_patch_set_signature(strong_sets[1].members)!=target_signature:
        raise AssertionError("triple training contexts did not reproduce same signature")
    policy=RepositoryPatchCardinalityOrgan(parent).policy()
    if policy.cardinality!=3 or policy.signature!=target_signature or len(policy.supporting_contexts)!=2:
        raise AssertionError(f"minimal triple policy was not learned: {policy}")
    if [assessment.cardinality for assessment in assessments] != [1,2]:
        raise AssertionError("developmental cardinality lineage skipped a lower repair size")

    checkpoint=checkpoint_dict(parent)
    verifierless=restore_runtime(checkpoint)
    verifierless_policy=RepositoryPatchCardinalityOrgan(verifierless).policy()
    if verifierless_policy.cardinality is not None:
        raise AssertionError("patch cardinality authority restored without external verifier")

    heldout=_make_task(rng,hidden_operator,"record-structure","k3-heldout-record",TARGET_ROLES)
    treatment=restore_runtime(checkpoint,world_verifier=verifier)
    remove=restore_runtime(checkpoint,world_verifier=verifier)
    treatment_singles=RepositoryTaskAcquisitionOrgan(treatment).propose(heldout.task_id,heldout.files)
    remove_singles=RepositoryTaskAcquisitionOrgan(remove).propose(heldout.task_id,heldout.files)
    treatment_organ=RepositoryPatchCardinalityOrgan(treatment)
    remove_organ=RepositoryPatchCardinalityOrgan(remove)
    treatment_sets=treatment_organ.propose(heldout.task_id,heldout.files,treatment_singles,3)
    remove_sets=remove_organ.propose(heldout.task_id,heldout.files,remove_singles,3)
    if tuple(item.proposal.experiment_id for item in treatment_sets)!=tuple(item.proposal.experiment_id for item in remove_sets):
        raise AssertionError("same-checkpoint triple candidate universe diverged")
    treatment_selection=treatment_organ.select(treatment_sets,max_candidates=1,apply_learned_policy=True)
    remove_selection=remove_organ.select(remove_sets,max_candidates=1,apply_learned_policy=False)
    treatment_effects=execute_set(treatment,heldout,treatment_selection.candidates[0],signers,verifier,50000)
    remove_effects=execute_set(remove,heldout,remove_selection.candidates[0],signers,verifier,50000)
    treatment_cap=float(min(treatment_effects)>=0.9)
    remove_cap=float(min(remove_effects)>=0.9)

    # Strong minimality control: every size-1 and size-2 subset of the selected triple fails heldout.
    selected=treatment_selection.candidates[0]
    selected_member_ids={member.proposal.experiment_id for member in selected.members}
    subset_control=restore_runtime(checkpoint,world_verifier=verifier)
    subset_singles=RepositoryTaskAcquisitionOrgan(subset_control).propose(heldout.task_id,heldout.files)
    generator=RepositoryPatchSetGenerator()
    subset_caps={1:[],2:[]}
    for cardinality in (1,2):
        all_sets=generator.generate(
            heldout.task_id,heldout.files,subset_singles,cardinality=cardinality,frontier_open=True
        )
        selected_subsets=[
            candidate for candidate in all_sets
            if {member.proposal.experiment_id for member in candidate.members}.issubset(selected_member_ids)
        ]
        expected_subset_count={1:3,2:3}[cardinality]
        if len(selected_subsets)!=expected_subset_count:
            raise AssertionError("selected triple subset enumeration drifted")
        for index,candidate in enumerate(selected_subsets):
            subset_control.memory.remember_experiment(candidate.proposal)
            effects=execute_set(
                subset_control,heldout,candidate,signers,verifier,
                60000+cardinality*1000+index*10,
            )
            cap=float(min(effects)>=0.9)
            subset_caps[cardinality].append(cap)
            if cap!=0.0:
                raise AssertionError("selected triple has a sufficient lower-cardinality subset")

    reset=PersistentCognitiveRuntime()
    reset_singles=RepositoryTaskAcquisitionOrgan(reset).propose(heldout.task_id,heldout.files)
    reset_sets=RepositoryPatchCardinalityOrgan(reset).propose(
        heldout.task_id,heldout.files,reset_singles,3
    )
    if reset_sets:
        raise AssertionError("RESET skipped directly to triple repair without lower-cardinality failure")
    reset_cap=0.0

    full=restore_runtime(checkpoint,world_verifier=verifier)
    full_singles=RepositoryTaskAcquisitionOrgan(full).propose(heldout.task_id,heldout.files)
    full_sets=RepositoryPatchCardinalityOrgan(full).propose(heldout.task_id,heldout.files,full_singles,3)
    full_success=False
    for index,candidate in enumerate(full_sets):
        effects=execute_set(full,heldout,candidate,signers,verifier,70000+index*10)
        full_success=full_success or min(effects)>=0.9
    full_cap=float(full_success)

    # Genuine lower-cardinality WRONG developmental state: same operator, but repeated
    # external tasks are solvable with exactly ROOT_IMPORTER + IMPORTED_LEAF.
    wrong_parent=PersistentCognitiveRuntime()
    wrong_numeric=_make_task(rng,hidden_operator,"numeric-triples","k2-wrong-numeric",WRONG_PAIR_ROLES)
    wrong_string=_make_task(rng,hidden_operator,"lexical-triples","k2-wrong-string",WRONG_PAIR_ROLES)
    _,wrong_assessments,wrong_strong,wrong_counts,_=train_to_cardinality(
        wrong_parent,(wrong_numeric,wrong_string),2,signers,verifier,80000,
    )
    wrong_policy=RepositoryPatchCardinalityOrgan(wrong_parent).policy()
    if wrong_policy.cardinality!=2 or len(wrong_assessments)!=1:
        raise AssertionError("wrong-control BODY failed to genuinely learn minimal cardinality 2")
    wrong=restore_runtime(checkpoint_dict(wrong_parent),world_verifier=verifier)
    wrong_singles=RepositoryTaskAcquisitionOrgan(wrong).propose(heldout.task_id,heldout.files)
    wrong_organ=RepositoryPatchCardinalityOrgan(wrong)
    wrong_sets=wrong_organ.propose(heldout.task_id,heldout.files,wrong_singles,2)
    wrong_selection=wrong_organ.select(wrong_sets,max_candidates=1,apply_learned_policy=True)
    wrong_effects=execute_set(wrong,heldout,wrong_selection.candidates[0],signers,verifier,95000)
    wrong_cap=float(min(wrong_effects)>=0.9)

    if treatment_cap!=1.0 or remove_cap!=0.0 or reset_cap!=0.0 or wrong_cap!=0.0 or full_cap!=1.0:
        raise AssertionError("world-driven patch cardinality causal controls failed")
    if selected.cardinality!=3 or selected.signature!=target_signature:
        raise AssertionError("Treatment did not select learned minimal triple repair")
    if wrong_selection.candidates[0].cardinality!=2:
        raise AssertionError("wrong-control resource cardinality drifted")

    repo_hashes={
        train_numeric.repo_hash,train_string.repo_hash,heldout.repo_hash,
        wrong_numeric.repo_hash,wrong_string.repo_hash,
    }
    if len(repo_hashes)!=5:
        raise AssertionError("cardinality tasks were not source-disjoint")

    result={
        "status":"PASS_BOUNDED_WORLD_FAILURE_DRIVEN_MINIMAL_REPOSITORY_REPAIR_CARDINALITY_1_TO_2_TO_3_AND_DESCENDANT_TRANSFER",
        "hidden_repair_operator":hidden_operator,
        "learned_minimal_cardinality":3,
        "learned_triple_signature":target_signature,
        "developmental_cardinality_sequence":[1,2,3],
        "lower_cardinality_falsified_contexts":{
            "1":len(assessments[0].falsified_contexts),
            "2":len(assessments[1].falsified_contexts),
        },
        "lower_cardinality_missing_candidate_counts":{
            "1":len(assessments[0].missing_experiment_ids),
            "2":len(assessments[1].missing_experiment_ids),
        },
        "lower_cardinality_evaluated_candidate_counts":{
            "1":assessments[0].evaluated_candidate_count,
            "2":assessments[1].evaluated_candidate_count,
        },
        "training_candidate_counts":{
            "1":list(counts[1]),"2":list(counts[2]),"3":list(counts[3])
        },
        "heldout_triple_candidate_count":len(full_sets),
        "selected_single_subset_capabilities":subset_caps[1],
        "selected_pair_subset_capabilities":subset_caps[2],
        "strict_minimality_proved_on_heldout":True,
        "treatment_candidate_count":1,
        "remove_candidate_count":1,
        "reset_candidate_count":0,
        "wrong_candidate_count":1,
        "full_external_pair_count":2*len(full_sets),
        "treatment_external_pair_count":2,
        "external_pair_reduction_vs_full":1.0-(2.0/(2.0*len(full_sets))),
        "treatment_capability":treatment_cap,
        "remove_same_checkpoint_capability":remove_cap,
        "reset_capability":reset_cap,
        "wrong_genuinely_learned_cardinality":wrong_policy.cardinality,
        "wrong_uses_same_repair_operator":True,
        "wrong_lower_cardinality_capability":wrong_cap,
        "full_exhaustive_capability":full_cap,
        "verifierless_cardinality_authority":False,
        "repository_files_per_task":4,
        "repository_hashes_disjoint":True,
        "concrete_filenames_random_post_checkout":True,
        "patch_set_candidate_content_uses_hidden_outcomes":False,
        "cardinality_frontier_world_failure_driven":True,
        "external_execution":"fresh_subprocess_materialized_four_file_repository",
        "hidden_tests_exposed_to_body_before_execution":False,
        "cardinality_expansion_rule_human_authored":True,
        "real_repository_autonomous_repair":False,
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
        raise SystemExit("usage: run_world_driven_repair_cardinality_genesis.py <seed_path>")
    main(sys.argv[1])
