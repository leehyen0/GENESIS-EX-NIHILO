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
from arte_cognition.repository_task_acquisition import (
    RepositoryTaskAcquisitionOrgan,
    SubprocessRepositoryRepairExecutor,
    derive_repository_file_roles,
    repository_hash,
)
from arte_cognition.repository_patch_composition import (
    RepositoryPatchCompositionOrgan,
    SubprocessRepositoryPatchPairExecutor,
    canonical_pair_signature,
)
from arte_cognition.world_coupling import HMACWorldReceiptSigner, HMACWorldReceiptVerifier


REPAIR_FAMILIES = {
    "COMPARE::GtE->Gt": (">=", operator.gt, "GREATER"),
    "COMPARE::Gt->GtE": (">", operator.ge, "GREATER"),
    "COMPARE::Lt->LtE": ("<", operator.le, "LESS"),
    "COMPARE::LtE->Lt": ("<=", operator.lt, "LESS"),
}

TARGET_PAIR = ("IMPORTED_LEAF", "INTERMEDIATE")
WRONG_PAIR = ("IMPORTED_LEAF", "ROOT_IMPORTER")


@dataclass(frozen=True)
class HiddenCoordinatedRepositoryTask:
    task_id: str
    domain: str
    files: dict[str, str]
    entry_module: str
    function_name: str
    cases: tuple[dict, ...]
    intended_roles: tuple[str, str]

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


def _domain_values(domain, direction, rng):
    if domain == "numeric-tuples":
        pivot_a = 0
        pivot_b = 0
        strict = 1 if direction == "GREATER" else -1
        opposite = -1 if direction == "GREATER" else 1
        items = [
            (0, strict),
            (strict, 0),
            (strict, strict),
            (0, 0),
            (opposite, strict),
            (strict, opposite),
        ]
        first = "item[0]"
        second = "item[1]"
        return items, pivot_a, pivot_b, first, second
    if domain == "lexical-tuples":
        pivot_a = "m"
        pivot_b = "m"
        strict = "z" if direction == "GREATER" else "a"
        opposite = "a" if direction == "GREATER" else "z"
        items = [
            ("m", strict),
            (strict, "m"),
            (strict, strict),
            ("m", "m"),
            (opposite, strict),
            (strict, opposite),
        ]
        return items, pivot_a, pivot_b, "item[0]", "item[1]"
    if domain == "record-structure":
        pivot_a = 0
        pivot_b = 0
        strict = 1 if direction == "GREATER" else -1
        opposite = -1 if direction == "GREATER" else 1
        key_a = _name(rng, "axis_a")
        key_b = _name(rng, "axis_b")
        key_id = _name(rng, "id")
        raw = [
            (0, strict, "eq-a"),
            (strict, 0, "eq-b"),
            (strict, strict, "strict"),
            (0, 0, "eq-both"),
            (opposite, strict, "fail-a"),
            (strict, opposite, "fail-b"),
        ]
        items = [{key_a:a, key_b:b, key_id:label} for a,b,label in raw]
        return items, pivot_a, pivot_b, f"item[{key_a!r}]", f"item[{key_b!r}]"
    raise ValueError(domain)


def _make_task(rng, repair_operator, domain, label, intended_roles):
    buggy_op, predicate, direction = REPAIR_FAMILIES[repair_operator]
    isolated_module, root_module, mid_module, leaf_module = _module_chain(rng)
    isolated_fn = _name(rng, "isolated")
    root_fn = _name(rng, "public")
    mid_fn = _name(rng, "middle")
    leaf_fn = _name(rng, "leaf")
    items, pivot_a, pivot_b, first_expr, second_expr = _domain_values(domain, direction, rng)

    leaf_source = f'''def {leaf_fn}(items, pivot_a):
    return [item for item in items if {first_expr} {buggy_op} pivot_a]
'''

    if "INTERMEDIATE" in intended_roles:
        mid_source = f'''from {leaf_module} import {leaf_fn}

def {mid_fn}(items, pivot_a, pivot_b):
    items = {leaf_fn}(items, pivot_a)
    return [item for item in items if {second_expr} {buggy_op} pivot_b]
'''
        root_source = f'''from {mid_module} import {mid_fn}

def {root_fn}(items, pivot_a, pivot_b):
    _unused = pivot_b {buggy_op} pivot_b
    return {mid_fn}(items, pivot_a, pivot_b)
'''
    else:
        mid_source = f'''from {leaf_module} import {leaf_fn}

def {mid_fn}(items, pivot_a, pivot_b):
    _unused = pivot_b {buggy_op} pivot_b
    return {leaf_fn}(items, pivot_a)
'''
        root_source = f'''from {mid_module} import {mid_fn}

def {root_fn}(items, pivot_a, pivot_b):
    items = {mid_fn}(items, pivot_a, pivot_b)
    return [item for item in items if {second_expr} {buggy_op} pivot_b]
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
    expected_role_counts = {"ISOLATED":1, "ROOT_IMPORTER":1, "INTERMEDIATE":1, "IMPORTED_LEAF":1}
    actual_role_counts = {role:list(roles.values()).count(role) for role in expected_role_counts}
    if actual_role_counts != expected_role_counts:
        raise AssertionError(f"coordinated repository lost unique role geometry: {roles}")

    def first_value(item):
        if domain == "record-structure":
            key = first_expr.split("[",1)[1].rsplit("]",1)[0]
            return item[eval(key)]
        return item[0]

    def second_value(item):
        if domain == "record-structure":
            key = second_expr.split("[",1)[1].rsplit("]",1)[0]
            return item[eval(key)]
        return item[1]

    expected = [
        item for item in items
        if predicate(first_value(item), pivot_a) and predicate(second_value(item), pivot_b)
    ]
    cases = ({"args":[items,pivot_a,pivot_b], "expected":expected},)
    return HiddenCoordinatedRepositoryTask(
        task_id=f"{label}-{rng.randrange(10**7,10**8)}",
        domain=domain,
        files=files,
        entry_module=root_module,
        function_name=root_fn,
        cases=tuple(cases),
        intended_roles=tuple(sorted(intended_roles)),
    )


def execute_single(body, task, candidate, signers, verifier, epoch_base):
    effects=[]
    for issuer_index,(issuer,signer) in enumerate(signers.items()):
        executor=SubprocessRepositoryRepairExecutor(
            baseline_files=task.files,
            candidate=candidate,
            entry_module=task.entry_module,
            function_name=task.function_name,
            hidden_cases=task.cases,
            signer=signer,
            source_id=f"{task.task_id}-single-{candidate.file_role}-{issuer}",
            context_id=task.task_id,
            challenge_id=f"{task.task_id}-single-hidden-{candidate.file_role}-{issuer}",
            epoch=epoch_base+issuer_index,
        )
        pair=body.execute_world_intervention(candidate.proposal,executor,verifier=verifier)
        if not pair.authority_verified:
            raise AssertionError("single-edit repository evidence lost authority")
        effects.append(float(pair.effect))
    return tuple(effects)


def execute_pair(body, task, candidate, signers, verifier, epoch_base):
    effects=[]
    for issuer_index,(issuer,signer) in enumerate(signers.items()):
        executor=SubprocessRepositoryPatchPairExecutor(
            baseline_files=task.files,
            candidate=candidate,
            entry_module=task.entry_module,
            function_name=task.function_name,
            hidden_cases=task.cases,
            signer=signer,
            source_id=f"{task.task_id}-pair-{hashlib.sha256(candidate.signature.encode()).hexdigest()[:8]}-{issuer}",
            context_id=task.task_id,
            challenge_id=f"{task.task_id}-pair-hidden-{candidate.proposal.experiment_id[-12:]}-{issuer}",
            epoch=epoch_base+issuer_index,
        )
        pair=body.execute_world_intervention(candidate.proposal,executor,verifier=verifier)
        if not pair.authority_verified:
            raise AssertionError("coordinated repository evidence lost authority")
        effects.append(float(pair.effect))
    return tuple(effects)


def train_two_context_composition(body, tasks, signers, verifier, epoch_base):
    single_map={}
    for task_index,task in enumerate(tasks):
        single_organ=RepositoryTaskAcquisitionOrgan(body)
        singles=single_organ.propose(task.task_id,task.files)
        if len(singles)!=4:
            raise AssertionError(f"expected four single-edit candidates, got {len(singles)}")
        for candidate_index,candidate in enumerate(singles):
            effects=execute_single(
                body,task,candidate,signers,verifier,
                epoch_base+task_index*1000+candidate_index*10,
            )
            if max(effects)>=0.9:
                raise AssertionError("a single edit unexpectedly repaired a coordinated task")
        single_map[task.task_id]=singles

    composition=RepositoryPatchCompositionOrgan(body)
    assessment=composition.assess_single_failure(single_map)
    if assessment.status!="SINGLE_EDIT_REPAIR_SPACE_FALSIFIED_OPEN_PAIR_COMPOSITION":
        raise AssertionError(f"complete single-edit failure did not open pair composition: {assessment}")
    if assessment.missing_experiment_ids:
        raise AssertionError("single-edit failure gate admitted missing evidence")

    strong_pairs=[]
    pair_counts=[]
    for task_index,task in enumerate(tasks):
        pairs=composition.propose(
            task.task_id,task.files,single_map[task.task_id],
            training_single_candidates_by_context=single_map,
        )
        if len(pairs)!=6:
            raise AssertionError(f"expected six exact two-file combinations, got {len(pairs)}")
        strong=[]
        for candidate_index,candidate in enumerate(pairs):
            effects=execute_pair(
                body,task,candidate,signers,verifier,
                epoch_base+5000+task_index*1000+candidate_index*10,
            )
            if min(effects)>=0.9:
                strong.append(candidate)
        if len(strong)!=1:
            raise AssertionError(f"expected exactly one coordinated repair pair: {strong}")
        roles=tuple(sorted(member.file_role for member in strong[0].members))
        if roles!=task.intended_roles:
            raise AssertionError(f"coordinated pair localized to wrong roles: {roles}")
        strong_pairs.append(strong[0])
        pair_counts.append(len(pairs))
    return single_map,tuple(strong_pairs),assessment,tuple(pair_counts)


def execute_one_pair(body,task,selection,signers,verifier,epoch_base):
    if len(selection.candidates)!=1:
        return ()
    return execute_pair(body,task,selection.candidates[0],signers,verifier,epoch_base)


def main(seed_path):
    seed=int(Path(seed_path).read_text().strip())
    rng=random.Random(seed)
    hidden_operator=rng.choice(tuple(sorted(REPAIR_FAMILIES)))

    issuer_a=f"pair-lab-{rng.randrange(10**7,10**8)}"
    issuer_b=f"pair-lab-{rng.randrange(10**7,10**8)}"
    key_a=hashlib.sha256(f"{seed}:pair:a".encode()).digest()
    key_b=hashlib.sha256(f"{seed}:pair:b".encode()).digest()
    signers={issuer_a:HMACWorldReceiptSigner(issuer_a,key_a),issuer_b:HMACWorldReceiptSigner(issuer_b,key_b)}
    verifier=HMACWorldReceiptVerifier(
        {issuer_a:key_a,issuer_b:key_b},
        independence_classes={issuer_a:"PAIR_LAB_A",issuer_b:"PAIR_LAB_B"},
    )

    parent=PersistentCognitiveRuntime()
    train_numeric=_make_task(rng,hidden_operator,"numeric-tuples","pair-train-numeric",TARGET_PAIR)
    train_string=_make_task(rng,hidden_operator,"lexical-tuples","pair-train-string",TARGET_PAIR)
    single_map,strong_pairs,assessment,pair_counts=train_two_context_composition(
        parent,(train_numeric,train_string),signers,verifier,10000,
    )
    target_signature=canonical_pair_signature(strong_pairs[0].members)
    if canonical_pair_signature(strong_pairs[1].members)!=target_signature:
        raise AssertionError("training repositories did not reproduce the same pair signature")
    policy=RepositoryPatchCompositionOrgan(parent).policy()
    if policy.signature!=target_signature or len(policy.supporting_contexts)!=2:
        raise AssertionError(f"coordinated patch policy was not learned: {policy}")

    checkpoint=checkpoint_dict(parent)
    verifierless=restore_runtime(checkpoint)
    if RepositoryPatchCompositionOrgan(verifierless).policy().signature is not None:
        raise AssertionError("coordinated patch authority restored without external verifier")

    heldout=_make_task(rng,hidden_operator,"record-structure","pair-heldout-record",TARGET_PAIR)
    treatment=restore_runtime(checkpoint,world_verifier=verifier)
    remove=restore_runtime(checkpoint,world_verifier=verifier)
    treatment_singles=RepositoryTaskAcquisitionOrgan(treatment).propose(heldout.task_id,heldout.files)
    remove_singles=RepositoryTaskAcquisitionOrgan(remove).propose(heldout.task_id,heldout.files)
    treatment_organ=RepositoryPatchCompositionOrgan(treatment)
    remove_organ=RepositoryPatchCompositionOrgan(remove)
    treatment_pairs=treatment_organ.propose(heldout.task_id,heldout.files,treatment_singles)
    remove_pairs=remove_organ.propose(heldout.task_id,heldout.files,remove_singles)
    if tuple(item.proposal.experiment_id for item in treatment_pairs)!=tuple(item.proposal.experiment_id for item in remove_pairs):
        raise AssertionError("same-checkpoint pair candidate universe diverged")
    treatment_selection=treatment_organ.select(treatment_pairs,max_candidates=1,apply_learned_policy=True)
    remove_selection=remove_organ.select(remove_pairs,max_candidates=1,apply_learned_policy=False)
    treatment_effects=execute_one_pair(treatment,heldout,treatment_selection,signers,verifier,40000)
    remove_effects=execute_one_pair(remove,heldout,remove_selection,signers,verifier,40000)
    treatment_cap=float(bool(treatment_effects) and min(treatment_effects)>=0.9)
    remove_cap=float(bool(remove_effects) and min(remove_effects)>=0.9)

    # Strong synergy control: each member of Treatment's selected pair fails alone.
    member_control=restore_runtime(checkpoint,world_verifier=verifier)
    selected_pair=treatment_selection.candidates[0]
    member_effects=[]
    for index,member in enumerate(selected_pair.members):
        effects=execute_single(member_control,heldout,member,signers,verifier,50000+index*100)
        member_effects.append(effects)
    if any(min(effects)>=0.9 for effects in member_effects):
        raise AssertionError("coordinated capability collapsed to a sufficient single edit")

    reset=PersistentCognitiveRuntime()
    reset_singles=RepositoryTaskAcquisitionOrgan(reset).propose(heldout.task_id,heldout.files)
    reset_organ=RepositoryPatchCompositionOrgan(reset)
    reset_pairs=reset_organ.propose(heldout.task_id,heldout.files,reset_singles)
    reset_cap=0.0
    if reset_pairs:
        raise AssertionError("RESET opened pair composition without complete single-edit failure or inherited authority")

    full=restore_runtime(checkpoint,world_verifier=verifier)
    full_singles=RepositoryTaskAcquisitionOrgan(full).propose(heldout.task_id,heldout.files)
    full_pairs=RepositoryPatchCompositionOrgan(full).propose(heldout.task_id,heldout.files,full_singles)
    full_success=False
    for index,candidate in enumerate(full_pairs):
        effects=execute_pair(full,heldout,candidate,signers,verifier,60000+index*10)
        full_success=full_success or min(effects)>=0.9
    full_cap=float(full_success)

    wrong_parent=PersistentCognitiveRuntime()
    wrong_numeric=_make_task(rng,hidden_operator,"numeric-tuples","wrong-pair-numeric",WRONG_PAIR)
    wrong_string=_make_task(rng,hidden_operator,"lexical-tuples","wrong-pair-string",WRONG_PAIR)
    _,wrong_strong,wrong_assessment,_=train_two_context_composition(
        wrong_parent,(wrong_numeric,wrong_string),signers,verifier,70000,
    )
    wrong_signature=canonical_pair_signature(wrong_strong[0].members)
    wrong_policy=RepositoryPatchCompositionOrgan(wrong_parent).policy()
    if wrong_policy.signature!=wrong_signature or wrong_signature==target_signature:
        raise AssertionError("wrong-control BODY failed to genuinely learn a distinct pair signature")
    wrong=restore_runtime(checkpoint_dict(wrong_parent),world_verifier=verifier)
    wrong_singles=RepositoryTaskAcquisitionOrgan(wrong).propose(heldout.task_id,heldout.files)
    wrong_organ=RepositoryPatchCompositionOrgan(wrong)
    wrong_pairs=wrong_organ.propose(heldout.task_id,heldout.files,wrong_singles)
    wrong_selection=wrong_organ.select(wrong_pairs,max_candidates=1,apply_learned_policy=True)
    wrong_effects=execute_one_pair(wrong,heldout,wrong_selection,signers,verifier,90000)
    wrong_cap=float(bool(wrong_effects) and min(wrong_effects)>=0.9)

    if treatment_cap!=1.0 or remove_cap!=0.0 or reset_cap!=0.0 or wrong_cap!=0.0 or full_cap!=1.0:
        raise AssertionError("coordinated multi-file patch causal controls failed")
    if treatment_selection.candidates[0].signature!=target_signature:
        raise AssertionError("learned pair policy did not select the reproduced coordinated signature")
    if remove_selection.candidates[0].signature==target_signature:
        raise AssertionError("REMOVE accidentally selected the target pair under one-candidate budget")
    repo_hashes={
        train_numeric.repo_hash,train_string.repo_hash,heldout.repo_hash,
        wrong_numeric.repo_hash,wrong_string.repo_hash,
    }
    if len(repo_hashes)!=5:
        raise AssertionError("coordinated repository tasks were not source-disjoint")

    result={
        "status":"PASS_BOUNDED_COMPLETE_SINGLE_EDIT_FAILURE_TO_COORDINATED_MULTIFILE_PATCH_COMPOSITION_AND_DESCENDANT_TRANSFER",
        "hidden_repair_operator":hidden_operator,
        "learned_pair_signature":target_signature,
        "wrong_genuinely_learned_pair_signature":wrong_signature,
        "wrong_uses_same_repair_operator":True,
        "training_domains":[train_numeric.domain,train_string.domain],
        "heldout_domain":heldout.domain,
        "repository_files_per_task":4,
        "single_candidates_per_training_context":4,
        "single_failure_evaluated_candidate_count":assessment.evaluated_candidate_count,
        "single_failure_missing_candidate_count":len(assessment.missing_experiment_ids),
        "single_failure_falsified_contexts":len(assessment.falsified_contexts),
        "pair_candidates_per_training_context":list(pair_counts),
        "heldout_pair_candidate_count":len(full_pairs),
        "treatment_candidate_count":1,
        "remove_candidate_count":1,
        "reset_candidate_count":0,
        "wrong_candidate_count":1,
        "full_external_pair_count":2*len(full_pairs),
        "treatment_external_pair_count":2,
        "external_pair_reduction_vs_full":1.0-(2.0/(2.0*len(full_pairs))),
        "treatment_capability":treatment_cap,
        "remove_same_checkpoint_capability":remove_cap,
        "reset_capability":reset_cap,
        "wrong_pair_capability":wrong_cap,
        "full_exhaustive_capability":full_cap,
        "selected_member_single_capabilities":[float(min(effects)>=0.9) for effects in member_effects],
        "pair_synergy_required":True,
        "verifierless_pair_authority":False,
        "repository_hashes_disjoint":True,
        "concrete_filenames_random_post_checkout":True,
        "pair_candidate_content_uses_hidden_outcomes":False,
        "pair_frontier_opened_by_authenticated_single_edit_failure":True,
        "external_execution":"fresh_subprocess_materialized_four_file_repository",
        "hidden_tests_exposed_to_body_before_execution":False,
        "coordinated_patch_schema_human_authored":True,
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
        raise SystemExit("usage: run_coordinated_multifile_patch_composition.py <seed_path>")
    main(sys.argv[1])
