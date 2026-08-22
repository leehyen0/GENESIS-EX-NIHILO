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
    RepositoryPatchCandidate,
    RepositoryTaskAcquisitionOrgan,
    SubprocessRepositoryRepairExecutor,
    derive_repository_file_roles,
    repository_hash,
)
from arte_cognition.world_coupling import HMACWorldReceiptSigner, HMACWorldReceiptVerifier


REPAIR_FAMILIES = {
    "COMPARE::GtE->Gt": (">=", operator.gt),
    "COMPARE::Gt->GtE": (">", operator.ge),
    "COMPARE::Lt->LtE": ("<", operator.le),
    "COMPARE::LtE->Lt": ("<=", operator.lt),
}


@dataclass(frozen=True)
class HiddenRepositoryTask:
    task_id: str
    domain: str
    files: dict[str, str]
    entry_module: str
    function_name: str
    cases: tuple[dict, ...]
    intended_role: str

    @property
    def repo_hash(self) -> str:
        return repository_hash(self.files)


def _name(rng, prefix):
    return f"{prefix}_{rng.randrange(10000000, 99999999)}"


def _module_triplet(rng):
    # Prefixes make the no-policy first candidate an isolated decoy without revealing
    # which randomized concrete filename is actually causally relevant.
    decoy = _name(rng, "a_mod")
    root = _name(rng, "m_mod")
    leaf = _name(rng, "z_mod")
    return decoy, root, leaf


def _expected_filter(values, pivot, predicate):
    return [value for value in values if predicate(value, pivot)]


def _make_repository_task(rng, repair_operator, domain, label, intended_role):
    buggy_op, predicate = REPAIR_FAMILIES[repair_operator]
    decoy_module, root_module, leaf_module = _module_triplet(rng)
    decoy_fn = _name(rng, "decoy")
    root_fn = _name(rng, "public")
    leaf_fn = _name(rng, "leaf")

    decoy_source = f'''def {decoy_fn}(value, pivot):
    return value if value {buggy_op} pivot else pivot
'''

    if domain == "numeric-sequence":
        values_cases = [([-3,-1,0,2,5],0), ([1,2,3,4],2), ([7,7,8,9],7)]
        if intended_role == "IMPORTED_LEAF":
            leaf_source = f'''def {leaf_fn}(values, pivot):
    return [value for value in values if value {buggy_op} pivot]
'''
            root_source = f'''from {leaf_module} import {leaf_fn}

def {root_fn}(values, pivot):
    _unused = pivot {buggy_op} pivot
    return {leaf_fn}(values, pivot)
'''
        else:
            leaf_source = f'''def {leaf_fn}(values, pivot):
    _unused = pivot {buggy_op} pivot
    return list(values)
'''
            root_source = f'''from {leaf_module} import {leaf_fn}

def {root_fn}(values, pivot):
    values = {leaf_fn}(values, pivot)
    return [value for value in values if value {buggy_op} pivot]
'''
        cases = tuple({"args":[values,pivot], "expected":_expected_filter(values,pivot,predicate)} for values,pivot in values_cases)

    elif domain == "lexical-string":
        values_cases = [(["aa","bb","bc","zz"],"bb"), (["ant","bee","cat"],"bee"), (["k","k","m"],"k")]
        if intended_role == "IMPORTED_LEAF":
            leaf_source = f'''def {leaf_fn}(tokens, pivot):
    return [token for token in tokens if token {buggy_op} pivot]
'''
            root_source = f'''from {leaf_module} import {leaf_fn}

def {root_fn}(tokens, pivot):
    _unused = pivot {buggy_op} pivot
    return {leaf_fn}(tokens, pivot)
'''
        else:
            leaf_source = f'''def {leaf_fn}(tokens, pivot):
    _unused = pivot {buggy_op} pivot
    return list(tokens)
'''
            root_source = f'''from {leaf_module} import {leaf_fn}

def {root_fn}(tokens, pivot):
    tokens = {leaf_fn}(tokens, pivot)
    return [token for token in tokens if token {buggy_op} pivot]
'''
        cases = tuple({"args":[values,pivot], "expected":_expected_filter(values,pivot,predicate)} for values,pivot in values_cases)

    elif domain == "record-structure":
        score_key = _name(rng, "score")
        id_key = _name(rng, "id")
        raw_cases = [
            ([{score_key:1,id_key:"a"},{score_key:2,id_key:"b"},{score_key:4,id_key:"c"}],2),
            ([{score_key:5,id_key:"q"},{score_key:5,id_key:"r"},{score_key:8,id_key:"s"}],5),
            ([{score_key:-1,id_key:"x"},{score_key:0,id_key:"y"},{score_key:1,id_key:"z"}],0),
        ]
        if intended_role == "IMPORTED_LEAF":
            leaf_source = f'''def {leaf_fn}(rows, pivot):
    return [row[{id_key!r}] for row in rows if row[{score_key!r}] {buggy_op} pivot]
'''
            root_source = f'''from {leaf_module} import {leaf_fn}

def {root_fn}(rows, pivot):
    _unused = pivot {buggy_op} pivot
    return {leaf_fn}(rows, pivot)
'''
        else:
            leaf_source = f'''def {leaf_fn}(rows, pivot):
    _unused = pivot {buggy_op} pivot
    return list(rows)
'''
            root_source = f'''from {leaf_module} import {leaf_fn}

def {root_fn}(rows, pivot):
    rows = {leaf_fn}(rows, pivot)
    return [row[{id_key!r}] for row in rows if row[{score_key!r}] {buggy_op} pivot]
'''
        cases = []
        for rows,pivot in raw_cases:
            expected=[row[id_key] for row in rows if predicate(row[score_key],pivot)]
            cases.append({"args":[rows,pivot],"expected":expected})
        cases = tuple(cases)
    else:
        raise ValueError(domain)

    files = {
        f"{decoy_module}.py": decoy_source,
        f"{root_module}.py": root_source,
        f"{leaf_module}.py": leaf_source,
    }
    roles = derive_repository_file_roles(files)
    role_counts = {role: list(roles.values()).count(role) for role in set(roles.values())}
    if role_counts.get("ROOT_IMPORTER") != 1 or role_counts.get("IMPORTED_LEAF") != 1 or role_counts.get("ISOLATED") != 1:
        raise AssertionError(f"repository geometry lost structural-role uniqueness: {roles}")
    return HiddenRepositoryTask(
        task_id=f"{label}-{rng.randrange(10**7,10**8)}",
        domain=domain,
        files=files,
        entry_module=root_module,
        function_name=root_fn,
        cases=cases,
        intended_role=intended_role,
    )


def execute_candidate(body, task, candidate, signers, verifier, epoch_base):
    effects=[]
    for issuer_index,(issuer,signer) in enumerate(signers.items()):
        executor=SubprocessRepositoryRepairExecutor(
            baseline_files=task.files,
            candidate=candidate,
            entry_module=task.entry_module,
            function_name=task.function_name,
            hidden_cases=task.cases,
            signer=signer,
            source_id=f"{task.task_id}-repo-{candidate.file_role}-{candidate.site_index}-{issuer}",
            context_id=task.task_id,
            challenge_id=f"{task.task_id}-repository-hidden-suite-{candidate.file_role}-{candidate.site_index}-{issuer}",
            epoch=epoch_base+issuer_index,
        )
        pair=body.execute_world_intervention(candidate.proposal,executor,verifier=verifier)
        if not pair.authority_verified:
            raise AssertionError("repository hidden-test receipt lost external authority")
        effects.append(float(pair.effect))
    return tuple(effects)


def train_repository(body, task, signers, verifier, epoch_base):
    organ=RepositoryTaskAcquisitionOrgan(body)
    candidates=organ.propose(task.task_id,task.files)
    if len(candidates)!=3:
        raise AssertionError(f"expected exactly three structural localization candidates, got {len(candidates)}")
    strong=[]
    for index,candidate in enumerate(candidates):
        effects=execute_candidate(body,task,candidate,signers,verifier,epoch_base+index*10)
        if min(effects)>=0.9:
            strong.append(candidate)
    if len(strong)!=1:
        raise AssertionError(f"repository task must have exactly one externally successful patch: {strong}")
    if strong[0].file_role!=task.intended_role:
        raise AssertionError("external repository tests selected the wrong structural role")
    return candidates,strong[0]


def execute_one(body, task, selection, signers, verifier, epoch_base):
    if len(selection.candidates)!=1:
        raise AssertionError("matched repository arm requires exactly one candidate")
    return execute_candidate(body,task,selection.candidates[0],signers,verifier,epoch_base)


def exhaustive_capability(body, task, signers, verifier, epoch_base):
    organ=RepositoryTaskAcquisitionOrgan(body)
    candidates=organ.propose(task.task_id,task.files)
    success=False
    for index,candidate in enumerate(candidates):
        effects=execute_candidate(body,task,candidate,signers,verifier,epoch_base+index*10)
        success=success or min(effects)>=0.9
    return float(success),len(candidates),2*len(candidates)


def main(seed_path):
    seed=int(Path(seed_path).read_text().strip())
    rng=random.Random(seed)
    hidden_operator=rng.choice(tuple(sorted(REPAIR_FAMILIES)))

    issuer_a=f"repo-lab-{rng.randrange(10**7,10**8)}"
    issuer_b=f"repo-lab-{rng.randrange(10**7,10**8)}"
    key_a=hashlib.sha256(f"{seed}:repo:a".encode()).digest()
    key_b=hashlib.sha256(f"{seed}:repo:b".encode()).digest()
    signers={issuer_a:HMACWorldReceiptSigner(issuer_a,key_a),issuer_b:HMACWorldReceiptSigner(issuer_b,key_b)}
    verifier=HMACWorldReceiptVerifier(
        {issuer_a:key_a,issuer_b:key_b},
        independence_classes={issuer_a:"REPOSITORY_LAB_A",issuer_b:"REPOSITORY_LAB_B"},
    )

    parent=PersistentCognitiveRuntime()
    train_numeric=_make_repository_task(rng,hidden_operator,"numeric-sequence","repo-train-numeric","IMPORTED_LEAF")
    train_string=_make_repository_task(rng,hidden_operator,"lexical-string","repo-train-string","IMPORTED_LEAF")
    numeric_candidates,numeric_patch=train_repository(parent,train_numeric,signers,verifier,10000)
    string_candidates,string_patch=train_repository(parent,train_string,signers,verifier,20000)
    policy=RepositoryTaskAcquisitionOrgan(parent).policy()
    if policy.file_role!="IMPORTED_LEAF" or policy.operator_id!=hidden_operator or len(policy.supporting_contexts)!=2:
        raise AssertionError(f"repository localization policy was not learned: {policy}")
    if numeric_patch.operator_id!=hidden_operator or string_patch.operator_id!=hidden_operator:
        raise AssertionError("repository training selected unexpected repair operator")

    checkpoint=checkpoint_dict(parent)
    verifierless=restore_runtime(checkpoint)
    if RepositoryTaskAcquisitionOrgan(verifierless).policy().file_role is not None:
        raise AssertionError("repository localization authority restored without external verifier")

    heldout=_make_repository_task(rng,hidden_operator,"record-structure","repo-heldout-record","IMPORTED_LEAF")
    treatment=restore_runtime(checkpoint,world_verifier=verifier)
    remove=restore_runtime(checkpoint,world_verifier=verifier)
    treatment_organ=RepositoryTaskAcquisitionOrgan(treatment)
    remove_organ=RepositoryTaskAcquisitionOrgan(remove)
    treatment_candidates=treatment_organ.propose(heldout.task_id,heldout.files)
    remove_candidates=remove_organ.propose(heldout.task_id,heldout.files)
    if tuple(c.proposal.experiment_id for c in treatment_candidates)!=tuple(c.proposal.experiment_id for c in remove_candidates):
        raise AssertionError("same-checkpoint repository candidate generation diverged")
    treatment_selection=treatment_organ.select(treatment_candidates,max_candidates=1,apply_learned_policy=True)
    remove_selection=remove_organ.select(remove_candidates,max_candidates=1,apply_learned_policy=False)
    treatment_effects=execute_one(treatment,heldout,treatment_selection,signers,verifier,40000)
    remove_effects=execute_one(remove,heldout,remove_selection,signers,verifier,40000)
    treatment_cap=float(min(treatment_effects)>=0.9)
    remove_cap=float(min(remove_effects)>=0.9)

    reset=PersistentCognitiveRuntime()
    reset_organ=RepositoryTaskAcquisitionOrgan(reset)
    reset_candidates=reset_organ.propose(heldout.task_id,heldout.files)
    reset_selection=reset_organ.select(reset_candidates,max_candidates=1,apply_learned_policy=True)
    reset_effects=execute_one(reset,heldout,reset_selection,signers,verifier,50000)
    reset_cap=float(min(reset_effects)>=0.9)

    full=PersistentCognitiveRuntime()
    full_cap,full_candidates,full_pairs=exhaustive_capability(full,heldout,signers,verifier,60000)

    # Genuine wrong-localization BODY: it learns the same repair operator, but repeated
    # external success places the causal edit in ROOT_IMPORTER rather than IMPORTED_LEAF.
    wrong_parent=PersistentCognitiveRuntime()
    wrong_numeric=_make_repository_task(rng,hidden_operator,"numeric-sequence","wrong-repo-numeric","ROOT_IMPORTER")
    wrong_string=_make_repository_task(rng,hidden_operator,"lexical-string","wrong-repo-string","ROOT_IMPORTER")
    train_repository(wrong_parent,wrong_numeric,signers,verifier,70000)
    train_repository(wrong_parent,wrong_string,signers,verifier,80000)
    wrong_policy=RepositoryTaskAcquisitionOrgan(wrong_parent).policy()
    if wrong_policy.file_role!="ROOT_IMPORTER" or wrong_policy.operator_id!=hidden_operator:
        raise AssertionError("wrong-control BODY failed to genuinely learn root-importer localization")
    wrong=restore_runtime(checkpoint_dict(wrong_parent),world_verifier=verifier)
    wrong_organ=RepositoryTaskAcquisitionOrgan(wrong)
    wrong_candidates=wrong_organ.propose(heldout.task_id,heldout.files)
    wrong_selection=wrong_organ.select(wrong_candidates,max_candidates=1,apply_learned_policy=True)
    wrong_effects=execute_one(wrong,heldout,wrong_selection,signers,verifier,90000)
    wrong_cap=float(min(wrong_effects)>=0.9)

    if treatment_cap!=1.0 or remove_cap!=0.0 or reset_cap!=0.0 or wrong_cap!=0.0 or full_cap!=1.0:
        raise AssertionError("multi-file repository acquisition causal controls failed")
    chosen=treatment_selection.candidates[0]
    if chosen.file_role!="IMPORTED_LEAF" or chosen.operator_id!=hidden_operator:
        raise AssertionError("learned repository policy failed to localize fresh causal patch")
    if remove_selection.candidates[0].file_role=="IMPORTED_LEAF":
        raise AssertionError("REMOVE accidentally localized the hidden leaf under one-candidate budget")
    repo_hashes={train_numeric.repo_hash,train_string.repo_hash,heldout.repo_hash,wrong_numeric.repo_hash,wrong_string.repo_hash}
    if len(repo_hashes)!=5:
        raise AssertionError("repository tasks were not source-disjoint")

    result={
        "status":"PASS_BOUNDED_SOURCE_DISJOINT_MULTIFILE_REPOSITORY_LOCALIZATION_AND_DESCENDANT_CAUSAL_TRANSFER",
        "hidden_repair_operator":hidden_operator,
        "learned_file_role":"IMPORTED_LEAF",
        "wrong_genuinely_learned_file_role":"ROOT_IMPORTER",
        "wrong_uses_same_repair_operator":True,
        "training_domains":[train_numeric.domain,train_string.domain],
        "heldout_domain":heldout.domain,
        "repository_files_per_task":3,
        "repository_hashes_disjoint":True,
        "concrete_filenames_random_post_checkout":True,
        "localization_feature":"STATIC_IMPORT_GRAPH_ROLE",
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
        "wrong_localization_capability":wrong_cap,
        "full_exhaustive_capability":full_cap,
        "verifierless_localization_authority":False,
        "external_execution":"fresh_subprocess_materialized_multifile_repository",
        "hidden_tests_exposed_to_body_before_execution":False,
        "candidate_generation_uses_hidden_outcomes":False,
        "repository_repair_language_human_authored":True,
        "repository_scale":"GENERATED_THREE_FILE_MICROREPOSITORY",
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
        raise SystemExit("usage: run_multifile_repository_repair_acquisition.py <seed_path>")
    main(sys.argv[1])
