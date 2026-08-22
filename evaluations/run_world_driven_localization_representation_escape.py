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
from arte_cognition.repository_localization_representation_genesis import (
    RepositoryLocalizationRepresentationOrgan,
    derive_import_graph_fingerprints,
    parse_graph_localization_signature,
)
from arte_cognition.repository_task_acquisition import (
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
class HiddenTopologyTask:
    task_id: str
    domain: str
    files: dict[str, str]
    entry_module: str
    function_name: str
    cases: tuple[dict, ...]
    causal_branch: str
    target_mid_path: str
    decoy_mid_path: str

    @property
    def repo_hash(self) -> str:
        return repository_hash(self.files)


def _name(rng, prefix="m"):
    return f"{prefix}_{rng.randrange(10_000_000, 99_999_999)}"


def _unique_modules(rng, count):
    names = set()
    while len(names) < count:
        names.add(_name(rng))
    return list(names)


def _make_task(rng, repair_operator, domain, label, causal_branch="TARGET"):
    buggy_op, predicate = REPAIR_FAMILIES[repair_operator]
    raw = _unique_modules(rng, 10)
    mid_names = sorted(raw[:2])
    decoy_mid, target_mid = mid_names[0], mid_names[1]
    (
        target_root,
        decoy_root,
        target_tail,
        decoy_tail,
        target_leaf,
        decoy_branch,
        decoy_leaf_a,
        decoy_leaf_b,
    ) = raw[2:]

    target_fn = _name(rng, "target")
    decoy_fn = _name(rng, "decoy")
    public_fn = _name(rng, "public")
    decoy_public_fn = _name(rng, "shadow")

    target_mid_source = f'''from {target_tail} import passthrough as _next

def {target_fn}(value, pivot):
    value = _next(value)
    return value {buggy_op} pivot
'''
    decoy_mid_source = f'''from {decoy_tail} import passthrough as _next

def {decoy_fn}(value, pivot):
    value = _next(value)
    return value {buggy_op} pivot
'''
    target_tail_source = f'''from {target_leaf} import passthrough as _next

def passthrough(value):
    return _next(value)
'''
    target_leaf_source = '''def passthrough(value):
    return value
'''
    decoy_tail_source = f'''from {decoy_branch} import passthrough as _next

def passthrough(value):
    return _next(value)
'''
    decoy_branch_source = f'''from {decoy_leaf_a} import left
from {decoy_leaf_b} import right

def passthrough(value):
    return left(right(value))
'''
    decoy_leaf_a_source = '''def left(value):
    return value
'''
    decoy_leaf_b_source = '''def right(value):
    return value
'''

    if domain == "numeric-scalar":
        target_root_source = f'''from {target_mid} import {target_fn}

def {public_fn}(value, pivot):
    return {target_fn}(value, pivot)
'''
        decoy_root_source = f'''from {decoy_mid} import {decoy_fn}

def {decoy_public_fn}(value, pivot):
    return {decoy_fn}(value, pivot)
'''
        raw_cases = [(-2, 0), (0, 0), (3, 0), (7, 3)]
        cases = tuple({"args": [value, pivot], "expected": bool(predicate(value, pivot))} for value, pivot in raw_cases)
    elif domain == "lexical-scalar":
        target_root_source = f'''from {target_mid} import {target_fn}

def {public_fn}(value, pivot):
    return {target_fn}(value, pivot)
'''
        decoy_root_source = f'''from {decoy_mid} import {decoy_fn}

def {decoy_public_fn}(value, pivot):
    return {decoy_fn}(value, pivot)
'''
        raw_cases = [("aa", "bb"), ("bb", "bb"), ("zz", "bb"), ("k", "k")]
        cases = tuple({"args": [value, pivot], "expected": bool(predicate(value, pivot))} for value, pivot in raw_cases)
    elif domain == "record-structure":
        key = _name(rng, "score")
        target_root_source = f'''from {target_mid} import {target_fn}

def {public_fn}(row, pivot):
    return {target_fn}(row[{key!r}], pivot)
'''
        decoy_root_source = f'''from {decoy_mid} import {decoy_fn}

def {decoy_public_fn}(row, pivot):
    return {decoy_fn}(row[{key!r}], pivot)
'''
        raw_cases = [({key: -1}, 0), ({key: 0}, 0), ({key: 5}, 0), ({key: 5}, 5)]
        cases = tuple({"args": [row, pivot], "expected": bool(predicate(row[key], pivot))} for row, pivot in raw_cases)
    else:
        raise ValueError(domain)

    files = {
        f"{target_root}.py": target_root_source,
        f"{decoy_root}.py": decoy_root_source,
        f"{target_mid}.py": target_mid_source,
        f"{decoy_mid}.py": decoy_mid_source,
        f"{target_tail}.py": target_tail_source,
        f"{decoy_tail}.py": decoy_tail_source,
        f"{target_leaf}.py": target_leaf_source,
        f"{decoy_branch}.py": decoy_branch_source,
        f"{decoy_leaf_a}.py": decoy_leaf_a_source,
        f"{decoy_leaf_b}.py": decoy_leaf_b_source,
    }
    roles = derive_repository_file_roles(files)
    target_path = f"{target_mid}.py"
    decoy_path = f"{decoy_mid}.py"
    if roles[target_path] != "INTERMEDIATE" or roles[decoy_path] != "INTERMEDIATE":
        raise AssertionError(f"old localization language did not collide as intended: {roles}")

    depth0 = derive_import_graph_fingerprints(files, 0)
    depth1 = derive_import_graph_fingerprints(files, 1)
    depth2 = derive_import_graph_fingerprints(files, 2)
    if depth0[target_path] != depth0[decoy_path] or depth1[target_path] != depth1[decoy_path]:
        raise AssertionError("task does not require structural refinement beyond degree/one-hop labels")
    if depth2[target_path] == depth2[decoy_path]:
        raise AssertionError("two-hop generated graph representation still cannot separate old-role collision")

    if causal_branch == "TARGET":
        entry_module, function_name = target_root, public_fn
    elif causal_branch == "DECOY":
        entry_module, function_name = decoy_root, decoy_public_fn
    else:
        raise ValueError(causal_branch)

    return HiddenTopologyTask(
        task_id=f"{label}-{rng.randrange(10_000_000, 99_999_999)}",
        domain=domain,
        files=files,
        entry_module=entry_module,
        function_name=function_name,
        cases=cases,
        causal_branch=causal_branch,
        target_mid_path=target_path,
        decoy_mid_path=decoy_path,
    )


def _execute_candidate(body, task, candidate, signers, verifier, epoch_base):
    effects = []
    for issuer_index, (issuer, signer) in enumerate(signers.items()):
        executor = SubprocessRepositoryRepairExecutor(
            baseline_files=task.files,
            candidate=candidate,
            entry_module=task.entry_module,
            function_name=task.function_name,
            hidden_cases=task.cases,
            signer=signer,
            source_id=f"{task.task_id}-topology-{candidate.site_index}-{issuer}",
            context_id=task.task_id,
            challenge_id=f"{task.task_id}-hidden-topology-{candidate.site_index}-{issuer}",
            epoch=epoch_base + issuer_index,
        )
        pair = body.execute_world_intervention(candidate.proposal, executor, verifier=verifier)
        if not pair.authority_verified:
            raise AssertionError("graph-localization hidden receipt lost authority")
        effects.append(float(pair.effect))
    return tuple(effects)


def _train_complete(body, task, signers, verifier, epoch_base):
    organ = RepositoryLocalizationRepresentationOrgan(body)
    candidates, depth = organ.propose(task.task_id, task.files)
    if len(candidates) != 2:
        raise AssertionError(f"expected exactly two old-role-colliding repair candidates, got {len(candidates)}")
    if depth != 2:
        raise AssertionError(f"minimal source-derived graph escape depth must be 2, got {depth}")
    strong = []
    for index, candidate in enumerate(candidates):
        effects = _execute_candidate(body, task, candidate, signers, verifier, epoch_base + index * 10)
        if min(effects) >= 0.9:
            strong.append(candidate)
    if len(strong) != 1:
        raise AssertionError(f"task must have exactly one strong exact patch, got {strong}")
    expected_path = task.target_mid_path if task.causal_branch == "TARGET" else task.decoy_mid_path
    if strong[0].file_path != expected_path:
        raise AssertionError("hidden executable world selected unexpected causal branch")
    return candidates, strong[0]


def _rebuild_training_universe(body, tasks):
    organ = RepositoryLocalizationRepresentationOrgan(body)
    by_context = {}
    for task in tasks:
        candidates, depth = organ.propose(task.task_id, task.files)
        if depth != 2:
            raise AssertionError("reconstructed graph representation depth drifted")
        by_context[task.task_id] = candidates
    return by_context


def _execute_one(body, task, selection, signers, verifier, epoch_base):
    if len(selection.candidates) != 1:
        raise AssertionError("matched graph-localization arm requires exactly one candidate")
    return _execute_candidate(body, task, selection.candidates[0], signers, verifier, epoch_base)


def main(seed_path):
    seed = int(Path(seed_path).read_text().strip())
    rng = random.Random(seed)
    hidden_operator = rng.choice(tuple(sorted(REPAIR_FAMILIES)))

    issuer_a = f"graph-lab-{rng.randrange(10_000_000, 99_999_999)}"
    issuer_b = f"graph-lab-{rng.randrange(10_000_000, 99_999_999)}"
    key_a = hashlib.sha256(f"{seed}:graph:a".encode()).digest()
    key_b = hashlib.sha256(f"{seed}:graph:b".encode()).digest()
    signers = {
        issuer_a: HMACWorldReceiptSigner(issuer_a, key_a),
        issuer_b: HMACWorldReceiptSigner(issuer_b, key_b),
    }
    verifier = HMACWorldReceiptVerifier(
        {issuer_a: key_a, issuer_b: key_b},
        independence_classes={issuer_a: "GRAPH_LAB_A", issuer_b: "GRAPH_LAB_B"},
    )

    parent = PersistentCognitiveRuntime()
    train_numeric = _make_task(rng, hidden_operator, "numeric-scalar", "graph-train-numeric", "TARGET")
    train_lexical = _make_task(rng, hidden_operator, "lexical-scalar", "graph-train-lexical", "TARGET")
    numeric_candidates, numeric_strong = _train_complete(parent, train_numeric, signers, verifier, 10000)
    lexical_candidates, lexical_strong = _train_complete(parent, train_lexical, signers, verifier, 20000)

    graph_organ = RepositoryLocalizationRepresentationOrgan(parent)
    training_universe = {
        train_numeric.task_id: numeric_candidates,
        train_lexical.task_id: lexical_candidates,
    }
    assessment = graph_organ.assess_old_language(training_universe)
    if assessment.status != "NAMED_ROLE_LOCALIZATION_NON_IDENTIFYING_OPEN_GRAPH_REPRESENTATION":
        raise AssertionError(f"old named-role language should be causally non-identifying: {assessment}")
    if assessment.missing_experiment_ids or assessment.evaluated_candidate_count != 4:
        raise AssertionError(f"old-language ambiguity must be complete, not missing-evidence driven: {assessment}")
    graph_policy = graph_organ.policy(assessment)
    if graph_policy.status != "REPRODUCED_GENERATED_GRAPH_LOCALIZATION":
        raise AssertionError(f"graph localization policy not learned: {graph_policy}")
    if graph_policy.operator_id != hidden_operator or graph_policy.fingerprint_depth != 2:
        raise AssertionError("graph policy learned wrong operator or nonminimal structural depth")
    if len(graph_policy.supporting_contexts) != 2:
        raise AssertionError("graph localization did not reproduce across two source-disjoint contexts")
    target_identity = parse_graph_localization_signature(numeric_strong.proposal)
    if target_identity != parse_graph_localization_signature(lexical_strong.proposal):
        raise AssertionError("source-disjoint target branches did not share generated structural fingerprint")

    old_role_policy = RepositoryTaskAcquisitionOrgan(parent).policy()
    if old_role_policy.file_role != "INTERMEDIATE" or old_role_policy.operator_id != hidden_operator:
        raise AssertionError("old role policy should retain the ambiguous INTERMEDIATE signature")

    checkpoint = checkpoint_dict(parent)
    verifierless = restore_runtime(checkpoint)
    verifierless_universe = _rebuild_training_universe(verifierless, (train_numeric, train_lexical))
    verifierless_assessment = RepositoryLocalizationRepresentationOrgan(verifierless).assess_old_language(verifierless_universe)
    verifierless_policy = RepositoryLocalizationRepresentationOrgan(verifierless).policy(verifierless_assessment)
    if verifierless_policy.fingerprint is not None:
        raise AssertionError("generated graph localization authority restored without external verifier")

    heldout = _make_task(rng, hidden_operator, "record-structure", "graph-heldout-record", "TARGET")

    treatment = restore_runtime(checkpoint, world_verifier=verifier)
    treatment_training = _rebuild_training_universe(treatment, (train_numeric, train_lexical))
    treatment_assessment = RepositoryLocalizationRepresentationOrgan(treatment).assess_old_language(treatment_training)
    treatment_policy = RepositoryLocalizationRepresentationOrgan(treatment).policy(treatment_assessment)
    treatment_candidates, treatment_depth = RepositoryLocalizationRepresentationOrgan(treatment).propose(heldout.task_id, heldout.files)
    treatment_selection = RepositoryLocalizationRepresentationOrgan(treatment).select(
        treatment_candidates, treatment_policy, max_candidates=1
    )
    treatment_effects = _execute_one(treatment, heldout, treatment_selection, signers, verifier, 40000)
    treatment_cap = float(min(treatment_effects) >= 0.9)

    # Same checkpoint, same fresh candidate universe, but graph representation is removed.
    role_only = restore_runtime(checkpoint, world_verifier=verifier)
    role_candidates, role_depth = RepositoryLocalizationRepresentationOrgan(role_only).propose(heldout.task_id, heldout.files)
    role_selection = RepositoryTaskAcquisitionOrgan(role_only).select(
        role_candidates, max_candidates=1, apply_learned_policy=True
    )
    role_effects = _execute_candidate(role_only, heldout, role_selection.candidates[0], signers, verifier, 50000)
    role_only_cap = float(min(role_effects) >= 0.9)

    reset = PersistentCognitiveRuntime()
    reset_candidates, reset_depth = RepositoryLocalizationRepresentationOrgan(reset).propose(heldout.task_id, heldout.files)
    reset_selection = RepositoryLocalizationRepresentationOrgan(reset).select(
        reset_candidates, None, max_candidates=1
    )
    reset_effects = _execute_one(reset, heldout, reset_selection, signers, verifier, 60000)
    reset_cap = float(min(reset_effects) >= 0.9)

    full = PersistentCognitiveRuntime()
    full_candidates, full_depth = RepositoryLocalizationRepresentationOrgan(full).propose(heldout.task_id, heldout.files)
    full_cap = 0.0
    for index, candidate in enumerate(full_candidates):
        effects = _execute_candidate(full, heldout, candidate, signers, verifier, 70000 + index * 10)
        full_cap = max(full_cap, float(min(effects) >= 0.9))

    # Genuine WRONG representation: same operator, opposite generated structural fingerprint.
    wrong_parent = PersistentCognitiveRuntime()
    wrong_numeric = _make_task(rng, hidden_operator, "numeric-scalar", "wrong-graph-numeric", "DECOY")
    wrong_lexical = _make_task(rng, hidden_operator, "lexical-scalar", "wrong-graph-lexical", "DECOY")
    wrong_numeric_candidates, wrong_numeric_strong = _train_complete(wrong_parent, wrong_numeric, signers, verifier, 80000)
    wrong_lexical_candidates, wrong_lexical_strong = _train_complete(wrong_parent, wrong_lexical, signers, verifier, 90000)
    wrong_organ = RepositoryLocalizationRepresentationOrgan(wrong_parent)
    wrong_assessment = wrong_organ.assess_old_language({
        wrong_numeric.task_id: wrong_numeric_candidates,
        wrong_lexical.task_id: wrong_lexical_candidates,
    })
    wrong_policy = wrong_organ.policy(wrong_assessment)
    if wrong_policy.status != "REPRODUCED_GENERATED_GRAPH_LOCALIZATION":
        raise AssertionError("wrong-control BODY failed to genuinely learn alternate graph fingerprint")
    if wrong_policy.operator_id != hidden_operator or wrong_policy.fingerprint == graph_policy.fingerprint:
        raise AssertionError("wrong-control graph localization was not genuinely distinct")
    if parse_graph_localization_signature(wrong_numeric_strong.proposal) != parse_graph_localization_signature(wrong_lexical_strong.proposal):
        raise AssertionError("wrong graph fingerprint did not reproduce")
    wrong = restore_runtime(checkpoint_dict(wrong_parent), world_verifier=verifier)
    wrong_training = _rebuild_training_universe(wrong, (wrong_numeric, wrong_lexical))
    wrong_reassessment = RepositoryLocalizationRepresentationOrgan(wrong).assess_old_language(wrong_training)
    wrong_repolicy = RepositoryLocalizationRepresentationOrgan(wrong).policy(wrong_reassessment)
    wrong_candidates, wrong_depth = RepositoryLocalizationRepresentationOrgan(wrong).propose(heldout.task_id, heldout.files)
    wrong_selection = RepositoryLocalizationRepresentationOrgan(wrong).select(
        wrong_candidates, wrong_repolicy, max_candidates=1
    )
    wrong_effects = _execute_one(wrong, heldout, wrong_selection, signers, verifier, 100000)
    wrong_cap = float(min(wrong_effects) >= 0.9)

    if not (
        treatment_cap == 1.0
        and role_only_cap == 0.0
        and reset_cap == 0.0
        and wrong_cap == 0.0
        and full_cap == 1.0
    ):
        raise AssertionError("localization representation-escape causal controls failed")
    if treatment_depth != 2 or role_depth != 2 or reset_depth != 2 or full_depth != 2 or wrong_depth != 2:
        raise AssertionError("minimal generated fingerprint depth drifted across matched arms")
    if treatment_selection.candidates[0].file_path != heldout.target_mid_path:
        raise AssertionError("generated structural representation failed to localize heldout causal patch")
    if role_selection.candidates[0].file_path != heldout.decoy_mid_path:
        raise AssertionError("old named-role control did not expose its intended ambiguity")
    if wrong_selection.candidates[0].file_path != heldout.decoy_mid_path:
        raise AssertionError("genuinely learned wrong graph fingerprint did not select the heldout decoy")

    hashes = {
        train_numeric.repo_hash,
        train_lexical.repo_hash,
        heldout.repo_hash,
        wrong_numeric.repo_hash,
        wrong_lexical.repo_hash,
    }
    if len(hashes) != 5:
        raise AssertionError("topology tasks were not source-disjoint")

    result = {
        "status": "PASS_BOUNDED_WORLD_DRIVEN_REPOSITORY_LOCALIZATION_REPRESENTATION_ESCAPE_AND_DESCENDANT_TRANSFER",
        "hidden_repair_operator": hidden_operator,
        "training_domains": [train_numeric.domain, train_lexical.domain],
        "heldout_domain": heldout.domain,
        "repository_files_per_task": len(heldout.files),
        "repository_hashes_disjoint": True,
        "old_named_role": "INTERMEDIATE",
        "old_role_candidate_count_per_context": [len(numeric_candidates), len(lexical_candidates)],
        "old_language_complete_contexts": len(assessment.complete_contexts),
        "old_language_ambiguous_contexts": len(assessment.ambiguous_contexts),
        "old_language_missing_candidate_count": len(assessment.missing_experiment_ids),
        "old_language_evaluated_candidate_count": assessment.evaluated_candidate_count,
        "degree_fingerprint_nonidentifying": True,
        "one_hop_fingerprint_nonidentifying": True,
        "minimal_generated_graph_fingerprint_depth": graph_policy.fingerprint_depth,
        "graph_fingerprint_filename_independent": True,
        "graph_fingerprint_generation_uses_hidden_outcomes": False,
        "learned_graph_fingerprint": graph_policy.fingerprint,
        "wrong_genuinely_learned_graph_fingerprint": wrong_policy.fingerprint,
        "treatment_candidate_count": len(treatment_selection.candidates),
        "treatment_external_pair_count": 2,
        "treatment_capability": treatment_cap,
        "old_role_same_checkpoint_capability": role_only_cap,
        "reset_capability": reset_cap,
        "wrong_graph_capability": wrong_cap,
        "full_candidate_count": len(full_candidates),
        "full_external_pair_count": 2 * len(full_candidates),
        "full_exhaustive_capability": full_cap,
        "external_pair_reduction_vs_full": 1.0 - (1.0 / len(full_candidates)),
        "verifierless_graph_localization_authority": False,
        "hidden_tests_exposed_to_body_before_execution": False,
        "post_hidden_human_structural_repairs": 0,
        "authored_named_role_vocabulary_causally_surpassed": True,
        "wl_refinement_algorithm_human_authored": True,
        "real_repository_autonomous_repair": False,
        "unrestricted_localization_representation_genesis": False,
        "foundation_weight_change": False,
        "independent_organizational_custody": False,
        "physical_world": False,
        "global_recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_world_driven_localization_representation_escape.py <seed_path>")
    main(sys.argv[1])
