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
from arte_cognition.repository_repair import RepositoryRepairOrgan, SubprocessRepositoryRepairExecutor, repository_hash
from arte_cognition.world_coupling import HMACWorldReceiptSigner, HMACWorldReceiptVerifier


REPAIRS = {
    "COMPARE::GtE->Gt": (">=", operator.gt),
    "COMPARE::LtE->Lt": ("<=", operator.lt),
}


@dataclass(frozen=True)
class RepoTask:
    task_id: str
    domain: str
    package: str
    files: dict[str, str]
    module: str
    function_name: str
    cases: tuple[dict, ...]

    @property
    def repo_hash(self) -> str:
        return repository_hash(self.files)


def _name(rng, prefix):
    return f"{prefix}_{rng.randrange(10**7, 10**8)}"


def make_repo(rng, repair, label, domain, bug_role="LEAF"):
    buggy, correct = REPAIRS[repair]
    package = _name(rng, "pkg")
    helper = f"{package}/helper.py"
    api = f"{package}/api.py"
    intermediate = f"{package}/a_decoy.py"
    spare = f"{package}/b_decoy.py"
    leaf = f"{package}/z_logic.py"

    intermediate_source = f'''from .helper import SENTINEL

def distract(value, threshold):
    return value {buggy} threshold
'''
    leaf_source = f'''def accept(value, threshold):
    return value {buggy} threshold
'''
    spare_source = f'''def spare(value, threshold):
    return value {buggy} threshold
'''

    if bug_role == "LEAF":
        callee_module, callee_fn = "z_logic", "accept"
    elif bug_role == "INTERMEDIATE":
        callee_module, callee_fn = "a_decoy", "distract"
    else:
        raise ValueError("unsupported bug role")

    if domain == "numeric":
        api_source = f'''from .{callee_module} import {callee_fn} as _impl
from .z_logic import accept as _leaf
from .a_decoy import distract as _middle

def decide(value, threshold):
    return _impl(value, threshold)
'''
        raw = [(1, 1), (2, 1), (-2, 0)]
        cases = tuple({"args": [a, b], "expected": bool(correct(a, b))} for a, b in raw)
    elif domain == "lexical":
        api_source = f'''from .{callee_module} import {callee_fn} as _impl
from .z_logic import accept as _leaf
from .a_decoy import distract as _middle

def decide(value, threshold):
    return _impl(value, threshold)
'''
        raw = [("m", "m"), ("z", "m"), ("a", "m")]
        cases = tuple({"args": [a, b], "expected": bool(correct(a, b))} for a, b in raw)
    elif domain == "record":
        key = f"score_{rng.randrange(100, 999)}"
        api_source = f'''from .{callee_module} import {callee_fn} as _impl
from .z_logic import accept as _leaf
from .a_decoy import distract as _middle

def decide(row, threshold):
    return _impl(row[{key!r}], threshold)
'''
        raw = [({key: 4}, 4), ({key: 9}, 4), ({key: -1}, 4)]
        cases = tuple({"args": [row, b], "expected": bool(correct(row[key], b))} for row, b in raw)
    else:
        raise ValueError("unsupported domain")

    files = {
        f"{package}/__init__.py": "",
        helper: "SENTINEL = 1\n",
        api: api_source,
        intermediate: intermediate_source,
        spare: spare_source,
        leaf: leaf_source,
    }
    return RepoTask(
        task_id=f"{label}-{rng.randrange(10**7, 10**8)}",
        domain=domain,
        package=package,
        files=files,
        module=f"{package}.api",
        function_name="decide",
        cases=cases,
    )


def execute_candidate(body, task, candidate, signers, verifier, epoch):
    effects = []
    patched = dict(candidate.patched_files)
    for index, (issuer, signer) in enumerate(signers.items()):
        executor = SubprocessRepositoryRepairExecutor(
            baseline_files=task.files,
            patched_files=patched,
            module=task.module,
            function_name=task.function_name,
            hidden_cases=task.cases,
            signer=signer,
            source_id=f"{task.task_id}-repo-{issuer}",
            context_id=task.task_id,
            challenge_id=f"{task.task_id}-hidden-ci-{candidate.target_path}-{issuer}",
            epoch=epoch + index,
        )
        pair = body.execute_world_intervention(candidate.proposal, executor, verifier=verifier)
        if not pair.authority_verified:
            raise AssertionError("repository repair world receipt lost authority")
        effects.append(float(pair.effect))
    return tuple(effects)


def train_localization(body, tasks, signers, verifier, epoch):
    all_candidates = []
    strong = []
    organ = RepositoryRepairOrgan(body)
    for task_index, task in enumerate(tasks):
        candidates = organ.propose(task.task_id, task.files)
        if len(candidates) < 3:
            raise AssertionError("repository task did not expose multi-file localization decoys")
        task_strong = []
        for candidate_index, candidate in enumerate(candidates):
            effects = execute_candidate(
                body, task, candidate, signers, verifier,
                epoch + task_index * 1000 + candidate_index * 10,
            )
            if min(effects) >= 0.9:
                task_strong.append(candidate)
        if len(task_strong) != 1:
            raise AssertionError(f"repository task must have exactly one strong patch: {task_strong}")
        all_candidates.extend(candidates)
        strong.extend(task_strong)
    policy = organ.policy()
    if policy.location_signature is None or len(policy.supporting_contexts) < 2:
        raise AssertionError(f"repository localization policy not learned: {policy}")
    if any(item.location_signature != policy.location_signature for item in strong):
        raise AssertionError("learned repository localization signature mismatched strong training files")
    return tuple(all_candidates), tuple(strong), policy


def main(seed_path):
    seed = int(Path(seed_path).read_text().strip())
    rng = random.Random(seed)
    hidden = rng.choice(tuple(sorted(REPAIRS)))

    issuer_a = f"repo-lab-{rng.randrange(10**7, 10**8)}"
    issuer_b = f"repo-lab-{rng.randrange(10**7, 10**8)}"
    key_a = hashlib.sha256(f"{seed}:repo:a".encode()).digest()
    key_b = hashlib.sha256(f"{seed}:repo:b".encode()).digest()
    signers = {
        issuer_a: HMACWorldReceiptSigner(issuer_a, key_a),
        issuer_b: HMACWorldReceiptSigner(issuer_b, key_b),
    }
    verifier = HMACWorldReceiptVerifier(
        {issuer_a: key_a, issuer_b: key_b},
        independence_classes={issuer_a: "REPO_LAB_A", issuer_b: "REPO_LAB_B"},
    )

    parent = PersistentCognitiveRuntime()
    train_a = make_repo(rng, hidden, "repo-train-numeric", "numeric", bug_role="LEAF")
    train_b = make_repo(rng, hidden, "repo-train-lexical", "lexical", bug_role="LEAF")
    _, strong, learned = train_localization(parent, (train_a, train_b), signers, verifier, 10000)

    checkpoint = checkpoint_dict(parent)
    verifierless = restore_runtime(checkpoint)
    if RepositoryRepairOrgan(verifierless).policy().location_signature is not None:
        raise AssertionError("repository localization authority restored without verifier")

    heldout = make_repo(rng, hidden, "repo-heldout-record", "record", bug_role="LEAF")
    treatment = restore_runtime(checkpoint, world_verifier=verifier)
    remove = restore_runtime(checkpoint, world_verifier=verifier)
    treatment_candidates = RepositoryRepairOrgan(treatment).propose(heldout.task_id, heldout.files)
    remove_candidates = RepositoryRepairOrgan(remove).propose(heldout.task_id, heldout.files)
    treatment_selection = RepositoryRepairOrgan(treatment).select(
        treatment_candidates, max_candidates=1, apply_learned_localization=True
    )
    remove_selection = RepositoryRepairOrgan(remove).select(
        remove_candidates, max_candidates=1, apply_learned_localization=False
    )
    treatment_effect = execute_candidate(
        treatment, heldout, treatment_selection.candidates[0], signers, verifier, 50000
    )
    remove_effect = execute_candidate(
        remove, heldout, remove_selection.candidates[0], signers, verifier, 50000
    )
    treatment_capability = float(min(treatment_effect) >= 0.9)
    remove_capability = float(min(remove_effect) >= 0.9)

    full = restore_runtime(checkpoint, world_verifier=verifier)
    full_candidates = RepositoryRepairOrgan(full).propose(heldout.task_id, heldout.files)
    full_capability = 0.0
    for index, candidate in enumerate(full_candidates):
        effects = execute_candidate(full, heldout, candidate, signers, verifier, 60000 + index * 10)
        full_capability = max(full_capability, float(min(effects) >= 0.9))

    reset = PersistentCognitiveRuntime()
    reset_candidates = RepositoryRepairOrgan(reset).propose(heldout.task_id, heldout.files)
    reset_selection = RepositoryRepairOrgan(reset).select(
        reset_candidates, max_candidates=1, apply_learned_localization=True
    )
    reset_effect = execute_candidate(
        reset, heldout, reset_selection.candidates[0], signers, verifier, 70000
    )
    reset_capability = float(min(reset_effect) >= 0.9)

    wrong = PersistentCognitiveRuntime()
    wrong_a = make_repo(rng, hidden, "wrong-repo-numeric", "numeric", bug_role="INTERMEDIATE")
    wrong_b = make_repo(rng, hidden, "wrong-repo-lexical", "lexical", bug_role="INTERMEDIATE")
    _, wrong_strong, wrong_policy = train_localization(
        wrong, (wrong_a, wrong_b), signers, verifier, 80000
    )
    if wrong_policy.location_signature == learned.location_signature:
        raise AssertionError("wrong-control BODY learned the same localization signature")
    wrong = restore_runtime(checkpoint_dict(wrong), world_verifier=verifier)
    wrong_candidates = RepositoryRepairOrgan(wrong).propose(heldout.task_id, heldout.files)
    wrong_selection = RepositoryRepairOrgan(wrong).select(
        wrong_candidates, max_candidates=1, apply_learned_localization=True
    )
    wrong_effect = execute_candidate(
        wrong, heldout, wrong_selection.candidates[0], signers, verifier, 110000
    )
    wrong_capability = float(min(wrong_effect) >= 0.9)

    if not (
        treatment_capability == 1.0
        and remove_capability == 0.0
        and reset_capability == 0.0
        and wrong_capability == 0.0
        and full_capability == 1.0
    ):
        raise AssertionError("repository localization causal controls failed")
    if treatment_selection.candidates[0].target_path == remove_selection.candidates[0].target_path:
        raise AssertionError("REMOVE did not remove learned repository localization")
    if wrong_selection.candidates[0].target_path == treatment_selection.candidates[0].target_path:
        raise AssertionError("genuinely trained wrong localization selected the treatment file")
    if len({train_a.repo_hash, train_b.repo_hash, heldout.repo_hash}) != 3:
        raise AssertionError("train/heldout repositories were not source-disjoint")
    if min(len(train_a.files), len(train_b.files), len(heldout.files)) < 5:
        raise AssertionError("repository tasks were not genuinely multi-file")

    result = {
        "status": "PASS_BOUNDED_MULTI_FILE_REPOSITORY_LOCALIZATION_PATCH_SYNTHESIS_AND_HIDDEN_CI_TRANSFER",
        "hidden_repair_operator": hidden,
        "training_domains": [train_a.domain, train_b.domain],
        "heldout_domain": heldout.domain,
        "repository_hashes_disjoint": True,
        "repository_file_count": len(heldout.files),
        "heldout_patch_candidate_count": len(full_candidates),
        "learned_location_signature": learned.location_signature,
        "wrong_location_signature": wrong_policy.location_signature,
        "treatment_target_path": treatment_selection.candidates[0].target_path,
        "remove_target_path": remove_selection.candidates[0].target_path,
        "wrong_target_path": wrong_selection.candidates[0].target_path,
        "treatment_capability": treatment_capability,
        "remove_same_checkpoint_localization_capability": remove_capability,
        "reset_capability": reset_capability,
        "wrong_genuinely_trained_localization_capability": wrong_capability,
        "full_exhaustive_capability": full_capability,
        "treatment_external_pairs": 2,
        "full_external_pairs": 2 * len(full_candidates),
        "external_pair_reduction_vs_full": 1.0 - (1.0 / len(full_candidates)),
        "verifierless_repository_localization_authority": False,
        "external_execution": "separate_python_subprocess_over_materialized_multifile_repository",
        "hidden_tests_exposed_to_body_before_execution": False,
        "patch_generation_uses_hidden_outcomes": False,
        "localization_features_source_derived": True,
        "repository_scale_arbitrary_bug_repair": False,
        "unrestricted_software_operator_invention": False,
        "post_hidden_human_structural_repairs": 0,
        "foundation_weight_change": False,
        "physical_world": False,
        "independent_organizational_custody": False,
        "global_recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_multifile_repository_repair.py <seed_path>")
    main(sys.argv[1])
