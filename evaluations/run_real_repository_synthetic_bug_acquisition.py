from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.software_task_acquisition import SoftwareTaskAcquisitionOrgan, SoftwarePatchCandidate
from arte_cognition.world_coupling import HMACWorldReceiptSigner, HMACWorldReceiptVerifier, WorldOutcomeReceipt

EXPECTED_REPAIR_OPERATOR = "COMPARE::Eq->NotEq"


@dataclass(frozen=True)
class RealRepositoryTask:
    task_id: str
    relative_path: str
    module_name: str
    pristine_source: str
    buggy_source: str

    @property
    def pristine_hash(self) -> str:
        return hashlib.sha256(self.pristine_source.encode()).hexdigest()

    @property
    def buggy_hash(self) -> str:
        return hashlib.sha256(self.buggy_source.encode()).hexdigest()


class _AuthorityPredicateBugInjector(ast.NodeTransformer):
    """Evaluator-only semantic mutation; BODY never receives pristine source or location."""

    def __init__(self) -> None:
        self.matches = 0

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        if (
            len(node.ops) == 1
            and isinstance(node.ops[0], ast.NotEq)
            and len(node.comparators) == 1
            and isinstance(node.left, ast.Attribute)
            and node.left.attr == "independence_class_id"
            and isinstance(node.comparators[0], ast.Constant)
            and node.comparators[0].value == "UNVERIFIED"
        ):
            self.matches += 1
            node.ops[0] = ast.Eq()
        return node


def inject_hidden_authority_bug(source: str) -> str:
    tree = ast.parse(source)
    injector = _AuthorityPredicateBugInjector()
    tree = injector.visit(tree)
    ast.fix_missing_locations(tree)
    if injector.matches != 1:
        raise AssertionError(f"expected one authority predicate target, got {injector.matches}")
    return ast.unparse(tree) + "\n"


class RealRepositoryAuthorityExecutor:
    """Evaluate a candidate in a fresh copy of the actual production package."""

    HARNESS = r'''
from types import SimpleNamespace
import importlib, json, sys
try:
    fn = getattr(importlib.import_module(sys.argv[1]), "_authoritative")
    cases = [
        (SimpleNamespace(matched_budget=True, externally_generated=True, authority_verified=True, independence_class_id="REAL_A"), True),
        (SimpleNamespace(matched_budget=True, externally_generated=True, authority_verified=True, independence_class_id="UNVERIFIED"), False),
        (SimpleNamespace(matched_budget=False, externally_generated=True, authority_verified=True, independence_class_id="REAL_A"), False),
        (SimpleNamespace(matched_budget=True, externally_generated=False, authority_verified=True, independence_class_id="REAL_A"), False),
        (SimpleNamespace(matched_budget=True, externally_generated=True, authority_verified=False, independence_class_id="REAL_A"), False),
    ]
    print(json.dumps({"ok": all(bool(fn(x)) is expected for x, expected in cases)}))
except Exception:
    print(json.dumps({"ok": False}))
'''

    def __init__(self, task, candidate, signer, source_id, context_id, challenge_id, epoch):
        self.task = task
        self.candidate = candidate
        self.signer = signer
        self.source_id = str(source_id)
        self.context_id = str(context_id)
        self.challenge_id = str(challenge_id)
        self.epoch = int(epoch)

    @staticmethod
    def run_source(task: RealRepositoryTask, source: str) -> float:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "arte_cognition", root / "arte_cognition")
            target = root / task.relative_path
            target.write_text(source, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", RealRepositoryAuthorityExecutor.HARNESS, task.module_name],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=8,
                    check=False,
                )
                if completed.returncode != 0:
                    return 0.0
                return 1.0 if json.loads(completed.stdout.strip().splitlines()[-1]).get("ok") else 0.0
            except Exception:
                return 0.0

    def execute(self, proposal, arm: str, value: float) -> WorldOutcomeReceipt:
        source = self.task.buggy_source if str(arm).upper() == "LOW" else self.candidate.patched_source
        outcome = self.run_source(self.task, source)
        return self.signer.sign(WorldOutcomeReceipt(
            receipt_id=f"{self.challenge_id}::{proposal.experiment_id}::{arm}",
            experiment_id=proposal.experiment_id,
            axis_id=proposal.axis_id,
            arm=str(arm).upper(),
            intervention_value=float(value),
            outcome=float(outcome),
            source_id=self.source_id,
            context_id=self.context_id,
            challenge_id=self.challenge_id,
            epoch=self.epoch,
            budget_token=f"real-repository-hidden-synthetic-bug::{self.challenge_id}",
            externally_generated=True,
        ))


def make_task(path: str, module: str, task_id: str) -> RealRepositoryTask:
    pristine = (ROOT / path).read_text(encoding="utf-8")
    task = RealRepositoryTask(task_id, path, module, pristine, inject_hidden_authority_bug(pristine))
    if RealRepositoryAuthorityExecutor.run_source(task, task.pristine_source) != 1.0:
        raise AssertionError(f"pristine production behavior not green: {path}")
    if RealRepositoryAuthorityExecutor.run_source(task, task.buggy_source) != 0.0:
        raise AssertionError(f"hidden synthetic mutation did not break behavior: {path}")
    return task


def execute_candidate(body, task, candidate, signers, verifier, epoch_base):
    effects = []
    token = hashlib.sha256(
        f"{task.relative_path}|{candidate.site_index}|{candidate.proposal.experiment_id}".encode()
    ).hexdigest()[:16]
    for issuer_index, (issuer, signer) in enumerate(signers.items()):
        executor = RealRepositoryAuthorityExecutor(
            task,
            candidate,
            signer,
            f"real-repo::{task.relative_path}::{token}::{issuer}",
            task.task_id,
            f"real-repo-hidden::{task.task_id}::{token}::{issuer}",
            epoch_base + issuer_index,
        )
        pair = body.execute_world_intervention(candidate.proposal, executor, verifier=verifier)
        if not pair.authority_verified:
            raise AssertionError("real-repository receipt lost authority")
        effects.append(float(pair.effect))
    return tuple(effects)


def execute_candidates(body, task, candidates, signers, verifier, epoch_base):
    strong = []
    for index, candidate in enumerate(candidates):
        effects = execute_candidate(body, task, candidate, signers, verifier, epoch_base + index * 10)
        if min(effects) >= 0.9:
            strong.append(candidate)
    return tuple(strong)


def propose(body, task):
    candidates = SoftwareTaskAcquisitionOrgan(body).propose(task.task_id, task.buggy_source)
    if not candidates:
        raise AssertionError(f"no AST repair candidates for actual source: {task.relative_path}")
    return candidates


def main():
    # These files are actual production modules from the checked-out repository.
    # The first is intentionally the smaller discovery context; the second uses the
    # first world's result only as a provisional search-order hint, not action authority.
    tasks = (
        make_task(
            "arte_cognition/repository_patch_composition.py",
            "arte_cognition.repository_patch_composition",
            "real-train-patch-composition",
        ),
        make_task(
            "arte_cognition/repository_patch_cardinality.py",
            "arte_cognition.repository_patch_cardinality",
            "real-train-patch-cardinality",
        ),
        make_task(
            "arte_cognition/repository_task_acquisition.py",
            "arte_cognition.repository_task_acquisition",
            "real-heldout-repository-acquisition",
        ),
    )
    if len({task.pristine_hash for task in tasks}) != 3:
        raise AssertionError("actual production source hashes are not distinct")

    issuer_a, issuer_b = "REAL_REPO_EVAL_A", "REAL_REPO_EVAL_B"
    key_a = hashlib.sha256(b"real-repo-hidden-a").digest()
    key_b = hashlib.sha256(b"real-repo-hidden-b").digest()
    signers = {
        issuer_a: HMACWorldReceiptSigner(issuer_a, key_a),
        issuer_b: HMACWorldReceiptSigner(issuer_b, key_b),
    }
    verifier = HMACWorldReceiptVerifier(
        {issuer_a: key_a, issuer_b: key_b},
        independence_classes={issuer_a: "REAL_REPO_CLASS_A", issuer_b: "REAL_REPO_CLASS_B"},
    )

    parent = PersistentCognitiveRuntime()

    # Developmental context 1: no repair knowledge yet, so exhaust the actual source surface.
    first_candidates = propose(parent, tasks[0])
    first_strong = execute_candidates(parent, tasks[0], first_candidates, signers, verifier, 10000)
    if len(first_strong) != 1 or first_strong[0].operator_id != EXPECTED_REPAIR_OPERATOR:
        raise AssertionError(
            f"first actual module did not yield one expected repair: "
            f"{[(c.site_index, c.operator_id) for c in first_strong]}"
        )
    provisional_operator = first_strong[0].operator_id

    # Context 2: one-context evidence may guide exploration but cannot yet become
    # reproduced authority. Execute only the matching operator family on the second
    # actual module, then require independent reproduction before policy promotion.
    second_candidates = propose(parent, tasks[1])
    second_search = tuple(c for c in second_candidates if c.operator_id == provisional_operator)
    if not second_search:
        raise AssertionError("provisional operator absent from second actual module")
    second_strong = execute_candidates(parent, tasks[1], second_search, signers, verifier, 20000)
    if len(second_strong) != 1 or second_strong[0].operator_id != provisional_operator:
        raise AssertionError(
            f"provisional repair failed real cross-module reproduction: "
            f"{[(c.site_index, c.operator_id) for c in second_strong]}"
        )

    policy = SoftwareTaskAcquisitionOrgan(parent).policy()
    if (
        policy.status != "REPRODUCED_SOFTWARE_REPAIR_OPERATOR"
        or policy.operator_id != EXPECTED_REPAIR_OPERATOR
        or len(policy.supporting_contexts) != 2
    ):
        raise AssertionError(f"real-repository operator not promoted after reproduction: {policy}")

    checkpoint = checkpoint_dict(parent)
    verifierless = restore_runtime(checkpoint)
    if SoftwareTaskAcquisitionOrgan(verifierless).policy().operator_id is not None:
        raise AssertionError("repair policy self-authorized without external verifier")

    heldout = tasks[2]
    treatment = restore_runtime(checkpoint, world_verifier=verifier)
    treatment_organ = SoftwareTaskAcquisitionOrgan(treatment)
    treatment_candidates = propose(treatment, heldout)
    matching_count = sum(c.operator_id == EXPECTED_REPAIR_OPERATOR for c in treatment_candidates)
    if matching_count < 1:
        raise AssertionError("heldout production source lacks learned-operator candidate")
    treatment_selection = treatment_organ.select(
        treatment_candidates, max_candidates=matching_count, apply_learned_policy=True
    )
    treatment_strong = execute_candidates(
        treatment, heldout, treatment_selection.candidates, signers, verifier, 40000
    )

    remove = restore_runtime(checkpoint, world_verifier=verifier)
    remove_candidates = propose(remove, heldout)
    remove_selection = SoftwareTaskAcquisitionOrgan(remove).select(
        remove_candidates, max_candidates=matching_count, apply_learned_policy=False
    )
    remove_strong = execute_candidates(
        remove, heldout, remove_selection.candidates, signers, verifier, 50000
    )

    full = PersistentCognitiveRuntime()
    full_candidates = propose(full, heldout)
    full_strong = execute_candidates(full, heldout, full_candidates, signers, verifier, 60000)

    if len(treatment_strong) != 1 or treatment_strong[0].operator_id != EXPECTED_REPAIR_OPERATOR:
        raise AssertionError("learned repair operator failed fresh actual production module")
    if remove_strong:
        raise AssertionError("same-budget REMOVE found actual repair without learned operator")
    if len(full_strong) != 1 or full_strong[0].operator_id != EXPECTED_REPAIR_OPERATOR:
        raise AssertionError("full actual-source search did not recover a unique repair")
    if len(full_candidates) <= matching_count:
        raise AssertionError("learned operator did not contract heldout actual-source search")

    result = {
        "status": "PASS_BOUNDED_REAL_REPOSITORY_SYNTHETIC_HIDDEN_BUG_REPAIR_OPERATOR_ACQUISITION_AND_TRANSFER",
        "repository": "leehyen0/GENESIS-EX-NIHILO",
        "real_repository_source": True,
        "natural_historical_bug": False,
        "synthetic_bug_injected_post_checkout": True,
        "synthetic_bug_semantics": "independence_class_id != UNVERIFIED -> == UNVERIFIED",
        "bug_location_exposed_to_body": False,
        "pristine_source_exposed_to_body_as_task_input": False,
        "production_paths": [task.relative_path for task in tasks],
        "production_source_hashes": [task.pristine_hash for task in tasks],
        "buggy_source_hashes": [task.buggy_hash for task in tasks],
        "cross_module_source_hashes_distinct": True,
        "pristine_behavior_verified_before_body": True,
        "buggy_behavior_verified_failed_before_body": True,
        "first_training_full_candidate_count": len(first_candidates),
        "first_training_executed_candidate_count": len(first_candidates),
        "first_training_strong_site_index": first_strong[0].site_index,
        "second_training_full_candidate_count": len(second_candidates),
        "second_training_provisional_operator_candidate_count": len(second_search),
        "second_training_executed_candidate_count": len(second_search),
        "second_training_search_guided_only_by_first_world_outcome": True,
        "learned_operator": policy.operator_id,
        "learned_operator_supporting_contexts": len(policy.supporting_contexts),
        "heldout_path": heldout.relative_path,
        "heldout_full_candidate_count": len(full_candidates),
        "heldout_learned_operator_candidate_count": matching_count,
        "treatment_candidate_count": len(treatment_selection.candidates),
        "treatment_external_pair_count": 2 * len(treatment_selection.candidates),
        "treatment_capability": 1.0,
        "remove_candidate_count": len(remove_selection.candidates),
        "remove_external_pair_count": 2 * len(remove_selection.candidates),
        "remove_same_checkpoint_capability": 0.0,
        "full_external_pair_count": 2 * len(full_candidates),
        "full_exhaustive_capability": 1.0,
        "external_pair_reduction_vs_full": 1.0 - (len(treatment_selection.candidates) / len(full_candidates)),
        "verifierless_policy_authority": False,
        "external_execution": "fresh_temp_copy_of_actual_arte_cognition_package_imported_in_python_subprocess",
        "hidden_behavior_exposed_to_body_before_execution": False,
        "repair_candidates_generated_from_buggy_production_source_ast": True,
        "post_hidden_human_structural_repairs": 0,
        "repair_mutation_alphabet_human_authored": True,
        "duplicate_authority_predicate_mechanism_across_modules": True,
        "arbitrary_real_repository_bug_repair": False,
        "unrestricted_software_operator_invention": False,
        "foundation_weight_change": False,
        "independent_organizational_custody": False,
        "physical_world": False,
        "global_recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
