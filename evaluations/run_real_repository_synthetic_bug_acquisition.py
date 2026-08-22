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
        return hashlib.sha256(self.pristine_source.encode("utf-8")).hexdigest()

    @property
    def buggy_hash(self) -> str:
        return hashlib.sha256(self.buggy_source.encode("utf-8")).hexdigest()


class _AuthorityPredicateBugInjector(ast.NodeTransformer):
    """Evaluator-owned synthetic bug: verified class != UNVERIFIED becomes ==.

    The semantic target is intentionally defined without line numbers or formatting
    so post-checkout source formatting cannot reveal or drift the challenge. Exactly
    one target must exist in each frozen production module.
    """

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
    mutated = injector.visit(tree)
    ast.fix_missing_locations(mutated)
    if injector.matches != 1:
        raise AssertionError(f"expected exactly one authority predicate target, got {injector.matches}")
    return ast.unparse(mutated) + "\n"


class RealRepositoryAuthorityExecutor:
    """Run one candidate inside a fresh copy of the actual arte_cognition package."""

    _HARNESS = r'''
from types import SimpleNamespace
import importlib, json, sys
module_name = sys.argv[1]
try:
    module = importlib.import_module(module_name)
    predicate = getattr(module, "_authoritative")
    cases = [
        (SimpleNamespace(matched_budget=True, externally_generated=True, authority_verified=True, independence_class_id="REAL_CLASS_A"), True),
        (SimpleNamespace(matched_budget=True, externally_generated=True, authority_verified=True, independence_class_id="UNVERIFIED"), False),
        (SimpleNamespace(matched_budget=False, externally_generated=True, authority_verified=True, independence_class_id="REAL_CLASS_A"), False),
        (SimpleNamespace(matched_budget=True, externally_generated=False, authority_verified=True, independence_class_id="REAL_CLASS_A"), False),
        (SimpleNamespace(matched_budget=True, externally_generated=True, authority_verified=False, independence_class_id="REAL_CLASS_A"), False),
    ]
    ok = all(bool(predicate(item)) is expected for item, expected in cases)
    print(json.dumps({"ok": bool(ok)}))
except Exception:
    print(json.dumps({"ok": False}))
'''

    def __init__(
        self,
        task: RealRepositoryTask,
        candidate: SoftwarePatchCandidate,
        signer,
        source_id: str,
        context_id: str,
        challenge_id: str,
        epoch: int,
        timeout_seconds: float = 8.0,
    ) -> None:
        self.task = task
        self.candidate = candidate
        self.signer = signer
        self.source_id = str(source_id)
        self.context_id = str(context_id)
        self.challenge_id = str(challenge_id)
        self.epoch = int(epoch)
        self.timeout_seconds = max(1.0, float(timeout_seconds))

    @staticmethod
    def _run_source(task: RealRepositoryTask, source: str, timeout_seconds: float = 8.0) -> float:
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            shutil.copytree(ROOT / "arte_cognition", temp_root / "arte_cognition")
            target = temp_root / task.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", RealRepositoryAuthorityExecutor._HARNESS, task.module_name],
                    cwd=temp_root,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                if completed.returncode != 0:
                    return 0.0
                result = json.loads(completed.stdout.strip().splitlines()[-1])
                return 1.0 if bool(result.get("ok")) else 0.0
            except Exception:
                return 0.0

    def execute(self, proposal, arm: str, value: float) -> WorldOutcomeReceipt:
        source = self.task.buggy_source if str(arm).upper() == "LOW" else self.candidate.patched_source
        outcome = self._run_source(self.task, source, timeout_seconds=self.timeout_seconds)
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


def make_real_task(relative_path: str, module_name: str, task_id: str) -> RealRepositoryTask:
    pristine = (ROOT / relative_path).read_text(encoding="utf-8")
    buggy = inject_hidden_authority_bug(pristine)
    task = RealRepositoryTask(
        task_id=str(task_id),
        relative_path=str(relative_path),
        module_name=str(module_name),
        pristine_source=pristine,
        buggy_source=buggy,
    )
    if RealRepositoryAuthorityExecutor._run_source(task, pristine) != 1.0:
        raise AssertionError(f"frozen production source is not behaviorally green: {relative_path}")
    if RealRepositoryAuthorityExecutor._run_source(task, buggy) != 0.0:
        raise AssertionError(f"evaluator-owned synthetic mutation did not break hidden behavior: {relative_path}")
    return task


def execute_candidate(body, task, candidate, signers, verifier, epoch_base):
    effects = []
    for issuer_index, (issuer, signer) in enumerate(signers.items()):
        token = hashlib.sha256(
            f"{task.relative_path}|{candidate.site_index}|{candidate.proposal.experiment_id}".encode()
        ).hexdigest()[:16]
        executor = RealRepositoryAuthorityExecutor(
            task=task,
            candidate=candidate,
            signer=signer,
            source_id=f"real-repo::{task.relative_path}::{token}::{issuer}",
            context_id=task.task_id,
            challenge_id=f"real-repo-hidden::{task.task_id}::{token}::{issuer}",
            epoch=epoch_base + issuer_index,
        )
        pair = body.execute_world_intervention(candidate.proposal, executor, verifier=verifier)
        if not pair.authority_verified:
            raise AssertionError("real-repository world receipt lost external authority")
        effects.append(float(pair.effect))
    return tuple(effects)


def train_real_task(body, task, signers, verifier, epoch_base):
    organ = SoftwareTaskAcquisitionOrgan(body)
    candidates = organ.propose(task.task_id, task.buggy_source)
    if not candidates:
        raise AssertionError("real production source exposed no AST repair candidates")
    strong = []
    for index, candidate in enumerate(candidates):
        effects = execute_candidate(body, task, candidate, signers, verifier, epoch_base + index * 10)
        if min(effects) >= 0.9:
            strong.append(candidate)
    if len(strong) != 1:
        raise AssertionError(
            f"real-repository hidden mutation must have exactly one externally strong repair; "
            f"path={task.relative_path} strong={[(c.site_index, c.operator_id) for c in strong]}"
        )
    if strong[0].operator_id != EXPECTED_REPAIR_OPERATOR:
        raise AssertionError(f"unexpected repair operator: {strong[0].operator_id}")
    return candidates, strong[0]


def run_selection(body, task, candidates: Sequence[SoftwarePatchCandidate], selection, signers, verifier, epoch_base):
    capability = 0.0
    strong_count = 0
    for index, candidate in enumerate(selection.candidates):
        effects = execute_candidate(body, task, candidate, signers, verifier, epoch_base + index * 10)
        strong = float(min(effects) >= 0.9)
        capability = max(capability, strong)
        strong_count += int(strong)
    return capability, strong_count


def main():
    specs = (
        ("arte_cognition/repository_task_acquisition.py", "arte_cognition.repository_task_acquisition", "real-train-repository-acquisition"),
        ("arte_cognition/repository_localization_representation_genesis.py", "arte_cognition.repository_localization_representation_genesis", "real-train-localization-representation"),
        ("arte_cognition/repository_patch_cardinality.py", "arte_cognition.repository_patch_cardinality", "real-heldout-patch-cardinality"),
    )
    tasks = tuple(make_real_task(*spec) for spec in specs)
    if len({task.pristine_hash for task in tasks}) != len(tasks):
        raise AssertionError("real production source hashes are not distinct")
    if any(task.pristine_hash == task.buggy_hash for task in tasks):
        raise AssertionError("synthetic hidden mutation did not change production source hash")

    issuer_a = "REAL_REPO_EVAL_A"
    issuer_b = "REAL_REPO_EVAL_B"
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
    train_counts = []
    strong_training = []
    for task_index, task in enumerate(tasks[:2]):
        candidates, strong = train_real_task(
            parent, task, signers, verifier, 10000 + task_index * 10000
        )
        train_counts.append(len(candidates))
        strong_training.append((task.relative_path, strong.site_index, strong.operator_id))

    policy = SoftwareTaskAcquisitionOrgan(parent).policy()
    if policy.status != "REPRODUCED_SOFTWARE_REPAIR_OPERATOR":
        raise AssertionError(f"real-repository repair operator did not reproduce: {policy}")
    if policy.operator_id != EXPECTED_REPAIR_OPERATOR or len(policy.supporting_contexts) != 2:
        raise AssertionError(f"wrong real-repository learned repair policy: {policy}")

    checkpoint = checkpoint_dict(parent)
    verifierless = restore_runtime(checkpoint)
    if SoftwareTaskAcquisitionOrgan(verifierless).policy().operator_id is not None:
        raise AssertionError("real-repository repair authority restored without external verifier")

    heldout = tasks[2]
    treatment = restore_runtime(checkpoint, world_verifier=verifier)
    treatment_organ = SoftwareTaskAcquisitionOrgan(treatment)
    heldout_candidates = treatment_organ.propose(heldout.task_id, heldout.buggy_source)
    treatment_policy = treatment_organ.policy()
    if treatment_policy.operator_id != EXPECTED_REPAIR_OPERATOR:
        raise AssertionError("reverified descendant lost real-repository repair operator")
    matching_count = sum(
        candidate.operator_id == EXPECTED_REPAIR_OPERATOR
        for candidate in heldout_candidates
    )
    if matching_count < 1:
        raise AssertionError("heldout actual source has no learned-operator candidates")

    treatment_selection = treatment_organ.select(
        heldout_candidates,
        max_candidates=matching_count,
        apply_learned_policy=True,
    )
    treatment_capability, treatment_strong_count = run_selection(
        treatment, heldout, heldout_candidates, treatment_selection,
        signers, verifier, 50000,
    )

    remove = restore_runtime(checkpoint, world_verifier=verifier)
    remove_organ = SoftwareTaskAcquisitionOrgan(remove)
    remove_candidates = remove_organ.propose(heldout.task_id, heldout.buggy_source)
    remove_selection = remove_organ.select(
        remove_candidates,
        max_candidates=matching_count,
        apply_learned_policy=False,
    )
    remove_capability, remove_strong_count = run_selection(
        remove, heldout, remove_candidates, remove_selection,
        signers, verifier, 60000,
    )

    full = PersistentCognitiveRuntime()
    full_organ = SoftwareTaskAcquisitionOrgan(full)
    full_candidates = full_organ.propose(heldout.task_id, heldout.buggy_source)
    full_selection = full_organ.select(full_candidates, apply_learned_policy=False)
    full_capability, full_strong_count = run_selection(
        full, heldout, full_candidates, full_selection,
        signers, verifier, 70000,
    )

    if treatment_capability != 1.0 or treatment_strong_count != 1:
        raise AssertionError("learned operator did not repair fresh actual production module")
    if full_capability != 1.0 or full_strong_count != 1:
        raise AssertionError("full actual-source repair search did not recover unique repair")
    if remove_capability != 0.0:
        raise AssertionError(
            "same-checkpoint REMOVE reached the hidden actual repair within the learned-operator matched budget; "
            "challenge does not isolate cross-module repair knowledge"
        )
    if len(full_candidates) <= matching_count:
        raise AssertionError("learned operator did not contract the actual heldout repair search")

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
        "training_full_candidate_counts": train_counts,
        "training_unique_strong_repairs": strong_training,
        "learned_operator": policy.operator_id,
        "learned_operator_supporting_contexts": len(policy.supporting_contexts),
        "heldout_path": heldout.relative_path,
        "heldout_full_candidate_count": len(full_candidates),
        "heldout_learned_operator_candidate_count": matching_count,
        "treatment_candidate_count": len(treatment_selection.candidates),
        "treatment_external_pair_count": 2 * len(treatment_selection.candidates),
        "treatment_capability": treatment_capability,
        "remove_candidate_count": len(remove_selection.candidates),
        "remove_external_pair_count": 2 * len(remove_selection.candidates),
        "remove_same_checkpoint_capability": remove_capability,
        "full_external_pair_count": 2 * len(full_candidates),
        "full_exhaustive_capability": full_capability,
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
