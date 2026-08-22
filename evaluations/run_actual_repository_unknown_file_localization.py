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
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.repository_wide_software_task_acquisition import (
    RepositoryWidePatchCandidate,
    RepositoryWideRepairPolicy,
    RepositoryWideSoftwareTaskAcquisitionOrgan,
)
from arte_cognition.world_coupling import HMACWorldReceiptSigner, HMACWorldReceiptVerifier, WorldOutcomeReceipt


EXPECTED_OPERATOR = "COMPARE::Eq->NotEq"


@dataclass(frozen=True)
class ActualRepositoryChallenge:
    task_id: str
    target_path: str
    target_module: str
    baseline_files: Mapping[str, str]
    pristine_target_source: str


class _AuthorityPredicateBugInjector(ast.NodeTransformer):
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


def inject_hidden_bug(source: str) -> str:
    tree = ast.parse(source)
    injector = _AuthorityPredicateBugInjector()
    mutated = injector.visit(tree)
    ast.fix_missing_locations(mutated)
    if injector.matches != 1:
        raise AssertionError(f"expected one authority predicate target, got {injector.matches}")
    return ast.unparse(mutated) + "\n"


class ActualRepositoryExecutor:
    """Execute one repository-wide candidate in a fresh actual-package subprocess."""

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
        challenge: ActualRepositoryChallenge,
        candidate: RepositoryWidePatchCandidate,
        signer,
        source_id: str,
        context_id: str,
        challenge_id: str,
        epoch: int,
        timeout_seconds: float = 8.0,
    ) -> None:
        self.challenge = challenge
        self.candidate = candidate
        self.signer = signer
        self.source_id = str(source_id)
        self.context_id = str(context_id)
        self.challenge_id = str(challenge_id)
        self.epoch = int(epoch)
        self.timeout_seconds = max(1.0, float(timeout_seconds))

    @staticmethod
    def _run_files(challenge: ActualRepositoryChallenge, files: Mapping[str, str], timeout_seconds: float = 8.0) -> float:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "arte_cognition", root / "arte_cognition")
            for relative_path, source in files.items():
                target = root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(str(source), encoding="utf-8")
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", ActualRepositoryExecutor._HARNESS, challenge.target_module],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                if completed.returncode != 0:
                    return 0.0
                payload = json.loads(completed.stdout.strip().splitlines()[-1])
                return 1.0 if bool(payload.get("ok")) else 0.0
            except Exception:
                return 0.0

    def _files_for_arm(self, arm: str) -> dict[str, str]:
        files = {str(path): str(source) for path, source in self.challenge.baseline_files.items()}
        if str(arm).upper() == "HIGH":
            files[self.candidate.file_path] = self.candidate.patched_source
        return files

    def execute(self, proposal, arm: str, value: float) -> WorldOutcomeReceipt:
        outcome = self._run_files(self.challenge, self._files_for_arm(arm), self.timeout_seconds)
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
            budget_token=f"actual-repository-unknown-file::{self.challenge_id}",
            externally_generated=True,
        ))


def make_challenge(task_id: str, target_path: str, target_module: str, corpus_paths: Sequence[str]) -> ActualRepositoryChallenge:
    pristine_files = {
        path: (ROOT / path).read_text(encoding="utf-8")
        for path in corpus_paths
    }
    pristine_target = pristine_files[target_path]
    buggy_target = inject_hidden_bug(pristine_target)
    baseline = dict(pristine_files)
    baseline[target_path] = buggy_target
    challenge = ActualRepositoryChallenge(
        task_id=str(task_id),
        target_path=str(target_path),
        target_module=str(target_module),
        baseline_files=baseline,
        pristine_target_source=pristine_target,
    )
    if ActualRepositoryExecutor._run_files(challenge, pristine_files) != 1.0:
        raise AssertionError(f"pristine actual repository behavior is not green: {target_path}")
    if ActualRepositoryExecutor._run_files(challenge, baseline) != 0.0:
        raise AssertionError(f"hidden mutation did not break actual repository behavior: {target_path}")
    return challenge


def execute_candidate(body, challenge, candidate, signers, verifier, epoch_base):
    effects = []
    for issuer_index, (issuer, signer) in enumerate(signers.items()):
        token = hashlib.sha256(
            f"{challenge.task_id}|{candidate.file_path_hash}|{candidate.site_index}|{candidate.proposal.experiment_id}".encode()
        ).hexdigest()[:16]
        executor = ActualRepositoryExecutor(
            challenge=challenge,
            candidate=candidate,
            signer=signer,
            source_id=f"actual-repo::{token}::{issuer}",
            context_id=challenge.task_id,
            challenge_id=f"actual-repo-unknown-file::{challenge.task_id}::{token}::{issuer}",
            epoch=epoch_base + issuer_index,
        )
        pair = body.execute_world_intervention(candidate.proposal, executor, verifier=verifier)
        if not pair.authority_verified:
            raise AssertionError("actual repository world outcome lost authority")
        effects.append(float(pair.effect))
    return tuple(effects)


def train_known_file_full(body, challenge, signers, verifier, epoch_base):
    organ = RepositoryWideSoftwareTaskAcquisitionOrgan(body)
    candidates = organ.propose(challenge.task_id, challenge.baseline_files)
    strong = []
    for index, candidate in enumerate(candidates):
        effects = execute_candidate(body, challenge, candidate, signers, verifier, epoch_base + index * 10)
        if min(effects) >= 0.9:
            strong.append(candidate)
    if len(strong) != 1:
        raise AssertionError(f"known-file training did not isolate one repair: {[(c.file_path, c.site_index, c.operator_id) for c in strong]}")
    return candidates, strong[0]


def train_known_file_guided(body, challenge, provisional, signers, verifier, epoch_base):
    organ = RepositoryWideSoftwareTaskAcquisitionOrgan(body)
    candidates = organ.propose(challenge.task_id, challenge.baseline_files)
    guided = tuple(
        candidate for candidate in candidates
        if candidate.operator_id == provisional.operator_id
        and candidate.context_fingerprint == provisional.context_fingerprint
    )
    if not guided or len(guided) >= len(candidates):
        raise AssertionError("first world outcome did not contract second-context structural search")
    strong = []
    for index, candidate in enumerate(guided):
        effects = execute_candidate(body, challenge, candidate, signers, verifier, epoch_base + index * 10)
        if min(effects) >= 0.9:
            strong.append(candidate)
    if len(strong) != 1:
        raise AssertionError(f"guided second context did not reproduce one repair: {[(c.file_path, c.site_index) for c in strong]}")
    return candidates, guided, strong[0]


def run_selection(body, challenge, selection, signers, verifier, epoch_base):
    capability = 0.0
    strong_count = 0
    strong_paths = []
    for index, candidate in enumerate(selection.candidates):
        effects = execute_candidate(body, challenge, candidate, signers, verifier, epoch_base + index * 10)
        strong = min(effects) >= 0.9
        if strong:
            strong_count += 1
            strong_paths.append(candidate.file_path)
            capability = 1.0
    return capability, strong_count, tuple(strong_paths)


def main():
    training_specs = (
        ("actual-known-pair-composition", "arte_cognition/repository_patch_composition.py", "arte_cognition.repository_patch_composition"),
        ("actual-known-cardinality", "arte_cognition/repository_patch_cardinality.py", "arte_cognition.repository_patch_cardinality"),
    )
    heldout_path = "arte_cognition/repository_task_acquisition.py"
    heldout_module = "arte_cognition.repository_task_acquisition"
    distractor_paths = (
        "arte_cognition/causal_credit.py",
        "arte_cognition/causal_identification.py",
        "arte_cognition/semantic_genesis.py",
        "arte_cognition/validation_matrix.py",
    )
    heldout_corpus_paths = tuple(sorted((
        training_specs[0][1],
        training_specs[1][1],
        heldout_path,
        *distractor_paths,
    )))

    issuer_a = "ACTUAL_REPO_LOCALIZER_A"
    issuer_b = "ACTUAL_REPO_LOCALIZER_B"
    key_a = hashlib.sha256(b"actual-repo-localizer-a").digest()
    key_b = hashlib.sha256(b"actual-repo-localizer-b").digest()
    signers = {
        issuer_a: HMACWorldReceiptSigner(issuer_a, key_a),
        issuer_b: HMACWorldReceiptSigner(issuer_b, key_b),
    }
    verifier = HMACWorldReceiptVerifier(
        {issuer_a: key_a, issuer_b: key_b},
        independence_classes={issuer_a: "ACTUAL_REPO_CLASS_A", issuer_b: "ACTUAL_REPO_CLASS_B"},
    )

    parent = PersistentCognitiveRuntime()
    first = make_challenge(training_specs[0][0], training_specs[0][1], training_specs[0][2], (training_specs[0][1],))
    first_candidates, first_strong = train_known_file_full(parent, first, signers, verifier, 10000)
    if first_strong.operator_id != EXPECTED_OPERATOR:
        raise AssertionError(f"unexpected first repair operator {first_strong.operator_id}")

    second = make_challenge(training_specs[1][0], training_specs[1][1], training_specs[1][2], (training_specs[1][1],))
    second_candidates, second_guided, second_strong = train_known_file_guided(
        parent, second, first_strong, signers, verifier, 30000
    )
    if (
        second_strong.operator_id != first_strong.operator_id
        or second_strong.context_fingerprint != first_strong.context_fingerprint
    ):
        raise AssertionError("filename-independent structural repair phenotype did not reproduce")

    policy = RepositoryWideSoftwareTaskAcquisitionOrgan(parent).policy()
    if policy.status != "REPRODUCED_REPOSITORY_WIDE_LOCALIZATION":
        raise AssertionError(f"repository-wide localization did not gain authority: {policy}")
    if policy.operator_id != EXPECTED_OPERATOR or policy.context_fingerprint != first_strong.context_fingerprint:
        raise AssertionError(f"wrong repository-wide policy: {policy}")
    if len(policy.supporting_contexts) != 2:
        raise AssertionError("repository-wide phenotype lacks two-context reproduction")

    checkpoint = checkpoint_dict(parent)
    verifierless = restore_runtime(checkpoint)
    if RepositoryWideSoftwareTaskAcquisitionOrgan(verifierless).policy().operator_id is not None:
        raise AssertionError("repository-wide localization authority restored without verifier")

    heldout = make_challenge(
        "actual-heldout-unknown-file",
        heldout_path,
        heldout_module,
        heldout_corpus_paths,
    )

    treatment = restore_runtime(checkpoint, world_verifier=verifier)
    treatment_organ = RepositoryWideSoftwareTaskAcquisitionOrgan(treatment)
    treatment_candidates = treatment_organ.propose(heldout.task_id, heldout.baseline_files)
    treatment_policy = treatment_organ.policy()
    signature_count = sum(
        candidate.operator_id == treatment_policy.operator_id
        and candidate.context_fingerprint == treatment_policy.context_fingerprint
        for candidate in treatment_candidates
    )
    operator_only_count = sum(
        candidate.operator_id == treatment_policy.operator_id
        for candidate in treatment_candidates
    )
    if signature_count < 1 or signature_count >= operator_only_count:
        raise AssertionError(
            f"structural localization failed to contract operator-only space: signature={signature_count} operator={operator_only_count}"
        )

    treatment_selection = treatment_organ.select(
        treatment_candidates,
        max_candidates=signature_count,
        apply_learned_policy=True,
        operator_only=False,
    )
    treatment_capability, treatment_strong_count, treatment_strong_paths = run_selection(
        treatment, heldout, treatment_selection, signers, verifier, 50000
    )

    remove = restore_runtime(checkpoint, world_verifier=verifier)
    remove_organ = RepositoryWideSoftwareTaskAcquisitionOrgan(remove)
    remove_candidates = remove_organ.propose(heldout.task_id, heldout.baseline_files)
    remove_selection = remove_organ.select(
        remove_candidates,
        max_candidates=signature_count,
        apply_learned_policy=True,
        operator_only=True,
    )
    remove_capability, remove_strong_count, remove_strong_paths = run_selection(
        remove, heldout, remove_selection, signers, verifier, 70000
    )

    full = PersistentCognitiveRuntime()
    full_organ = RepositoryWideSoftwareTaskAcquisitionOrgan(full)
    full_candidates = full_organ.propose(heldout.task_id, heldout.baseline_files)
    full_selection = full_organ.select(full_candidates, apply_learned_policy=False)
    full_capability, full_strong_count, full_strong_paths = run_selection(
        full, heldout, full_selection, signers, verifier, 90000
    )

    if treatment_capability != 1.0 or treatment_strong_count != 1 or treatment_strong_paths != (heldout_path,):
        raise AssertionError(
            f"learned localization did not repair the hidden actual file: cap={treatment_capability} paths={treatment_strong_paths}"
        )
    if remove_capability != 0.0:
        raise AssertionError(
            "operator-only REMOVE found the hidden file under the matched localization budget; localization phenotype not causally isolated"
        )
    if full_capability != 1.0 or full_strong_count != 1 or full_strong_paths != (heldout_path,):
        raise AssertionError("full repository-wide search did not isolate the unique actual-file repair")
    if len(full_candidates) != len(treatment_candidates):
        raise AssertionError("FULL and Treatment did not receive the same source-derived candidate universe")

    receipt = {
        "status": "PASS_BOUNDED_ACTUAL_REPOSITORY_UNKNOWN_FILE_LOCALIZATION_AND_PATCH_TRANSFER",
        "repository": "leehyen0/GENESIS-EX-NIHILO",
        "real_repository_source": True,
        "synthetic_bug_injected_post_checkout": True,
        "natural_historical_bug": False,
        "heldout_target_file_exposed_to_body": False,
        "heldout_target_module_exposed_to_candidate_generator": False,
        "heldout_repository_file_count": len(heldout.baseline_files),
        "heldout_repository_paths": sorted(heldout.baseline_files),
        "heldout_target_path": heldout_path,
        "known_file_training": True,
        "training_paths": [training_specs[0][1], training_specs[1][1]],
        "training_first_full_candidate_count": len(first_candidates),
        "training_second_full_candidate_count": len(second_candidates),
        "training_second_guided_candidate_count": len(second_guided),
        "training_guidance_from_first_world_outcome_only": True,
        "learned_operator": policy.operator_id,
        "learned_context_fingerprint": policy.context_fingerprint,
        "learned_context_supporting_contexts": len(policy.supporting_contexts),
        "fingerprint_filename_independent": True,
        "fingerprint_identifier_spelling_erased": True,
        "fingerprint_literal_values_erased": True,
        "fingerprint_comparison_direction_erased": True,
        "fingerprint_shape_grammar_human_authored": True,
        "heldout_full_candidate_count": len(full_candidates),
        "heldout_operator_only_candidate_count": operator_only_count,
        "heldout_context_plus_operator_candidate_count": signature_count,
        "treatment_candidate_count": len(treatment_selection.candidates),
        "treatment_external_pair_count": 2 * len(treatment_selection.candidates),
        "treatment_capability": treatment_capability,
        "remove_definition": "same checkpoint and repair operator; remove learned AST-context localization phenotype",
        "remove_candidate_count": len(remove_selection.candidates),
        "remove_external_pair_count": 2 * len(remove_selection.candidates),
        "remove_same_checkpoint_capability": remove_capability,
        "full_external_pair_count": 2 * len(full_selection.candidates),
        "full_exhaustive_capability": full_capability,
        "external_pair_reduction_vs_full": 1.0 - (len(treatment_selection.candidates) / max(1, len(full_selection.candidates))),
        "localization_contraction_vs_operator_only": 1.0 - (signature_count / max(1, operator_only_count)),
        "candidate_generation_uses_hidden_outcomes": False,
        "hidden_behavior_exposed_to_body_before_execution": False,
        "external_execution": "fresh_temp_copy_of_actual_arte_cognition_package_imported_in_python_subprocess",
        "verifierless_localization_authority": False,
        "arbitrary_real_repository_bug_repair": False,
        "unrestricted_localization_representation_genesis": False,
        "repair_mutation_alphabet_human_authored": True,
        "foundation_weight_change": False,
        "independent_organizational_custody": False,
        "physical_world": False,
        "global_recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
    }
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
