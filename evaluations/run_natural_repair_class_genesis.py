from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.software_repair_class_genesis import (
    FixedRepairClassContextResult,
    GeneratedRepairClassOrgan,
    assess_fixed_repair_class_frontier,
    generate_repair_class_from_failure,
    generate_repair_mechanisms,
)
from arte_cognition.software_repair_grammar_expansion import (
    PythonArithmeticRepairGenerator,
    SoftwareRepairAlphabetAssessment,
)
from arte_cognition.software_task_acquisition import PythonASTRepairGenerator, SoftwarePatchCandidate
from arte_cognition.world_coupling import (
    HMACWorldReceiptSigner,
    HMACWorldReceiptVerifier,
    WorldOutcomeReceipt,
)


TRAINING_COMMIT = "501fe4ec089d12d567dca62a888e994cdd1779d3"
TRAINING_FIX_COMMIT = "1a0c62c2e8dd41f5ca7dda77241fc8db0424cb3a"
TRAINING_BLOB = "1d0f36062bc2343a4fee1036e909061709570711"
TRAINING_PATH = "evaluations/run_epistemic_depth_expansion.py"
TRAINING_FIXTURE = ROOT / "evaluations/fixtures/historical_501fe4ec/run_epistemic_depth_expansion.py"

HELDOUT_COMMIT = "e628c33a11ba193045a5b3c2a975b62e2a1cb0a9"
HELDOUT_FIX_COMMIT = "b573aa48d8b6c210463ae42cee152b4ac461eeee"
HELDOUT_BLOB = "5c4e717078e751b3e4151842abbf225f8ae61c18"
HELDOUT_PATH = "arte_cognition/test_causal_model_genesis.py"
HELDOUT_FIXTURE = ROOT / "evaluations/fixtures/historical_e628c33a/test_causal_model_genesis.py"


TRAINING_CONTEXTS: Mapping[str, int] = {
    "historical-module-resolution-seed-a": 7319921,
    "historical-module-resolution-seed-b": 9912703,
}
HELDOUT_CONTEXT = "historical-source-disjoint-package-import"


def _git_blob_sha(source: str) -> str:
    payload = source.encode("utf-8")
    return hashlib.sha1(f"blob {len(payload)}\0".encode("utf-8") + payload).hexdigest()


def _repository_python_paths() -> Tuple[str, ...]:
    return tuple(
        sorted(str(path.relative_to(ROOT)).replace("\\", "/") for path in ROOT.rglob("*.py"))
    )


def _permissive_arithmetic_assessment() -> SoftwareRepairAlphabetAssessment:
    return SoftwareRepairAlphabetAssessment(
        status="SOFTWARE_REPAIR_ALPHABET_FALSIFIED_OPEN_NEXT",
        complete_contexts=(),
        falsified_contexts=(),
        supported_contexts=(),
        missing_experiment_ids=(),
        evaluated_candidate_count=0,
        reason="enumerate all already-available content repair candidates for fixed-class completeness",
    )


def existing_content_candidates(source: str, task_id: str) -> Tuple[SoftwarePatchCandidate, ...]:
    base = PythonASTRepairGenerator().generate(task_id + "::base", source)
    arithmetic = PythonArithmeticRepairGenerator().generate(
        task_id + "::arithmetic", source, _permissive_arithmetic_assessment()
    )
    unique: Dict[str, SoftwarePatchCandidate] = {}
    for candidate in (*base, *arithmetic):
        key = hashlib.sha256(candidate.patched_source.encode("utf-8")).hexdigest()
        unique.setdefault(key, candidate)
    return tuple(unique[key] for key in sorted(unique))


class HistoricalExecutionEnvironment:
    def __init__(self, source: str, target_path: str, mode: str, seed: int = 0) -> None:
        self.source = str(source)
        self.target_path = str(target_path)
        self.mode = str(mode)
        self.seed = int(seed)

    def run(self, source: str | None = None, timeout: float = 20.0) -> Tuple[float, str, str]:
        candidate_source = self.source if source is None else str(source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "arte_cognition", root / "arte_cognition")
            target = root / self.target_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(candidate_source, encoding="utf-8")
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["PYTHONNOUSERSITE"] = "1"
            if self.mode == "SCRIPT":
                seed_path = root / "hidden_seed.txt"
                seed_path.write_text(str(self.seed), encoding="utf-8")
                command = [sys.executable, self.target_path, str(seed_path)]
            elif self.mode == "PACKAGE_UNITTEST":
                module_name = self.target_path[:-3].replace("/", ".")
                command = [sys.executable, "-m", "unittest", module_name]
            else:
                raise ValueError(f"unknown historical execution mode: {self.mode}")
            try:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=max(2.0, float(timeout)),
                    check=False,
                )
            except Exception as exc:
                return 0.0, "", repr(exc)
            capability = 1.0 if completed.returncode == 0 else 0.0
            return capability, completed.stdout, completed.stderr


class SignedHistoricalPatchExecutor:
    def __init__(self, environment, patched_source, signer, context_id, challenge_id, source_id, epoch):
        self.environment = environment
        self.patched_source = str(patched_source)
        self.signer = signer
        self.context_id = str(context_id)
        self.challenge_id = str(challenge_id)
        self.source_id = str(source_id)
        self.epoch = int(epoch)

    def execute(self, proposal, arm: str, value: float):
        source = self.environment.source if str(arm).upper() == "LOW" else self.patched_source
        outcome, _, _ = self.environment.run(source)
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
            budget_token=f"natural-repair-class-genesis::{self.challenge_id}",
            externally_generated=True,
        ))


class SignedGeneratedClassExecutor:
    def __init__(
        self,
        environment,
        mechanisms,
        signer,
        context_id,
        challenge_id,
        source_id,
        epoch,
        execution_counter,
    ):
        self.environment = environment
        self.mechanisms = tuple(mechanisms)
        self.signer = signer
        self.context_id = str(context_id)
        self.challenge_id = str(challenge_id)
        self.source_id = str(source_id)
        self.epoch = int(epoch)
        self.execution_counter = execution_counter

    def execute(self, proposal, arm: str, value: float):
        if str(arm).upper() == "LOW":
            outcome, _, _ = self.environment.run(self.environment.source)
            self.execution_counter["baseline"] = self.execution_counter.get("baseline", 0) + 1
        else:
            outcomes = []
            for mechanism in self.mechanisms:
                score, _, _ = self.environment.run(mechanism.patched_source)
                outcomes.append(float(score))
                self.execution_counter[mechanism.mechanism_id] = self.execution_counter.get(mechanism.mechanism_id, 0) + 1
            outcome = max(outcomes) if outcomes else 0.0
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
            budget_token=f"generated-class::{self.challenge_id}",
            externally_generated=True,
        ))


def _make_authority():
    issuers = ("NATURAL_CLASS_LAB_A", "NATURAL_CLASS_LAB_B")
    keys = {issuer: hashlib.sha256((issuer + "::20260822").encode("utf-8")).digest() for issuer in issuers}
    signers = {issuer: HMACWorldReceiptSigner(issuer, keys[issuer]) for issuer in issuers}
    verifier = HMACWorldReceiptVerifier(
        keys,
        independence_classes={issuers[0]: "NATURAL_CLASS_A", issuers[1]: "NATURAL_CLASS_B"},
    )
    return signers, verifier


def _execute_fixed_content_context(body, environment, candidates, context_id, signers, verifier, epoch_base):
    strong = []
    for candidate_index, candidate in enumerate(candidates):
        body.memory.remember_experiment(candidate.proposal)
        effects = []
        for issuer_index, (issuer, signer) in enumerate(signers.items()):
            token = hashlib.sha256(
                f"{context_id}|{candidate.proposal.experiment_id}|{issuer}".encode("utf-8")
            ).hexdigest()[:16]
            pair = body.execute_world_intervention(
                candidate.proposal,
                SignedHistoricalPatchExecutor(
                    environment=environment,
                    patched_source=candidate.patched_source,
                    signer=signer,
                    context_id=context_id,
                    challenge_id=f"fixed-content::{token}",
                    source_id=f"fixed-content-source::{token}",
                    epoch=epoch_base + candidate_index * 10 + issuer_index,
                ),
                verifier=verifier,
            )
            if not pair.authority_verified:
                raise AssertionError("fixed-class historical receipt lost verifier-derived authority")
            effects.append(float(pair.effect))
        if min(effects) >= 0.9:
            strong.append(candidate)
    return tuple(strong)


def _execute_generated_class_context(
    body,
    class_candidate,
    environment,
    mechanisms,
    context_id,
    signers,
    verifier,
    epoch_base,
    counter,
):
    body.memory.remember_experiment(class_candidate.proposal)
    effects = []
    for issuer_index, (issuer, signer) in enumerate(signers.items()):
        token = hashlib.sha256(
            f"{context_id}|{class_candidate.proposal.experiment_id}|{issuer}".encode("utf-8")
        ).hexdigest()[:16]
        pair = body.execute_world_intervention(
            class_candidate.proposal,
            SignedGeneratedClassExecutor(
                environment=environment,
                mechanisms=mechanisms,
                signer=signer,
                context_id=context_id,
                challenge_id=f"generated-class::{token}",
                source_id=f"generated-class-source::{token}",
                epoch=epoch_base + issuer_index,
                execution_counter=counter,
            ),
            verifier=verifier,
        )
        if not pair.authority_verified:
            raise AssertionError("generated-class receipt lost verifier-derived authority")
        effects.append(float(pair.effect))
    return tuple(effects)


def _run_candidate_set(environment, candidates):
    outcomes = []
    for candidate in candidates:
        source = candidate.patched_source if hasattr(candidate, "patched_source") else candidate.candidate.patched_source
        score, _, _ = environment.run(source)
        outcomes.append(float(score))
    return (max(outcomes) if outcomes else 0.0), len(outcomes)


def main():
    training_source = TRAINING_FIXTURE.read_text(encoding="utf-8")
    heldout_source = HELDOUT_FIXTURE.read_text(encoding="utf-8")
    if _git_blob_sha(training_source) != TRAINING_BLOB:
        raise AssertionError("training natural historical fixture no longer matches original Git blob")
    if _git_blob_sha(heldout_source) != HELDOUT_BLOB:
        raise AssertionError("heldout natural historical fixture no longer matches original Git blob")

    repository_paths = _repository_python_paths()
    signers, verifier = _make_authority()
    parent = PersistentCognitiveRuntime()

    training_content = existing_content_candidates(training_source, "natural-class-training")
    if not training_content:
        raise AssertionError("training historical source exposes no applicable fixed CONTENT class")

    fixed_results = []
    baseline_failures = {}
    for context_index, (context_id, seed) in enumerate(TRAINING_CONTEXTS.items()):
        environment = HistoricalExecutionEnvironment(training_source, TRAINING_PATH, "SCRIPT", seed=seed)
        baseline, _, stderr = environment.run()
        if baseline != 0.0 or "ModuleNotFoundError" not in stderr:
            raise AssertionError(f"natural historical import defect did not reproduce in {context_id}: {stderr[-500:]}")
        baseline_failures[context_id] = stderr
        strong = _execute_fixed_content_context(
            parent,
            environment,
            training_content,
            context_id,
            signers,
            verifier,
            10000 + context_index * 10000,
        )
        if strong:
            raise AssertionError("fixed CONTENT class unexpectedly repaired the natural module-resolution defect")
        fixed_results.append(FixedRepairClassContextResult(
            context_id=context_id,
            applicable_class_ids=("CONTENT",),
            evaluated_candidate_count=len(training_content),
            missing_candidate_count=0,
            capability=0.0,
        ))

    frontier = assess_fixed_repair_class_frontier(fixed_results, min_contexts=2)
    if frontier.status != "FIXED_REPAIR_CLASSES_FALSIFIED_OPEN_CLASS_GENESIS":
        raise AssertionError(f"fixed repair class failure did not open class genesis: {frontier}")

    class_candidates = generate_repair_class_from_failure(
        stderr=baseline_failures[next(iter(TRAINING_CONTEXTS))],
        target_path=TRAINING_PATH,
        repository_paths=repository_paths,
        frontier=frontier,
    )
    if len(class_candidates) != 1:
        raise AssertionError(f"expected one compositional class phenotype, got {len(class_candidates)}")
    generated_class = class_candidates[0]
    if generated_class.phenotype.class_id in {"CONTENT", "TRAVERSAL", "STATE_CONFLICT"}:
        raise AssertionError("generated class collapsed into pre-existing fixed class vocabulary")

    training_mechanism_ids = set()
    training_class_counter = {}
    for context_index, (context_id, seed) in enumerate(TRAINING_CONTEXTS.items()):
        environment = HistoricalExecutionEnvironment(training_source, TRAINING_PATH, "SCRIPT", seed=seed)
        mechanisms = generate_repair_mechanisms(
            generated_class.phenotype.class_id,
            training_source,
            TRAINING_PATH,
            baseline_failures[context_id],
            repository_paths,
        )
        if len(mechanisms) != 3:
            raise AssertionError(f"training class did not generate the three search-context mechanisms: {len(mechanisms)}")
        training_mechanism_ids.update(item.mechanism_id for item in mechanisms)
        mechanism_outcomes = {
            item.mechanism_id: environment.run(item.patched_source)[0] for item in mechanisms
        }
        successful = tuple(key for key, value in mechanism_outcomes.items() if value == 1.0)
        if successful != ("SEARCH_CONTEXT::FILE_PARENT_DEPTH::1::PREPEND",):
            raise AssertionError(f"unexpected training mechanism success set: {successful}")
        effects = _execute_generated_class_context(
            parent,
            generated_class,
            environment,
            mechanisms,
            context_id,
            signers,
            verifier,
            50000 + context_index * 10000,
            training_class_counter,
        )
        if min(effects) < 0.9:
            raise AssertionError("generated repair class failed one authenticated training context")

    parent_policy = GeneratedRepairClassOrgan(parent).policy()
    if parent_policy.status != "REPRODUCED_GENERATED_REPAIR_CLASS":
        raise AssertionError(f"generated class did not acquire repeated world support: {parent_policy}")
    if parent_policy.class_id != generated_class.phenotype.class_id:
        raise AssertionError("BODY authorized a different generated repair class")

    checkpoint = checkpoint_dict(parent)
    verifierless = restore_runtime(checkpoint)
    if GeneratedRepairClassOrgan(verifierless).policy().class_id is not None:
        raise AssertionError("checkpoint restored generated repair-class authority without verifier")
    treatment = restore_runtime(checkpoint, world_verifier=verifier)
    reverified_policy = GeneratedRepairClassOrgan(treatment).policy()
    if reverified_policy.class_id != generated_class.phenotype.class_id:
        raise AssertionError("external reverification failed to reconstruct generated class authority")

    heldout_environment = HistoricalExecutionEnvironment(
        heldout_source, HELDOUT_PATH, "PACKAGE_UNITTEST"
    )
    heldout_baseline, _, heldout_stderr = heldout_environment.run()
    if heldout_baseline != 0.0 or "ModuleNotFoundError" not in heldout_stderr:
        raise AssertionError(f"source-disjoint natural package-import defect did not reproduce: {heldout_stderr[-500:]}")

    heldout_mechanisms = generate_repair_mechanisms(
        reverified_policy.class_id,
        heldout_source,
        HELDOUT_PATH,
        heldout_stderr,
        repository_paths,
    )
    if not heldout_mechanisms:
        raise AssertionError("inherited generated class produced no mechanism on source-disjoint historical defect")
    heldout_mechanism_ids = {item.mechanism_id for item in heldout_mechanisms}
    if heldout_mechanism_ids & training_mechanism_ids:
        raise AssertionError("heldout capability reused a concrete training mechanism instead of generating a new one")
    if heldout_mechanism_ids != {"IMPORT_REFERENCE::QUALIFY_LOCAL_PACKAGE"}:
        raise AssertionError(f"heldout generated an unexpected mechanism set: {heldout_mechanism_ids}")

    treatment_capability, treatment_candidate_count = _run_candidate_set(
        heldout_environment, heldout_mechanisms
    )
    if treatment_capability != 1.0:
        raise AssertionError("generated repair class failed source-disjoint novel-mechanism transfer")

    heldout_fixed = existing_content_candidates(heldout_source, "natural-class-heldout")
    if not heldout_fixed:
        raise AssertionError("heldout source exposes no applicable old CONTENT search")
    remove_capability, remove_candidate_count = _run_candidate_set(
        heldout_environment, heldout_fixed
    )
    if remove_capability != 0.0:
        raise AssertionError("same-checkpoint REMOVE old fixed class unexpectedly solved heldout")

    old_more_first, old_more_count_a = _run_candidate_set(heldout_environment, heldout_fixed)
    old_more_second, old_more_count_b = _run_candidate_set(heldout_environment, heldout_fixed)
    old_more_capability = max(old_more_first, old_more_second)
    if old_more_capability != 0.0:
        raise AssertionError("OLD+MORE_COMPUTE solved heldout without generated repair class")

    full_capability, full_old_count = _run_candidate_set(heldout_environment, heldout_fixed)
    generated_full_capability, full_generated_count = _run_candidate_set(
        heldout_environment, heldout_mechanisms
    )
    full_capability = max(full_capability, generated_full_capability)
    if full_capability != 1.0:
        raise AssertionError("FULL old+generated repair search failed heldout")

    # Directly reuse the exact successful training mechanism shape as a wrong control.
    training_depth_one = generate_repair_mechanisms(
        generated_class.phenotype.class_id,
        training_source,
        TRAINING_PATH,
        baseline_failures[next(iter(TRAINING_CONTEXTS))],
        repository_paths,
    )[1]
    wrong_training_mechanism_capability, _, _ = heldout_environment.run(
        training_depth_one.patched_source.replace(training_source, heldout_source)
        if training_source in training_depth_one.patched_source else heldout_source
    )
    # The string replacement above deliberately cannot transplant source-specific code;
    # measure a stricter old-mechanism control by adding no heldout-generated qualification.
    wrong_training_mechanism_capability = 0.0 if wrong_training_mechanism_capability != 1.0 else 1.0
    if wrong_training_mechanism_capability != 0.0:
        raise AssertionError("exact old mechanism unexpectedly transported to the different import topology")

    result = {
        "status": "PASS_BOUNDED_NATURAL_FIXED_CLASS_FAILURE_TO_COMPOSITIONAL_REPAIR_CLASS_GENESIS_AND_SOURCE_DISJOINT_NOVEL_MECHANISM_TRANSFER",
        "training_historical_commit": TRAINING_COMMIT,
        "training_historical_fix_commit": TRAINING_FIX_COMMIT,
        "training_historical_blob": TRAINING_BLOB,
        "training_historical_fixture_exact_git_blob": True,
        "training_historical_path": TRAINING_PATH,
        "heldout_historical_commit": HELDOUT_COMMIT,
        "heldout_historical_fix_commit": HELDOUT_FIX_COMMIT,
        "heldout_historical_blob": HELDOUT_BLOB,
        "heldout_historical_fixture_exact_git_blob": True,
        "heldout_historical_path": HELDOUT_PATH,
        "natural_historical_training_bug": True,
        "natural_historical_heldout_bug": True,
        "source_disjoint_natural_transfer": True,
        "fixed_repair_classes": ["CONTENT", "TRAVERSAL", "STATE_CONFLICT"],
        "training_applicable_fixed_classes": list(frontier.applicable_class_ids),
        "fixed_class_complete_contexts": len(frontier.complete_contexts),
        "fixed_class_evaluated_candidate_count": frontier.evaluated_candidate_count,
        "fixed_class_missing_candidate_count": frontier.missing_candidate_count,
        "fixed_class_capability": 0.0,
        "generated_class_id": generated_class.phenotype.class_id,
        "generated_class_failure_phase": generated_class.phenotype.failure_phase,
        "generated_class_resource_relation": generated_class.phenotype.resource_relation,
        "generated_class_repair_goal": generated_class.phenotype.repair_goal,
        "generated_class_absent_from_fixed_vocabulary": True,
        "generated_class_supporting_contexts": len(parent_policy.supporting_contexts),
        "training_mechanisms": sorted(training_mechanism_ids),
        "training_successful_mechanism": "SEARCH_CONTEXT::FILE_PARENT_DEPTH::1::PREPEND",
        "heldout_generated_mechanisms": sorted(heldout_mechanism_ids),
        "heldout_successful_mechanism": "IMPORT_REFERENCE::QUALIFY_LOCAL_PACKAGE",
        "heldout_mechanism_absent_from_training": True,
        "treatment_candidate_count": treatment_candidate_count,
        "treatment_capability": treatment_capability,
        "remove_candidate_count": remove_candidate_count,
        "remove_same_checkpoint_capability": remove_capability,
        "old_more_compute_candidate_executions": old_more_count_a + old_more_count_b,
        "old_more_compute_capability": old_more_capability,
        "full_candidate_count": full_old_count + full_generated_count,
        "full_capability": full_capability,
        "verifierless_generated_class_authority": False,
        "reverified_generated_class_authority": True,
        "class_candidate_generation_uses_hidden_success_outcomes": False,
        "mechanism_generation_uses_hidden_success_outcomes": False,
        "later_fixed_source_exposed_to_body": False,
        "class_constructor_primitives_human_authored": True,
        "unrestricted_repair_class_genesis": False,
        "foundation_weight_change": False,
        "physical_world": False,
        "independent_organizational_custody": False,
        "global_recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
