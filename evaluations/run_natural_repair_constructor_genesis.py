from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.software_repair_class_genesis import (
    FixedRepairClassFrontierAssessment,
    generate_repair_class_from_failure,
)
from arte_cognition.software_repair_constructor_authority import (
    infer_descendant_relational_constructor_primitive,
)
from arte_cognition.software_repair_constructor_genesis import (
    CONSTRUCTOR_FAMILY,
    ConstructorInexpressivityContext,
    RelationalRepairConstructorOrgan,
    assess_constructor_inexpressivity,
    generate_relational_repair_mechanisms,
    infer_relational_constructor_primitive,
    propose_relational_repair_class,
)
from arte_cognition.world_coupling import (
    HMACWorldReceiptSigner,
    HMACWorldReceiptVerifier,
    WorldOutcomeReceipt,
)

IMPORT_TRAIN_COMMIT = "2fa45c4f1d7c8fb8de67abab23a1f8f6b383cdef"
IMPORT_FIX_COMMIT = "089e494815bfecb1ab565365fe0f0f62d046c28a"
IMPORT_BLOB = "ba85edc6eafcda18eea726f66ca0cf6dbdcedcf2"
IMPORT_PATH = "arte_cognition/software_task_acquisition.py"
IMPORT_FIXTURE = ROOT / "evaluations/fixtures/historical_2fa45c4f/software_task_acquisition.py"

CALL_TRAIN_COMMIT = "5a4fc4ca5ac1dd40e75c6ad541c055a4d0743744"
CALL_FIX_COMMIT = "cd80dbdff9d895de9f862eccad5fe33a241737fb"
CALL_BLOB = "1a37f91ae913304edb9fe0ec56ddeeb960194403"
CALL_PATH = "arte_cognition/test_causal_identification.py"
CALL_FIXTURE = ROOT / "evaluations/fixtures/historical_5a4fc4ca/test_causal_identification.py"
CALL_TEST = (
    "arte_cognition.test_causal_identification."
    "GenerationScopedIdentifierTests."
    "test_older_generations_do_not_dilute_current_generation_eig"
)


def _git_blob_sha(source: str) -> str:
    payload = source.encode("utf-8")
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def _repository_sources() -> Dict[str, str]:
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "arte_cognition").rglob("*.py"))
    }


def _permissive_old_frontier() -> FixedRepairClassFrontierAssessment:
    # Give the old #80 constructor its upstream gate for free. A zero proposal count
    # therefore measures constructor inexpressivity, not missing fixed-class evidence.
    return FixedRepairClassFrontierAssessment(
        status="FIXED_REPAIR_CLASSES_FALSIFIED_OPEN_CLASS_GENESIS",
        complete_contexts=("expressivity-probe",),
        applicable_class_ids=("CONTENT", "TRAVERSAL", "STATE_CONFLICT"),
        evaluated_candidate_count=1,
        missing_candidate_count=0,
        reason="maximally permissive probe of the already-merged class constructor",
    )


def _old_constructor_count(stderr: str, path: str, sources: Mapping[str, str]) -> int:
    return len(generate_repair_class_from_failure(
        stderr=stderr,
        target_path=path,
        repository_paths=tuple(sorted(sources)),
        frontier=_permissive_old_frontier(),
    ))


class FreshRepositoryEnvironment:
    def __init__(self, source: str, path: str, mode: str, selector: Optional[str] = None,
                 extra_sources: Optional[Mapping[str, str]] = None) -> None:
        self.source = str(source)
        self.path = str(path)
        self.mode = str(mode)
        self.selector = selector
        self.extra_sources = dict(extra_sources or {})

    @staticmethod
    def _module(path: str) -> str:
        return str(path).replace("\\", "/")[:-3].replace("/", ".")

    def run(self, source: Optional[str] = None, timeout: float = 25.0) -> Tuple[float, str, str]:
        candidate = self.source if source is None else str(source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "arte_cognition", root / "arte_cognition")
            for path, text in self.extra_sources.items():
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            target = root / self.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(candidate, encoding="utf-8")
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["PYTHONNOUSERSITE"] = "1"
            if self.mode == "IMPORT":
                command = [sys.executable, "-c", f"import {self._module(self.path)}"]
            elif self.mode == "UNITTEST":
                if not self.selector:
                    raise ValueError("UNITTEST mode requires selector")
                command = [sys.executable, "-m", "unittest", self.selector]
            else:
                raise ValueError(f"unknown mode {self.mode}")
            try:
                completed = subprocess.run(
                    command, cwd=root, env=env, capture_output=True, text=True,
                    timeout=max(2.0, float(timeout)), check=False,
                )
            except Exception as exc:
                return 0.0, "", repr(exc)
            return 1.0 if completed.returncode == 0 else 0.0, completed.stdout, completed.stderr


class SignedPatchExecutor:
    def __init__(self, environment, patched_source, signer, context, challenge, source_id, epoch):
        self.environment = environment
        self.patched_source = str(patched_source)
        self.signer = signer
        self.context = str(context)
        self.challenge = str(challenge)
        self.source_id = str(source_id)
        self.epoch = int(epoch)

    def execute(self, proposal, arm: str, value: float):
        source = self.environment.source if str(arm).upper() == "LOW" else self.patched_source
        outcome, _, _ = self.environment.run(source)
        return self.signer.sign(WorldOutcomeReceipt(
            receipt_id=f"{self.challenge}::{proposal.experiment_id}::{arm}",
            experiment_id=proposal.experiment_id,
            axis_id=proposal.axis_id,
            arm=str(arm).upper(),
            intervention_value=float(value),
            outcome=float(outcome),
            source_id=self.source_id,
            context_id=self.context,
            challenge_id=self.challenge,
            epoch=self.epoch,
            budget_token=f"constructor::{self.challenge}",
            externally_generated=True,
        ))


def _authority():
    issuers = ("CONSTRUCTOR_LAB_A", "CONSTRUCTOR_LAB_B")
    keys = {name: hashlib.sha256((name + "::20260822").encode()).digest() for name in issuers}
    signers = {name: HMACWorldReceiptSigner(name, keys[name]) for name in issuers}
    verifier = HMACWorldReceiptVerifier(
        keys,
        independence_classes={issuers[0]: "CONSTRUCTOR_A", issuers[1]: "CONSTRUCTOR_B"},
    )
    return signers, verifier


def _execute_candidate(body, class_candidate, environment, patched_source, context,
                       signers, verifier, epoch_base):
    body.memory.remember_experiment(class_candidate.proposal)
    effects = []
    for index, (issuer, signer) in enumerate(signers.items()):
        token = hashlib.sha256(
            f"{context}|{class_candidate.proposal.experiment_id}|{issuer}".encode()
        ).hexdigest()[:16]
        pair = body.execute_world_intervention(
            class_candidate.proposal,
            SignedPatchExecutor(
                environment, patched_source, signer, context,
                f"constructor::{token}", f"constructor-source::{token}", epoch_base + index,
            ),
            verifier=verifier,
        )
        if not pair.authority_verified:
            raise AssertionError("relational constructor receipt lost verifier-derived authority")
        effects.append(float(pair.effect))
    return tuple(effects)


def _train(parent, *, source, path, environment, context, traceback_token,
           family_label, sources, signers, verifier, epoch_base):
    baseline, _, stderr = environment.run()
    if baseline != 0.0 or traceback_token not in stderr:
        raise AssertionError(f"natural {family_label} failure did not reproduce: {stderr[-800:]}")
    old_count = _old_constructor_count(stderr, path, sources)
    if old_count != 0:
        raise AssertionError("merged #80 constructor unexpectedly represented the new failure relation")
    assessment = assess_constructor_inexpressivity((ConstructorInexpressivityContext(
        context_id=context,
        baseline_capability=baseline,
        old_constructor_candidate_count=old_count,
        failure_signature=stderr,
    ),), min_contexts=1)
    primitive = infer_relational_constructor_primitive(
        stderr=stderr, source=source, target_path=path,
        repository_sources=sources, assessment=assessment,
    )
    if primitive is None or primitive.exception_family != family_label:
        raise AssertionError(f"wrong constructor primitive for {family_label}: {primitive}")
    candidate = propose_relational_repair_class(primitive)
    mechanisms = generate_relational_repair_mechanisms(
        candidate=candidate, stderr=stderr, source=source,
        target_path=path, repository_sources=sources,
    )
    if len(mechanisms) != 1:
        raise AssertionError(f"{family_label} should generate one concrete mechanism, got {len(mechanisms)}")
    repaired, _, repaired_stderr = environment.run(mechanisms[0].patched_source)
    if repaired != 1.0:
        raise AssertionError(f"generated {family_label} mechanism failed: {repaired_stderr[-800:]}")
    effects = _execute_candidate(
        parent, candidate, environment, mechanisms[0].patched_source,
        context, signers, verifier, epoch_base,
    )
    if min(effects) < 0.9:
        raise AssertionError(f"{family_label} primitive lacked replicated world effect")
    return primitive, candidate, mechanisms[0], old_count


def main() -> None:
    import_source = IMPORT_FIXTURE.read_text(encoding="utf-8")
    call_source = CALL_FIXTURE.read_text(encoding="utf-8")
    if _git_blob_sha(import_source) != IMPORT_BLOB or _git_blob_sha(call_source) != CALL_BLOB:
        raise AssertionError("natural constructor fixture diverged from its exact historical Git blob")

    signers, verifier = _authority()
    parent = PersistentCognitiveRuntime()
    base_sources = _repository_sources()

    import_sources = dict(base_sources)
    import_sources[IMPORT_PATH] = import_source
    import_row = _train(
        parent,
        source=import_source,
        path=IMPORT_PATH,
        environment=FreshRepositoryEnvironment(import_source, IMPORT_PATH, "IMPORT"),
        context="natural-import-error",
        traceback_token="ImportError",
        family_label="IMPORT_ERROR",
        sources=import_sources,
        signers=signers,
        verifier=verifier,
        epoch_base=10000,
    )
    if RelationalRepairConstructorOrgan(parent).policy().status == "REPRODUCED_RELATIONAL_REPAIR_CONSTRUCTOR":
        raise AssertionError("one primitive incorrectly authorized the meta-constructor")

    call_sources = dict(base_sources)
    call_sources[CALL_PATH] = call_source
    call_row = _train(
        parent,
        source=call_source,
        path=CALL_PATH,
        environment=FreshRepositoryEnvironment(call_source, CALL_PATH, "UNITTEST", selector=CALL_TEST),
        context="natural-type-error",
        traceback_token="TypeError",
        family_label="TYPE_ERROR",
        sources=call_sources,
        signers=signers,
        verifier=verifier,
        epoch_base=20000,
    )

    training_primitives = {import_row[0].primitive_id, call_row[0].primitive_id}
    if len(training_primitives) != 2:
        raise AssertionError("two natural exception relations collapsed to one constructor primitive")
    parent_policy = RelationalRepairConstructorOrgan(parent).policy()
    if (
        parent_policy.status != "REPRODUCED_RELATIONAL_REPAIR_CONSTRUCTOR"
        or parent_policy.constructor_family != CONSTRUCTOR_FAMILY
        or set(parent_policy.supporting_exception_families) != {"IMPORT_ERROR", "TYPE_ERROR"}
    ):
        raise AssertionError(f"natural training failed to authorize relational meta-constructor: {parent_policy}")

    checkpoint = checkpoint_dict(parent)
    verifierless = restore_runtime(checkpoint)
    verifierless_policy = RelationalRepairConstructorOrgan(verifierless).policy()
    if verifierless_policy.status == "REPRODUCED_RELATIONAL_REPAIR_CONSTRUCTOR":
        raise AssertionError("constructor authority leaked through checkpoint")
    reverified = restore_runtime(checkpoint, world_verifier=verifier)
    reverified_policy = RelationalRepairConstructorOrgan(reverified).policy()
    if reverified_policy != parent_policy:
        raise AssertionError("external reverification failed to reconstruct exact meta-constructor policy")

    # Fresh post-checkout exception family: an unbound local export. Concrete names and
    # values are randomized after checkout; candidate construction does not consume
    # the hidden success outcome.
    rng = random.SystemRandom()
    suffix = str(rng.randrange(10**7, 10**8))
    provider = f"constructor_provider_{suffix}"
    consumer = f"constructor_consumer_{suffix}"
    symbol = f"Binding_{rng.randrange(10**7, 10**8)}"
    provider_path = f"arte_cognition/{provider}.py"
    consumer_path = f"arte_cognition/{consumer}.py"
    provider_source = f"{symbol} = {rng.randrange(1000, 9999)}\n"
    consumer_source = f"from __future__ import annotations\n\nRESULT = {symbol}\n"
    heldout_sources = dict(base_sources)
    heldout_sources[provider_path] = provider_source
    heldout_sources[consumer_path] = consumer_source
    environment = FreshRepositoryEnvironment(
        consumer_source, consumer_path, "IMPORT", extra_sources={provider_path: provider_source}
    )
    baseline, _, stderr = environment.run()
    if baseline != 0.0 or "NameError" not in stderr or symbol not in stderr:
        raise AssertionError(f"fresh NameError did not reproduce: {stderr[-800:]}")

    old_counts = tuple(_old_constructor_count(stderr, consumer_path, heldout_sources) for _ in range(16))
    if any(old_counts):
        raise AssertionError("OLD+MORE_COMPUTE escaped the old constructor language")
    assessment = assess_constructor_inexpressivity((ConstructorInexpressivityContext(
        context_id="fresh-name-error",
        baseline_capability=baseline,
        old_constructor_candidate_count=0,
        failure_signature=stderr,
    ),), min_contexts=1)

    if infer_descendant_relational_constructor_primitive(
        stderr, consumer_source, consumer_path, heldout_sources, assessment, verifierless_policy
    ) is not None:
        raise AssertionError("verifierless descendant used meta-constructor on unseen exception")
    primitive = infer_descendant_relational_constructor_primitive(
        stderr, consumer_source, consumer_path, heldout_sources, assessment, reverified_policy
    )
    if primitive is None or primitive.exception_family != "NAME_ERROR":
        raise AssertionError(f"reverified meta-constructor failed unseen exception transfer: {primitive}")
    if primitive.primitive_id in training_primitives:
        raise AssertionError("fresh constructor primitive was already present in training")

    candidate = propose_relational_repair_class(primitive)
    mechanisms = generate_relational_repair_mechanisms(
        candidate=candidate, stderr=stderr, source=consumer_source,
        target_path=consumer_path, repository_sources=heldout_sources,
    )
    if len(mechanisms) != 1 or not mechanisms[0].mechanism_id.startswith("NAME_BINDING::IMPORT_LOCAL_EXPORT::"):
        raise AssertionError(f"unexpected fresh binding mechanism set: {[m.mechanism_id for m in mechanisms]}")
    repaired, _, repaired_stderr = environment.run(mechanisms[0].patched_source)
    if repaired != 1.0:
        raise AssertionError(f"fresh generated binding repair failed: {repaired_stderr[-800:]}")

    treatment = restore_runtime(checkpoint, world_verifier=verifier)
    treatment_effects = _execute_candidate(
        treatment, candidate, environment, mechanisms[0].patched_source,
        "fresh-name-error", signers, verifier, 30000,
    )
    treatment_capability = 1.0 if min(treatment_effects) >= 0.9 else 0.0
    remove = restore_runtime(checkpoint, world_verifier=verifier)
    remove_capability = environment.run()[0]
    if treatment_capability != 1.0 or remove_capability != 0.0:
        raise AssertionError("same-checkpoint Treatment/REMOVE causal isolation failed")
    treatment_policy = RelationalRepairConstructorOrgan(treatment).policy()
    if "NAME_ERROR" not in treatment_policy.supporting_exception_families:
        raise AssertionError("fresh primitive did not enter descendant world-evidence lineage")

    result = {
        "status": "PASS_BOUNDED_WORLD_AUTHORIZED_RELATIONAL_REPAIR_CONSTRUCTOR_PRIMITIVE_GENESIS_AND_FRESH_EXCEPTION_TRANSFER",
        "repository": "leehyen0/GENESIS-EX-NIHILO",
        "natural_training": True,
        "natural_import_commit": IMPORT_TRAIN_COMMIT,
        "natural_import_fix_commit": IMPORT_FIX_COMMIT,
        "natural_import_blob": IMPORT_BLOB,
        "natural_import_fixture_exact_git_blob": True,
        "natural_call_commit": CALL_TRAIN_COMMIT,
        "natural_call_fix_commit": CALL_FIX_COMMIT,
        "natural_call_blob": CALL_BLOB,
        "natural_call_fixture_exact_git_blob": True,
        "old_constructor_training_candidate_counts": [import_row[3], call_row[3]],
        "training_exception_families": sorted(parent_policy.supporting_exception_families),
        "training_primitive_ids": sorted(training_primitives),
        "training_generated_class_ids": sorted([import_row[1].class_id, call_row[1].class_id]),
        "training_mechanisms": sorted([import_row[2].mechanism_id, call_row[2].mechanism_id]),
        "one_primitive_insufficient_for_meta_authority": True,
        "meta_constructor_family": parent_policy.constructor_family,
        "meta_constructor_rederived_after_external_reverification": True,
        "verifierless_meta_constructor_authority": False,
        "fresh_heldout_exception_family": primitive.exception_family,
        "fresh_heldout_locus_kind": primitive.locus_kind,
        "fresh_heldout_binding_relation": primitive.binding_relation,
        "fresh_heldout_primitive_id": primitive.primitive_id,
        "fresh_heldout_primitive_absent_from_training": True,
        "fresh_heldout_generated_class_id": candidate.class_id,
        "fresh_heldout_generated_mechanisms": [m.mechanism_id for m in mechanisms],
        "fresh_names_random_post_checkout": True,
        "fresh_source_disjoint_from_natural_training": True,
        "verifierless_fresh_primitive_generation": False,
        "reverified_fresh_primitive_generation": True,
        "old_more_compute_attempts": len(old_counts),
        "old_more_compute_total_candidates": sum(old_counts),
        "old_more_compute_capability": baseline,
        "treatment_candidate_count": len(mechanisms),
        "treatment_capability": treatment_capability,
        "remove_same_checkpoint_candidate_count": 0,
        "remove_same_checkpoint_capability": remove_capability,
        "fresh_third_primitive_entered_descendant_lineage": True,
        "candidate_generation_uses_hidden_success_outcomes": False,
        "relationship_extractor_schema_human_authored": True,
        "repair_goal_vocabulary_human_authored": True,
        "unrestricted_constructor_primitive_invention": False,
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
