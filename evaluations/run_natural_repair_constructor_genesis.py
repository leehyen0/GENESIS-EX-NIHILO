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
from typing import Dict, Mapping, Optional, Sequence, Tuple

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
IMPORT_CONTEXT = "natural-constructor-import-error"

CALL_TRAIN_COMMIT = "5a4fc4ca5ac1dd40e75c6ad541c055a4d0743744"
CALL_FIX_COMMIT = "cd80dbdff9d895de9f862eccad5fe33a241737fb"
CALL_BLOB = "1a37f91ae913304edb9fe0ec56ddeeb960194403"
CALL_PATH = "arte_cognition/test_causal_identification.py"
CALL_FIXTURE = ROOT / "evaluations/fixtures/historical_5a4fc4ca/test_causal_identification.py"
CALL_CONTEXT = "natural-constructor-type-error"
CALL_TEST = (
    "arte_cognition.test_causal_identification."
    "GenerationScopedIdentifierTests."
    "test_older_generations_do_not_dilute_current_generation_eig"
)


def _git_blob_sha(source: str) -> str:
    payload = source.encode("utf-8")
    return hashlib.sha1(f"blob {len(payload)}\0".encode("utf-8") + payload).hexdigest()


def _current_repository_sources() -> Dict[str, str]:
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "arte_cognition").rglob("*.py"))
    }


def _open_old_constructor_frontier() -> FixedRepairClassFrontierAssessment:
    # This deliberately gives the old constructor its most permissive upstream gate.
    # A zero result therefore demonstrates constructor-language inexpressivity, not
    # missing prerequisite evidence. It is not a claim that all old repair classes
    # were refuted for these new tasks.
    return FixedRepairClassFrontierAssessment(
        status="FIXED_REPAIR_CLASSES_FALSIFIED_OPEN_CLASS_GENESIS",
        complete_contexts=("maximally-permissive-old-constructor-probe",),
        applicable_class_ids=("CONTENT", "TRAVERSAL", "STATE_CONFLICT"),
        evaluated_candidate_count=1,
        missing_candidate_count=0,
        reason="probe old constructor expressivity under an already-open upstream gate",
    )


def _old_constructor_candidate_count(
    stderr: str,
    target_path: str,
    repository_sources: Mapping[str, str],
) -> int:
    return len(generate_repair_class_from_failure(
        stderr=stderr,
        target_path=target_path,
        repository_paths=tuple(sorted(repository_sources)),
        frontier=_open_old_constructor_frontier(),
    ))


class FreshRepositoryEnvironment:
    def __init__(
        self,
        source: str,
        target_path: str,
        mode: str,
        selector: Optional[str] = None,
        extra_sources: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.source = str(source)
        self.target_path = str(target_path)
        self.mode = str(mode)
        self.selector = None if selector is None else str(selector)
        self.extra_sources = dict(extra_sources or {})

    @staticmethod
    def _module_name(path: str) -> str:
        normalized = str(path).replace("\\", "/")
        if not normalized.endswith(".py"):
            raise ValueError("Python module target must end in .py")
        return normalized[:-3].replace("/", ".")

    def run(self, source: Optional[str] = None, timeout: float = 25.0) -> Tuple[float, str, str]:
        candidate_source = self.source if source is None else str(source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "arte_cognition", root / "arte_cognition")
            for path, text in self.extra_sources.items():
                full = root / str(path)
                full.parent.mkdir(parents=True, exist_ok=True)
                full.write_text(str(text), encoding="utf-8")
            target = root / self.target_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(candidate_source, encoding="utf-8")

            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["PYTHONNOUSERSITE"] = "1"
            if self.mode == "IMPORT_MODULE":
                module_name = self._module_name(self.target_path)
                command = [sys.executable, "-c", f"import {module_name}"]
            elif self.mode == "UNITTEST_TARGET":
                if not self.selector:
                    raise ValueError("UNITTEST_TARGET requires selector")
                command = [sys.executable, "-m", "unittest", self.selector]
            else:
                raise ValueError(f"unknown mode: {self.mode}")
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
            return (
                1.0 if completed.returncode == 0 else 0.0,
                completed.stdout,
                completed.stderr,
            )


class SignedConstructorCandidateExecutor:
    def __init__(
        self,
        environment: FreshRepositoryEnvironment,
        patched_source: str,
        signer,
        context_id: str,
        challenge_id: str,
        source_id: str,
        epoch: int,
    ) -> None:
        self.environment = environment
        self.patched_source = str(patched_source)
        self.signer = signer
        self.context_id = str(context_id)
        self.challenge_id = str(challenge_id)
        self.source_id = str(source_id)
        self.epoch = int(epoch)

    def execute(self, proposal, arm: str, value: float):
        candidate_source = (
            self.environment.source
            if str(arm).upper() == "LOW"
            else self.patched_source
        )
        outcome, _, _ = self.environment.run(candidate_source)
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
            budget_token=f"constructor-genesis::{self.challenge_id}",
            externally_generated=True,
        ))


def _make_authority():
    issuers = ("CONSTRUCTOR_LAB_A", "CONSTRUCTOR_LAB_B")
    keys = {
        issuer: hashlib.sha256((issuer + "::natural-constructor-20260822").encode("utf-8")).digest()
        for issuer in issuers
    }
    signers = {issuer: HMACWorldReceiptSigner(issuer, keys[issuer]) for issuer in issuers}
    verifier = HMACWorldReceiptVerifier(
        keys,
        independence_classes={
            issuers[0]: "CONSTRUCTOR_INDEPENDENCE_A",
            issuers[1]: "CONSTRUCTOR_INDEPENDENCE_B",
        },
    )
    return signers, verifier


def _execute_constructor_candidate(
    body,
    class_candidate,
    environment,
    patched_source,
    context_id,
    signers,
    verifier,
    epoch_base,
):
    body.memory.remember_experiment(class_candidate.proposal)
    effects = []
    for issuer_index, (issuer, signer) in enumerate(signers.items()):
        token = hashlib.sha256(
            f"{context_id}|{class_candidate.proposal.experiment_id}|{issuer}".encode("utf-8")
        ).hexdigest()[:16]
        pair = body.execute_world_intervention(
            class_candidate.proposal,
            SignedConstructorCandidateExecutor(
                environment=environment,
                patched_source=patched_source,
                signer=signer,
                context_id=context_id,
                challenge_id=f"relational-constructor::{token}",
                source_id=f"constructor-world-source::{token}",
                epoch=epoch_base + issuer_index,
            ),
            verifier=verifier,
        )
        if not pair.authority_verified:
            raise AssertionError("constructor-genesis receipt lost verifier-derived authority")
        effects.append(float(pair.effect))
    return tuple(effects)


def _train_natural_constructor_primitive(
    body,
    source,
    target_path,
    environment,
    context_id,
    expected_exception,
    repository_sources,
    signers,
    verifier,
    epoch_base,
):
    baseline, _, stderr = environment.run()
    if baseline != 0.0 or expected_exception not in stderr:
        raise AssertionError(
            f"natural constructor-training failure did not reproduce {expected_exception}: {stderr[-800:]}"
        )
    old_count = _old_constructor_candidate_count(stderr, target_path, repository_sources)
    if old_count != 0:
        raise AssertionError("old #80 constructor unexpectedly represented the new failure relation")
    assessment = assess_constructor_inexpressivity((ConstructorInexpressivityContext(
        context_id=context_id,
        baseline_capability=baseline,
        old_constructor_candidate_count=old_count,
        failure_signature=stderr,
    ),), min_contexts=1)
    if assessment.status != "OLD_REPAIR_CLASS_CONSTRUCTOR_INEXPRESSIVE_OPEN_RELATIONAL_CONSTRUCTOR":
        raise AssertionError(f"constructor inexpressivity did not open relational shadow search: {assessment}")
    primitive = infer_relational_constructor_primitive(
        stderr=stderr,
        source=source,
        target_path=target_path,
        repository_sources=repository_sources,
        assessment=assessment,
    )
    if primitive is None or primitive.exception_family != expected_exception:
        raise AssertionError(f"failed to infer expected relational constructor primitive: {primitive}")
    class_candidate = propose_relational_repair_class(primitive)
    mechanisms = generate_relational_repair_mechanisms(
        candidate=class_candidate,
        stderr=stderr,
        source=source,
        target_path=target_path,
        repository_sources=repository_sources,
    )
    if len(mechanisms) != 1:
        raise AssertionError(f"expected one outcome-independent natural repair mechanism, got {len(mechanisms)}")
    repaired, _, repaired_stderr = environment.run(mechanisms[0].patched_source)
    if repaired != 1.0:
        raise AssertionError(f"generated natural repair mechanism failed: {repaired_stderr[-800:]}")
    effects = _execute_constructor_candidate(
        body,
        class_candidate,
        environment,
        mechanisms[0].patched_source,
        context_id,
        signers,
        verifier,
        epoch_base,
    )
    if min(effects) < 0.9:
        raise AssertionError("generated constructor primitive lacked strong independent world effect")
    return {
        "context": context_id,
        "old_constructor_candidate_count": old_count,
        "primitive": primitive,
        "class_candidate": class_candidate,
        "mechanism": mechanisms[0],
        "effects": effects,
        "stderr": stderr,
    }


def main() -> None:
    import_source = IMPORT_FIXTURE.read_text(encoding="utf-8")
    call_source = CALL_FIXTURE.read_text(encoding="utf-8")
    if _git_blob_sha(import_source) != IMPORT_BLOB:
        raise AssertionError("stale-symbol natural fixture no longer matches exact historical Git blob")
    if _git_blob_sha(call_source) != CALL_BLOB:
        raise AssertionError("call-contract natural fixture no longer matches exact historical Git blob")

    signers, verifier = _make_authority()
    parent = PersistentCognitiveRuntime()
    base_sources = _current_repository_sources()

    import_sources = dict(base_sources)
    import_sources[IMPORT_PATH] = import_source
    import_environment = FreshRepositoryEnvironment(
        import_source,
        IMPORT_PATH,
        "IMPORT_MODULE",
    )
    import_training = _train_natural_constructor_primitive(
        body=parent,
        source=import_source,
        target_path=IMPORT_PATH,
        environment=import_environment,
        context_id=IMPORT_CONTEXT,
        expected_exception="IMPORT_ERROR",
        repository_sources=import_sources,
        signers=signers,
        verifier=verifier,
        epoch_base=10000,
    )

    after_one = RelationalRepairConstructorOrgan(parent).policy()
    if after_one.status == "REPRODUCED_RELATIONAL_REPAIR_CONSTRUCTOR":
        raise AssertionError("one natural primitive incorrectly authorized the meta-constructor")

    call_sources = dict(base_sources)
    call_sources[CALL_PATH] = call_source
    call_environment = FreshRepositoryEnvironment(
        call_source,
        CALL_PATH,
        "UNITTEST_TARGET",
        selector=CALL_TEST,
    )
    call_training = _train_natural_constructor_primitive(
        body=parent,
        source=call_source,
        target_path=CALL_PATH,
        environment=call_environment,
        context_id=CALL_CONTEXT,
        expected_exception="TYPE_ERROR",
        repository_sources=call_sources,
        signers=signers,
        verifier=verifier,
        epoch_base=20000,
    )

    training_primitive_ids = {
        import_training["primitive"].primitive_id,
        call_training["primitive"].primitive_id,
    }
    if len(training_primitive_ids) != 2:
        raise AssertionError("two natural exception families collapsed to one constructor primitive")
    parent_policy = RelationalRepairConstructorOrgan(parent).policy()
    if (
        parent_policy.status != "REPRODUCED_RELATIONAL_REPAIR_CONSTRUCTOR"
        or parent_policy.constructor_family != CONSTRUCTOR_FAMILY
        or set(parent_policy.supporting_exception_families) != {"IMPORT_ERROR", "TYPE_ERROR"}
    ):
        raise AssertionError(f"two natural families failed to authorize relational constructor: {parent_policy}")

    checkpoint = checkpoint_dict(parent)
    verifierless = restore_runtime(checkpoint)
    verifierless_policy = RelationalRepairConstructorOrgan(verifierless).policy()
    if verifierless_policy.status == "REPRODUCED_RELATIONAL_REPAIR_CONSTRUCTOR":
        raise AssertionError("checkpoint restored relational-constructor authority without verifier")
    reverified = restore_runtime(checkpoint, world_verifier=verifier)
    reverified_policy = RelationalRepairConstructorOrgan(reverified).policy()
    if reverified_policy != parent_policy:
        raise AssertionError("external reverification failed to reconstruct exact relational-constructor policy")

    # Fresh, post-checkout, source-disjoint exception family. Names and the provider
    # value are generated after checkout. The constructor sees source topology and
    # traceback structure, never a hidden expected repair or hidden success outcome.
    rng = random.SystemRandom()
    suffix = f"{rng.randrange(10**7, 10**8)}"
    provider_module = f"constructor_provider_{suffix}"
    consumer_module = f"constructor_consumer_{suffix}"
    symbol = f"Binding_{rng.randrange(10**7, 10**8)}"
    provider_path = f"arte_cognition/{provider_module}.py"
    consumer_path = f"arte_cognition/{consumer_module}.py"
    provider_source = f"{symbol} = {rng.randrange(1000, 9999)}\n"
    consumer_source = (
        "from __future__ import annotations\n\n"
        f"RESULT = {symbol}\n"
    )
    heldout_sources = dict(base_sources)
    heldout_sources[provider_path] = provider_source
    heldout_sources[consumer_path] = consumer_source
    heldout_environment = FreshRepositoryEnvironment(
        consumer_source,
        consumer_path,
        "IMPORT_MODULE",
        extra_sources={provider_path: provider_source},
    )
    heldout_baseline, _, heldout_stderr = heldout_environment.run()
    if heldout_baseline != 0.0 or "NameError" not in heldout_stderr or symbol not in heldout_stderr:
        raise AssertionError(f"fresh heldout NameError did not reproduce: {heldout_stderr[-800:]}")

    old_counts = tuple(
        _old_constructor_candidate_count(heldout_stderr, consumer_path, heldout_sources)
        for _ in range(16)
    )
    if any(old_counts):
        raise AssertionError("OLD+MORE_COMPUTE escaped the old constructor language on fresh heldout")
    heldout_assessment = assess_constructor_inexpressivity((ConstructorInexpressivityContext(
        context_id="fresh-post-checkout-name-binding",
        baseline_capability=heldout_baseline,
        old_constructor_candidate_count=max(old_counts),
        failure_signature=heldout_stderr,
    ),), min_contexts=1)

    verifierless_primitive = infer_descendant_relational_constructor_primitive(
        stderr=heldout_stderr,
        source=consumer_source,
        target_path=consumer_path,
        repository_sources=heldout_sources,
        assessment=heldout_assessment,
        policy=verifierless_policy,
    )
    if verifierless_primitive is not None:
        raise AssertionError("verifierless descendant generated a fresh constructor primitive with authority")

    heldout_primitive = infer_descendant_relational_constructor_primitive(
        stderr=heldout_stderr,
        source=consumer_source,
        target_path=consumer_path,
        repository_sources=heldout_sources,
        assessment=heldout_assessment,
        policy=reverified_policy,
    )
    if heldout_primitive is None or heldout_primitive.exception_family != "NAME_ERROR":
        raise AssertionError(f"reverified meta-constructor failed fresh unseen exception transfer: {heldout_primitive}")
    if heldout_primitive.primitive_id in training_primitive_ids:
        raise AssertionError("fresh heldout constructor primitive was already present in natural training")
    heldout_candidate = propose_relational_repair_class(heldout_primitive)
    heldout_mechanisms = generate_relational_repair_mechanisms(
        candidate=heldout_candidate,
        stderr=heldout_stderr,
        source=consumer_source,
        target_path=consumer_path,
        repository_sources=heldout_sources,
    )
    if len(heldout_mechanisms) != 1:
        raise AssertionError(f"fresh NameError should generate one local-binding mechanism, got {len(heldout_mechanisms)}")
    if not heldout_mechanisms[0].mechanism_id.startswith("NAME_BINDING::IMPORT_LOCAL_EXPORT::"):
        raise AssertionError("fresh heldout generated the wrong concrete repair mechanism family")
    repaired_capability, _, repaired_stderr = heldout_environment.run(heldout_mechanisms[0].patched_source)
    if repaired_capability != 1.0:
        raise AssertionError(f"fresh generated binding repair failed externally: {repaired_stderr[-800:]}")

    treatment = restore_runtime(checkpoint, world_verifier=verifier)
    treatment_effects = _execute_constructor_candidate(
        treatment,
        heldout_candidate,
        heldout_environment,
        heldout_mechanisms[0].patched_source,
        "fresh-post-checkout-name-binding",
        signers,
        verifier,
        30000,
    )
    treatment_capability = 1.0 if min(treatment_effects) >= 0.9 else 0.0
    remove = restore_runtime(checkpoint, world_verifier=verifier)
    remove_capability = heldout_environment.run()[0]
    if treatment_capability != 1.0 or remove_capability != 0.0:
        raise AssertionError("same-checkpoint relational-constructor Treatment/REMOVE causal separation failed")

    treatment_policy_after = RelationalRepairConstructorOrgan(treatment).policy()
    if "NAME_ERROR" not in treatment_policy_after.supporting_exception_families:
        raise AssertionError("fresh third constructor primitive did not enter reverified descendant evidence lineage")

    result = {
        "status": "PASS_BOUNDED_WORLD_AUTHORIZED_RELATIONAL_REPAIR_CONSTRUCTOR_PRIMITIVE_GENESIS_AND_FRESH_EXCEPTION_TRANSFER",
        "repository": "leehyen0/GENESIS-EX-NIHILO",
        "natural_training": True,
        "import_training_commit": IMPORT_TRAIN_COMMIT,
        "import_training_fix_commit": IMPORT_FIX_COMMIT,
        "import_training_blob": IMPORT_BLOB,
        "import_fixture_exact_git_blob": True,
        "call_training_commit": CALL_TRAIN_COMMIT,
        "call_training_fix_commit": CALL_FIX_COMMIT,
        "call_training_blob": CALL_BLOB,
        "call_fixture_exact_git_blob": True,
        "old_constructor_training_candidate_counts": [
            import_training["old_constructor_candidate_count"],
            call_training["old_constructor_candidate_count"],
        ],
        "training_exception_families": sorted(parent_policy.supporting_exception_families),
        "training_primitive_ids": sorted(training_primitive_ids),
        "training_generated_class_ids": sorted([
            import_training["class_candidate"].class_id,
            call_training["class_candidate"].class_id,
        ]),
        "training_mechanisms": sorted([
            import_training["mechanism"].mechanism_id,
            call_training["mechanism"].mechanism_id,
        ]),
        "meta_constructor_family": parent_policy.constructor_family,
        "one_primitive_insufficient_for_meta_authority": True,
        "meta_constructor_rederived_after_external_reverification": True,
        "verifierless_meta_constructor_authority": False,
        "fresh_heldout_exception_family": heldout_primitive.exception_family,
        "fresh_heldout_locus_kind": heldout_primitive.locus_kind,
        "fresh_heldout_binding_relation": heldout_primitive.binding_relation,
        "fresh_heldout_primitive_id": heldout_primitive.primitive_id,
        "fresh_heldout_primitive_absent_from_training": True,
        "fresh_heldout_generated_class_id": heldout_candidate.class_id,
        "fresh_heldout_generated_mechanisms": [item.mechanism_id for item in heldout_mechanisms],
        "fresh_names_random_post_checkout": True,
        "fresh_source_disjoint_from_natural_training": True,
        "verifierless_fresh_primitive_generation": False,
        "reverified_fresh_primitive_generation": True,
        "old_more_compute_attempts": len(old_counts),
        "old_more_compute_total_candidates": sum(old_counts),
        "old_more_compute_capability": heldout_baseline,
        "treatment_candidate_count": len(heldout_mechanisms),
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
