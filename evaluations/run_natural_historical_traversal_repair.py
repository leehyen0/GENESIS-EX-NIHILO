from __future__ import annotations

import ast
import hashlib
import json
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
from arte_cognition.software_repair_grammar_expansion import (
    SoftwareRepairAlphabetAssessment,
    PythonArithmeticRepairGenerator,
    assess_software_repair_alphabet_failure,
    derive_generated_software_repair_policy,
    select_generated_software_repairs,
)
from arte_cognition.software_structural_repair import PythonTraversalStrategyRepairGenerator
from arte_cognition.software_task_acquisition import PythonASTRepairGenerator, SoftwarePatchCandidate
from arte_cognition.world_coupling import HMACWorldReceiptSigner, HMACWorldReceiptVerifier, WorldOutcomeReceipt


HISTORICAL_COMMIT = "88f2e74f5164731b1370f74d76f8c58b584a959c"
HISTORICAL_BLOB = "cbc73d77b34380d19f3540a52c1f77e4e90f9fbf"
HISTORICAL_PATH = "arte_cognition/software_task_acquisition.py"
FIXTURE = ROOT / "evaluations/fixtures/historical_88f2e74/software_task_acquisition.py"
EXPECTED_STRATEGY = "TRAVERSAL::DFS_POST"


PROBE_CONTEXTS: Mapping[str, Tuple[str, ...]] = {
    "historical-regression-a": (
        '''
def probe(a, b, c, d):
    if (a == b and c != d) or a > c:
        return a <= d
    return b < c
''',
        '''
def probe(a, b, c):
    return (a >= b or b == c) and a != c
''',
    ),
    "historical-regression-b": (
        '''
def probe(xs, limit):
    return [x for x in xs if (x >= limit or x == 0) and x != -1]
''',
        '''
def probe(a, b, c, d):
    left = a < b and (c >= d or a == c)
    return left or b != d
''',
    ),
    "historical-heldout-c": (
        '''
def probe(a, b, c, d, e):
    flag = a < b or (c >= d and e == a)
    return flag and b != d
''',
        '''
def probe(a, b, c, d):
    return (a <= b and c > d) or (a != d and b == c)
''',
    ),
}


def _git_blob_sha(source: str) -> str:
    payload = source.encode("utf-8")
    return hashlib.sha1(f"blob {len(payload)}\0".encode("utf-8") + payload).hexdigest()


def _permissive_arithmetic_assessment() -> SoftwareRepairAlphabetAssessment:
    return SoftwareRepairAlphabetAssessment(
        status="SOFTWARE_REPAIR_ALPHABET_FALSIFIED_OPEN_NEXT",
        complete_contexts=(),
        falsified_contexts=(),
        supported_contexts=(),
        missing_experiment_ids=(),
        evaluated_candidate_count=0,
        reason="enumerate the already-existing arithmetic repair alphabet for historical completeness control",
    )


def generate_existing_content_alphabet(source: str) -> Tuple[SoftwarePatchCandidate, ...]:
    base = PythonASTRepairGenerator().generate("historical-old-alphabet", source)
    arithmetic = PythonArithmeticRepairGenerator().generate(
        "historical-old-arithmetic",
        source,
        _permissive_arithmetic_assessment(),
    )
    unique: Dict[str, SoftwarePatchCandidate] = {}
    for candidate in (*base, *arithmetic):
        unique[candidate.proposal.experiment_id] = candidate
    return tuple(unique[key] for key in sorted(unique))


class HistoricalTraversalExecutor:
    """Execute a historical generator candidate against evaluator-owned nested AST regressions."""

    _HARNESS = r'''
import ast, json, sys
from arte_cognition.software_task_acquisition import PythonASTRepairGenerator

probes = json.loads(sys.stdin.read())["probes"]

COMPARE_IDS = {
    "Eq": "COMPARE::Eq->NotEq",
    "NotEq": "COMPARE::NotEq->Eq",
    "Gt": "COMPARE::Gt->GtE",
    "GtE": "COMPARE::GtE->Gt",
    "Lt": "COMPARE::Lt->LtE",
    "LtE": "COMPARE::LtE->Lt",
}
BOOL_IDS = {"And": "BOOL::And->Or", "Or": "BOOL::Or->And"}


def sites(source):
    records = []
    def visit(node):
        for child in ast.iter_child_nodes(node):
            visit(child)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op_name = type(node.ops[0]).__name__
            if op_name in COMPARE_IDS:
                records.append((op_name, COMPARE_IDS[op_name]))
        elif isinstance(node, ast.BoolOp):
            op_name = type(node.op).__name__
            if op_name in BOOL_IDS:
                records.append((op_name, BOOL_IDS[op_name]))
    visit(ast.parse(source))
    return records


def patch_is_exact(source, candidate, expected_records):
    patched_records = sites(candidate.patched_source)
    if len(patched_records) != len(expected_records):
        return False
    changed = [i for i, (before, after) in enumerate(zip(expected_records, patched_records)) if before[0] != after[0]]
    if changed != [candidate.site_index]:
        return False
    if candidate.operator_id != expected_records[candidate.site_index][1]:
        return False
    before = expected_records[candidate.site_index][0]
    after = patched_records[candidate.site_index][0]
    allowed = {
        ("Eq", "NotEq"), ("NotEq", "Eq"),
        ("Gt", "GtE"), ("GtE", "Gt"),
        ("Lt", "LtE"), ("LtE", "Lt"),
        ("And", "Or"), ("Or", "And"),
    }
    return (before, after) in allowed


ok = True
for index, source in enumerate(probes):
    expected = sites(source)
    try:
        candidates = PythonASTRepairGenerator().generate(f"hidden-regression-{index}", source)
    except Exception:
        ok = False
        break
    if [candidate.operator_id for candidate in candidates] != [item[1] for item in expected]:
        ok = False
        break
    if len(candidates) != len(expected):
        ok = False
        break
    if not all(patch_is_exact(source, candidate, expected) for candidate in candidates):
        ok = False
        break
print(json.dumps({"ok": bool(ok)}))
'''

    def __init__(
        self,
        baseline_source: str,
        patched_source: str,
        probes: Sequence[str],
        signer,
        source_id: str,
        context_id: str,
        challenge_id: str,
        epoch: int,
        timeout_seconds: float = 8.0,
    ) -> None:
        self.baseline_source = str(baseline_source)
        self.patched_source = str(patched_source)
        self.probes = tuple(str(item) for item in probes)
        self.signer = signer
        self.source_id = str(source_id)
        self.context_id = str(context_id)
        self.challenge_id = str(challenge_id)
        self.epoch = int(epoch)
        self.timeout_seconds = max(1.0, float(timeout_seconds))

    @staticmethod
    def _run(source: str, probes: Sequence[str], timeout_seconds: float = 8.0) -> float:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "arte_cognition", root / "arte_cognition")
            target = root / HISTORICAL_PATH
            target.write_text(str(source), encoding="utf-8")
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", HistoricalTraversalExecutor._HARNESS],
                    input=json.dumps({"probes": list(probes)}),
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

    def execute(self, proposal, arm: str, value: float) -> WorldOutcomeReceipt:
        source = self.baseline_source if str(arm).upper() == "LOW" else self.patched_source
        outcome = self._run(source, self.probes, self.timeout_seconds)
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
            budget_token=f"historical-natural-traversal::{self.challenge_id}",
            externally_generated=True,
        ))


def execute_candidate(body, source, candidate, context_id, signers, verifier, epoch_base):
    effects = []
    for issuer_index, (issuer, signer) in enumerate(signers.items()):
        token = hashlib.sha256(
            f"{context_id}|{candidate.proposal.experiment_id}|{issuer}".encode("utf-8")
        ).hexdigest()[:16]
        executor = HistoricalTraversalExecutor(
            baseline_source=source,
            patched_source=candidate.patched_source,
            probes=PROBE_CONTEXTS[context_id],
            signer=signer,
            source_id=f"historical-regression::{context_id}::{token}::{issuer}",
            context_id=context_id,
            challenge_id=f"historical-natural::{context_id}::{token}::{issuer}",
            epoch=epoch_base + issuer_index,
        )
        pair = body.execute_world_intervention(candidate.proposal, executor, verifier=verifier)
        if not pair.authority_verified:
            raise AssertionError("historical world receipt lost verifier-derived authority")
        effects.append(float(pair.effect))
    return tuple(effects)


def run_candidates(body, source, candidates, context_id, signers, verifier, epoch_base):
    strong = []
    for index, candidate in enumerate(candidates):
        effects = execute_candidate(
            body, source, candidate, context_id, signers, verifier, epoch_base + index * 10
        )
        if min(effects) >= 0.9:
            strong.append(candidate)
    return tuple(strong)


def main():
    historical_source = FIXTURE.read_text(encoding="utf-8")
    if _git_blob_sha(historical_source) != HISTORICAL_BLOB:
        raise AssertionError("historical source fixture no longer matches the original Git blob")

    for context_id, probes in PROBE_CONTEXTS.items():
        if HistoricalTraversalExecutor._run(historical_source, probes) != 0.0:
            raise AssertionError(f"historical bug no longer reproduces in {context_id}")

    issuer_a = "HISTORICAL_REGRESSION_A"
    issuer_b = "HISTORICAL_REGRESSION_B"
    key_a = hashlib.sha256(b"historical-natural-a").digest()
    key_b = hashlib.sha256(b"historical-natural-b").digest()
    signers = {
        issuer_a: HMACWorldReceiptSigner(issuer_a, key_a),
        issuer_b: HMACWorldReceiptSigner(issuer_b, key_b),
    }
    verifier = HMACWorldReceiptVerifier(
        {issuer_a: key_a, issuer_b: key_b},
        independence_classes={issuer_a: "HISTORICAL_CLASS_A", issuer_b: "HISTORICAL_CLASS_B"},
    )

    parent = PersistentCognitiveRuntime()
    old_candidates = generate_existing_content_alphabet(historical_source)
    if not old_candidates:
        raise AssertionError("historical source exposed no existing content-repair candidates")
    for candidate in old_candidates:
        parent.memory.remember_experiment(candidate.proposal)

    old_candidates_by_context = {
        "historical-regression-a": old_candidates,
        "historical-regression-b": old_candidates,
    }
    old_strong_by_context = {}
    for context_index, context_id in enumerate(old_candidates_by_context):
        strong = run_candidates(
            parent,
            historical_source,
            old_candidates,
            context_id,
            signers,
            verifier,
            10000 + context_index * 20000,
        )
        old_strong_by_context[context_id] = strong
        if strong:
            raise AssertionError(
                f"existing content-repair language unexpectedly solved natural historical defect in {context_id}: "
                f"{[(c.site_index, c.operator_id) for c in strong]}"
            )

    assessment = assess_software_repair_alphabet_failure(
        old_candidates_by_context=old_candidates_by_context,
        world_pairs=parent.world_coupling.pairs,
        min_independent_classes=parent.world_coupling.min_independent_classes,
        strong_effect_threshold=0.9,
        min_contexts=2,
    )
    if assessment.status != "SOFTWARE_REPAIR_ALPHABET_FALSIFIED_OPEN_NEXT":
        raise AssertionError(f"historical content-repair alphabet did not fail closed: {assessment}")
    if assessment.missing_experiment_ids:
        raise AssertionError("historical old-alphabet failure used absence as refutation")

    structural_generator = PythonTraversalStrategyRepairGenerator()
    structural_candidates = structural_generator.generate(
        "historical-natural-traversal",
        historical_source,
        assessment,
    )
    if len(structural_candidates) != 4:
        raise AssertionError("bounded traversal metalanguage did not generate four outcome-independent strategies")
    for candidate in structural_candidates:
        parent.memory.remember_experiment(candidate.proposal)

    first_strong = run_candidates(
        parent,
        historical_source,
        structural_candidates,
        "historical-regression-a",
        signers,
        verifier,
        60000,
    )
    if len(first_strong) != 1 or first_strong[0].operator_id != EXPECTED_STRATEGY:
        raise AssertionError(
            f"first natural regression did not uniquely identify DFS_POST: {[c.operator_id for c in first_strong]}"
        )

    second_strong = run_candidates(
        parent,
        historical_source,
        first_strong,
        "historical-regression-b",
        signers,
        verifier,
        70000,
    )
    if len(second_strong) != 1 or second_strong[0].operator_id != EXPECTED_STRATEGY:
        raise AssertionError("second regression context did not reproduce the traversal repair")

    policy = derive_generated_software_repair_policy(
        proposals=(record.proposal for record in parent.memory.experiments.values()),
        world_pairs=parent.world_coupling.pairs,
        min_independent_classes=parent.world_coupling.min_independent_classes,
        strong_effect_threshold=0.9,
        min_contexts=2,
    )
    if policy.status != "REPRODUCED_EXPANDED_SOFTWARE_REPAIR_OPERATOR" or policy.operator_id != EXPECTED_STRATEGY:
        raise AssertionError(f"natural historical structural repair did not enter BODY policy: {policy}")

    checkpoint = checkpoint_dict(parent)
    verifierless = restore_runtime(checkpoint)
    verifierless_policy = derive_generated_software_repair_policy(
        proposals=(record.proposal for record in verifierless.memory.experiments.values()),
        world_pairs=verifierless.world_coupling.pairs,
        min_independent_classes=verifierless.world_coupling.min_independent_classes,
        strong_effect_threshold=0.9,
        min_contexts=2,
    )
    if verifierless_policy.operator_id is not None:
        raise AssertionError("historical structural repair authority restored without external verifier")

    treatment = restore_runtime(checkpoint, world_verifier=verifier)
    treatment_policy = derive_generated_software_repair_policy(
        proposals=(record.proposal for record in treatment.memory.experiments.values()),
        world_pairs=treatment.world_coupling.pairs,
        min_independent_classes=treatment.world_coupling.min_independent_classes,
        strong_effect_threshold=0.9,
        min_contexts=2,
    )
    treatment_candidates = structural_generator.generate(
        "historical-natural-traversal",
        historical_source,
        assessment,
    )
    treatment_selection = select_generated_software_repairs(
        treatment_candidates,
        treatment_policy,
        max_candidates=1,
    )
    treatment_strong = run_candidates(
        treatment,
        historical_source,
        treatment_selection.candidates,
        "historical-heldout-c",
        signers,
        verifier,
        80000,
    )

    remove = restore_runtime(checkpoint, world_verifier=verifier)
    remove_selection = select_generated_software_repairs(
        treatment_candidates,
        policy=None,
        max_candidates=1,
    )
    remove_strong = run_candidates(
        remove,
        historical_source,
        remove_selection.candidates,
        "historical-heldout-c",
        signers,
        verifier,
        90000,
    )

    full = restore_runtime(checkpoint, world_verifier=verifier)
    full_selection = select_generated_software_repairs(
        treatment_candidates,
        policy=None,
        max_candidates=None,
    )
    full_strong = run_candidates(
        full,
        historical_source,
        full_selection.candidates,
        "historical-heldout-c",
        signers,
        verifier,
        100000,
    )

    if len(treatment_strong) != 1 or treatment_strong[0].operator_id != EXPECTED_STRATEGY:
        raise AssertionError("descendant treatment failed fresh hidden regression")
    if remove_strong:
        raise AssertionError("same-budget REMOVE solved heldout without learned traversal strategy")
    if len(full_strong) != 1 or full_strong[0].operator_id != EXPECTED_STRATEGY:
        raise AssertionError(f"full structural search did not uniquely isolate DFS_POST: {[c.operator_id for c in full_strong]}")

    receipt = {
        "status": "PASS_BOUNDED_NATURAL_HISTORICAL_SOFTWARE_BUG_TO_STRUCTURAL_REPAIR_LANGUAGE_AND_DESCENDANT",
        "repository": "leehyen0/GENESIS-EX-NIHILO",
        "natural_historical_bug": True,
        "historical_commit_sha": HISTORICAL_COMMIT,
        "historical_path": HISTORICAL_PATH,
        "historical_blob_sha": HISTORICAL_BLOB,
        "historical_fixture_exact_git_blob": True,
        "later_fixed_source_exposed_to_body": False,
        "bug_family": "AST_REPAIR_SITE_IDENTITY_TRAVERSAL_ORDER_MISMATCH",
        "historical_failure_reproduced_in_contexts": len(PROBE_CONTEXTS),
        "existing_content_repair_candidate_count": len(old_candidates),
        "existing_content_repair_evaluated_candidate_count": assessment.evaluated_candidate_count,
        "existing_content_repair_complete_contexts": len(assessment.complete_contexts),
        "existing_content_repair_falsified_contexts": len(assessment.falsified_contexts),
        "existing_content_repair_missing_candidate_count": len(assessment.missing_experiment_ids),
        "existing_content_repair_capability": 0.0,
        "structural_grammar_opened_only_after_complete_old_failure": True,
        "structural_strategy_candidates": [candidate.operator_id for candidate in structural_candidates],
        "structural_candidate_generation_uses_hidden_outcomes": False,
        "learned_structural_strategy": policy.operator_id,
        "learned_structural_supporting_contexts": len(policy.supporting_contexts),
        "treatment_candidate_count": len(treatment_selection.candidates),
        "treatment_external_pair_count": 2 * len(treatment_selection.candidates),
        "treatment_capability": 1.0,
        "remove_definition": "same checkpoint and same one-candidate budget; remove learned traversal-strategy selection",
        "remove_candidate_count": len(remove_selection.candidates),
        "remove_external_pair_count": 2 * len(remove_selection.candidates),
        "remove_same_checkpoint_capability": 0.0,
        "full_structural_candidate_count": len(full_selection.candidates),
        "full_structural_external_pair_count": 2 * len(full_selection.candidates),
        "full_structural_capability": 1.0,
        "fresh_hidden_regression_context": "historical-heldout-c",
        "verifierless_structural_authority": False,
        "traversal_strategy_metalanguage_human_authored": True,
        "unrestricted_software_operator_invention": False,
        "arbitrary_historical_bug_repair": False,
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
