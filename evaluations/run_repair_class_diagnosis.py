from __future__ import annotations

from dataclasses import dataclass
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
from arte_cognition.software_repair_class_diagnosis import (
    RepairClassCandidate,
    assess_repair_class_applicability,
    derive_repair_class_diagnosis_policy,
    generate_repair_class_candidates,
    normalized_structural_fingerprint,
    select_repair_classes,
)
from arte_cognition.software_repair_grammar_expansion import SoftwareRepairAlphabetAssessment
from arte_cognition.software_state_conflict_repair import PythonStateConflictRepairGenerator
from arte_cognition.software_structural_repair import PythonTraversalStrategyRepairGenerator
from arte_cognition.world_coupling import HMACWorldReceiptSigner, HMACWorldReceiptVerifier, WorldOutcomeReceipt

from evaluations.run_natural_historical_traversal_repair import (
    FIXTURE as TRAVERSAL_FIXTURE,
    PROBE_CONTEXTS as TRAVERSAL_CONTEXTS,
    HistoricalTraversalExecutor,
    generate_existing_content_alphabet as generate_traversal_content,
)
from evaluations.run_natural_historical_provenance_repair import (
    FIXTURE as STATE_FIXTURE,
    CONTEXTS as STATE_CONTEXTS,
    HistoricalStateConflictExecutor,
    generate_existing_content_alphabet as generate_state_content,
)


OPEN_SPECIALIZED = SoftwareRepairAlphabetAssessment(
    status="SOFTWARE_REPAIR_ALPHABET_FALSIFIED_OPEN_NEXT",
    complete_contexts=("prior-world-falsification-a", "prior-world-falsification-b"),
    falsified_contexts=("prior-world-falsification-a", "prior-world-falsification-b"),
    supported_contexts=(),
    missing_experiment_ids=(),
    evaluated_candidate_count=42,
    reason=(
        "repair-class diagnosis may directly reopen a previously validated specialized class for a matching "
        "structural fingerprint; class-level world evidence is revalidated independently"
    ),
)

CONTENT_SOURCES: Mapping[str, str] = {
    "content-train-a": "def alpha(left, right):\n    return left >= right\n",
    "content-train-b": "def beta(value, cutoff):\n    return value >= cutoff\n",
    "content-heldout-c": "def gamma(score, threshold):\n    return score >= threshold\n",
}
CONTENT_CASES = ((0, 0, False), (1, 0, True), (-1, 0, False), (5, 5, False))


@dataclass
class SearchStats:
    class_id: str
    candidate_count: int = 0
    candidate_external_executions: int = 0
    strong_candidate_count: int = 0


def _run_content_source(source: str) -> float:
    harness = r'''
import importlib.util, json
spec = importlib.util.spec_from_file_location("task", "task.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
func = next(value for key, value in vars(module).items() if callable(value) and not key.startswith("__"))
cases = json.loads(open("cases.json", "r", encoding="utf-8").read())
ok = all(bool(func(a,b)) == bool(expected) for a,b,expected in cases)
print(json.dumps({"ok": bool(ok)}))
'''
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "task.py").write_text(str(source), encoding="utf-8")
        (root / "cases.json").write_text(json.dumps(CONTENT_CASES), encoding="utf-8")
        try:
            completed = subprocess.run([sys.executable, "-c", harness], cwd=root, capture_output=True, text=True, timeout=5.0, check=False)
            if completed.returncode != 0:
                return 0.0
            return 1.0 if json.loads(completed.stdout.strip().splitlines()[-1]).get("ok") else 0.0
        except Exception:
            return 0.0


def _content_candidates(source: str):
    from arte_cognition.software_task_acquisition import PythonASTRepairGenerator
    from arte_cognition.software_repair_grammar_expansion import PythonArithmeticRepairGenerator
    base = PythonASTRepairGenerator().generate("class-diagnosis-content", source)
    arithmetic = PythonArithmeticRepairGenerator().generate("class-diagnosis-content-arithmetic", source, OPEN_SPECIALIZED)
    unique = {}
    for item in (*base, *arithmetic):
        unique[item.proposal.experiment_id] = item
    return tuple(unique[key] for key in sorted(unique))


def _run_one_patched(family: str, source: str, patched_source: str, context_id: str) -> float:
    if family == "TRAVERSAL":
        return HistoricalTraversalExecutor._run(patched_source, TRAVERSAL_CONTEXTS[context_id])
    if family == "STATE_CONFLICT":
        return HistoricalStateConflictExecutor._run(patched_source, STATE_CONTEXTS[context_id])
    if family == "CONTENT":
        return _run_content_source(patched_source)
    raise ValueError(f"unknown family: {family}")


def _baseline_capability(family: str, source: str, context_id: str) -> float:
    if family == "TRAVERSAL":
        return HistoricalTraversalExecutor._run(source, TRAVERSAL_CONTEXTS[context_id])
    if family == "STATE_CONFLICT":
        return HistoricalStateConflictExecutor._run(source, STATE_CONTEXTS[context_id])
    if family == "CONTENT":
        return _run_content_source(source)
    raise ValueError(f"unknown family: {family}")


def _class_candidates(family: str, source: str, class_id: str):
    if class_id == "CONTENT":
        if family == "TRAVERSAL":
            return generate_traversal_content(source)
        if family == "STATE_CONFLICT":
            return generate_state_content(source)
        return _content_candidates(source)
    if class_id == "TRAVERSAL":
        return PythonTraversalStrategyRepairGenerator().generate("repair-class-traversal", source, OPEN_SPECIALIZED)
    if class_id == "STATE_CONFLICT":
        return PythonStateConflictRepairGenerator().generate("repair-class-state", source, OPEN_SPECIALIZED)
    raise ValueError(f"unknown repair class: {class_id}")


def execute_class_search_once(family: str, source: str, class_id: str, context_id: str) -> Tuple[float, SearchStats]:
    applicability = assess_repair_class_applicability(source)[class_id]
    if applicability.status != "APPLICABLE":
        return 0.0, SearchStats(class_id=class_id)
    candidates = _class_candidates(family, source, class_id)
    stats = SearchStats(class_id=class_id, candidate_count=len(candidates))
    strong = 0
    for candidate in candidates:
        stats.candidate_external_executions += 1
        if _run_one_patched(family, source, candidate.patched_source, context_id) >= 1.0:
            strong += 1
    stats.strong_candidate_count = strong
    return (1.0 if strong else 0.0), stats


class RepairClassExecutor:
    def __init__(self, family: str, source: str, class_id: str, context_id: str, signer, source_id: str, challenge_id: str, epoch: int):
        self.family = family; self.source = source; self.class_id = class_id; self.context_id = context_id
        self.signer = signer; self.source_id = source_id; self.challenge_id = challenge_id; self.epoch = int(epoch)
        self.high_stats = SearchStats(class_id=class_id)
    def execute(self, proposal, arm: str, value: float) -> WorldOutcomeReceipt:
        if str(arm).upper() == "LOW":
            outcome = _baseline_capability(self.family, self.source, self.context_id)
        else:
            outcome, self.high_stats = execute_class_search_once(self.family, self.source, self.class_id, self.context_id)
        return self.signer.sign(WorldOutcomeReceipt(
            receipt_id=f"{self.challenge_id}::{proposal.experiment_id}::{arm}", experiment_id=proposal.experiment_id,
            axis_id=proposal.axis_id, arm=str(arm).upper(), intervention_value=float(value), outcome=float(outcome),
            source_id=self.source_id, context_id=self.context_id, challenge_id=self.challenge_id, epoch=self.epoch,
            budget_token=f"repair-class-diagnosis::{self.challenge_id}", externally_generated=True))


def execute_class_candidate(body, family: str, source: str, candidate: RepairClassCandidate, context_id: str, signers, verifier, epoch_base: int):
    effects = []; candidate_execs = 0; candidate_count = None
    for issuer_index, (issuer, signer) in enumerate(signers.items()):
        token = hashlib.sha256(f"{family}|{context_id}|{candidate.class_id}|{issuer}".encode()).hexdigest()[:16]
        executor = RepairClassExecutor(family, source, candidate.class_id, context_id, signer,
            f"repair-class::{family}::{context_id}::{token}::{issuer}",
            f"repair-class::{family}::{context_id}::{token}::{issuer}", epoch_base + issuer_index)
        pair = body.execute_world_intervention(candidate.proposal, executor, verifier=verifier)
        if not pair.authority_verified:
            raise AssertionError("repair-class world receipt lost authority")
        effects.append(float(pair.effect)); candidate_execs += executor.high_stats.candidate_external_executions
        if candidate_count is None: candidate_count = executor.high_stats.candidate_count
    return tuple(effects), int(candidate_count or 0), candidate_execs


def train_family(body, family: str, source: str, context_ids: Sequence[str], signers, verifier, epoch_base: int):
    candidates = generate_repair_class_candidates(source)
    for candidate in candidates: body.memory.remember_experiment(candidate.proposal)
    trace = []
    for context_index, context_id in enumerate(context_ids):
        for class_index, candidate in enumerate(candidates):
            effects, count, execs = execute_class_candidate(body, family, source, candidate, context_id, signers, verifier,
                epoch_base + context_index * 10000 + class_index * 1000)
            trace.append({"context": context_id, "class_id": candidate.class_id, "effects": effects,
                "candidate_count": count, "candidate_external_executions": execs})
    policy = derive_repair_class_diagnosis_policy((r.proposal for r in body.memory.experiments.values()),
        body.world_coupling.pairs, normalized_structural_fingerprint(source),
        body.world_coupling.min_independent_classes, min_contexts=2)
    return candidates, policy, trace


def heldout_arm(body, family: str, source: str, context_id: str, selected_class_ids: Sequence[str], signers, verifier, epoch_base: int):
    total_execs = 0; total_candidates = 0; capability = 0.0; trace = []
    by_id = {item.class_id: item for item in generate_repair_class_candidates(source)}
    for class_index, class_id in enumerate(selected_class_ids):
        candidate = by_id.get(class_id)
        if candidate is None:
            trace.append({"class_id": class_id, "status": "INAPPLICABLE", "candidate_count": 0, "candidate_external_executions": 0, "capability": 0.0})
            continue
        effects, count, execs = execute_class_candidate(body, family, source, candidate, context_id, signers, verifier,
            epoch_base + class_index * 1000)
        class_cap = 1.0 if min(effects) >= 0.9 else 0.0
        capability = max(capability, class_cap); total_candidates += count; total_execs += execs
        trace.append({"class_id": class_id, "status": "APPLICABLE", "candidate_count": count,
            "candidate_external_executions": execs, "capability": class_cap})
    return capability, total_candidates, total_execs, trace


def main():
    traversal_source = TRAVERSAL_FIXTURE.read_text(encoding="utf-8")
    state_source = STATE_FIXTURE.read_text(encoding="utf-8")
    content_source_a = CONTENT_SOURCES["content-train-a"]
    content_source_b = CONTENT_SOURCES["content-train-b"]
    content_source_c = CONTENT_SOURCES["content-heldout-c"]

    if normalized_structural_fingerprint(content_source_a) != normalized_structural_fingerprint(content_source_b):
        raise AssertionError("identifier-renamed synthetic content controls should share one structural fingerprint")
    if normalized_structural_fingerprint(content_source_a) != normalized_structural_fingerprint(content_source_c):
        raise AssertionError("heldout synthetic content control should preserve structural fingerprint")
    if len({normalized_structural_fingerprint(traversal_source), normalized_structural_fingerprint(state_source), normalized_structural_fingerprint(content_source_a)}) != 3:
        raise AssertionError("three repair families collided under bounded structural fingerprint")

    issuer_a = "REPAIR_CLASS_A"; issuer_b = "REPAIR_CLASS_B"
    key_a = hashlib.sha256(b"repair-class-a").digest(); key_b = hashlib.sha256(b"repair-class-b").digest()
    signers = {issuer_a: HMACWorldReceiptSigner(issuer_a, key_a), issuer_b: HMACWorldReceiptSigner(issuer_b, key_b)}
    verifier = HMACWorldReceiptVerifier({issuer_a: key_a, issuer_b: key_b},
        independence_classes={issuer_a: "REPAIR_CLASS_INDEPENDENCE_A", issuer_b: "REPAIR_CLASS_INDEPENDENCE_B"})

    parent = PersistentCognitiveRuntime()
    traversal_candidates, traversal_policy, traversal_trace = train_family(parent, "TRAVERSAL", traversal_source,
        ("historical-regression-a", "historical-regression-b"), signers, verifier, 10000)
    if traversal_policy.class_id != "TRAVERSAL": raise AssertionError(f"traversal diagnosis failed: {traversal_policy}")
    state_candidates, state_policy, state_trace = train_family(parent, "STATE_CONFLICT", state_source,
        ("historical-provenance-a", "historical-provenance-b"), signers, verifier, 80000)
    if state_policy.class_id != "STATE_CONFLICT": raise AssertionError(f"state diagnosis failed: {state_policy}")
    content_candidates, content_policy, content_trace = train_family(parent, "CONTENT", content_source_a,
        ("content-train-a", "content-train-b"), signers, verifier, 150000)
    if content_policy.class_id != "CONTENT": raise AssertionError(f"content control diagnosis failed: {content_policy}")

    checkpoint = checkpoint_dict(parent)
    verifierless = restore_runtime(checkpoint)
    for source in (traversal_source, state_source, content_source_c):
        policy = derive_repair_class_diagnosis_policy((r.proposal for r in verifierless.memory.experiments.values()),
            verifierless.world_coupling.pairs, normalized_structural_fingerprint(source),
            verifierless.world_coupling.min_independent_classes, min_contexts=2)
        if policy.class_id is not None: raise AssertionError("verifierless repair-class authority leak")

    treatment = restore_runtime(checkpoint, world_verifier=verifier)
    traversal_tp = derive_repair_class_diagnosis_policy((r.proposal for r in treatment.memory.experiments.values()),
        treatment.world_coupling.pairs, normalized_structural_fingerprint(traversal_source), treatment.world_coupling.min_independent_classes, min_contexts=2)
    state_tp = derive_repair_class_diagnosis_policy((r.proposal for r in treatment.memory.experiments.values()),
        treatment.world_coupling.pairs, normalized_structural_fingerprint(state_source), treatment.world_coupling.min_independent_classes, min_contexts=2)
    content_tp = derive_repair_class_diagnosis_policy((r.proposal for r in treatment.memory.experiments.values()),
        treatment.world_coupling.pairs, normalized_structural_fingerprint(content_source_c), treatment.world_coupling.min_independent_classes, min_contexts=2)

    heldout_results = {}
    for index, (family, source, context_id, policy) in enumerate((
        ("TRAVERSAL", traversal_source, "historical-heldout-c", traversal_tp),
        ("STATE_CONFLICT", state_source, "historical-provenance-heldout-c", state_tp),
    )):
        class_candidates = generate_repair_class_candidates(source)
        treatment_sel = select_repair_classes(class_candidates, policy, max_classes=1)
        if treatment_sel.class_ids != (family,): raise AssertionError(f"wrong learned class on heldout: {treatment_sel}")
        treatment_cap, treatment_n, treatment_execs, treatment_trace = heldout_arm(treatment, family, source, context_id,
            treatment_sel.class_ids, signers, verifier, 200000 + index * 50000)
        full_body = restore_runtime(checkpoint, world_verifier=verifier)
        full_sel = select_repair_classes(class_candidates, None, max_classes=None)
        full_cap, full_n, full_execs, full_trace = heldout_arm(full_body, family, source, context_id,
            full_sel.class_ids, signers, verifier, 220000 + index * 50000)
        remove_body = restore_runtime(checkpoint, world_verifier=verifier)
        remove_sel = select_repair_classes(class_candidates, None, max_classes=1)
        remove_cap, remove_n, remove_execs, remove_trace = heldout_arm(remove_body, family, source, context_id,
            remove_sel.class_ids, signers, verifier, 240000 + index * 50000)
        if treatment_cap != 1.0 or full_cap != 1.0 or remove_cap != 0.0:
            raise AssertionError(f"causal class diagnosis failed for {family}: treatment={treatment_cap} full={full_cap} remove={remove_cap}")
        if treatment_n != 4 or full_n != 25 or remove_n != 21:
            raise AssertionError(f"unexpected class-search candidate counts for {family}: {treatment_n}, {full_n}, {remove_n}")
        heldout_results[family] = {
            "applicable_classes": [item.class_id for item in class_candidates],
            "diagnosed_class": policy.class_id,
            "treatment_candidate_count": treatment_n, "treatment_candidate_external_executions": treatment_execs,
            "treatment_capability": treatment_cap, "treatment_trace": treatment_trace,
            "full_candidate_count": full_n, "full_candidate_external_executions": full_execs,
            "full_capability": full_cap, "full_trace": full_trace,
            "remove_selected_classes": list(remove_sel.class_ids), "remove_candidate_count": remove_n,
            "remove_candidate_external_executions": remove_execs, "remove_capability": remove_cap,
            "remove_trace": remove_trace,
            "candidate_reduction_vs_full": 1.0 - treatment_n / full_n,
            "candidate_external_execution_reduction_vs_full": 1.0 - treatment_execs / full_execs,
        }

    content_body = restore_runtime(checkpoint, world_verifier=verifier)
    content_class_candidates = generate_repair_class_candidates(content_source_c)
    content_sel = select_repair_classes(content_class_candidates, content_tp, max_classes=1)
    content_cap, content_n, content_execs, content_heldout_trace = heldout_arm(content_body, "CONTENT", content_source_c,
        "content-heldout-c", content_sel.class_ids, signers, verifier, 350000)
    if content_sel.class_ids != ("CONTENT",) or content_cap != 1.0:
        raise AssertionError("synthetic content control did not route to CONTENT")

    receipt = {
        "status": "PASS_BOUNDED_WORLD_LEARNED_REPAIR_CLASS_DIAGNOSIS_ACROSS_TWO_NATURAL_HISTORICAL_FAMILIES",
        "natural_historical_training_families": ["TRAVERSAL", "STATE_CONFLICT"],
        "synthetic_content_control": True,
        "repair_class_candidates": ["CONTENT", "TRAVERSAL", "STATE_CONFLICT"],
        "diagnostic_fingerprint_filename_independent": True,
        "diagnostic_fingerprint_source_hash_independent": True,
        "diagnostic_fingerprint_identifier_spelling_erased": True,
        "diagnostic_fingerprint_literal_values_erased": True,
        "diagnostic_fingerprint_schema_human_authored": True,
        "source_disjoint_natural_class_diagnosis": False,
        "class_candidate_generation_uses_hidden_outcomes": False,
        "inapplicable_is_not_refuted": True,
        "within_class_learned_operator_used_for_heldout_selection": False,
        "traversal_training_policy": traversal_policy.class_id,
        "state_training_policy": state_policy.class_id,
        "content_control_policy": content_policy.class_id,
        "traversal_training_trace": traversal_trace,
        "state_training_trace": state_trace,
        "content_control_training_trace": content_trace,
        "heldout": heldout_results,
        "content_control_heldout": {
            "diagnosed_class": content_tp.class_id, "selected_classes": list(content_sel.class_ids),
            "candidate_count": content_n, "candidate_external_executions": content_execs,
            "capability": content_cap, "trace": content_heldout_trace,
            "renamed_source_structural_transfer": True,
        },
        "verifierless_repair_class_authority": False,
        "repair_class_policy_rederived_after_external_reverification": True,
        "autonomous_repair_class_genesis": False,
        "arbitrary_historical_bug_repair": False,
        "independent_organizational_custody": False,
        "foundation_weight_change": False,
        "physical_world": False,
        "global_recursive_acceleration": False,
        "AGI": False, "ASI": False,
    }
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
