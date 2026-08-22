from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Mapping, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.software_repair_grammar_expansion import (
    SoftwareRepairAlphabetAssessment,
    PythonArithmeticRepairGenerator,
    assess_software_repair_alphabet_failure,
)
from arte_cognition.software_state_conflict_repair import (
    PythonStateConflictRepairGenerator,
    derive_state_conflict_repair_policy,
    select_state_conflict_repairs,
)
from arte_cognition.software_task_acquisition import PythonASTRepairGenerator, SoftwarePatchCandidate
from arte_cognition.world_coupling import HMACWorldReceiptSigner, HMACWorldReceiptVerifier, WorldOutcomeReceipt

HISTORICAL_COMMIT = "b3da1c3db7eb29995a6d06f696016cf52c89cd77"
HISTORICAL_BLOB = "d75b171f9cb79b7089492bd15fb6d64cd6f5690c"
HISTORICAL_PATH = "arte_cognition/epistemic_memory.py"
FIXTURE = ROOT / "evaluations/fixtures/historical_b3da1c3/epistemic_memory.py"
EXPECTED_STRATEGY = "STATE_CONFLICT::EXACT_ACTION_PROVENANCE_UNION"

CONTEXTS: Mapping[str, Mapping[str, object]] = {
    "historical-provenance-a": {
        "experiment_id": "EXPERIMENT::ALIAS::A", "axis_id": "AXIS::P::A", "variable": "x_a",
        "fixed_name": "z_a", "fixed_value": -0.5, "low": -2057.4569594629756,
        "high": 2057.4569594629756, "changed_high": 2058.4569594629756,
        "program_a": "GENERATOR_AST::LOG::ALPHA::0.75",
        "program_b": "GENERATOR_AST::INV>LOG::ALPHA::0.75",
    },
    "historical-provenance-b": {
        "experiment_id": "EXPERIMENT::ALIAS::B", "axis_id": "AXIS::P::B", "variable": "x_b",
        "fixed_name": "z_b", "fixed_value": 3.25, "low": -17.125, "high": 91.75,
        "changed_high": 94.5, "program_a": "GENERATOR_AST::LOG>LOG::ALPHA::0.25",
        "program_b": "GENERATOR_AST::INV>LOG>LOG::ALPHA::0.25",
    },
    "historical-provenance-heldout-c": {
        "experiment_id": "EXPERIMENT::ALIAS::C", "axis_id": "AXIS::P::C", "variable": "x_c",
        "fixed_name": "z_c", "fixed_value": 8.0, "low": -0.125, "high": 0.875,
        "changed_high": 1.625, "program_a": "GENERATOR_AST::POWER::ALPHA::0.5",
        "program_b": "GENERATOR_AST::LOG>POWER::ALPHA::0.5",
    },
}


def _git_blob_sha(source: str) -> str:
    payload = source.encode("utf-8")
    return hashlib.sha1(f"blob {len(payload)}\0".encode("utf-8") + payload).hexdigest()


def _permissive_arithmetic_assessment() -> SoftwareRepairAlphabetAssessment:
    return SoftwareRepairAlphabetAssessment(
        status="SOFTWARE_REPAIR_ALPHABET_FALSIFIED_OPEN_NEXT", complete_contexts=(), falsified_contexts=(),
        supported_contexts=(), missing_experiment_ids=(), evaluated_candidate_count=0,
        reason="enumerate already-existing arithmetic content repairs for historical completeness control",
    )


def generate_existing_content_alphabet(source: str) -> Tuple[SoftwarePatchCandidate, ...]:
    base = PythonASTRepairGenerator().generate("historical-state-old-alphabet", source)
    arithmetic = PythonArithmeticRepairGenerator().generate(
        "historical-state-old-arithmetic", source, _permissive_arithmetic_assessment()
    )
    unique: Dict[str, SoftwarePatchCandidate] = {}
    for candidate in (*base, *arithmetic):
        unique[candidate.proposal.experiment_id] = candidate
    return tuple(unique[key] for key in sorted(unique))


def _has_traversal_repair_target(source: str) -> bool:
    tree = ast.parse(source)
    for class_node in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        if class_node.name != "PythonASTRepairGenerator":
            continue
        if any(isinstance(n, ast.FunctionDef) and n.name == "_site_operator_ids" for n in class_node.body):
            return True
    return False


class HistoricalStateConflictExecutor:
    _HARNESS = r'''
import json, sys
from arte_cognition.epistemic_memory import EpistemicMemory
from arte_cognition.experiment_genesis import InterventionProposal
cfg = json.loads(sys.stdin.read())["config"]
MARKER = "generator_transform_programs="
def ids(reason):
    text = str(reason)
    if MARKER not in text: return ()
    tail = text.split(MARKER, 1)[1].strip().split()[0].rstrip(",;)")
    return tuple(sorted(set(item for item in tail.split("|") if item)))
def proposal(program, high=None):
    return InterventionProposal(
        experiment_id=str(cfg["experiment_id"]), axis_id=str(cfg["axis_id"]),
        manipulated_variable=str(cfg["variable"]), held_fixed=((str(cfg["fixed_name"]), float(cfg["fixed_value"])),),
        low_value=float(cfg["low"]), high_value=float(cfg["high"] if high is None else high),
        predicted_low_side="LE_THRESHOLD", predicted_high_side="GT_THRESHOLD",
        reason=f"probe_scale={abs(float(cfg['high'])) + 1.0} {MARKER}{program}", status="PROPOSAL_ONLY")
first = proposal(cfg["program_a"]); alias = proposal(cfg["program_b"])
memory = EpistemicMemory(); memory.remember_experiment(first); memory.remember_experiment(alias)
stored_alias = memory.experiments[first.experiment_id].proposal
alias_ok = (set(ids(stored_alias.reason)) == {str(cfg["program_a"]), str(cfg["program_b"])}
            and float(stored_alias.low_value) == float(first.low_value)
            and float(stored_alias.high_value) == float(first.high_value))
collision_id = str(cfg["experiment_id"]) + "::COLLISION"
base_collision = InterventionProposal(
    experiment_id=collision_id, axis_id=str(cfg["axis_id"]), manipulated_variable=str(cfg["variable"]),
    held_fixed=((str(cfg["fixed_name"]), float(cfg["fixed_value"])),), low_value=float(cfg["low"]),
    high_value=float(cfg["high"]), predicted_low_side="LE_THRESHOLD", predicted_high_side="GT_THRESHOLD",
    reason=f"probe_scale=1 {MARKER}{cfg['program_a']}", status="PROPOSAL_ONLY")
changed_collision = InterventionProposal(
    experiment_id=collision_id, axis_id=str(cfg["axis_id"]), manipulated_variable=str(cfg["variable"]),
    held_fixed=((str(cfg["fixed_name"]), float(cfg["fixed_value"])),), low_value=float(cfg["low"]),
    high_value=float(cfg["changed_high"]), predicted_low_side="LE_THRESHOLD", predicted_high_side="GT_THRESHOLD",
    reason=f"probe_scale=2 {MARKER}{cfg['program_b']}", status="PROPOSAL_ONLY")
memory2 = EpistemicMemory(); memory2.remember_experiment(base_collision); memory2.remember_experiment(changed_collision)
stored_changed = memory2.experiments[collision_id].proposal
changed_ok = (float(stored_changed.high_value) == float(cfg["changed_high"])
              and ids(stored_changed.reason) == (str(cfg["program_b"]),))
print(json.dumps({"ok": bool(alias_ok and changed_ok), "alias_ok": bool(alias_ok), "changed_ok": bool(changed_ok)}))
'''
    def __init__(self, baseline_source, patched_source, config, signer, source_id, context_id, challenge_id, epoch, timeout_seconds=8.0):
        self.baseline_source = str(baseline_source); self.patched_source = str(patched_source); self.config = dict(config)
        self.signer = signer; self.source_id = str(source_id); self.context_id = str(context_id)
        self.challenge_id = str(challenge_id); self.epoch = int(epoch); self.timeout_seconds = max(1.0, float(timeout_seconds))
    @staticmethod
    def _run(source: str, config, timeout_seconds: float = 8.0) -> float:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); shutil.copytree(ROOT / "arte_cognition", root / "arte_cognition")
            (root / HISTORICAL_PATH).write_text(str(source), encoding="utf-8")
            try:
                completed = subprocess.run([sys.executable, "-c", HistoricalStateConflictExecutor._HARNESS],
                    input=json.dumps({"config": dict(config)}), cwd=root, capture_output=True, text=True,
                    timeout=timeout_seconds, check=False)
                if completed.returncode != 0: return 0.0
                payload = json.loads(completed.stdout.strip().splitlines()[-1])
                return 1.0 if bool(payload.get("ok")) else 0.0
            except Exception:
                return 0.0
    def execute(self, proposal, arm: str, value: float) -> WorldOutcomeReceipt:
        source = self.baseline_source if str(arm).upper() == "LOW" else self.patched_source
        outcome = self._run(source, self.config, self.timeout_seconds)
        return self.signer.sign(WorldOutcomeReceipt(
            receipt_id=f"{self.challenge_id}::{proposal.experiment_id}::{arm}", experiment_id=proposal.experiment_id,
            axis_id=proposal.axis_id, arm=str(arm).upper(), intervention_value=float(value), outcome=float(outcome),
            source_id=self.source_id, context_id=self.context_id, challenge_id=self.challenge_id, epoch=self.epoch,
            budget_token=f"historical-natural-state-conflict::{self.challenge_id}", externally_generated=True))


def execute_candidate(body, source, candidate, context_id, signers, verifier, epoch_base):
    effects = []
    for issuer_index, (issuer, signer) in enumerate(signers.items()):
        token = hashlib.sha256(f"{context_id}|{candidate.proposal.experiment_id}|{issuer}".encode("utf-8")).hexdigest()[:16]
        executor = HistoricalStateConflictExecutor(source, candidate.patched_source, CONTEXTS[context_id], signer,
            f"historical-state::{context_id}::{token}::{issuer}", context_id,
            f"historical-state::{context_id}::{token}::{issuer}", epoch_base + issuer_index)
        pair = body.execute_world_intervention(candidate.proposal, executor, verifier=verifier)
        if not pair.authority_verified: raise AssertionError("historical state-conflict receipt lost authority")
        effects.append(float(pair.effect))
    return tuple(effects)


def run_candidates(body, source, candidates, context_id, signers, verifier, epoch_base):
    strong = []
    for index, candidate in enumerate(candidates):
        effects = execute_candidate(body, source, candidate, context_id, signers, verifier, epoch_base + index * 10)
        if min(effects) >= 0.9: strong.append(candidate)
    return tuple(strong)


def main():
    historical_source = FIXTURE.read_text(encoding="utf-8")
    if _git_blob_sha(historical_source) != HISTORICAL_BLOB: raise AssertionError("historical fixture Git blob drift")
    if _has_traversal_repair_target(historical_source): raise AssertionError("unexpected traversal target")
    for context_id, config in CONTEXTS.items():
        if HistoricalStateConflictExecutor._run(historical_source, config) != 0.0:
            raise AssertionError(f"historical provenance bug no longer reproduces in {context_id}")

    issuer_a = "HISTORICAL_STATE_A"; issuer_b = "HISTORICAL_STATE_B"
    key_a = hashlib.sha256(b"historical-state-a").digest(); key_b = hashlib.sha256(b"historical-state-b").digest()
    signers = {issuer_a: HMACWorldReceiptSigner(issuer_a, key_a), issuer_b: HMACWorldReceiptSigner(issuer_b, key_b)}
    verifier = HMACWorldReceiptVerifier({issuer_a: key_a, issuer_b: key_b},
        independence_classes={issuer_a: "HISTORICAL_STATE_CLASS_A", issuer_b: "HISTORICAL_STATE_CLASS_B"})

    parent = PersistentCognitiveRuntime(); old_candidates = generate_existing_content_alphabet(historical_source)
    if not old_candidates: raise AssertionError("no old content candidates")
    for candidate in old_candidates: parent.memory.remember_experiment(candidate.proposal)
    old_candidates_by_context = {"historical-provenance-a": old_candidates, "historical-provenance-b": old_candidates}
    for context_index, context_id in enumerate(old_candidates_by_context):
        strong = run_candidates(parent, historical_source, old_candidates, context_id, signers, verifier,
            10000 + context_index * 30000)
        if strong: raise AssertionError(f"old content repair unexpectedly solved state conflict: {[(c.site_index,c.operator_id) for c in strong]}")
    assessment = assess_software_repair_alphabet_failure(old_candidates_by_context, parent.world_coupling.pairs,
        parent.world_coupling.min_independent_classes, strong_effect_threshold=0.9, min_contexts=2)
    if assessment.status != "SOFTWARE_REPAIR_ALPHABET_FALSIFIED_OPEN_NEXT" or assessment.missing_experiment_ids:
        raise AssertionError(f"old content language did not fail completely: {assessment}")

    state_generator = PythonStateConflictRepairGenerator()
    state_candidates = state_generator.generate("historical-natural-state-conflict", historical_source, assessment)
    expected = ["STATE_CONFLICT::LAST_WRITE", "STATE_CONFLICT::KEEP_FIRST",
        "STATE_CONFLICT::ID_ONLY_PROVENANCE_UNION", EXPECTED_STRATEGY]
    if [c.operator_id for c in state_candidates] != expected: raise AssertionError("state-conflict metalanguage drift")
    for candidate in state_candidates: parent.memory.remember_experiment(candidate.proposal)
    first_strong = run_candidates(parent, historical_source, state_candidates, "historical-provenance-a", signers, verifier, 80000)
    if len(first_strong) != 1 or first_strong[0].operator_id != EXPECTED_STRATEGY:
        raise AssertionError(f"first context not unique: {[c.operator_id for c in first_strong]}")
    second_strong = run_candidates(parent, historical_source, first_strong, "historical-provenance-b", signers, verifier, 90000)
    if len(second_strong) != 1 or second_strong[0].operator_id != EXPECTED_STRATEGY: raise AssertionError("second reproduction failed")
    policy = derive_state_conflict_repair_policy((r.proposal for r in parent.memory.experiments.values()),
        parent.world_coupling.pairs, parent.world_coupling.min_independent_classes, min_contexts=2)
    if policy.status != "REPRODUCED_STATE_CONFLICT_REPAIR_STRATEGY" or f"STATE_CONFLICT::{policy.strategy_id}" != EXPECTED_STRATEGY:
        raise AssertionError(f"policy failed: {policy}")

    checkpoint = checkpoint_dict(parent)
    verifierless = restore_runtime(checkpoint)
    vp = derive_state_conflict_repair_policy((r.proposal for r in verifierless.memory.experiments.values()),
        verifierless.world_coupling.pairs, verifierless.world_coupling.min_independent_classes, min_contexts=2)
    if vp.strategy_id is not None: raise AssertionError("verifierless authority leak")

    treatment = restore_runtime(checkpoint, world_verifier=verifier)
    tp = derive_state_conflict_repair_policy((r.proposal for r in treatment.memory.experiments.values()),
        treatment.world_coupling.pairs, treatment.world_coupling.min_independent_classes, min_contexts=2)
    fresh_candidates = state_generator.generate("historical-natural-state-conflict", historical_source, assessment)
    treatment_selection = select_state_conflict_repairs(fresh_candidates, tp, max_candidates=1)
    treatment_strong = run_candidates(treatment, historical_source, treatment_selection.candidates,
        "historical-provenance-heldout-c", signers, verifier, 100000)
    remove = restore_runtime(checkpoint, world_verifier=verifier)
    remove_selection = select_state_conflict_repairs(fresh_candidates, None, max_candidates=1)
    remove_strong = run_candidates(remove, historical_source, remove_selection.candidates,
        "historical-provenance-heldout-c", signers, verifier, 110000)
    full = restore_runtime(checkpoint, world_verifier=verifier)
    full_selection = select_state_conflict_repairs(fresh_candidates, None, max_candidates=None)
    full_strong = run_candidates(full, historical_source, full_selection.candidates,
        "historical-provenance-heldout-c", signers, verifier, 120000)
    if len(treatment_strong) != 1 or treatment_strong[0].operator_id != EXPECTED_STRATEGY: raise AssertionError("treatment failed")
    if remove_strong: raise AssertionError("REMOVE unexpectedly passed")
    if len(full_strong) != 1 or full_strong[0].operator_id != EXPECTED_STRATEGY: raise AssertionError("FULL not unique")

    receipt = {
        "status": "PASS_BOUNDED_NATURAL_HISTORICAL_PROVENANCE_BUG_TO_STATE_CONFLICT_REPAIR_LANGUAGE_AND_DESCENDANT",
        "repository": "leehyen0/GENESIS-EX-NIHILO", "natural_historical_bug": True,
        "historical_commit_sha": HISTORICAL_COMMIT, "historical_path": HISTORICAL_PATH,
        "historical_blob_sha": HISTORICAL_BLOB, "historical_fixture_exact_git_blob": True,
        "later_fixed_source_exposed_to_body": False,
        "bug_family": "EXACT_INTERVENTION_IDENTITY_PROVENANCE_ALIAS_OVERWRITE",
        "historical_failure_reproduced_in_contexts": len(CONTEXTS),
        "existing_content_repair_candidate_count": len(old_candidates),
        "existing_content_repair_evaluated_candidate_count": assessment.evaluated_candidate_count,
        "existing_content_repair_complete_contexts": len(assessment.complete_contexts),
        "existing_content_repair_falsified_contexts": len(assessment.falsified_contexts),
        "existing_content_repair_missing_candidate_count": len(assessment.missing_experiment_ids),
        "existing_content_repair_capability": 0.0, "traversal_repair_target_present": False,
        "traversal_inapplicable_not_refuted": True,
        "state_conflict_grammar_opened_only_after_complete_content_failure": True,
        "state_conflict_strategy_candidates": [c.operator_id for c in state_candidates],
        "state_conflict_candidate_generation_uses_hidden_outcomes": False,
        "learned_state_conflict_strategy": f"STATE_CONFLICT::{policy.strategy_id}",
        "learned_state_conflict_supporting_contexts": len(policy.supporting_contexts),
        "treatment_candidate_count": len(treatment_selection.candidates),
        "treatment_external_pair_count": 2 * len(treatment_selection.candidates), "treatment_capability": 1.0,
        "remove_definition": "same checkpoint and same one-candidate budget; remove learned state-conflict strategy selection",
        "remove_candidate_count": len(remove_selection.candidates),
        "remove_external_pair_count": 2 * len(remove_selection.candidates), "remove_same_checkpoint_capability": 0.0,
        "full_state_conflict_candidate_count": len(full_selection.candidates),
        "full_state_conflict_external_pair_count": 2 * len(full_selection.candidates), "full_state_conflict_capability": 1.0,
        "fresh_hidden_regression_context": "historical-provenance-heldout-c", "verifierless_state_conflict_authority": False,
        "state_conflict_metalanguage_human_authored": True, "unrestricted_software_operator_invention": False,
        "arbitrary_historical_bug_repair": False, "foundation_weight_change": False,
        "independent_organizational_custody": False, "physical_world": False,
        "global_recursive_acceleration": False, "AGI": False, "ASI": False,
    }
    print(json.dumps(receipt, sort_keys=True))

if __name__ == "__main__":
    main()
