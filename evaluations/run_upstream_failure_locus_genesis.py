from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import secrets
import sys
from typing import Dict, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.software_failure_extractor_program_genesis import (
    apply_failure_extractor_program,
    generate_failure_extractor_programs,
)
from arte_cognition.software_repair_grammar_expansion import (
    PythonArithmeticRepairGenerator,
    SoftwareRepairAlphabetAssessment,
)
from arte_cognition.software_task_acquisition import PythonASTRepairGenerator
from arte_cognition.software_upstream_failure_locus_genesis import (
    UpstreamFailureProgramOrgan,
    generate_upstream_failure_programs,
    generate_upstream_patch_candidates,
    locate_upstream_list_assignment,
    oracle_fingerprint_sha256,
    oracle_preserved,
    propose_upstream_failure_program,
    select_authorized_upstream_failure_program,
    target_frame_line,
)
from evaluations.run_natural_repair_constructor_genesis import (
    FreshRepositoryEnvironment,
    _authority,
    _execute_candidate,
    _git_blob_sha,
)


HISTORICAL_PARENT_COMMIT = "2ef608040135bbfc0991f73ecb50f83303a5a957"
HISTORICAL_FIX_COMMIT = "c2a6ea079d04a63f2abbebe2809e8c9af8e59f12"
HISTORICAL_BLOB = "a5f9753636135d7de653fda8293fad0911c067b1"
HISTORICAL_PATH = "arte_cognition/test_cognitive_runtime.py"
HISTORICAL_FIXTURE = ROOT / "evaluations/fixtures/historical_2ef60804/test_cognitive_runtime.py"
HISTORICAL_SELECTOR = (
    "arte_cognition.test_cognitive_runtime."
    "CognitiveRuntimeTests."
    "test_cycle_integrates_routing_possibility_and_semantic_genesis"
)


def _list_name_and_count(source: str, failure_line: int) -> Tuple[str, int]:
    locus = locate_upstream_list_assignment(source, failure_line)
    if locus is None:
        raise AssertionError("failure backslice did not reach an upstream list assignment")
    _, _, name = locus
    tree = ast.parse(source)
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name and isinstance(node.value, ast.List):
            matches.append(node)
    if not matches:
        raise AssertionError("backsliced upstream list disappeared")
    matches.sort(key=lambda node: int(node.lineno))
    return name, len(matches[-1].value.elts)


def _list_count_by_name(source: str, name: str) -> Optional[int]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name and isinstance(node.value, ast.List):
            matches.append(node)
    if not matches:
        return None
    matches.sort(key=lambda node: int(node.lineno))
    return len(matches[-1].value.elts)


class MinimalUpstreamEnvironment(FreshRepositoryEnvironment):
    """External semantic world: executable success + frozen oracle + cardinality preservation."""

    def __init__(self, source: str, path: str, selector: str, failure_line: int) -> None:
        super().__init__(source, path, "UNITTEST", selector=selector)
        self.failure_line = int(failure_line)
        self.upstream_name, self.reference_cardinality = _list_name_and_count(source, failure_line)
        self.oracle_fingerprint = oracle_fingerprint_sha256(source)

    def run(self, source: Optional[str] = None, timeout: float = 25.0):
        candidate = self.source if source is None else str(source)
        executable, stdout, stderr = super().run(candidate, timeout=timeout)
        if executable != 1.0:
            return 0.0, stdout, stderr
        if not oracle_preserved(self.source, candidate, self.failure_line):
            return 0.0, stdout, "oracle firewall rejected candidate"
        count = _list_count_by_name(candidate, self.upstream_name)
        if count != self.reference_cardinality:
            return 0.0, stdout, "upstream cardinality changed"
        return 1.0, stdout, stderr


def _baseline_environment(source: str, path: str, selector: str):
    environment = FreshRepositoryEnvironment(source, path, "UNITTEST", selector=selector)
    capability, _, stderr = environment.run()
    if capability != 0.0 or "AssertionError" not in stderr:
        raise AssertionError(f"expected natural/support-gate AssertionError, got: {stderr[-1200:]}")
    failure_line = target_frame_line(stderr, path)
    if failure_line is None:
        raise AssertionError("traceback did not expose target failure line")
    return environment, stderr, failure_line


def _permissive_arithmetic_assessment() -> SoftwareRepairAlphabetAssessment:
    # Give the already-merged arithmetic grammar its upstream opening gate for free.
    # This isolates expressive reach under the oracle firewall rather than prior evidence.
    return SoftwareRepairAlphabetAssessment(
        status="SOFTWARE_REPAIR_ALPHABET_FALSIFIED_OPEN_NEXT",
        complete_contexts=("oracle-firewall-probe",),
        falsified_contexts=("oracle-firewall-probe",),
        supported_contexts=(),
        missing_experiment_ids=(),
        evaluated_candidate_count=1,
        reason="maximally permissive old-grammar expressivity probe",
    )


def _old_local_repair_probe(source: str, path: str, selector: str, stderr: str, failure_line: int):
    environment = FreshRepositoryEnvironment(source, path, "UNITTEST", selector=selector)
    rows = []

    local = PythonASTRepairGenerator().generate("upstream-old-local", source)
    for candidate in local:
        rows.append(("COMPARE_BOOL", candidate.patched_source))

    arithmetic = PythonArithmeticRepairGenerator().generate(
        "upstream-old-arithmetic", source, _permissive_arithmetic_assessment()
    )
    for candidate in arithmetic:
        rows.append(("ARITHMETIC", candidate.patched_source))

    for program in generate_failure_extractor_programs():
        for candidate in apply_failure_extractor_program(program, stderr, source, path):
            rows.append(("TRACEBACK_CALL_ARGUMENT", candidate.patched_source))

    admissible = [row for row in rows if oracle_preserved(source, row[1], failure_line)]
    effects = []
    for _, patched in admissible:
        effect, _, _ = environment.run(patched)
        effects.append(float(effect))
    return {
        "total_candidate_count": len(rows),
        "oracle_safe_candidate_count": len(admissible),
        "capability": max(effects) if effects else 0.0,
        "families": sorted({family for family, _ in rows}),
        "oracle_safe_families": sorted({family for family, _ in admissible}),
        "candidate_hashes": tuple(sorted(hashlib.sha256(text.encode()).hexdigest() for _, text in rows)),
    }


def _randomized_source(token: str) -> Tuple[str, str, str]:
    module = f"test_upstream_random_{token}"
    path = f"arte_cognition/{module}.py"
    class_name = f"RandomizedUpstream{token.upper()}Tests"
    method = "test_randomized_support_gate_requires_upstream_repair"
    selector = f"arte_cognition.{module}.{class_name}.{method}"
    positive_a = f"positive_a_{token}"
    positive_b = f"positive_b_{token}"
    negative = f"negative_{token}"
    outcome_positive = f"OUTCOME_POS_{token}"
    outcome_negative = f"OUTCOME_NEG_{token}"
    source = f'''import unittest

from arte_cognition import Hypothesis, TaskState
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.possibility_space import Fact, OperatorSpec
from arte_cognition.semantic_genesis import ResidualObservation


class {class_name}(unittest.TestCase):
    def {method}(self):
        runtime = PersistentCognitiveRuntime()
        task = TaskState(
            goal={('randomized upstream support ' + token)!r},
            hypotheses=[Hypothesis({('h1_' + token)!r}), Hypothesis({('h2_' + token)!r})],
            residuals=[{('pressure_1_' + token)!r}, {('pressure_2_' + token)!r}],
            novelty=0.8,
        )
        residuals = [
            ResidualObservation({('r1_' + token)!r}, ({positive_a!r}, {positive_b!r}), {outcome_positive!r}),
            ResidualObservation({('r2_' + token)!r}, ({positive_a!r}, {positive_b!r}), {outcome_positive!r}),
            ResidualObservation({('r3_' + token)!r}, ({negative!r},), {outcome_negative!r}),
            ResidualObservation({('r4_' + token)!r}, ({negative!r},), {outcome_negative!r}),
            ResidualObservation({('held_' + token)!r}, ({positive_a!r}, {positive_b!r}), {outcome_positive!r}, heldout=True),
        ]
        cycle = runtime.cycle(
            task,
            facts=[Fact({('entity_' + token)!r}, "state", {positive_a!r})],
            residuals=residuals,
            operator_spec=OperatorSpec(relation_opposites={{"state": "not_state"}}),
        )
        self.assertTrue(any(law.status == "BOUNDED_LAW" for law in cycle.laws))
        self.assertTrue(cycle.active_generated_concepts)


if __name__ == "__main__":
    unittest.main()
'''
    return source, path, selector


def _search_program(program, stderr: str, environment: MinimalUpstreamEnvironment):
    candidates = generate_upstream_patch_candidates(
        program, stderr, environment.source, environment.path, max_candidates=256
    )
    effects = []
    successful = []
    for candidate in candidates:
        effect, _, _ = environment.run(candidate.patched_source)
        effects.append(float(effect))
        if effect == 1.0:
            successful.append(candidate)
    return candidates, tuple(effects), tuple(successful)


def _train_context(body, proposals, *, source: str, path: str, selector: str,
                   context: str, signers, verifier, epoch_base: int):
    _, stderr, failure_line = _baseline_environment(source, path, selector)
    environment = MinimalUpstreamEnvironment(source, path, selector, failure_line)
    rows = []
    for index, proposal in enumerate(proposals):
        candidates, effects, successful = _search_program(proposal.program, stderr, environment)
        rows.append((proposal, candidates, effects, successful))
        if not successful:
            continue
        # Search may use external outcomes to find a witness, but that witness is
        # independently re-executed by two verifier classes before program authority.
        authority_effects = _execute_candidate(
            body,
            proposal,
            environment,
            successful[0].patched_source,
            context,
            signers,
            verifier,
            epoch_base + index * 10,
        )
        if min(authority_effects) < 0.9:
            raise AssertionError("program search witness failed independent re-execution")
    strong = [row for row in rows if row[3]]
    if len(strong) != 1:
        detail = [(row[0].program.edit_operator, len(row[1]), len(row[3])) for row in rows]
        raise AssertionError(f"semantic upstream world did not isolate one program: {detail}")
    return {
        "stderr": stderr,
        "failure_line": failure_line,
        "environment": environment,
        "program": strong[0][0].program,
        "candidate_count": len(strong[0][1]),
        "successful_count": len(strong[0][3]),
        "rows": rows,
    }


def _matched_wrong_capability(program, stderr: str, environment: MinimalUpstreamEnvironment, budget: int):
    candidates = generate_upstream_patch_candidates(
        program, stderr, environment.source, environment.path, max_candidates=256
    )
    effects = []
    for candidate in candidates[:budget]:
        effect, _, _ = environment.run(candidate.patched_source)
        effects.append(float(effect))
    while len(effects) < budget:
        effect, _, _ = environment.run()
        effects.append(float(effect))
    return max(effects) if effects else 0.0, len(candidates), len(effects)


def main() -> None:
    historical_source = HISTORICAL_FIXTURE.read_text(encoding="utf-8")
    observed_blob = _git_blob_sha(historical_source)
    if observed_blob != HISTORICAL_BLOB:
        raise AssertionError(f"historical fixture diverged from exact Git blob: {observed_blob}")

    # Prove the inherited traceback-local and one-node AST alphabets cannot reach
    # an oracle-safe repair. Repeating deterministic generation 16x must not reveal
    # new candidates: more compute cannot change this representation boundary.
    _, natural_stderr, natural_failure_line = _baseline_environment(
        historical_source, HISTORICAL_PATH, HISTORICAL_SELECTOR
    )
    old_probe = _old_local_repair_probe(
        historical_source, HISTORICAL_PATH, HISTORICAL_SELECTOR,
        natural_stderr, natural_failure_line,
    )
    if old_probe["capability"] != 0.0:
        raise AssertionError("existing local repair language solved the natural upstream failure")
    repeated_hashes = []
    for _ in range(16):
        probe = _old_local_repair_probe(
            historical_source, HISTORICAL_PATH, HISTORICAL_SELECTOR,
            natural_stderr, natural_failure_line,
        )
        repeated_hashes.append(probe["candidate_hashes"])
        if probe["candidate_hashes"] != old_probe["candidate_hashes"] or probe["capability"] != 0.0:
            raise AssertionError("OLD+MORE_COMPUTE changed the bounded local repair frontier")

    programs = generate_upstream_failure_programs()
    proposals = tuple(propose_upstream_failure_program(item) for item in programs)
    if len(programs) != 3 or len({item.program_id for item in programs}) != 3:
        raise AssertionError("upstream program shadow language is not the intended bounded three-program search")

    signers, verifier = _authority()
    parent = PersistentCognitiveRuntime()
    organ = UpstreamFailureProgramOrgan(parent)
    organ.remember(proposals)

    natural = _train_context(
        parent,
        proposals,
        source=historical_source,
        path=HISTORICAL_PATH,
        selector=HISTORICAL_SELECTOR,
        context="natural-frozen-law-support-upstream",
        signers=signers,
        verifier=verifier,
        epoch_base=51000,
    )
    if organ.policy().status == "REPRODUCED_UPSTREAM_FAILURE_PROGRAM":
        raise AssertionError("one natural context incorrectly created program authority")

    train_token = secrets.token_hex(4)
    train_source, train_path, train_selector = _randomized_source(train_token)
    randomized_train = _train_context(
        parent,
        proposals,
        source=train_source,
        path=train_path,
        selector=train_selector,
        context="randomized-upstream-train",
        signers=signers,
        verifier=verifier,
        epoch_base=52000,
    )
    if natural["program"].program_id != randomized_train["program"].program_id:
        raise AssertionError("natural and randomized training did not converge on one upstream program")

    policy = organ.policy()
    if (
        policy.status != "REPRODUCED_UPSTREAM_FAILURE_PROGRAM"
        or policy.program_id != natural["program"].program_id
        or set(policy.supporting_contexts) != {
            "natural-frozen-law-support-upstream", "randomized-upstream-train"
        }
    ):
        raise AssertionError(f"world evidence did not authorize the upstream program: {policy}")

    checkpoint = checkpoint_dict(parent)
    verifierless = restore_runtime(checkpoint)
    verifierless_policy = UpstreamFailureProgramOrgan(verifierless).policy()
    if verifierless_policy.status == "REPRODUCED_UPSTREAM_FAILURE_PROGRAM":
        raise AssertionError("upstream program authority leaked through checkpoint")
    reverified = restore_runtime(checkpoint, world_verifier=verifier)
    reverified_policy = UpstreamFailureProgramOrgan(reverified).policy()
    if reverified_policy != policy:
        raise AssertionError("external reverification did not reconstruct upstream program policy")
    learned = select_authorized_upstream_failure_program(programs, reverified_policy)
    if learned is None:
        raise AssertionError("reverified descendant could not select learned upstream program")

    heldout_token = secrets.token_hex(4)
    while heldout_token == train_token:
        heldout_token = secrets.token_hex(4)
    heldout_source, heldout_path, heldout_selector = _randomized_source(heldout_token)
    if hashlib.sha256(train_source.encode()).hexdigest() == hashlib.sha256(heldout_source.encode()).hexdigest():
        raise AssertionError("randomized train and heldout sources were not source-disjoint")
    _, heldout_stderr, heldout_failure_line = _baseline_environment(
        heldout_source, heldout_path, heldout_selector
    )
    heldout_environment = MinimalUpstreamEnvironment(
        heldout_source, heldout_path, heldout_selector, heldout_failure_line
    )

    heldout_old = _old_local_repair_probe(
        heldout_source, heldout_path, heldout_selector,
        heldout_stderr, heldout_failure_line,
    )
    if heldout_old["capability"] != 0.0:
        raise AssertionError("old repair language unexpectedly solved randomized heldout")

    treatment_candidates, treatment_effects, treatment_successes = _search_program(
        learned, heldout_stderr, heldout_environment
    )
    if not treatment_successes:
        raise AssertionError("learned upstream program failed source-disjoint heldout transfer")
    treatment_capability = max(treatment_effects) if treatment_effects else 0.0
    budget = max(1, len(treatment_candidates))

    # Same-checkpoint REMOVE: no upstream program; spend identical external search
    # budget on the unchanged failing source rather than silently substituting a new language.
    remove_effects = tuple(heldout_environment.run()[0] for _ in range(budget))
    remove_capability = max(remove_effects) if remove_effects else 0.0

    wrong_programs = [item for item in programs if item.program_id != learned.program_id]
    wrong_rows = []
    for wrong in wrong_programs:
        capability, candidate_count, executed_count = _matched_wrong_capability(
            wrong, heldout_stderr, heldout_environment, budget
        )
        wrong_rows.append((wrong, capability, candidate_count, executed_count))
    if treatment_capability != 1.0 or remove_capability != 0.0 or any(row[1] != 0.0 for row in wrong_rows):
        raise AssertionError("Treatment/REMOVE/WRONG upstream-locus causal isolation failed")

    result = {
        "status": "PASS_BOUNDED_WORLD_FALSIFICATION_DRIVEN_UPSTREAM_FAILURE_LOCUS_PROGRAM_AND_SOURCE_DISJOINT_TRANSFER",
        "repository": "leehyen0/GENESIS-EX-NIHILO",
        "natural_historical_failure": True,
        "historical_parent_commit": HISTORICAL_PARENT_COMMIT,
        "historical_fix_commit": HISTORICAL_FIX_COMMIT,
        "historical_blob": HISTORICAL_BLOB,
        "historical_fixture_exact_git_blob": True,
        "later_human_fix_exposed_to_body": False,
        "old_control_scope": "traceback-local call-argument plus compare/bool plus arithmetic single-node AST alphabets",
        "old_total_candidate_count": old_probe["total_candidate_count"],
        "old_oracle_safe_candidate_count": old_probe["oracle_safe_candidate_count"],
        "old_candidate_families": old_probe["families"],
        "old_oracle_safe_families": old_probe["oracle_safe_families"],
        "old_more_compute_attempts": len(repeated_hashes),
        "old_more_compute_frontier_identical": True,
        "old_more_compute_capability": old_probe["capability"],
        "oracle_firewall": "STRUCTURAL_AST_FINGERPRINT_OF_ALL_ASSERT_AND_UNITTEST_ASSERT_CALLS",
        "natural_oracle_fingerprint": natural["environment"].oracle_fingerprint,
        "semantic_minimality": "UPSTREAM_LIST_CARDINALITY_PRESERVED",
        "generated_upstream_program_count": len(programs),
        "learned_program_id": learned.program_id,
        "learned_edit_operator": learned.edit_operator,
        "learned_locus_selector": learned.locus_selector,
        "learned_program_supporting_contexts": list(policy.supporting_contexts),
        "one_natural_context_insufficient_for_program_authority": True,
        "program_candidate_generation_uses_world_outcomes": False,
        "program_search_uses_world_outcomes": True,
        "program_search_witness_independently_reexecuted": True,
        "natural_candidate_count": natural["candidate_count"],
        "natural_successful_candidate_count": natural["successful_count"],
        "randomized_training_post_checkout": True,
        "randomized_training_source_hash": hashlib.sha256(train_source.encode()).hexdigest(),
        "randomized_training_candidate_count": randomized_train["candidate_count"],
        "randomized_training_successful_candidate_count": randomized_train["successful_count"],
        "verifierless_upstream_program_authority": False,
        "meta_policy_rederived_after_external_reverification": True,
        "heldout_post_checkout_randomized": True,
        "heldout_source_disjoint_from_randomized_training": True,
        "heldout_source_hash": hashlib.sha256(heldout_source.encode()).hexdigest(),
        "heldout_old_capability": heldout_old["capability"],
        "heldout_treatment_candidate_count": len(treatment_candidates),
        "heldout_treatment_successful_candidate_count": len(treatment_successes),
        "heldout_treatment_external_search_budget": budget,
        "treatment_capability": treatment_capability,
        "remove_same_checkpoint_candidate_count": 0,
        "remove_same_checkpoint_external_search_budget": budget,
        "remove_same_checkpoint_capability": remove_capability,
        "wrong_programs": [
            {
                "program_id": row[0].program_id,
                "edit_operator": row[0].edit_operator,
                "candidate_count": row[2],
                "external_search_budget": row[3],
                "capability": row[1],
            }
            for row in wrong_rows
        ],
        "upstream_locus_algorithm_human_authored": True,
        "upstream_edit_alphabet_human_authored": True,
        "unrestricted_failure_locus_genesis": False,
        "unrestricted_software_operator_invention": False,
        "foundation_weight_change": False,
        "global_recursive_acceleration": False,
        "independent_organizational_custody": False,
        "physical_world": False,
        "AGI": False,
        "ASI": False,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
