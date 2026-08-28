from __future__ import annotations

import unittest

from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.executable_morphology import ExperienceArchive, MorphologyGenome, OrganKind, OrganSpec
from arte_cognition.meta_acceleration import MutationProgramDevelopmentState, MutationStrategyState
from arte_cognition.native_recursive_research import (
    NativeResearchCycle,
    NativeResearchEvaluation,
    NativeResearchLearner,
    NativeRecursiveResearchLedger,
    choose_native_meta_target,
    commit_native_research_to_body,
    discover_native_research_problems,
)
from arte_cognition.self_evolving_body_checkpoint import SelfEvolvingResearchBody


H = lambda c: c * 64


def make_body() -> SelfEvolvingResearchBody:
    morphology = MorphologyGenome(
        organs=(
            OrganSpec("governor", OrganKind.GOVERNOR),
            OrganSpec("archive", OrganKind.ARCHIVE),
        ),
        edges=(),
        event_order=("governor", "archive"),
    )
    return SelfEvolvingResearchBody(
        PersistentCognitiveRuntime(),
        morphology,
        MutationStrategyState(lineage_hash=H("0")),
        MutationProgramDevelopmentState(),
        ExperienceArchive(),
        None,
    )


def sources(native: bool = False):
    payload = {
        "arte_cognition/meta_acceleration.py": (
            "class GenerationMetrics:\n"
            "    research_invention_score: float\n"
            "class MetaMutationLearner:\n"
            "    def update(self, row):\n"
            "        if row.externally_generated and row.authority_verified and row.benchmark_disjoint:\n"
            "            pass\n"
        ),
        "arte_cognition/executable_morphology.py": (
            "human_dependency: float = 0.0\n"
            "class MutationLevel:\n"
            "    GENERATOR_MUTATOR = 3\n"
        ),
        "arte_cognition/morphology_genesis.py": (
            "def generate(self, residual):\n"
            "    level = MutationLevel.TOPOLOGY\n"
        ),
    }
    if native:
        payload["arte_cognition/native_recursive_research.py"] = (
            "class NativeResearchLearner: pass\n"
            "class Row:\n"
            "    @property\n"
            "    def research_productivity(self): return 1.0\n"
        )
    return payload


class NativeRecursiveResearchTests(unittest.TestCase):
    def test_problem_finder_discovers_real_native_credit_and_generator_reachability_gaps(self):
        problems = discover_native_research_problems(sources(False), body_hash=H("a"))
        detectors = {problem.detector_id for problem in problems}
        self.assertEqual(
            detectors,
            {
                "EXTERNAL_ONLY_META_CREDIT",
                "UNCREDITED_RESEARCH_INVENTION",
                "GENERATOR_MUTATOR_PRESSURE_UNREACHABLE",
            },
        )
        self.assertTrue(all(not problem.human_seeded for problem in problems))

    def test_after_native_credit_repair_next_frontier_is_generator_mutator_reachability(self):
        problems = discover_native_research_problems(sources(True), body_hash=H("b"))
        self.assertEqual(
            {problem.detector_id for problem in problems},
            {"GENERATOR_MUTATOR_PRESSURE_UNREACHABLE"},
        )

    def test_verified_native_research_can_change_inherited_search_prior_without_official_benchmark(self):
        problem = discover_native_research_problems(sources(False), body_hash=H("a"))[0]
        row = NativeResearchEvaluation(
            evaluation_id="ev-native",
            problem_sha256=problem.fingerprint(),
            operation_family="MUTATE_SEARCH_POLICY",
            context_id="self-source-parent",
            evidence_class="REPOSITORY_SELF_RESEARCH",
            solved=True,
            precommitted=True,
            evaluator_reverified=True,
            removal_effect=1.0,
            wrong_swap_effect=1.0,
            retained_competence_delta=0.0,
            calibration_delta=0.0,
            problem_discovery_delta=1.0,
            research_invention_delta=1.0,
            meta_improvement_delta=1.0,
            compute_cost=1.0,
            evidence_cost=1.0,
            human_structural_intervention=1.0,
            outcome_receipt_sha256=H("c"),
            official_benchmark_used=False,
        )
        state = NativeResearchLearner().update(MutationStrategyState(lineage_hash=H("0")), (row,))
        self.assertGreater(state.score("MUTATE_SEARCH_POLICY"), 0.0)
        self.assertEqual(state.support_map()["MUTATE_SEARCH_POLICY"], 1)
        self.assertNotEqual(state.lineage_hash, H("0"))
        self.assertFalse(row.official_benchmark_used)

    def test_remove_or_wrong_control_failure_cannot_receive_positive_credit(self):
        problem = discover_native_research_problems(sources(False), body_hash=H("a"))[0]
        rows = (
            NativeResearchEvaluation(
                "remove", problem.fingerprint(), "MUTATE_SEARCH_POLICY", "ctx-r", "REPOSITORY_SELF_RESEARCH",
                True, True, True, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, H("d"), False,
            ),
            NativeResearchEvaluation(
                "wrong", problem.fingerprint(), "MUTATE_SEARCH_POLICY", "ctx-w", "REPOSITORY_SELF_RESEARCH",
                True, True, True, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, H("e"), False,
            ),
        )
        state = NativeResearchLearner().update(MutationStrategyState(), rows)
        self.assertLess(state.score("MUTATE_SEARCH_POLICY"), 0.0)
        self.assertIn("MUTATE_SEARCH_POLICY", state.fossilized_operations)
        self.assertTrue(all(not row.admissible_native_credit for row in rows))

    def test_native_credit_can_change_next_meta_target(self):
        baseline = choose_native_meta_target(
            human_dependency=0.80,
            candidate_search_cost=0.79,
            evaluator_uncertainty=0.1,
            transfer_failure=0.1,
        )
        learned = MutationStrategyState(operation_scores=(("MUTATE_SEARCH_POLICY", 1.0),))
        descendant = choose_native_meta_target(
            human_dependency=0.80,
            candidate_search_cost=0.79,
            evaluator_uncertainty=0.1,
            transfer_failure=0.1,
            strategy=learned,
        )
        self.assertEqual(baseline, "MUTATE_MUTATOR")
        self.assertEqual(descendant, "MUTATE_SEARCH_POLICY")

    def test_native_research_commits_into_existing_body_strategy_and_experience_archive(self):
        body = make_body()
        problem = discover_native_research_problems(sources(False), body_hash=body.morphology.fingerprint())[0]
        row = NativeResearchEvaluation(
            "commit", problem.fingerprint(), "MUTATE_SEARCH_POLICY", "ctx", "REPOSITORY_SELF_RESEARCH",
            True, True, True, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, H("f"), False,
        )
        self.assertTrue(commit_native_research_to_body(body, problem, row, H("1")))
        self.assertGreater(body.mutation_strategy.score("MUTATE_SEARCH_POLICY"), 0.0)
        self.assertIn("NATIVE_RESEARCH::commit", body.experience_archive.episodes)
        episode = body.experience_archive.episodes["NATIVE_RESEARCH::commit"]
        self.assertEqual(episode.benchmark_family, "NATIVE_SELF_RESEARCH")
        self.assertIn("external_claim_authority=false", episode.notes)

    def test_one_valid_cycle_is_not_mislabeled_recursive_acceleration(self):
        ledger = NativeRecursiveResearchLedger()
        self.assertTrue(
            ledger.append(
                NativeResearchCycle(
                    0, H("1"), H("a"), H("b"), 1.0, 1.0, 1.0, 1.0, 0.9,
                    1.0, 1.0, 1.0, True, True,
                )
            )
        )
        result = ledger.assess()
        self.assertEqual(result.status, "PASS_BOUNDED_NATIVE_RESEARCH_CYCLE_NOT_RECURSIVE")
        self.assertFalse(result.global_recursive_acceleration)

    def test_three_contiguous_native_cycles_can_reach_only_bounded_recursive_candidate(self):
        ledger = NativeRecursiveResearchLedger()
        rows = (
            NativeResearchCycle(0, H("1"), H("a"), H("b"), 0.5, 0.5, 0.5, 1.0, 0.90, 2.0, 1.0, 2.0, True, True),
            NativeResearchCycle(1, H("2"), H("b"), H("c"), 0.8, 0.8, 0.8, 1.0, 0.91, 1.5, 0.8, 1.0, True, True),
            NativeResearchCycle(2, H("3"), H("c"), H("d"), 1.0, 1.0, 1.0, 1.0, 0.92, 1.0, 0.5, 0.0, True, True),
        )
        for row in rows:
            self.assertTrue(ledger.append(row))
        result = ledger.assess()
        self.assertEqual(result.status, "PASS_BOUNDED_RECURSIVE_NATIVE_RESEARCH_CANDIDATE")
        self.assertTrue(result.strict_research_productivity_growth)
        self.assertTrue(result.nonincreasing_human_intervention)
        self.assertFalse(result.global_recursive_acceleration)


if __name__ == "__main__":
    unittest.main()
