from __future__ import annotations

import unittest

from arte_cognition.executable_morphology import EdgeSpec, MorphologyGenome, OrganKind, OrganSpec, PressureVector
from arte_cognition.meta_acceleration import (
    GenerationMetrics,
    MetaAccelerationLedger,
    MetaMutationLearner,
    MutationStrategyState,
    apply_mutation_program,
    choose_meta_improvement_target,
    generate_mutation_programs,
)
from arte_cognition.morphology_genesis import MorphologyEvaluation, MorphologyGenesisEngine, MorphologyResidual


def body() -> MorphologyGenome:
    organs = (
        OrganSpec("source", OrganKind.SOURCE, produces=("evidence",)),
        OrganSpec("rep_a", OrganKind.REPRESENTATION, consumes=("evidence",), produces=("feature",)),
        OrganSpec("rep_b", OrganKind.REPRESENTATION, consumes=("evidence",), produces=("feature",)),
        OrganSpec("planner_a", OrganKind.PLANNER, consumes=("feature",), produces=("action",)),
        OrganSpec("planner_b", OrganKind.PLANNER, consumes=("feature",), produces=("action",)),
        OrganSpec("world", OrganKind.WORLD, consumes=("action",), produces=("outcome",)),
        OrganSpec("verifier", OrganKind.VERIFIER, consumes=("outcome",), produces=("authorized_outcome",)),
        OrganSpec("governor", OrganKind.GOVERNOR, consumes=("authorized_outcome",), produces=("decision",)),
        OrganSpec("archive", OrganKind.ARCHIVE, consumes=("decision",), produces=("experience",)),
    )
    edges = (
        EdgeSpec("e0", "source", "rep_a", "evidence"),
        EdgeSpec("e1", "rep_a", "planner_a", "feature"),
        EdgeSpec("e2", "planner_a", "world", "action"),
        EdgeSpec("e3", "world", "verifier", "outcome"),
        EdgeSpec("e4", "verifier", "governor", "authorized_outcome", authority_required=True),
        EdgeSpec("e5", "governor", "archive", "decision"),
    )
    return MorphologyGenome(organs, edges, tuple(o.organ_id for o in organs))


def candidates():
    residual = MorphologyResidual(
        "meta-r1",
        PressureVector(transfer_failure=1.0, efficiency_pressure=0.5),
        failed_edge_ids=("e0", "e1"),
        implicated_organ_ids=("rep_a", "rep_b", "planner_a", "planner_b"),
    )
    return MorphologyGenesisEngine(candidate_budget=128).generate(body(), (residual,))


class MetaAccelerationTests(unittest.TestCase):
    def test_past_external_evidence_changes_future_mutation_search_prior(self):
        pool = candidates()
        target = next(candidate for candidate in pool if candidate.operation_family == "REWIRE_EDGE")
        row = MorphologyEvaluation(
            "ev1", target.candidate_id, "ctx-a", "independent-a",
            1.0, 0.0, 0.0, 0.5, True, True, True,
        )
        state = MetaMutationLearner().update(MutationStrategyState(), pool, (row,))
        ranked = MetaMutationLearner.rank(state, pool)
        self.assertEqual(ranked[0].operation_family, "REWIRE_EDGE")
        self.assertGreater(state.score("REWIRE_EDGE"), 0.0)

    def test_unverified_current_like_feedback_cannot_train_strategy(self):
        pool = candidates()
        target = pool[0]
        row = MorphologyEvaluation(
            "ev0", target.candidate_id, "ctx-a", "UNVERIFIED",
            10.0, 1.0, 1.0, 10.0, True, False, True,
        )
        state = MetaMutationLearner().update(MutationStrategyState(), pool, (row,))
        self.assertEqual(state.operation_scores, ())
        self.assertEqual(state.operation_support, ())

    def test_mutation_program_composes_multiple_structural_edits_without_current_outcomes(self):
        pool = candidates()
        add_edges = [candidate for candidate in pool if candidate.operation_family == "ADD_EDGE"]
        self.assertGreaterEqual(len(add_edges), 2)
        strategy = MutationStrategyState(operation_scores=(("ADD_EDGE", 2.0),), lineage_hash="past-evidence")
        programs = generate_mutation_programs(add_edges[:3], strategy, max_depth=2, budget=32)
        program = next(program for program in programs if program.depth == 2)
        self.assertFalse(program.generation_uses_current_outcomes)
        descendant = apply_mutation_program(body(), program)
        self.assertGreaterEqual(len(descendant.edges), len(body().edges) + 2)
        self.assertNotEqual(descendant.fingerprint(), body().fingerprint())

    def test_acceleration_requires_three_real_transitions_not_one_cheap_success(self):
        ledger = MetaAccelerationLedger()
        first = GenerationMetrics(0, "b0", "", 1.0, 1.0, 1.0, 0.9, 0.2, 0.2, 4.0, 2.0, 3.0, True, True, "s0")
        second = GenerationMetrics(1, "b1", "b0", 2.0, 1.0, 1.0, 0.9, 0.3, 0.3, 3.0, 1.5, 2.0, True, True, "s1")
        self.assertTrue(ledger.append(first))
        self.assertTrue(ledger.append(second))
        assessment = ledger.assess()
        self.assertEqual(assessment.status, "INSUFFICIENT_PROSPECTIVE_META_ACCELERATION_EVIDENCE")
        self.assertFalse(assessment.global_recursive_acceleration)

    def test_four_generation_lineage_can_only_reach_bounded_candidate_status(self):
        ledger = MetaAccelerationLedger()
        rows = (
            GenerationMetrics(0, "b0", "", 1.0, 1.0, 1.0, 0.90, 0.2, 0.20, 5.0, 2.0, 3.0, True, True, "s0"),
            GenerationMetrics(1, "b1", "b0", 2.0, 1.0, 1.0, 0.91, 0.3, 0.35, 4.0, 1.5, 2.0, True, True, "s1"),
            GenerationMetrics(2, "b2", "b1", 4.0, 1.0, 1.0, 0.92, 0.5, 0.55, 3.0, 1.0, 1.0, True, True, "s2"),
            GenerationMetrics(3, "b3", "b2", 8.0, 1.0, 1.0, 0.93, 0.8, 0.85, 2.0, 0.5, 0.0, True, True, "s3"),
        )
        for row in rows:
            self.assertTrue(ledger.append(row))
        assessment = ledger.assess()
        self.assertEqual(assessment.status, "PASS_BOUNDED_PROSPECTIVE_META_ACCELERATION_CANDIDATE")
        self.assertTrue(assessment.strict_frontier_growth)
        self.assertTrue(assessment.strict_meta_productivity_growth)
        self.assertTrue(assessment.nonincreasing_human_intervention)
        self.assertTrue(assessment.meta_ability_improved)
        self.assertFalse(assessment.global_recursive_acceleration)

    def test_broken_lineage_is_rejected_at_append(self):
        ledger = MetaAccelerationLedger()
        self.assertTrue(ledger.append(GenerationMetrics(0, "b0", "", 1.0, 1.0, 1.0, 0.9, 0.2, 0.2, 1.0, 1.0, 1.0, True, True)))
        self.assertFalse(ledger.append(GenerationMetrics(1, "b1", "wrong", 2.0, 1.0, 1.0, 0.9, 0.3, 0.3, 1.0, 1.0, 1.0, True, True)))

    def test_meta_target_moves_upward_when_human_dependency_dominates(self):
        self.assertEqual(choose_meta_improvement_target(1.0, 0.5, 0.4, 0.3), "MUTATE_MUTATOR")
        self.assertEqual(choose_meta_improvement_target(0.1, 0.9, 0.4, 0.3), "MUTATE_SEARCH_POLICY")


if __name__ == "__main__":
    unittest.main()
