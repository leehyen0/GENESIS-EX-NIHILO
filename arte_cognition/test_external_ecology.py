from __future__ import annotations

import unittest

from arte_cognition.external_ecology import (
    AcquisitionStatus,
    ExternalEcologyAcquisitionGate,
    ExternalEcologyBatchScheduler,
    ExternalGenerationTransition,
    ExternalResidualPersistenceGate,
    ExternalWorldCandidate,
    ExternalWorldInexpressibilityGate,
    FrozenBodyProbe,
    ObservationOutcome,
    PersistenceStatus,
    ProspectiveExternalMetaLedger,
    ProspectiveObservation,
)
from arte_cognition.external_experience import ExposureClass


class ExternalEcologyTests(unittest.TestCase):
    def candidate(self, candidate_id="c1", ecology="software", **overrides):
        data = dict(
            candidate_id=candidate_id,
            repository="external/repo",
            issue_ref="#1",
            commit_sha="a" * 40,
            exact_command="python -m pytest tests/test_target.py::test_target -q",
            ecology_family=ecology,
            source_class="PUBLIC_REPOSITORY_CI",
            exposure=ExposureClass.PUBLIC_DEV,
            original_failure_signature="AssertionError::target",
            observation_cost=1.0,
            deterministic_expected=True,
            seed_controlled=False,
            repeatable_contract=False,
            repository_wide_contamination_search_complete=True,
            answer_seen_before_freeze=False,
            patch_seen_before_freeze=False,
            root_cause_seen_before_freeze=False,
            related_fix_link_seen_before_freeze=False,
            dependency_lock_frozen=True,
            runtime_frozen=True,
            hardware_reconstructable=True,
            external_service_dependencies=0,
            external_service_state_reconstructable=True,
            independent_external_origin=True,
        )
        data.update(overrides)
        return ExternalWorldCandidate(**data)

    def observation(self, candidate_id="c1", observation_id="o1", execution_id="r1", outcome=ObservationOutcome.FAIL, signature="AssertionError::target", distance=0.0):
        return ProspectiveObservation(
            candidate_id=candidate_id,
            observation_id=observation_id,
            independent_execution_id=execution_id,
            outcome=outcome,
            semantic_signature=signature,
            contract_match=True,
            environment_distance=distance,
            infrastructure_ready=True,
            post_freeze_solution_leakage=False,
        )

    def test_repository_wide_solution_leakage_rejects_before_observation(self):
        candidate = self.candidate(root_cause_seen_before_freeze=True)
        decision = ExternalEcologyAcquisitionGate.evaluate(candidate)
        self.assertFalse(decision.eligible)
        self.assertEqual(decision.status, AcquisitionStatus.CONTAMINATED)
        self.assertIn("root_cause_seen_before_freeze", decision.reasons)

    def test_unknown_external_service_state_rejects_environment_contract(self):
        candidate = self.candidate(
            external_service_dependencies=1,
            external_service_state_reconstructable=False,
        )
        decision = ExternalEcologyAcquisitionGate.evaluate(candidate)
        self.assertFalse(decision.eligible)
        self.assertEqual(decision.status, AcquisitionStatus.ENVIRONMENT_UNRESOLVED)

    def test_stochastic_candidate_requires_seed_or_repeatable_contract(self):
        candidate = self.candidate(deterministic_expected=False)
        decision = ExternalEcologyAcquisitionGate.evaluate(candidate)
        self.assertFalse(decision.eligible)
        self.assertEqual(decision.status, AcquisitionStatus.STOCHASTICITY_UNRESOLVED)

    def test_batch_scheduler_enforces_ecology_diversity_pre_outcome(self):
        candidates = (
            self.candidate("a1", "software"),
            self.candidate("a2", "software", observation_cost=1.1),
            self.candidate("b1", "scientific"),
            self.candidate("b2", "scientific", observation_cost=1.1),
            self.candidate("c1", "interactive"),
            self.candidate("c2", "interactive", observation_cost=1.1),
        )
        selected = ExternalEcologyBatchScheduler().select(candidates, budget=3, max_per_ecology=1)
        self.assertEqual(len(selected), 3)
        self.assertEqual({row.ecology_family for row in selected}, {"software", "scientific", "interactive"})

    def test_deterministic_residual_requires_two_independent_matching_failures(self):
        candidate = self.candidate()
        rows = (
            self.observation(observation_id="o1", execution_id="runner-a"),
            self.observation(observation_id="o2", execution_id="runner-b"),
        )
        assessment = ExternalResidualPersistenceGate.assess(candidate, rows)
        self.assertTrue(assessment.persistent)
        self.assertEqual(assessment.status, PersistenceStatus.PERSISTS)
        self.assertEqual(assessment.matching_failure_count, 2)

    def test_deterministic_pass_prevents_persistent_failure_claim(self):
        candidate = self.candidate()
        rows = (
            self.observation(observation_id="o1", execution_id="runner-a"),
            self.observation(observation_id="o2", execution_id="runner-b", outcome=ObservationOutcome.PASS, signature=""),
        )
        assessment = ExternalResidualPersistenceGate.assess(candidate, rows)
        self.assertFalse(assessment.persistent)
        self.assertEqual(assessment.status, PersistenceStatus.NOT_REPRODUCED)

    def test_environment_drift_is_not_converted_into_failure_evidence(self):
        candidate = self.candidate()
        rows = (
            self.observation(observation_id="o1", execution_id="runner-a", distance=0.5),
            self.observation(observation_id="o2", execution_id="runner-b", distance=0.5),
        )
        assessment = ExternalResidualPersistenceGate.assess(candidate, rows)
        self.assertFalse(assessment.persistent)
        self.assertEqual(assessment.status, PersistenceStatus.OBSERVATION_UNAVAILABLE)

    def test_repeatable_stochastic_residual_needs_high_recurrence(self):
        candidate = self.candidate(deterministic_expected=False, repeatable_contract=True)
        rows = tuple(
            self.observation(observation_id=f"o{i}", execution_id=f"runner-{i}")
            for i in range(4)
        ) + (
            self.observation(observation_id="o4", execution_id="runner-4", outcome=ObservationOutcome.PASS, signature=""),
        )
        assessment = ExternalResidualPersistenceGate.assess(candidate, rows)
        self.assertTrue(assessment.persistent)
        self.assertEqual(assessment.matching_failure_count, 4)
        self.assertEqual(assessment.pass_count, 1)

    def test_public_persistent_inexpressibility_opens_development_pressure_not_promotion(self):
        candidate = self.candidate(exposure=ExposureClass.PUBLIC_DEV)
        persistence = ExternalResidualPersistenceGate.assess(
            candidate,
            (
                self.observation(observation_id="o1", execution_id="runner-a"),
                self.observation(observation_id="o2", execution_id="runner-b"),
            ),
        )
        probe = FrozenBodyProbe(
            candidate_id="c1",
            frozen_body_hash="body-104",
            old_language_candidate_count=0,
            old_language_search_complete=True,
            more_compute_repeats=16,
            more_compute_candidate_count=0,
            current_outcome_used_for_generation=False,
            post_freeze_human_structural_repairs=0,
            solution_or_root_cause_leakage=False,
        )
        assessment = ExternalWorldInexpressibilityGate.assess(candidate, persistence, probe)
        pressure = ExternalWorldInexpressibilityGate.pressure(candidate, assessment, probe)
        self.assertTrue(assessment.language_pressure_open)
        self.assertFalse(assessment.promotion_authority)
        self.assertIsNotNone(pressure)
        self.assertTrue(pressure.development_pressure_only)

    def test_heldout_persistent_inexpressibility_can_carry_promotion_authority(self):
        candidate = self.candidate(exposure=ExposureClass.FROZEN_HELDOUT)
        persistence = ExternalResidualPersistenceGate.assess(
            candidate,
            (
                self.observation(observation_id="o1", execution_id="runner-a"),
                self.observation(observation_id="o2", execution_id="runner-b"),
            ),
        )
        probe = FrozenBodyProbe("c1", "body", 0, True, 16, 0, False, 0, False)
        assessment = ExternalWorldInexpressibilityGate.assess(candidate, persistence, probe)
        self.assertTrue(assessment.language_pressure_open)
        self.assertTrue(assessment.promotion_authority)

    def test_inexpressibility_fails_if_more_compute_control_is_not_closed(self):
        candidate = self.candidate()
        persistence = ExternalResidualPersistenceGate.assess(
            candidate,
            (
                self.observation(observation_id="o1", execution_id="runner-a"),
                self.observation(observation_id="o2", execution_id="runner-b"),
            ),
        )
        probe = FrozenBodyProbe("c1", "body", 0, True, 8, 0, False, 0, False)
        assessment = ExternalWorldInexpressibilityGate.assess(candidate, persistence, probe)
        self.assertFalse(assessment.language_pressure_open)
        self.assertIn("more_compute_control_too_small", assessment.reasons)

    def test_three_ecology_delta_frontier_acceleration_candidate(self):
        ledger = ProspectiveExternalMetaLedger()
        rows = (
            ExternalGenerationTransition(1, "b0", "b1", "software", 0.0, 1.0, 4.0, 4.0, 2.0, 1.0, 0.95, 1.0, True, True),
            ExternalGenerationTransition(2, "b1", "b2", "interactive", 1.0, 3.0, 3.0, 3.0, 1.0, 1.0, 0.95, 2.0, True, True),
            ExternalGenerationTransition(3, "b2", "b3", "scientific", 3.0, 6.0, 2.0, 2.0, 0.0, 1.0, 0.95, 3.0, True, True),
        )
        self.assertTrue(all(ledger.append(row) for row in rows))
        assessment = ledger.assess()
        self.assertEqual(assessment.status, "PASS_BOUNDED_MULTI_ECOLOGY_META_ACCELERATION_CANDIDATE")
        self.assertEqual(assessment.frontier_delta_trajectory, (1.0, 2.0, 3.0))
        self.assertTrue(assessment.strict_transition_productivity_growth)
        self.assertFalse(assessment.global_recursive_acceleration)

    def test_absolute_frontier_growth_does_not_mask_declining_delta_productivity(self):
        ledger = ProspectiveExternalMetaLedger()
        rows = (
            ExternalGenerationTransition(1, "b0", "b1", "software", 0.0, 2.0, 1.0, 1.0, 0.0, 1.0, 0.95, 1.0, True, True),
            ExternalGenerationTransition(2, "b1", "b2", "interactive", 2.0, 3.0, 5.0, 5.0, 0.0, 1.0, 0.95, 2.0, True, True),
            ExternalGenerationTransition(3, "b2", "b3", "scientific", 3.0, 4.0, 10.0, 10.0, 0.0, 1.0, 0.95, 3.0, True, True),
        )
        for row in rows:
            self.assertTrue(ledger.append(row))
        assessment = ledger.assess()
        self.assertTrue(assessment.strict_frontier_growth)
        self.assertFalse(assessment.strict_transition_productivity_growth)
        self.assertEqual(assessment.status, "INSUFFICIENT_MULTI_ECOLOGY_META_ACCELERATION_EVIDENCE")

    def test_ledger_rejects_broken_body_lineage(self):
        ledger = ProspectiveExternalMetaLedger()
        self.assertTrue(ledger.append(ExternalGenerationTransition(1, "b0", "b1", "software", 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.95, 1.0, True, True)))
        self.assertFalse(ledger.append(ExternalGenerationTransition(2, "wrong-parent", "b2", "interactive", 1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 0.95, 2.0, True, True)))


if __name__ == "__main__":
    unittest.main()
