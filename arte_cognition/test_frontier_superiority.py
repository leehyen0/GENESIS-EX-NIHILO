from __future__ import annotations

import unittest

from arte_cognition.frontier_superiority import (
    ArteStructuralRun,
    ExternalFrontierSuperiorityGate,
    FrontierModelReceipt,
    OPEN_BUDGET_CAPABILITY_PRIMARY,
)


class FrontierSuperiorityTests(unittest.TestCase):
    def baselines(self, contract="heldout-contract"):
        return (
            FrontierModelReceipt(
                provider="OpenAI",
                model_id="GPT-5.6-Sol-max",
                task_contract_hash=contract,
                score=0.81,
                sample_count=100,
                external_receipt_id="openai-live-receipt",
                benchmark_disjoint=True,
                hidden_until_freeze=True,
                same_task_contract=True,
                authority_verified=True,
                compute_cost=10.0,
                reasoning_effort="max",
            ),
            FrontierModelReceipt(
                provider="Anthropic",
                model_id="Claude-Fable-5-max",
                task_contract_hash=contract,
                score=0.83,
                sample_count=100,
                external_receipt_id="anthropic-live-receipt",
                benchmark_disjoint=True,
                hidden_until_freeze=True,
                same_task_contract=True,
                authority_verified=True,
                compute_cost=12.0,
                reasoning_effort="max",
            ),
            FrontierModelReceipt(
                provider="Google",
                model_id="Gemini-3.7-Flash-high",
                task_contract_hash=contract,
                score=0.79,
                sample_count=100,
                external_receipt_id="google-live-receipt",
                benchmark_disjoint=True,
                hidden_until_freeze=True,
                same_task_contract=True,
                authority_verified=True,
                compute_cost=5.0,
                reasoning_effort="high",
            ),
        )

    def arte(self, score=0.86, compute_cost=1000.0, **overrides):
        data = dict(
            parent_body_hash="body-parent",
            child_body_hash="body-child",
            task_contract_hash="heldout-contract",
            score=score,
            sample_count=100,
            world_caused_mutation_receipt_id="world-mutation-receipt",
            generated_representations=2,
            generated_operators=1,
            generated_topology_changes=1,
            structural_frontier_delta=1.0,
            retained_competence=0.98,
            calibration_score=0.90,
            authority_verified=True,
            benchmark_disjoint=True,
            frozen_before_outcomes=True,
            current_hidden_outcomes_used_for_generation=False,
            post_freeze_human_structural_intervention=0,
            compute_cost=compute_cost,
            evidence_cost=100.0,
        )
        data.update(overrides)
        return ArteStructuralRun(**data)

    def test_more_expensive_higher_structure_can_pass(self):
        assessment = ExternalFrontierSuperiorityGate.assess(
            self.arte(score=0.86, compute_cost=1000.0),
            self.baselines(),
            min_absolute_margin=0.02,
        )
        self.assertEqual(
            assessment.status,
            "PASS_EXTERNAL_FRONTIER_SUPERIORITY_WITH_STRUCTURAL_ASCENT",
        )
        self.assertEqual(assessment.budget_mode, OPEN_BUDGET_CAPABILITY_PRIMARY)
        self.assertFalse(assessment.cost_efficiency_is_promotion_gate)
        self.assertGreater(assessment.arte_compute_cost, assessment.strongest_frontier_compute_cost)
        self.assertTrue(assessment.beats_all_frontier_models)
        self.assertTrue(assessment.structural_ascent)

    def test_lower_cost_is_not_a_substitute_for_frontier_superiority(self):
        assessment = ExternalFrontierSuperiorityGate.assess(
            self.arte(score=0.82, compute_cost=0.1),
            self.baselines(),
            min_absolute_margin=0.02,
        )
        self.assertEqual(
            assessment.status,
            "INSUFFICIENT_EXTERNAL_FRONTIER_SUPERIORITY_EVIDENCE",
        )
        self.assertFalse(assessment.beats_all_frontier_models)

    def test_high_score_without_generated_structure_does_not_pass(self):
        assessment = ExternalFrontierSuperiorityGate.assess(
            self.arte(
                score=0.95,
                generated_representations=0,
                generated_operators=0,
                generated_topology_changes=0,
                structural_frontier_delta=0.0,
            ),
            self.baselines(),
        )
        self.assertFalse(assessment.structural_ascent)
        self.assertEqual(
            assessment.status,
            "INSUFFICIENT_EXTERNAL_FRONTIER_SUPERIORITY_EVIDENCE",
        )

    def test_post_freeze_human_repair_is_causal_invalidity_not_cost_penalty(self):
        assessment = ExternalFrontierSuperiorityGate.assess(
            self.arte(score=0.90, post_freeze_human_structural_intervention=1),
            self.baselines(),
        )
        self.assertFalse(assessment.post_freeze_human_free)
        self.assertIn("post_freeze_human_structural_intervention", assessment.reasons)
        self.assertFalse(assessment.cost_efficiency_is_promotion_gate)

    def test_three_external_provider_receipts_are_required(self):
        assessment = ExternalFrontierSuperiorityGate.assess(
            self.arte(),
            self.baselines()[:2],
            min_frontier_providers=3,
        )
        self.assertEqual(assessment.valid_frontier_provider_count, 2)
        self.assertIn("insufficient_external_frontier_provider_diversity", assessment.reasons)

    def test_task_contract_mismatch_fails_closed(self):
        rows = list(self.baselines())
        rows[2] = FrontierModelReceipt(
            provider="Google",
            model_id="Gemini-3.7-Flash-high",
            task_contract_hash="different-contract",
            score=0.79,
            sample_count=100,
            external_receipt_id="google-live-receipt",
            benchmark_disjoint=True,
            hidden_until_freeze=True,
            same_task_contract=True,
            authority_verified=True,
        )
        assessment = ExternalFrontierSuperiorityGate.assess(self.arte(), tuple(rows))
        self.assertFalse(assessment.same_task_contract)
        self.assertIn("frontier_and_arte_task_contract_mismatch", assessment.reasons)


if __name__ == "__main__":
    unittest.main()
