from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence, Tuple


OPEN_BUDGET_CAPABILITY_PRIMARY = "OPEN_BUDGET_CAPABILITY_PRIMARY"


@dataclass(frozen=True)
class FrontierModelReceipt:
    provider: str
    model_id: str
    task_contract_hash: str
    score: float
    sample_count: int
    external_receipt_id: str
    benchmark_disjoint: bool
    hidden_until_freeze: bool
    same_task_contract: bool
    authority_verified: bool
    compute_cost: float = 0.0
    reasoning_effort: str = ""

    def valid(self) -> bool:
        return bool(
            self.provider.strip()
            and self.model_id.strip()
            and self.task_contract_hash.strip()
            and self.external_receipt_id.strip()
            and self.sample_count > 0
            and isfinite(float(self.score))
            and 0.0 <= float(self.score) <= 1.0
            and self.benchmark_disjoint
            and self.hidden_until_freeze
            and self.same_task_contract
            and self.authority_verified
        )


@dataclass(frozen=True)
class ArteStructuralRun:
    parent_body_hash: str
    child_body_hash: str
    task_contract_hash: str
    score: float
    sample_count: int
    world_caused_mutation_receipt_id: str
    generated_representations: int
    generated_operators: int
    generated_topology_changes: int
    structural_frontier_delta: float
    retained_competence: float
    calibration_score: float
    authority_verified: bool
    benchmark_disjoint: bool
    frozen_before_outcomes: bool
    current_hidden_outcomes_used_for_generation: bool
    post_freeze_human_structural_intervention: int
    compute_cost: float = 0.0
    evidence_cost: float = 0.0

    @property
    def generated_structure_count(self) -> int:
        return int(self.generated_representations) + int(self.generated_operators) + int(
            self.generated_topology_changes
        )

    def structurally_ascended(self) -> bool:
        return bool(
            self.parent_body_hash
            and self.child_body_hash
            and self.parent_body_hash != self.child_body_hash
            and self.world_caused_mutation_receipt_id.strip()
            and self.generated_structure_count > 0
            and float(self.structural_frontier_delta) > 0.0
        )


@dataclass(frozen=True)
class FrontierSuperiorityAssessment:
    status: str
    budget_mode: str
    valid_frontier_provider_count: int
    valid_frontier_model_count: int
    strongest_frontier_provider: str
    strongest_frontier_model: str
    strongest_frontier_score: float
    arte_score: float
    absolute_advantage: float
    minimum_required_margin: float
    beats_all_frontier_models: bool
    structural_ascent: bool
    retained_competence: bool
    calibrated: bool
    same_task_contract: bool
    external_authority_valid: bool
    post_freeze_human_free: bool
    outcome_leakage_free: bool
    cost_efficiency_is_promotion_gate: bool
    arte_compute_cost: float
    strongest_frontier_compute_cost: float
    reasons: Tuple[str, ...]


class ExternalFrontierSuperiorityGate:
    """Primary promotion gate for structural capability and live frontier superiority.

    Cost is recorded but is deliberately not a promotion criterion. A more expensive
    child can pass if it generates a genuinely higher structure and beats the strongest
    valid external frontier baseline on the exact same frozen held-out task contract.
    """

    @staticmethod
    def assess(
        arte: ArteStructuralRun,
        frontier_receipts: Sequence[FrontierModelReceipt],
        *,
        min_frontier_providers: int = 3,
        min_absolute_margin: float = 0.02,
        retained_floor: float = 0.95,
        calibration_floor: float = 0.80,
    ) -> FrontierSuperiorityAssessment:
        reasons = []
        valid = tuple(row for row in frontier_receipts if row.valid())
        providers = {row.provider for row in valid}

        same_contract_rows = tuple(
            row for row in valid if row.task_contract_hash == arte.task_contract_hash
        )
        same_contract = bool(
            arte.task_contract_hash
            and same_contract_rows
            and len(same_contract_rows) == len(valid)
        )

        enough_providers = len(providers) >= max(1, int(min_frontier_providers))
        if not enough_providers:
            reasons.append("insufficient_external_frontier_provider_diversity")
        if not same_contract:
            reasons.append("frontier_and_arte_task_contract_mismatch")

        if valid:
            strongest = max(valid, key=lambda row: (float(row.score), row.provider, row.model_id))
            strongest_score = float(strongest.score)
            strongest_provider = strongest.provider
            strongest_model = strongest.model_id
            strongest_cost = float(strongest.compute_cost)
        else:
            strongest_score = 0.0
            strongest_provider = ""
            strongest_model = ""
            strongest_cost = 0.0
            reasons.append("no_valid_external_frontier_receipts")

        arte_score_valid = bool(
            arte.sample_count > 0
            and isfinite(float(arte.score))
            and 0.0 <= float(arte.score) <= 1.0
        )
        if not arte_score_valid:
            reasons.append("invalid_arte_score")

        structural_ascent = arte.structurally_ascended()
        if not structural_ascent:
            reasons.append("no_world_caused_structural_ascent")

        retained = float(arte.retained_competence) >= float(retained_floor)
        if not retained:
            reasons.append("retained_competence_below_floor")

        calibrated = float(arte.calibration_score) >= float(calibration_floor)
        if not calibrated:
            reasons.append("calibration_below_floor")

        authority = bool(arte.authority_verified and arte.benchmark_disjoint)
        if not authority:
            reasons.append("arte_external_authority_invalid")

        frozen = bool(arte.frozen_before_outcomes)
        if not frozen:
            reasons.append("arte_not_frozen_before_outcomes")

        human_free = int(arte.post_freeze_human_structural_intervention) == 0
        if not human_free:
            reasons.append("post_freeze_human_structural_intervention")

        leakage_free = not arte.current_hidden_outcomes_used_for_generation
        if not leakage_free:
            reasons.append("hidden_outcome_used_for_generation")

        advantage = float(arte.score) - strongest_score if arte_score_valid and valid else 0.0
        beats = bool(
            arte_score_valid
            and valid
            and same_contract
            and advantage >= float(min_absolute_margin)
        )
        if not beats:
            reasons.append("arte_does_not_exceed_strongest_frontier_margin")

        passed = all(
            (
                enough_providers,
                same_contract,
                arte_score_valid,
                structural_ascent,
                retained,
                calibrated,
                authority,
                frozen,
                human_free,
                leakage_free,
                beats,
            )
        )

        return FrontierSuperiorityAssessment(
            status=(
                "PASS_EXTERNAL_FRONTIER_SUPERIORITY_WITH_STRUCTURAL_ASCENT"
                if passed
                else "INSUFFICIENT_EXTERNAL_FRONTIER_SUPERIORITY_EVIDENCE"
            ),
            budget_mode=OPEN_BUDGET_CAPABILITY_PRIMARY,
            valid_frontier_provider_count=len(providers),
            valid_frontier_model_count=len(valid),
            strongest_frontier_provider=strongest_provider,
            strongest_frontier_model=strongest_model,
            strongest_frontier_score=strongest_score,
            arte_score=float(arte.score),
            absolute_advantage=advantage,
            minimum_required_margin=float(min_absolute_margin),
            beats_all_frontier_models=beats,
            structural_ascent=structural_ascent,
            retained_competence=retained,
            calibrated=calibrated,
            same_task_contract=same_contract,
            external_authority_valid=authority,
            post_freeze_human_free=human_free,
            outcome_leakage_free=leakage_free,
            cost_efficiency_is_promotion_gate=False,
            arte_compute_cost=float(arte.compute_cost),
            strongest_frontier_compute_cost=strongest_cost,
            reasons=tuple(reasons),
        )
