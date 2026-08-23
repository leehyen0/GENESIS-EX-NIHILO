from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Sequence, Tuple
import hashlib
import json

from .external_experience import ExposureClass


class ObservationOutcome(str, Enum):
    FAIL = "FAIL"
    PASS = "PASS"
    UNAVAILABLE = "UNAVAILABLE"
    DRIFTED = "DRIFTED"


class AcquisitionStatus(str, Enum):
    ELIGIBLE = "ELIGIBLE_FOR_PROSPECTIVE_REOBSERVATION"
    CONTAMINATED = "REJECT_CONTAMINATED"
    PROVENANCE_UNFROZEN = "REJECT_PROVENANCE_UNFROZEN"
    ENVIRONMENT_UNRESOLVED = "REJECT_ENVIRONMENT_UNRESOLVED"
    STOCHASTICITY_UNRESOLVED = "REJECT_STOCHASTICITY_UNRESOLVED"


class PersistenceStatus(str, Enum):
    PERSISTS = "EXTERNAL_RESIDUAL_PERSISTS"
    NOT_REPRODUCED = "EXTERNAL_RESIDUAL_NOT_REPRODUCED"
    OBSERVATION_UNAVAILABLE = "EXTERNAL_OBSERVATION_UNAVAILABLE"
    INSUFFICIENT = "INSUFFICIENT_PROSPECTIVE_REOBSERVATION"
    BLOCKED = "PERSISTENCE_GATE_BLOCKED"


@dataclass(frozen=True)
class ExternalWorldCandidate:
    candidate_id: str
    repository: str
    issue_ref: str
    commit_sha: str
    exact_command: str
    ecology_family: str
    source_class: str
    exposure: ExposureClass
    original_failure_signature: str
    observation_cost: float = 1.0
    deterministic_expected: bool = True
    seed_controlled: bool = False
    repeatable_contract: bool = False
    repository_wide_contamination_search_complete: bool = False
    answer_seen_before_freeze: bool = False
    patch_seen_before_freeze: bool = False
    root_cause_seen_before_freeze: bool = False
    related_fix_link_seen_before_freeze: bool = False
    dependency_lock_frozen: bool = False
    runtime_frozen: bool = False
    hardware_reconstructable: bool = True
    external_service_dependencies: int = 0
    external_service_state_reconstructable: bool = True
    independent_external_origin: bool = True

    def provenance_frozen(self) -> bool:
        return bool(
            self.repository
            and self.issue_ref
            and len(self.commit_sha) >= 7
            and self.exact_command.strip()
            and self.original_failure_signature.strip()
        )

    def contamination_free(self) -> bool:
        return bool(
            self.repository_wide_contamination_search_complete
            and not self.answer_seen_before_freeze
            and not self.patch_seen_before_freeze
            and not self.root_cause_seen_before_freeze
            and not self.related_fix_link_seen_before_freeze
        )

    def reconstructability(self) -> float:
        checks = [
            self.dependency_lock_frozen,
            self.runtime_frozen,
            self.hardware_reconstructable,
        ]
        if self.external_service_dependencies > 0:
            checks.append(self.external_service_state_reconstructable)
        return sum(1.0 for value in checks if value) / float(len(checks))


@dataclass(frozen=True)
class PreObservationDecision:
    candidate_id: str
    status: AcquisitionStatus
    eligible: bool
    contamination_free: bool
    reconstructability: float
    quality_score: float
    reasons: Tuple[str, ...] = ()


class ExternalEcologyAcquisitionGate:
    """Fail closed before any current external candidate consequence is observed."""

    @staticmethod
    def evaluate(candidate: ExternalWorldCandidate) -> PreObservationDecision:
        reasons = []
        if not candidate.provenance_frozen():
            return PreObservationDecision(
                candidate.candidate_id,
                AcquisitionStatus.PROVENANCE_UNFROZEN,
                False,
                candidate.contamination_free(),
                candidate.reconstructability(),
                0.0,
                ("missing_exact_source_or_command_contract",),
            )

        if not candidate.contamination_free():
            if not candidate.repository_wide_contamination_search_complete:
                reasons.append("repository_wide_contamination_search_incomplete")
            if candidate.answer_seen_before_freeze:
                reasons.append("answer_seen_before_freeze")
            if candidate.patch_seen_before_freeze:
                reasons.append("patch_seen_before_freeze")
            if candidate.root_cause_seen_before_freeze:
                reasons.append("root_cause_seen_before_freeze")
            if candidate.related_fix_link_seen_before_freeze:
                reasons.append("related_fix_link_seen_before_freeze")
            return PreObservationDecision(
                candidate.candidate_id,
                AcquisitionStatus.CONTAMINATED,
                False,
                False,
                candidate.reconstructability(),
                0.0,
                tuple(reasons),
            )

        reconstructability = candidate.reconstructability()
        if reconstructability < 1.0:
            return PreObservationDecision(
                candidate.candidate_id,
                AcquisitionStatus.ENVIRONMENT_UNRESOLVED,
                False,
                True,
                reconstructability,
                0.0,
                ("environment_contract_not_fully_reconstructable",),
            )

        if not candidate.deterministic_expected and not (
            candidate.seed_controlled or candidate.repeatable_contract
        ):
            return PreObservationDecision(
                candidate.candidate_id,
                AcquisitionStatus.STOCHASTICITY_UNRESOLVED,
                False,
                True,
                reconstructability,
                0.0,
                ("stochastic_failure_without_seed_or_repeatable_contract",),
            )

        determinism = 1.0 if candidate.deterministic_expected or candidate.seed_controlled else 0.75
        service_penalty = 1.0 / (1.0 + 0.5 * max(0, int(candidate.external_service_dependencies)))
        cost_penalty = 1.0 / max(1.0, float(candidate.observation_cost))
        independence = 1.0 if candidate.independent_external_origin else 0.25
        quality = reconstructability * determinism * service_penalty * cost_penalty * independence
        return PreObservationDecision(
            candidate.candidate_id,
            AcquisitionStatus.ELIGIBLE,
            True,
            True,
            reconstructability,
            float(quality),
            (),
        )


class ExternalEcologyBatchScheduler:
    """Select a diverse pre-outcome cohort without candidate consequences."""

    def select(
        self,
        candidates: Sequence[ExternalWorldCandidate],
        *,
        budget: int,
        max_per_ecology: int = 2,
    ) -> Tuple[ExternalWorldCandidate, ...]:
        evaluated = [
            (candidate, ExternalEcologyAcquisitionGate.evaluate(candidate))
            for candidate in candidates
        ]
        eligible = [
            (candidate, decision)
            for candidate, decision in evaluated
            if decision.eligible
        ]
        eligible.sort(
            key=lambda row: (
                -row[1].quality_score,
                row[0].ecology_family,
                row[0].candidate_id,
            )
        )

        selected = []
        counts: Dict[str, int] = {}
        limit = max(1, int(max_per_ecology))
        for candidate, _ in eligible:
            if len(selected) >= max(0, int(budget)):
                break
            family = candidate.ecology_family
            if counts.get(family, 0) >= limit:
                continue
            selected.append(candidate)
            counts[family] = counts.get(family, 0) + 1
        return tuple(selected)


@dataclass(frozen=True)
class ProspectiveObservation:
    candidate_id: str
    observation_id: str
    independent_execution_id: str
    outcome: ObservationOutcome
    semantic_signature: str
    contract_match: bool
    environment_distance: float
    infrastructure_ready: bool
    post_freeze_solution_leakage: bool = False


@dataclass(frozen=True)
class PersistenceAssessment:
    candidate_id: str
    status: PersistenceStatus
    persistent: bool
    valid_observation_count: int
    matching_failure_count: int
    pass_count: int
    environment_distances: Tuple[float, ...]
    reason: str


class ExternalResidualPersistenceGate:
    @staticmethod
    def assess(
        candidate: ExternalWorldCandidate,
        observations: Sequence[ProspectiveObservation],
        *,
        max_environment_distance: float = 0.10,
    ) -> PersistenceAssessment:
        pre = ExternalEcologyAcquisitionGate.evaluate(candidate)
        if not pre.eligible:
            return PersistenceAssessment(
                candidate.candidate_id,
                PersistenceStatus.BLOCKED,
                False,
                0,
                0,
                0,
                (),
                pre.status.value,
            )

        valid = [
            row
            for row in observations
            if row.candidate_id == candidate.candidate_id
            and row.contract_match
            and row.infrastructure_ready
            and not row.post_freeze_solution_leakage
            and float(row.environment_distance) <= float(max_environment_distance)
            and row.outcome not in {ObservationOutcome.UNAVAILABLE, ObservationOutcome.DRIFTED}
        ]
        if not valid:
            return PersistenceAssessment(
                candidate.candidate_id,
                PersistenceStatus.OBSERVATION_UNAVAILABLE,
                False,
                0,
                0,
                0,
                (),
                "no_sufficiently_matched_prospective_world",
            )

        unique_execs = {row.independent_execution_id for row in valid}
        required = 2 if candidate.deterministic_expected or candidate.seed_controlled else 3
        if len(unique_execs) < required or len(valid) < required:
            return PersistenceAssessment(
                candidate.candidate_id,
                PersistenceStatus.INSUFFICIENT,
                False,
                len(valid),
                0,
                sum(1 for row in valid if row.outcome == ObservationOutcome.PASS),
                tuple(float(row.environment_distance) for row in valid),
                "insufficient_independent_prospective_observations",
            )

        matching_failures = [
            row
            for row in valid
            if row.outcome == ObservationOutcome.FAIL
            and row.semantic_signature == candidate.original_failure_signature
        ]
        passes = [row for row in valid if row.outcome == ObservationOutcome.PASS]
        other_failures = [
            row
            for row in valid
            if row.outcome == ObservationOutcome.FAIL
            and row.semantic_signature != candidate.original_failure_signature
        ]
        if other_failures:
            return PersistenceAssessment(
                candidate.candidate_id,
                PersistenceStatus.OBSERVATION_UNAVAILABLE,
                False,
                len(valid),
                len(matching_failures),
                len(passes),
                tuple(float(row.environment_distance) for row in valid),
                "failure_signature_drift",
            )

        if candidate.deterministic_expected or candidate.seed_controlled:
            if passes:
                return PersistenceAssessment(
                    candidate.candidate_id,
                    PersistenceStatus.NOT_REPRODUCED,
                    False,
                    len(valid),
                    len(matching_failures),
                    len(passes),
                    tuple(float(row.environment_distance) for row in valid),
                    "matched_world_pass_observed",
                )
            persistent = len(matching_failures) >= required
        else:
            fail_rate = len(matching_failures) / float(len(valid))
            persistent = len(matching_failures) >= required and fail_rate >= 0.8
            if not persistent and len(passes) >= required:
                return PersistenceAssessment(
                    candidate.candidate_id,
                    PersistenceStatus.NOT_REPRODUCED,
                    False,
                    len(valid),
                    len(matching_failures),
                    len(passes),
                    tuple(float(row.environment_distance) for row in valid),
                    "repeatable_world_did_not_preserve_failure",
                )

        return PersistenceAssessment(
            candidate.candidate_id,
            PersistenceStatus.PERSISTS if persistent else PersistenceStatus.INSUFFICIENT,
            bool(persistent),
            len(valid),
            len(matching_failures),
            len(passes),
            tuple(float(row.environment_distance) for row in valid),
            "same_semantic_failure_reproduced" if persistent else "mixed_or_insufficient_recurrence",
        )


@dataclass(frozen=True)
class FrozenBodyProbe:
    candidate_id: str
    frozen_body_hash: str
    old_language_candidate_count: int
    old_language_search_complete: bool
    more_compute_repeats: int
    more_compute_candidate_count: int
    current_outcome_used_for_generation: bool
    post_freeze_human_structural_repairs: int
    solution_or_root_cause_leakage: bool


@dataclass(frozen=True)
class InexpressibilityAssessment:
    candidate_id: str
    status: str
    language_pressure_open: bool
    promotion_authority: bool
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class WorldCausedLanguagePressure:
    pressure_id: str
    candidate_id: str
    frozen_body_hash: str
    failure_signature: str
    ecology_family: str
    development_pressure_only: bool = True
    promotion_authority: bool = False


class ExternalWorldInexpressibilityGate:
    @staticmethod
    def assess(
        candidate: ExternalWorldCandidate,
        persistence: PersistenceAssessment,
        probe: FrozenBodyProbe,
    ) -> InexpressibilityAssessment:
        reasons = []
        if not persistence.persistent:
            reasons.append("external_residual_not_persistent")
        if probe.candidate_id != candidate.candidate_id:
            reasons.append("probe_candidate_mismatch")
        if not probe.frozen_body_hash:
            reasons.append("body_not_frozen")
        if not probe.old_language_search_complete:
            reasons.append("old_language_search_not_complete")
        if int(probe.old_language_candidate_count) != 0:
            reasons.append("old_language_has_candidates")
        if int(probe.more_compute_repeats) < 16:
            reasons.append("more_compute_control_too_small")
        if int(probe.more_compute_candidate_count) != 0:
            reasons.append("more_compute_found_candidates")
        if probe.current_outcome_used_for_generation:
            reasons.append("current_outcome_leakage")
        if int(probe.post_freeze_human_structural_repairs) != 0:
            reasons.append("post_freeze_human_repair")
        if probe.solution_or_root_cause_leakage:
            reasons.append("solution_or_root_cause_leakage")

        open_pressure = not reasons
        promotion_authority = bool(
            open_pressure
            and candidate.exposure
            in {
                ExposureClass.FROZEN_HELDOUT,
                ExposureClass.PRIVATE_EXTERNAL,
                ExposureClass.SOURCE_DISJOINT_TRANSFER,
            }
        )
        return InexpressibilityAssessment(
            candidate.candidate_id,
            "CURRENT_BODY_LANGUAGE_INEXPRESSIBLE_ON_EXTERNAL_RESIDUAL"
            if open_pressure
            else "INEXPRESSIBILITY_NOT_ESTABLISHED",
            open_pressure,
            promotion_authority,
            tuple(reasons),
        )

    @staticmethod
    def pressure(
        candidate: ExternalWorldCandidate,
        assessment: InexpressibilityAssessment,
        probe: FrozenBodyProbe,
    ) -> WorldCausedLanguagePressure | None:
        if not assessment.language_pressure_open:
            return None
        raw = "|".join(
            (
                candidate.candidate_id,
                probe.frozen_body_hash,
                candidate.original_failure_signature,
                candidate.ecology_family,
            )
        )
        pressure_id = "EXTERNAL_LANGUAGE_PRESSURE::" + hashlib.sha256(raw.encode()).hexdigest()[:20]
        return WorldCausedLanguagePressure(
            pressure_id=pressure_id,
            candidate_id=candidate.candidate_id,
            frozen_body_hash=probe.frozen_body_hash,
            failure_signature=candidate.original_failure_signature,
            ecology_family=candidate.ecology_family,
            development_pressure_only=not assessment.promotion_authority,
            promotion_authority=assessment.promotion_authority,
        )


@dataclass(frozen=True)
class ExternalGenerationTransition:
    generation: int
    parent_body_hash: str
    child_body_hash: str
    ecology_family: str
    parent_frontier: float
    child_frontier: float
    compute_cost: float
    evidence_cost: float
    human_structural_intervention: float
    retained_competence: float
    calibration_score: float
    meta_improvement_ability: float
    benchmark_disjoint: bool
    authority_verified: bool
    current_outcome_used_to_generate_mutation: bool = False

    @property
    def frontier_delta(self) -> float:
        return float(self.child_frontier) - float(self.parent_frontier)

    @property
    def transition_productivity(self) -> float:
        denominator = max(
            1e-12,
            float(self.compute_cost)
            + float(self.evidence_cost)
            + float(self.human_structural_intervention),
        )
        return self.frontier_delta / denominator


@dataclass(frozen=True)
class MultiEcologyAccelerationAssessment:
    status: str
    transition_count: int
    ecology_count: int
    frontier_delta_trajectory: Tuple[float, ...]
    transition_productivity_trajectory: Tuple[float, ...]
    human_intervention_trajectory: Tuple[float, ...]
    strict_frontier_growth: bool
    strict_transition_productivity_growth: bool
    nonincreasing_human_intervention: bool
    meta_ability_improved: bool
    lineage_continuous: bool
    ecology_diverse: bool
    all_authority_verified: bool
    all_benchmark_disjoint: bool
    no_current_outcome_generation: bool
    global_recursive_acceleration: bool = False


@dataclass
class ProspectiveExternalMetaLedger:
    transitions: Dict[int, ExternalGenerationTransition] = field(default_factory=dict)

    def append(self, transition: ExternalGenerationTransition) -> bool:
        if transition.generation in self.transitions:
            return False
        if self.transitions:
            last_generation = max(self.transitions)
            last = self.transitions[last_generation]
            if transition.generation != last_generation + 1:
                return False
            if transition.parent_body_hash != last.child_body_hash:
                return False
        self.transitions[transition.generation] = transition
        return True

    def ordered(self) -> Tuple[ExternalGenerationTransition, ...]:
        return tuple(self.transitions[key] for key in sorted(self.transitions))

    def assess(
        self,
        *,
        min_transitions: int = 3,
        min_ecologies: int = 3,
        retained_floor: float = 0.95,
        calibration_floor: float = 0.80,
    ) -> MultiEcologyAccelerationAssessment:
        rows = self.ordered()
        enough = len(rows) >= max(1, int(min_transitions))
        deltas = tuple(row.frontier_delta for row in rows)
        productivity = tuple(row.transition_productivity for row in rows)
        humans = tuple(float(row.human_structural_intervention) for row in rows)
        ecologies = {row.ecology_family for row in rows}
        strict_frontier = enough and all(delta > 0.0 for delta in deltas)
        strict_productivity = enough and len(productivity) >= 2 and all(
            b > a for a, b in zip(productivity, productivity[1:])
        )
        human_nonincrease = enough and all(b <= a for a, b in zip(humans, humans[1:]))
        retained = enough and all(row.retained_competence >= retained_floor for row in rows)
        calibration = enough and all(row.calibration_score >= calibration_floor for row in rows)
        meta_ability = enough and all(
            b.meta_improvement_ability >= a.meta_improvement_ability
            for a, b in zip(rows, rows[1:])
        ) and any(
            b.meta_improvement_ability > a.meta_improvement_ability
            for a, b in zip(rows, rows[1:])
        )
        lineage = enough and all(
            child.parent_body_hash == parent.child_body_hash
            for parent, child in zip(rows, rows[1:])
        )
        ecology_diverse = enough and len(ecologies) >= max(1, int(min_ecologies))
        authority = enough and all(row.authority_verified for row in rows)
        disjoint = enough and all(row.benchmark_disjoint for row in rows)
        no_leakage = enough and all(
            not row.current_outcome_used_to_generate_mutation for row in rows
        )
        passed = all(
            (
                enough,
                strict_frontier,
                strict_productivity,
                human_nonincrease,
                retained,
                calibration,
                meta_ability,
                lineage,
                ecology_diverse,
                authority,
                disjoint,
                no_leakage,
            )
        )
        return MultiEcologyAccelerationAssessment(
            status=(
                "PASS_BOUNDED_MULTI_ECOLOGY_META_ACCELERATION_CANDIDATE"
                if passed
                else "INSUFFICIENT_MULTI_ECOLOGY_META_ACCELERATION_EVIDENCE"
            ),
            transition_count=len(rows),
            ecology_count=len(ecologies),
            frontier_delta_trajectory=deltas,
            transition_productivity_trajectory=productivity,
            human_intervention_trajectory=humans,
            strict_frontier_growth=bool(strict_frontier),
            strict_transition_productivity_growth=bool(strict_productivity),
            nonincreasing_human_intervention=bool(human_nonincrease),
            meta_ability_improved=bool(meta_ability),
            lineage_continuous=bool(lineage),
            ecology_diverse=bool(ecology_diverse),
            all_authority_verified=bool(authority),
            all_benchmark_disjoint=bool(disjoint),
            no_current_outcome_generation=bool(no_leakage),
            global_recursive_acceleration=False,
        )


def cohort_fingerprint(candidates: Sequence[ExternalWorldCandidate]) -> str:
    payload = [
        {
            "candidate_id": candidate.candidate_id,
            "repository": candidate.repository,
            "commit_sha": candidate.commit_sha,
            "ecology_family": candidate.ecology_family,
            "source_class": candidate.source_class,
            "exact_command": candidate.exact_command,
        }
        for candidate in sorted(candidates, key=lambda row: row.candidate_id)
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
