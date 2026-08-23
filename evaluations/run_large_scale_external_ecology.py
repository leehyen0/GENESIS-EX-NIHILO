from __future__ import annotations

import json
import random
import sys
from pathlib import Path

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
    cohort_fingerprint,
)
from arte_cognition.external_experience import ExposureClass


ECOLOGIES = (
    "software",
    "interactive",
    "scientific",
    "planning",
    "data_system",
    "tool_use",
)


def make_candidate(
    rng: random.Random,
    candidate_id: str,
    ecology: str,
    *,
    exposure: ExposureClass = ExposureClass.PUBLIC_DEV,
    contamination: bool = False,
    environment_unresolved: bool = False,
    stochastic_unresolved: bool = False,
    observation_cost: float | None = None,
) -> ExternalWorldCandidate:
    deterministic = not stochastic_unresolved
    return ExternalWorldCandidate(
        candidate_id=candidate_id,
        repository=f"external/{ecology}-{candidate_id}",
        issue_ref=f"#{rng.randint(1, 9999)}",
        commit_sha="".join(rng.choice("0123456789abcdef") for _ in range(40)),
        exact_command=f"python -m pytest tests/{ecology}/test_{candidate_id}.py::test_residual -q",
        ecology_family=ecology,
        source_class=f"PUBLIC_{ecology.upper()}_CI",
        exposure=exposure,
        original_failure_signature=f"RESIDUAL::{ecology}::{candidate_id}",
        observation_cost=float(observation_cost if observation_cost is not None else rng.uniform(1.0, 3.0)),
        deterministic_expected=deterministic,
        seed_controlled=False,
        repeatable_contract=False,
        repository_wide_contamination_search_complete=True,
        answer_seen_before_freeze=False,
        patch_seen_before_freeze=False,
        root_cause_seen_before_freeze=bool(contamination),
        related_fix_link_seen_before_freeze=False,
        dependency_lock_frozen=True,
        runtime_frozen=True,
        hardware_reconstructable=True,
        external_service_dependencies=1 if environment_unresolved else 0,
        external_service_state_reconstructable=not environment_unresolved,
        independent_external_origin=True,
    )


def obs(candidate: ExternalWorldCandidate, index: int, outcome: ObservationOutcome, *, distance: float = 0.0, signature: str | None = None) -> ProspectiveObservation:
    return ProspectiveObservation(
        candidate_id=candidate.candidate_id,
        observation_id=f"{candidate.candidate_id}::obs::{index}",
        independent_execution_id=f"runner::{candidate.candidate_id}::{index}",
        outcome=outcome,
        semantic_signature=(candidate.original_failure_signature if signature is None else signature),
        contract_match=True,
        environment_distance=float(distance),
        infrastructure_ready=True,
        post_freeze_solution_leakage=False,
    )


def main(seed_path: str) -> int:
    seed = int(Path(seed_path).read_text(encoding="utf-8").strip())
    rng = random.Random(seed)

    candidates = []
    expected_eligible = set()
    rejected_kinds = {"contaminated": 0, "environment": 0, "stochastic": 0}
    for ecology in ECOLOGIES:
        for index in range(12):
            candidate_id = f"{ecology}-{index:02d}"
            mode = index % 6
            if mode == 0:
                candidate = make_candidate(rng, candidate_id, ecology, contamination=True)
                rejected_kinds["contaminated"] += 1
            elif mode == 1:
                candidate = make_candidate(rng, candidate_id, ecology, environment_unresolved=True)
                rejected_kinds["environment"] += 1
            elif mode == 2:
                candidate = make_candidate(rng, candidate_id, ecology, stochastic_unresolved=True)
                rejected_kinds["stochastic"] += 1
            else:
                candidate = make_candidate(rng, candidate_id, ecology)
                expected_eligible.add(candidate_id)
            candidates.append(candidate)

    decisions = {row.candidate_id: ExternalEcologyAcquisitionGate.evaluate(row) for row in candidates}
    eligible_ids = {key for key, value in decisions.items() if value.eligible}
    assert eligible_ids == expected_eligible, (len(eligible_ids), len(expected_eligible))
    assert all(decisions[row.candidate_id].status == AcquisitionStatus.CONTAMINATED for row in candidates if row.root_cause_seen_before_freeze)
    assert all(decisions[row.candidate_id].status == AcquisitionStatus.ENVIRONMENT_UNRESOLVED for row in candidates if row.external_service_dependencies > 0)
    assert all(decisions[row.candidate_id].status == AcquisitionStatus.STOCHASTICITY_UNRESOLVED for row in candidates if not row.deterministic_expected)

    scheduler = ExternalEcologyBatchScheduler()
    selected = scheduler.select(candidates, budget=12, max_per_ecology=2)
    assert len(selected) == 12
    assert {row.ecology_family for row in selected} == set(ECOLOGIES)
    assert all(sum(1 for row in selected if row.ecology_family == ecology) == 2 for ecology in ECOLOGIES)
    assert all(row.candidate_id in expected_eligible for row in selected)

    persistent_candidate = selected[0]
    persistent = ExternalResidualPersistenceGate.assess(
        persistent_candidate,
        (
            obs(persistent_candidate, 0, ObservationOutcome.FAIL, distance=0.02),
            obs(persistent_candidate, 1, ObservationOutcome.FAIL, distance=0.03),
        ),
    )
    assert persistent.status == PersistenceStatus.PERSISTS
    assert persistent.persistent

    pass_candidate = selected[1]
    not_reproduced = ExternalResidualPersistenceGate.assess(
        pass_candidate,
        (
            obs(pass_candidate, 0, ObservationOutcome.FAIL, distance=0.01),
            obs(pass_candidate, 1, ObservationOutcome.PASS, distance=0.01, signature=""),
        ),
    )
    assert not_reproduced.status == PersistenceStatus.NOT_REPRODUCED
    assert not not_reproduced.persistent

    drift_candidate = selected[2]
    unavailable = ExternalResidualPersistenceGate.assess(
        drift_candidate,
        (
            obs(drift_candidate, 0, ObservationOutcome.FAIL, distance=0.50),
            obs(drift_candidate, 1, ObservationOutcome.FAIL, distance=0.50),
        ),
    )
    assert unavailable.status == PersistenceStatus.OBSERVATION_UNAVAILABLE

    public_probe = FrozenBodyProbe(
        candidate_id=persistent_candidate.candidate_id,
        frozen_body_hash="CANONICAL_104",
        old_language_candidate_count=0,
        old_language_search_complete=True,
        more_compute_repeats=16,
        more_compute_candidate_count=0,
        current_outcome_used_for_generation=False,
        post_freeze_human_structural_repairs=0,
        solution_or_root_cause_leakage=False,
    )
    public_inexpressible = ExternalWorldInexpressibilityGate.assess(persistent_candidate, persistent, public_probe)
    assert public_inexpressible.language_pressure_open
    assert not public_inexpressible.promotion_authority
    assert public_inexpressible.status == "CURRENT_BODY_LANGUAGE_INEXPRESSIBLE_ON_EXTERNAL_RESIDUAL"

    heldout_candidate = make_candidate(
        rng,
        "sealed-heldout",
        "scientific",
        exposure=ExposureClass.FROZEN_HELDOUT,
        observation_cost=1.0,
    )
    heldout_persistence = ExternalResidualPersistenceGate.assess(
        heldout_candidate,
        (
            obs(heldout_candidate, 0, ObservationOutcome.FAIL, distance=0.01),
            obs(heldout_candidate, 1, ObservationOutcome.FAIL, distance=0.01),
        ),
    )
    heldout_probe = FrozenBodyProbe(
        candidate_id=heldout_candidate.candidate_id,
        frozen_body_hash="CANONICAL_104",
        old_language_candidate_count=0,
        old_language_search_complete=True,
        more_compute_repeats=16,
        more_compute_candidate_count=0,
        current_outcome_used_for_generation=False,
        post_freeze_human_structural_repairs=0,
        solution_or_root_cause_leakage=False,
    )
    heldout_inexpressible = ExternalWorldInexpressibilityGate.assess(
        heldout_candidate,
        heldout_persistence,
        heldout_probe,
    )
    assert heldout_inexpressible.language_pressure_open
    assert heldout_inexpressible.promotion_authority

    wrong_probe = FrozenBodyProbe(
        candidate_id=heldout_candidate.candidate_id,
        frozen_body_hash="CANONICAL_104",
        old_language_candidate_count=0,
        old_language_search_complete=True,
        more_compute_repeats=16,
        more_compute_candidate_count=1,
        current_outcome_used_for_generation=False,
        post_freeze_human_structural_repairs=0,
        solution_or_root_cause_leakage=False,
    )
    wrong_inexpressible = ExternalWorldInexpressibilityGate.assess(
        heldout_candidate,
        heldout_persistence,
        wrong_probe,
    )
    assert not wrong_inexpressible.language_pressure_open
    assert not wrong_inexpressible.promotion_authority

    ledger = ProspectiveExternalMetaLedger()
    positive_rows = (
        ExternalGenerationTransition(1, "b0", "b1", "software", 0.0, 1.0, 5.0, 4.0, 2.0, 1.0, 0.95, 1.0, True, True),
        ExternalGenerationTransition(2, "b1", "b2", "interactive", 1.0, 3.0, 4.0, 3.0, 1.0, 1.0, 0.95, 2.0, True, True),
        ExternalGenerationTransition(3, "b2", "b3", "scientific", 3.0, 6.0, 2.0, 2.0, 0.0, 1.0, 0.95, 3.0, True, True),
    )
    assert all(ledger.append(row) for row in positive_rows)
    positive = ledger.assess()
    assert positive.status == "PASS_BOUNDED_MULTI_ECOLOGY_META_ACCELERATION_CANDIDATE"
    assert positive.frontier_delta_trajectory == (1.0, 2.0, 3.0)
    assert positive.strict_transition_productivity_growth
    assert positive.ecology_diverse
    assert positive.global_recursive_acceleration is False

    wrong_ledger = ProspectiveExternalMetaLedger()
    wrong_rows = (
        ExternalGenerationTransition(1, "w0", "w1", "software", 0.0, 2.0, 1.0, 1.0, 0.0, 1.0, 0.95, 1.0, True, True),
        ExternalGenerationTransition(2, "w1", "w2", "software", 2.0, 3.0, 4.0, 4.0, 0.0, 1.0, 0.95, 2.0, True, True),
        ExternalGenerationTransition(3, "w2", "w3", "software", 3.0, 4.0, 8.0, 8.0, 0.0, 1.0, 0.95, 3.0, True, True),
    )
    assert all(wrong_ledger.append(row) for row in wrong_rows)
    wrong = wrong_ledger.assess()
    assert wrong.status == "INSUFFICIENT_MULTI_ECOLOGY_META_ACCELERATION_EVIDENCE"
    assert not wrong.strict_transition_productivity_growth
    assert not wrong.ecology_diverse
    assert wrong.global_recursive_acceleration is False

    output = {
        "status": "PASS_LARGE_SCALE_EXTERNAL_ECOLOGY_SHADOW_CONTRACT",
        "seed": seed,
        "candidate_count": len(candidates),
        "eligible_count": len(eligible_ids),
        "selected_count": len(selected),
        "selected_ecologies": sorted({row.ecology_family for row in selected}),
        "cohort_fingerprint": cohort_fingerprint(selected),
        "rejected": rejected_kinds,
        "persistent_positive_status": persistent.status.value,
        "not_reproduced_control_status": not_reproduced.status.value,
        "environment_drift_control_status": unavailable.status.value,
        "public_inexpressibility_status": public_inexpressible.status,
        "public_promotion_authority": public_inexpressible.promotion_authority,
        "heldout_promotion_authority": heldout_inexpressible.promotion_authority,
        "wrong_more_compute_pressure_open": wrong_inexpressible.language_pressure_open,
        "multi_ecology_meta_status": positive.status,
        "frontier_deltas": list(positive.frontier_delta_trajectory),
        "transition_productivity": list(positive.transition_productivity_trajectory),
        "wrong_meta_status": wrong.status,
        "AGI": False,
        "ASI": False,
        "global_recursive_acceleration": False,
        "physical_world_closure": False,
        "foundation_weight_change": False,
        "independent_organizational_custody": False,
    }
    print(json.dumps(output, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
