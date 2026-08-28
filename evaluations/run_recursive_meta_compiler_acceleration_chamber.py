from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

from arte_cognition.body_policy_generation import generate_contextual_body_candidate
from arte_cognition.executable_morphology import (
    EdgeSpec,
    MorphologyGenome,
    MorphologyMutator,
    OrganKind,
    OrganSpec,
    PressureVector,
)
from arte_cognition.meta_acceleration import GenerationMetrics, MetaAccelerationLedger
from arte_cognition.morphology_genesis import MorphologyGenesisEngine, MorphologyResidual
from arte_cognition.mutable_meta_compiler import (
    FAMILY_ABSTAIN,
    FAMILY_GENERATOR,
    FAMILY_MUTATOR,
    FAMILY_TOPOLOGY,
    ROUTABLE_FAMILIES,
    learn_meta_compiler_rule,
    meta_compiler_policy_from_body,
)
from arte_cognition.native_recursive_research import NativeMetaMorphologyGenesisEngine
from arte_cognition.native_representation_generator_language import (
    CompositionalRepresentationGenesisEngine,
    derive_generator_language_mutation,
    expression_representation_programs,
)
from arte_cognition.self_evolving_body_checkpoint import checkpoint_dict, restore_body
from evaluations.run_native_recursive_research_cycle7 import _eligible_target_specs, _make_query, _make_support


REGIME_GENERATOR = "GENERATOR_LANGUAGE"
REGIME_MUTATOR = "MUTATOR_POLICY"
REGIME_TOPOLOGY = "TOPOLOGY_REWIRE"
REGIME_ABSTAIN = "ABSTAIN"
REGIMES = (REGIME_GENERATOR, REGIME_MUTATOR, REGIME_TOPOLOGY, REGIME_ABSTAIN)
REGIME_TO_FAMILY = {
    REGIME_GENERATOR: FAMILY_GENERATOR,
    REGIME_MUTATOR: FAMILY_MUTATOR,
    REGIME_TOPOLOGY: FAMILY_TOPOLOGY,
    REGIME_ABSTAIN: FAMILY_ABSTAIN,
}


@dataclass(frozen=True)
class ChamberTask:
    task_id: str
    signal_slot: str
    regime: str
    generator_spec: object | None = None
    generator_support: Tuple[object, ...] = ()
    generator_query: Tuple[int, int] = (0, 0)
    generator_hidden_output: int = 0


def _read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _make_task(rng: random.Random, task_id: str, signal_slot: str, regime: str) -> ChamberTask:
    if regime != REGIME_GENERATOR:
        return ChamberTask(task_id=task_id, signal_slot=signal_slot, regime=regime)
    specs = list(_eligible_target_specs())
    spec = specs[rng.randrange(len(specs))]
    support = _make_support(rng, spec, 6)
    query = _make_query(rng, spec)
    return ChamberTask(
        task_id=task_id,
        signal_slot=signal_slot,
        regime=regime,
        generator_spec=spec,
        generator_support=tuple(support),
        generator_query=query,
        generator_hidden_output=spec.execute(query),
    )


def _tasks_for_regime(
    rng: random.Random,
    *,
    signal_slot: str,
    regime: str,
    prefix: str,
    count: int,
) -> Tuple[ChamberTask, ...]:
    return tuple(_make_task(rng, f"{prefix}-{index:03d}", signal_slot, regime) for index in range(count))


def _generator_success(body, task: ChamberTask) -> bool:
    origin = f"chamber-generator::{task.task_id}"
    mutation = derive_generator_language_mutation(
        body.morphology,
        origin_residual_id=origin,
        failure_fossil="chamber-hidden-generator-language-gap",
    )
    generator_child = MorphologyMutator().apply(body.morphology, mutation)
    residual = MorphologyResidual(
        residual_id=f"repr::{task.task_id}",
        pressure=PressureVector(transfer_failure=1.0, theory_blindspot=1.0),
        same_frozen_phenotype_different_outcome=True,
        more_compute_still_aliased=True,
        missing_artifact_types=(f"artifact_{task.task_id}",),
    )
    candidate = CompositionalRepresentationGenesisEngine(candidate_budget=1).generate(
        generator_child,
        residual,
        task.generator_support,
        expected_generator_origin_residual_id=origin,
    )[0]
    representation_child = MorphologyMutator().apply(generator_child, candidate.mutation)
    program = expression_representation_programs(
        representation_child,
        expected_artifact_type=f"artifact_{task.task_id}",
        expected_residual_id=residual.residual_id,
    )[0]
    return program.execute(task.generator_query) == task.generator_hidden_output


def _mutator_success(body, task: ChamberTask) -> bool:
    origin = f"mutator-origin::{task.task_id}"
    residual = MorphologyResidual(
        residual_id=origin,
        pressure=PressureVector(human_dependency=1.0, theory_blindspot=0.5),
    )
    rows = NativeMetaMorphologyGenesisEngine(candidate_budget=32).generate(body.morphology, (residual,))
    selected = [row for row in rows if row.operation_family == "CHANGE_MUTATOR_POLICY"]
    if len(selected) != 1:
        return False
    child = MorphologyMutator().apply(body.morphology, selected[0].mutation)
    fresh = MorphologyResidual(
        residual_id=f"mutator-fresh::{task.task_id}",
        pressure=PressureVector(human_dependency=1.0, novelty_pressure=0.1, theory_blindspot=0.1),
    )
    result = generate_contextual_body_candidate(
        child,
        fresh,
        expected_policy_origin_residual_id=origin,
    )
    return result.selected_operation_family == "CHANGE_MUTATOR_POLICY" and result.selected_candidate_budget == 1


def _topology_success(task: ChamberTask) -> bool:
    fixture = MorphologyGenome(
        organs=(
            OrganSpec("src", OrganKind.SOURCE, produces=("typed_signal",), implementation_ref="fixture://source"),
            OrganSpec("bad", OrganKind.PERCEPTOR, consumes=("typed_signal",), implementation_ref="fixture://bad"),
            OrganSpec("good", OrganKind.PERCEPTOR, consumes=("typed_signal",), implementation_ref="fixture://good"),
            OrganSpec("governor", OrganKind.GOVERNOR),
            OrganSpec("archive", OrganKind.ARCHIVE),
        ),
        edges=(EdgeSpec("failed-edge", "src", "bad", "typed_signal"),),
        event_order=(),
    )
    residual = MorphologyResidual(
        residual_id=f"topology::{task.task_id}",
        pressure=PressureVector(transfer_failure=1.0),
        failed_edge_ids=("failed-edge",),
        implicated_organ_ids=("good",),
    )
    rows = MorphologyGenesisEngine(candidate_budget=256).generate(fixture, (residual,))
    candidates = [
        row
        for row in rows
        if row.mutation.operation == "REWIRE_EDGE"
        and dict(row.mutation.payload.get("edge", {})).get("target") == "good"
    ]
    if not candidates:
        return False
    child = MorphologyMutator().apply(fixture, candidates[0].mutation)
    edge = next(row for row in child.edges if row.edge_id == "failed-edge")
    return edge.target == "good" and edge.artifact_type == "typed_signal"


def execute_family(body, family: str, task: ChamberTask) -> bool:
    if family == FAMILY_ABSTAIN:
        return task.regime == REGIME_ABSTAIN
    if family == FAMILY_GENERATOR:
        return task.regime == REGIME_GENERATOR and _generator_success(body, task)
    if family == FAMILY_MUTATOR:
        return task.regime == REGIME_MUTATOR and _mutator_success(body, task)
    if family == FAMILY_TOPOLOGY:
        return task.regime == REGIME_TOPOLOGY and _topology_success(task)
    raise ValueError("UNKNOWN_CHAMBER_PROPOSAL_FAMILY")


def proposal_outcomes(body, tasks: Sequence[ChamberTask]) -> Dict[str, float]:
    if not tasks:
        raise ValueError("CHAMBER_TRAINING_TASKS_REQUIRED")
    return {
        family: sum(int(execute_family(body, family, task)) for task in tasks) / len(tasks)
        for family in ROUTABLE_FAMILIES
    }


def validate_body(body, validation: Mapping[str, Sequence[ChamberTask]], learned_slots: Sequence[str]) -> dict:
    correct = 0
    total = 0
    by_regime = {}
    confident_errors = 0
    policy = meta_compiler_policy_from_body(body)
    for regime in REGIMES:
        rows = validation[regime]
        successes = 0
        for task in rows:
            family, confidence = policy.route(task.signal_slot)
            success = execute_family(body, family, task)
            successes += int(success)
            correct += int(success)
            total += 1
            if confidence >= 1.0 and not success:
                confident_errors += 1
        by_regime[regime] = successes / len(rows)
    learned_retained = all(
        all(execute_family(body, policy.route(task.signal_slot)[0], task) for task in validation[regime])
        for regime in REGIMES
        if validation[regime] and validation[regime][0].signal_slot in set(learned_slots)
    )
    return {
        "useful_rate": correct / total,
        "by_regime": by_regime,
        "retained_learned_competence": 1.0 if learned_retained else 0.0,
        "calibration": 1.0 if confident_errors == 0 else 0.0,
    }


def _rotated_outcomes(outcomes: Mapping[str, float]) -> Dict[str, float]:
    families = list(ROUTABLE_FAMILIES)
    rotated = {}
    for index, family in enumerate(families):
        rotated[families[(index + 1) % len(families)]] = float(outcomes[family])
    return rotated


def _restore_roundtrip(body):
    return restore_body(checkpoint_dict(body))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", required=True)
    parser.add_argument("--cycle8-receipt", required=True)
    parser.add_argument("--chamber-freeze", required=True)
    parser.add_argument("--candidate-head", required=True)
    parser.add_argument("--hidden-seed-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    precommit = _read(args.precommit)
    cycle8 = _read(args.cycle8_receipt)
    freeze = _read(args.chamber_freeze)
    if precommit.get("parent_outcome_receipt_sha256") != cycle8.get("outcome_receipt_sha256"):
        raise SystemExit("meta-compiler chamber parent receipt mismatch")
    if freeze.get("candidate_source_head_sha") != args.candidate_head:
        raise SystemExit("meta-compiler chamber source freeze mismatch")
    if not freeze.get("source_frozen_before_hidden_seed") or freeze.get("future_generation_source_edits_allowed"):
        raise SystemExit("meta-compiler chamber source-freeze contract invalid")

    rng = random.Random(int(Path(args.hidden_seed_file).read_text(encoding="utf-8").strip()))
    slots = [f"opaque-signal-{index}" for index in range(4)]
    shuffled_regimes = list(REGIMES)
    rng.shuffle(shuffled_regimes)
    slot_to_regime = dict(zip(slots, shuffled_regimes))
    regime_to_slot = {regime: slot for slot, regime in slot_to_regime.items()}
    training_order = [REGIME_GENERATOR, REGIME_MUTATOR, REGIME_TOPOLOGY]
    rng.shuffle(training_order)

    validation_count = int(precommit["resource_contract"]["validation_tasks_per_regime_per_generation"])
    training_count = int(precommit["resource_contract"]["training_tasks_per_new_regime"])

    full_body = restore_body(dict(freeze["checkpoint"]))
    remove_body = restore_body(dict(freeze["checkpoint"]))
    wrong_body = restore_body(dict(freeze["checkpoint"]))
    shuffle_body = restore_body(dict(freeze["checkpoint"]))

    full_learned_slots: list[str] = []
    wrong_slot_order = [regime_to_slot[regime] for regime in training_order]
    full_rows = []
    remove_rows = []
    wrong_rows = []
    shuffle_rows = []
    body_hashes = []
    policy_fingerprints = []
    learning_receipts = []
    ledger = MetaAccelerationLedger()

    for state_index in range(4):
        generation = 9 + state_index
        validation = {
            regime: _tasks_for_regime(
                rng,
                signal_slot=regime_to_slot[regime],
                regime=regime,
                prefix=f"validation-g{generation}-{regime.lower()}",
                count=validation_count,
            )
            for regime in REGIMES
        }
        full_metrics = validate_body(full_body, validation, full_learned_slots)
        remove_metrics = validate_body(remove_body, validation, ())
        wrong_metrics = validate_body(wrong_body, validation, ())
        shuffle_metrics = validate_body(shuffle_body, validation, ())

        expected_rate = 0.25 * (state_index + 1)
        if abs(full_metrics["useful_rate"] - expected_rate) > 1e-12:
            raise SystemExit(
                f"chamber strict prospective rate mismatch generation={generation} "
                f"actual={full_metrics['useful_rate']} expected={expected_rate}"
            )
        if abs(remove_metrics["useful_rate"] - 0.25) > 1e-12:
            raise SystemExit("chamber REMOVE frontier unexpectedly changed")
        if state_index > 0 and not (wrong_metrics["useful_rate"] < full_metrics["useful_rate"]):
            raise SystemExit("chamber WRONG failed to degrade")
        if state_index > 0 and not (shuffle_metrics["useful_rate"] < full_metrics["useful_rate"]):
            raise SystemExit("chamber SHUFFLE failed to degrade")

        policy = meta_compiler_policy_from_body(full_body)
        body_hash = full_body.morphology.fingerprint()
        body_hashes.append(body_hash)
        policy_fingerprints.append(policy.fingerprint())
        frontier = 1.0 + len(full_learned_slots)
        human_intervention = 1.0 if state_index == 0 else 0.0
        metrics = GenerationMetrics(
            generation=generation,
            body_hash=body_hash,
            parent_body_hash=(freeze["initial_body_hash"] if state_index == 0 else body_hashes[state_index - 1]),
            external_capability_frontier=frontier,
            transfer_score=float(full_metrics["useful_rate"]),
            retained_competence=float(full_metrics["retained_learned_competence"]),
            calibration_score=float(full_metrics["calibration"]),
            research_invention_score=frontier,
            meta_improvement_ability=float(full_metrics["useful_rate"]),
            compute_cost=float(precommit["resource_contract"]["generation_compute_cost"]),
            evidence_cost=float(precommit["resource_contract"]["generation_evidence_cost"]),
            human_structural_intervention=human_intervention,
            benchmark_disjoint=True,
            authority_verified=False,
            strategy_hash=full_body.mutation_strategy.lineage_hash,
        )
        if not ledger.append(metrics):
            raise SystemExit("chamber meta-acceleration lineage append failed")

        full_rows.append({"generation": generation, "body_hash": body_hash, "policy_fingerprint": policy.fingerprint(), **full_metrics})
        remove_rows.append({"generation": generation, **remove_metrics})
        wrong_rows.append({"generation": generation, **wrong_metrics})
        shuffle_rows.append({"generation": generation, **shuffle_metrics})

        if state_index == 3:
            break

        train_regime = training_order[state_index]
        train_slot = regime_to_slot[train_regime]
        training_tasks = _tasks_for_regime(
            rng,
            signal_slot=train_slot,
            regime=train_regime,
            prefix=f"training-transition-{generation}-to-{generation + 1}-{train_regime.lower()}",
            count=training_count,
        )
        outcomes = proposal_outcomes(full_body, training_tasks)
        receipt = learn_meta_compiler_rule(
            full_body,
            signal_slot=train_slot,
            proposal_outcomes=outcomes,
            evidence_ref=f"hidden-training://{generation}->{generation + 1}/{train_regime}",
        )
        if receipt.winning_family != REGIME_TO_FAMILY[train_regime]:
            raise SystemExit("chamber learned causal winner does not match executed behavior")
        full_learned_slots.append(train_slot)
        learning_receipts.append(receipt.__dict__)

        # WRONG: causal winner is credited to a different opaque signal slot.
        wrong_slot = wrong_slot_order[(state_index + 1) % len(wrong_slot_order)]
        learn_meta_compiler_rule(
            wrong_body,
            signal_slot=wrong_slot,
            proposal_outcomes=outcomes,
            evidence_ref=f"wrong-slot-control://{generation}->{generation + 1}",
        )

        # SHUFFLE: current slot receives a permuted proposal outcome vector.
        learn_meta_compiler_rule(
            shuffle_body,
            signal_slot=train_slot,
            proposal_outcomes=_rotated_outcomes(outcomes),
            evidence_ref=f"shuffled-outcome-control://{generation}->{generation + 1}",
        )

        full_body = _restore_roundtrip(full_body)
        wrong_body = _restore_roundtrip(wrong_body)
        shuffle_body = _restore_roundtrip(shuffle_body)
        remove_body = _restore_roundtrip(remove_body)

        if meta_compiler_policy_from_body(full_body).route(train_slot)[0] != REGIME_TO_FAMILY[train_regime]:
            raise SystemExit("chamber learned route lost after checkpoint restart")

    assessment = ledger.assess(min_transitions=3, retained_floor=1.0, calibration_floor=1.0)
    rows = ledger.ordered()
    internal_candidate = bool(
        assessment.strict_frontier_growth
        and assessment.strict_meta_productivity_growth
        and assessment.nonincreasing_human_intervention
        and assessment.retained_competence_viable
        and assessment.calibration_viable
        and assessment.meta_ability_improved
        and assessment.lineage_continuous
        and assessment.all_benchmark_disjoint
        and not assessment.all_authority_verified
    )
    expected_frontier = tuple(float(value) for value in precommit["required_growth"]["strict_distinct_regime_frontier"])
    expected_rates = tuple(float(value) for value in precommit["required_growth"]["strict_validation_useful_rate"])
    actual_frontier = tuple(row.external_capability_frontier for row in rows)
    actual_rates = tuple(row.transfer_score for row in rows)
    if actual_frontier != expected_frontier or actual_rates != expected_rates:
        raise SystemExit("chamber precommitted trajectory mismatch")
    if not internal_candidate:
        raise SystemExit("bounded internal recursive meta-compiler acceleration candidate not established")

    receipt = {
        "schema": "arte.recursive_meta_compiler_acceleration_chamber/v1",
        "trajectory_generations": [9, 10, 11, 12],
        "candidate_source_head_sha": args.candidate_head,
        "source_head_constant_across_trajectory": True,
        "source_edits_between_internal_generations": [0, 0, 0],
        "hidden_slot_mapping_sha256": __import__("hashlib").sha256(
            json.dumps(slot_to_regime, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "training_regime_order": training_order,
        "full_validation_useful_rate_trajectory": [row["useful_rate"] for row in full_rows],
        "remove_validation_useful_rate_trajectory": [row["useful_rate"] for row in remove_rows],
        "wrong_validation_useful_rate_trajectory": [row["useful_rate"] for row in wrong_rows],
        "shuffle_validation_useful_rate_trajectory": [row["useful_rate"] for row in shuffle_rows],
        "distinct_regime_frontier_trajectory": list(actual_frontier),
        "meta_productivity_trajectory": list(assessment.meta_productivity_trajectory),
        "human_structural_intervention_trajectory": list(assessment.human_intervention_trajectory),
        "body_hash_trajectory": body_hashes,
        "policy_fingerprint_trajectory": policy_fingerprints,
        "experience_count_trajectory": [0, 1, 2, 3],
        "strategy_support_total_trajectory": [
            sum(value for _, value in restore_body(checkpoint_dict(full_body)).mutation_strategy.operation_support)
            if index == 3 else index
            for index in range(4)
        ],
        "learning_receipts": learning_receipts,
        "strict_frontier_growth": assessment.strict_frontier_growth,
        "strict_meta_productivity_growth": assessment.strict_meta_productivity_growth,
        "nonincreasing_external_structural_intervention": assessment.nonincreasing_human_intervention,
        "retained_competence_viable": assessment.retained_competence_viable,
        "calibration_viable": assessment.calibration_viable,
        "meta_ability_improved": assessment.meta_ability_improved,
        "lineage_continuous": assessment.lineage_continuous,
        "fresh_validation_disjoint": assessment.all_benchmark_disjoint,
        "external_independent_authority_verified": assessment.all_authority_verified,
        "standard_meta_acceleration_assessment_status": assessment.status,
        "bounded_internal_recursive_meta_acceleration_candidate": internal_candidate,
        "eligible_claim": "PASS_BOUNDED_INTERNAL_RECURSIVE_META_COMPILER_ACCELERATION_CANDIDATE",
        "global_recursive_acceleration": False,
        "source_code_autonomous_self_modification_established": False,
        "external_structural_intervention_after_substrate_freeze": 0,
        "official_benchmark_used": False,
        "AGI": False,
        "ASI": False,
        "next_problem_detectors": ["META_COMPILER_POLICY_LANGUAGE_SELF_EXPANSION_UNPROVEN"],
        "full_rows": full_rows,
        "remove_rows": remove_rows,
        "wrong_rows": wrong_rows,
        "shuffle_rows": shuffle_rows,
    }
    material = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    receipt["outcome_receipt_sha256"] = __import__("hashlib").sha256(material).hexdigest()
    Path(args.output).write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key not in {"full_rows", "remove_rows", "wrong_rows", "shuffle_rows", "learning_receipts"}}, sort_keys=True))


if __name__ == "__main__":
    main()
