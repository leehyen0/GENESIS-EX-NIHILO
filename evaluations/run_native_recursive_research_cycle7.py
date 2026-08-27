from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

from arte_cognition.executable_morphology import MorphologyMutator, PressureVector
from arte_cognition.morphology_genesis import MorphologyResidual
from arte_cognition.native_recursive_research import NativeMetaMorphologyGenesisEngine
from arte_cognition.native_representation_genesis import (
    PRIMITIVE_FAMILIES,
    NativeRepresentationGenesisEngine,
    RepresentationSupportExample,
    apply_representation_primitive,
)
from arte_cognition.native_representation_generator_language import (
    CompositionalRepresentationGenesisEngine,
    ExpressionSpec,
    expression_language,
    expression_representation_programs,
    generator_policies,
    infer_expression_spec,
)
from arte_cognition.self_evolving_body_checkpoint import checkpoint_dict, restore_body


@dataclass(frozen=True)
class HiddenExpressionTask:
    task_id: str
    spec: ExpressionSpec
    artifact_type: str
    support: Tuple[RepresentationSupportExample, ...]
    query: Tuple[int, int]
    hidden_output: int


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(("git",) + args, text=True).strip()


def _read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pair_grid(limit: int = 8) -> Tuple[Tuple[int, int], ...]:
    return tuple((x, y) for x in range(limit) for y in range(limit))


def _fixed_signatures(pairs: Sequence[Tuple[int, int]]) -> set[Tuple[int, ...]]:
    out = set()
    for family in PRIMITIVE_FAMILIES:
        try:
            out.add(tuple(apply_representation_primitive(family, pair) for pair in pairs))
        except ValueError:
            continue
    return out


def _eligible_target_specs() -> Tuple[ExpressionSpec, ...]:
    probes = _pair_grid(8)
    fixed = _fixed_signatures(probes)
    by_signature: dict[Tuple[int, ...], list[ExpressionSpec]] = {}
    for spec in expression_language():
        try:
            signature = tuple(spec.execute(pair) for pair in probes)
        except ValueError:
            continue
        by_signature.setdefault(signature, []).append(spec)
    unique = [rows[0] for signature, rows in by_signature.items() if len(rows) == 1 and signature not in fixed]
    if len(unique) < 8:
        raise RuntimeError("expression language has insufficient non-parent unique targets")
    return tuple(sorted(unique))


def _fixed_parent_matches(support: Sequence[RepresentationSupportExample]) -> bool:
    for family in PRIMITIVE_FAMILIES:
        try:
            if all(
                apply_representation_primitive(family, row.input_pair) == int(row.output_value)
                for row in support
            ):
                return True
        except ValueError:
            continue
    return False


def _make_support(rng: random.Random, spec: ExpressionSpec, count: int) -> Tuple[RepresentationSupportExample, ...]:
    for _ in range(4000):
        rows = []
        while len(rows) < count:
            pair = (rng.randint(0, 31), rng.randint(0, 31))
            try:
                value = spec.execute(pair)
            except ValueError:
                continue
            rows.append(RepresentationSupportExample(pair, value))
        support = tuple(rows)
        try:
            identified = infer_expression_spec(support)
        except ValueError:
            continue
        if identified == spec and not _fixed_parent_matches(support):
            return support
    raise RuntimeError("unable to generate uniquely identified non-parent expression support")


def _make_query(rng: random.Random, spec: ExpressionSpec) -> Tuple[int, int]:
    for _ in range(4000):
        pair = (rng.randint(0, 31), rng.randint(0, 31))
        try:
            target = spec.execute(pair)
        except ValueError:
            continue
        alternatives = 0
        for other in _eligible_target_specs():
            if other == spec:
                continue
            try:
                alternatives += int(other.execute(pair) != target)
            except ValueError:
                continue
        if alternatives >= 5:
            return pair
    raise RuntimeError("unable to generate discriminating expression query")


def _hidden_tasks(seed: int, count: int, support_count: int) -> Tuple[HiddenExpressionTask, ...]:
    rng = random.Random(seed)
    specs = list(_eligible_target_specs())
    rng.shuffle(specs)
    selected = [specs[index % len(specs)] for index in range(count)]
    rng.shuffle(selected)
    tasks = []
    for index, spec in enumerate(selected):
        support = _make_support(rng, spec, support_count)
        query = _make_query(rng, spec)
        tasks.append(
            HiddenExpressionTask(
                task_id=f"cycle7-hidden-{index:03d}",
                spec=spec,
                artifact_type=f"expression_artifact_{index:03d}",
                support=support,
                query=query,
                hidden_output=spec.execute(query),
            )
        )
    return tuple(tasks)


def _residual(task: HiddenExpressionTask) -> MorphologyResidual:
    return MorphologyResidual(
        residual_id=task.task_id,
        pressure=PressureVector(transfer_failure=1.0, theory_blindspot=1.0),
        same_frozen_phenotype_different_outcome=True,
        more_compute_still_aliased=True,
        missing_artifact_types=(task.artifact_type,),
        source_refs=("hidden-expression-support://pre-outcome",),
    )


def _wrong_spec(task: HiddenExpressionTask) -> ExpressionSpec:
    for spec in _eligible_target_specs():
        if spec == task.spec:
            continue
        try:
            if spec.execute(task.query) != task.hidden_output:
                return spec
        except ValueError:
            continue
    raise RuntimeError("no discriminating wrong expression")


def _donor(tasks: Sequence[HiddenExpressionTask], task: HiddenExpressionTask) -> HiddenExpressionTask:
    for donor in tasks:
        if donor.task_id == task.task_id or donor.spec == task.spec:
            continue
        try:
            if donor.spec.execute(task.query) != task.hidden_output:
                return donor
        except ValueError:
            continue
    raise RuntimeError("no discriminating shuffle donor")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-head", required=True)
    parser.add_argument("--candidate-head", required=True)
    parser.add_argument("--precommit", required=True)
    parser.add_argument("--cycle6-receipt", required=True)
    parser.add_argument("--generator-freeze", required=True)
    parser.add_argument("--hidden-seed-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    precommit = _read(args.precommit)
    cycle6 = _read(args.cycle6_receipt)
    freeze = _read(args.generator_freeze)
    if precommit.get("parent_evidence_head_sha") != args.parent_head:
        raise SystemExit("cycle7 parent head mismatch")
    if precommit.get("parent_outcome_receipt_sha256") != cycle6.get("outcome_receipt_sha256"):
        raise SystemExit("cycle7 parent receipt mismatch")
    if precommit.get("parent_core_body_sha256") != cycle6.get("candidate_core_body_hash"):
        raise SystemExit("cycle7 parent core BODY mismatch")
    if freeze.get("candidate_head_sha") != args.candidate_head:
        raise SystemExit("cycle7 generator freeze candidate head mismatch")
    if freeze.get("hidden_task_information_consumed") or freeze.get("current_outcomes_consumed"):
        raise SystemExit("cycle7 generator freeze contaminated")
    _git("cat-file", "-e", f"{args.parent_head}^{{commit}}")

    restored_generator_body = restore_body(dict(freeze["checkpoint"]))
    origin = str(freeze["generator_policy_origin_residual_id"])
    policies = generator_policies(restored_generator_body.morphology, expected_origin_residual_id=origin)
    if len(policies) != 1 or policies[0].fingerprint() != freeze.get("generator_policy_fingerprint"):
        raise SystemExit("cycle7 restored generator policy mismatch")

    seed = int(Path(args.hidden_seed_file).read_text(encoding="utf-8").strip())
    task_count = int(precommit["resource_contract"]["hidden_task_count"])
    support_count = int(precommit["resource_contract"]["support_examples_per_task"])
    tasks = _hidden_tasks(seed, task_count, support_count)
    engine = CompositionalRepresentationGenesisEngine(candidate_budget=1)

    full = 0
    wrong = 0
    shuffle = 0
    parent_fixed_success = 0
    parent_native_expr_count = 0
    parent_l1_expr_count = 0
    restart_equal = True
    task_representation_checkpoint_equal = True

    for task in tasks:
        residual = _residual(task)

        # REMOVE/MORE_COMPUTE: cycle-6 fixed-family representation language cannot
        # identify or emit a compositional expression representation.
        try:
            NativeRepresentationGenesisEngine(candidate_budget=4096).generate(
                restored_generator_body.morphology,
                residual,
                task.support,
            )
            parent_fixed_success += 1
        except ValueError:
            pass
        rows = NativeMetaMorphologyGenesisEngine(candidate_budget=4096).generate(
            restored_generator_body.morphology,
            (residual,),
        )
        for row in rows:
            if row.mutation.level.value == 1:
                parent_l1_expr_count += 1
            child_probe = MorphologyMutator().apply(restored_generator_body.morphology, row.mutation)
            if any(str(organ.implementation_ref).startswith("native-repr-expr://") for organ in child_probe.organs):
                parent_native_expr_count += 1

        candidate = engine.generate(
            restored_generator_body.morphology,
            residual,
            task.support,
            expected_generator_origin_residual_id=origin,
        )[0]
        representation_child = MorphologyMutator().apply(restored_generator_body.morphology, candidate.mutation)
        body_checkpoint = dict(freeze["checkpoint"])
        # Reuse the frozen BODY state but replace morphology with the task-specific
        # descendant through the normal checkpoint path rather than editing bytes.
        task_body = restore_body(body_checkpoint)
        task_body.morphology = representation_child
        task_checkpoint = checkpoint_dict(task_body)
        restored_task_body = restore_body(task_checkpoint)
        task_representation_checkpoint_equal = task_representation_checkpoint_equal and (
            restored_task_body.morphology.fingerprint() == representation_child.fingerprint()
        )
        program = expression_representation_programs(
            restored_task_body.morphology,
            expected_artifact_type=task.artifact_type,
            expected_residual_id=task.task_id,
        )[0]
        prediction = program.execute(task.query)
        full += int(prediction == task.hidden_output)

        restarted = restore_body(checkpoint_dict(restored_task_body))
        restart_program = expression_representation_programs(
            restarted.morphology,
            expected_artifact_type=task.artifact_type,
            expected_residual_id=task.task_id,
        )[0]
        restart_equal = restart_equal and (
            restart_program.fingerprint() == program.fingerprint()
            and restart_program.execute(task.query) == prediction
        )

        wrong_candidate = engine.generate(
            restored_generator_body.morphology,
            residual,
            task.support,
            expected_generator_origin_residual_id=origin,
            force_spec=_wrong_spec(task),
        )[0]
        wrong_child = MorphologyMutator().apply(restored_generator_body.morphology, wrong_candidate.mutation)
        wrong_program = expression_representation_programs(
            wrong_child,
            expected_artifact_type=task.artifact_type,
            expected_residual_id=task.task_id,
        )[0]
        wrong += int(wrong_program.execute(task.query) == task.hidden_output)

        donor = _donor(tasks, task)
        shuffle_candidate = engine.generate(
            restored_generator_body.morphology,
            residual,
            donor.support,
            expected_generator_origin_residual_id=origin,
        )[0]
        shuffle_child = MorphologyMutator().apply(restored_generator_body.morphology, shuffle_candidate.mutation)
        shuffle_program = expression_representation_programs(
            shuffle_child,
            expected_artifact_type=task.artifact_type,
            expected_residual_id=task.task_id,
        )[0]
        shuffle += int(shuffle_program.execute(task.query) == task.hidden_output)

    parent_more_compute_unreachable = bool(
        parent_fixed_success == 0 and parent_native_expr_count == 0 and parent_l1_expr_count == 0
    )
    pass_contract = bool(
        full == task_count
        and parent_more_compute_unreachable
        and wrong < full
        and shuffle < full
        and restart_equal
        and task_representation_checkpoint_equal
    )
    if not pass_contract:
        raise SystemExit(
            f"cycle7 failed full={full} wrong={wrong} shuffle={shuffle} "
            f"parent_fixed={parent_fixed_success} parent_expr={parent_native_expr_count} "
            f"parent_l1={parent_l1_expr_count} restart={restart_equal}"
        )

    target_tokens = sorted({task.spec.token() for task in tasks})
    receipt_material = {
        "generation": 7,
        "candidate_head": args.candidate_head,
        "generator_policy": policies[0].fingerprint(),
        "targets": target_tokens,
        "full": full,
        "wrong": wrong,
        "shuffle": shuffle,
        "parent_fixed": parent_fixed_success,
        "parent_expr": parent_native_expr_count,
    }
    receipt_hash = _sha(receipt_material)
    receipt = {
        "schema": "arte.native_recursive_research_cycle7/v1",
        "generation": 7,
        "problem_detector": "REPRESENTATION_GENESIS_SELF_MODIFICATION_UNPROVEN",
        "parent_evidence_head_sha": args.parent_head,
        "candidate_implementation_head_sha": args.candidate_head,
        "parent_core_body_hash": precommit["parent_core_body_sha256"],
        "generator_policy_mutation_level": int(freeze["mutation_level"]),
        "generator_policy_fingerprint": policies[0].fingerprint(),
        "generator_policy_checkpointed_before_hidden_tasks": True,
        "hidden_task_count": task_count,
        "unique_hidden_expression_spec_count": len(target_tokens),
        "full_hidden_useful_count": full,
        "full_hidden_useful_rate": full / task_count,
        "wrong_hidden_useful_count": wrong,
        "wrong_hidden_useful_rate": wrong / task_count,
        "shuffle_hidden_useful_count": shuffle,
        "shuffle_hidden_useful_rate": shuffle / task_count,
        "parent_fixed_family_success_count": parent_fixed_success,
        "parent_more_compute_candidate_budget": int(precommit["resource_contract"]["parent_more_compute_candidate_budget"]),
        "parent_more_compute_l1_expression_candidate_count": parent_l1_expr_count,
        "parent_more_compute_native_expression_count": parent_native_expr_count,
        "parent_more_compute_expression_unreachable": parent_more_compute_unreachable,
        "task_representation_checkpoint_equal": task_representation_checkpoint_equal,
        "restart_reconstruction_equal": restart_equal,
        "current_outcomes_consumed_by_generation": False,
        "generator_self_modification_bounded": True,
        "source_code_autonomous_self_modification_established": False,
        "new_generator_language_reachability_established": True,
        "bounded_positive_descendant_generative_effect": True,
        "effect_type": "INHERITED_GENERATOR_POLICY_ENABLES_COMPOSITIONAL_REPRESENTATION_SYNTHESIS",
        "eligible_claim": "BOUNDED_INHERITED_REPRESENTATION_GENERATOR_LANGUAGE_EXPANSION",
        "broad_capability_improvement_established": False,
        "recursive_acceleration_established": False,
        "official_benchmark_used": False,
        "external_claim_authority": False,
        "next_problem_detectors": ["AUTONOMOUS_GENERATOR_MUTATION_CREDIT_UNPROVEN"],
        "outcome_receipt_sha256": receipt_hash,
        "AGI": False,
        "ASI": False,
    }
    Path(args.output).write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
