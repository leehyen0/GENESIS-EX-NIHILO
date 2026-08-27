from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess

from arte_cognition.body_policy_generation import generate_contextual_body_candidate
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.executable_morphology import (
    ExperienceArchive,
    MorphologyCompiler,
    MorphologyGenome,
    MorphologyMutator,
    MutationLevel,
    OrganKind,
    OrganSpec,
    PressureVector,
)
from arte_cognition.meta_acceleration import MutationProgramDevelopmentState, MutationStrategyState
from arte_cognition.morphology_genesis import MorphologyResidual
from arte_cognition.native_recursive_research import NativeMetaMorphologyGenesisEngine
from arte_cognition.self_evolving_body_checkpoint import SelfEvolvingResearchBody, checkpoint_dict, restore_body


TRACKED = (
    "arte_cognition/meta_acceleration.py",
    "arte_cognition/executable_morphology.py",
    "arte_cognition/morphology_genesis.py",
    "arte_cognition/native_recursive_research.py",
    "arte_cognition/native_meta_policy_runtime.py",
    "arte_cognition/body_policy_generation.py",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bundle(sources: dict[str, str]) -> str:
    return _sha(json.dumps(sources, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _git_show(ref: str, path: str) -> str:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _parent() -> MorphologyGenome:
    organs = (
        OrganSpec("generator", OrganKind.GENERATOR, produces=("candidate",), implementation_ref="bootstrap://generator"),
        OrganSpec("mutator", OrganKind.MUTATOR, consumes=("candidate",), produces=("mutation",), implementation_ref="bootstrap://mutator"),
        OrganSpec("governor", OrganKind.GOVERNOR),
        OrganSpec("archive", OrganKind.ARCHIVE),
    )
    return MorphologyGenome(organs=organs, edges=(), event_order=("generator", "mutator", "governor", "archive"))


def _policy_child(parent: MorphologyGenome, origin: str) -> MorphologyGenome:
    residual = MorphologyResidual(origin, PressureVector(human_dependency=1.0, theory_blindspot=0.25))
    rows = NativeMetaMorphologyGenesisEngine(candidate_budget=16).generate(parent, (residual,))
    mutation = next(row.mutation for row in rows if row.operation_family == "CHANGE_MUTATOR_POLICY")
    return MorphologyMutator().apply(parent, mutation)


def _target_family(residual: MorphologyResidual) -> str:
    p = residual.pressure.normalized()
    if p.human_dependency > p.novelty_pressure:
        return "CHANGE_MUTATOR_POLICY"
    if p.novelty_pressure > p.human_dependency:
        return "CHANGE_GENERATOR_POLICY"
    raise ValueError("INVALID_HIDDEN_TASK_TIE")


def _fresh_task(task_id: str, kind: str) -> MorphologyResidual:
    if kind == "HUMAN":
        pressure = PressureVector(human_dependency=1.0, novelty_pressure=0.1, theory_blindspot=0.1)
    elif kind == "NOVELTY":
        pressure = PressureVector(human_dependency=0.1, novelty_pressure=1.0, theory_blindspot=0.1)
    else:
        raise ValueError(kind)
    return MorphologyResidual(task_id, pressure)


def _raw_candidates(genome: MorphologyGenome, residual: MorphologyResidual, budget: int = 64):
    rows = NativeMetaMorphologyGenesisEngine(candidate_budget=budget).generate(genome, (residual,))
    unique = {}
    for row in rows:
        unique.setdefault(row.candidate_id, row)
    return tuple(sorted(unique.values(), key=lambda row: (row.operation_family, row.candidate_id)))


def _candidate_is_useful(
    genome: MorphologyGenome,
    residual: MorphologyResidual,
    candidate_id: str,
    target_family: str,
    *,
    raw_budget: int = 64,
) -> bool:
    rows = {row.candidate_id: row for row in _raw_candidates(genome, residual, raw_budget)}
    candidate = rows.get(candidate_id)
    if candidate is None:
        return False
    if candidate.operation_family != target_family:
        return False
    if candidate.mutation.level != MutationLevel.GENERATOR_MUTATOR:
        return False
    if candidate.generation_uses_outcomes:
        return False
    descendant = MorphologyMutator().apply(genome, candidate.mutation)
    runtime = MorphologyCompiler.compile_runtime(descendant)
    return any(binding.preferred_operation_family == target_family for binding in runtime.native_meta_policies)


def _more_compute_parent_success(parent: MorphologyGenome, residual: MorphologyResidual, target: str) -> bool:
    rows = _raw_candidates(parent, residual, 64)[:2]
    return any(_candidate_is_useful(parent, residual, row.candidate_id, target) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-head", required=True)
    parser.add_argument("--candidate-freeze-head", required=True)
    parser.add_argument("--precommit", required=True)
    parser.add_argument("--candidate-freeze", required=True)
    parser.add_argument("--cycle4-receipt", required=True)
    parser.add_argument("--hidden-seed-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    precommit = json.loads(Path(args.precommit).read_text(encoding="utf-8"))
    freeze = json.loads(Path(args.candidate_freeze).read_text(encoding="utf-8"))
    cycle4 = json.loads(Path(args.cycle4_receipt).read_text(encoding="utf-8"))
    if precommit["problem_detector"] != "INHERITED_L3_POLICY_CONTEXT_BLIND":
        raise SystemExit("cycle5 precommit problem changed")
    if freeze["inherited_policy_operation_family"] != "CHANGE_MUTATOR_POLICY":
        raise SystemExit("cycle5 child policy family changed")
    if freeze["fresh_task_seed_known_at_freeze"] is not False:
        raise SystemExit("cycle5 hidden seed was not frozen as unknown")
    if cycle4["useful_candidate_quality_improvement_established"] is not False:
        raise SystemExit("cycle4 already claimed useful candidate gain")

    parent_sources = {path: _git_show(args.parent_head, path) for path in TRACKED}
    parent_hash = _bundle(parent_sources)
    if parent_hash != precommit["parent_core_body_sha256"]:
        raise SystemExit(f"cycle5 parent mismatch: expected={precommit['parent_core_body_sha256']} actual={parent_hash}")
    if parent_hash != cycle4["candidate_core_body_hash"]:
        raise SystemExit("cycle4 receipt and cycle5 precommit disagree on parent BODY")

    frozen_candidate_sources = {path: _git_show(args.candidate_freeze_head, path) for path in TRACKED}
    current_sources = {path: _read(path) for path in TRACKED}
    frozen_candidate_hash = _bundle(frozen_candidate_sources)
    current_candidate_hash = _bundle(current_sources)
    if current_candidate_hash != frozen_candidate_hash:
        raise SystemExit("cycle5 candidate core changed after hidden-task candidate freeze")

    seed = int(Path(args.hidden_seed_file).read_text(encoding="utf-8").strip())
    rng = random.Random(seed)
    kinds = ["HUMAN"] * 12 + ["NOVELTY"] * 12
    rng.shuffle(kinds)
    tasks = []
    for index, kind in enumerate(kinds):
        task_token = _sha(f"{seed}|cycle5|{index}|{kind}".encode("utf-8"))[:20]
        tasks.append(_fresh_task(f"hidden-{task_token}", kind))

    parent = _parent()
    origin = freeze["inherited_policy_origin_residual_id"]
    child = _policy_child(parent, origin)

    # Strong heredity control: the child that enters hidden tasks is restored from the
    # canonical self-evolving BODY checkpoint, not the transient pre-checkpoint object.
    body = SelfEvolvingResearchBody(
        runtime=PersistentCognitiveRuntime(),
        morphology=child,
        mutation_strategy=MutationStrategyState(),
        mutation_program_state=MutationProgramDevelopmentState(),
        experience_archive=ExperienceArchive(),
    )
    restored = restore_body(checkpoint_dict(body))
    restored_child = restored.morphology
    if restored_child.fingerprint() != child.fingerprint():
        raise SystemExit("cycle5 inherited policy morphology did not survive BODY checkpoint")

    full_success = 0
    remove_success = 0
    wrong_success = 0
    shuffle_success = 0
    more_compute_success = 0
    full_trace = []

    human_tasks = [task for task in tasks if _target_family(task) == "CHANGE_MUTATOR_POLICY"]
    novelty_tasks = [task for task in tasks if _target_family(task) == "CHANGE_GENERATOR_POLICY"]
    if len(human_tasks) != 12 or len(novelty_tasks) != 12:
        raise SystemExit("hidden task balance contract failed")
    opposite_pressure = {}
    for human, novelty in zip(human_tasks, novelty_tasks):
        opposite_pressure[human.residual_id] = novelty.pressure
        opposite_pressure[novelty.residual_id] = human.pressure

    for task in tasks:
        target = _target_family(task)
        full = generate_contextual_body_candidate(
            restored_child,
            task,
            raw_candidate_budget=64,
            expected_policy_origin_residual_id=origin,
        )
        remove = generate_contextual_body_candidate(parent, task, raw_candidate_budget=64)

        full_ok = _candidate_is_useful(restored_child, task, full.selected_candidate_id, target)
        remove_ok = _candidate_is_useful(parent, task, remove.selected_candidate_id, target)
        full_success += int(full_ok)
        remove_success += int(remove_ok)

        # WRONG mapping: select one candidate from the opposite frozen family.
        wrong_target = "CHANGE_GENERATOR_POLICY" if target == "CHANGE_MUTATOR_POLICY" else "CHANGE_MUTATOR_POLICY"
        wrong_rows = [row for row in _raw_candidates(restored_child, task, 64) if row.operation_family == wrong_target]
        wrong_ok = bool(wrong_rows) and _candidate_is_useful(
            restored_child,
            task,
            wrong_rows[0].candidate_id,
            target,
        )
        wrong_success += int(wrong_ok)

        # SHUFFLE: bind another hidden task's opposite pressure to this task ID while
        # retaining this task's verifier target.
        shuffled_task = MorphologyResidual(task.residual_id, opposite_pressure[task.residual_id])
        shuffled = generate_contextual_body_candidate(
            restored_child,
            shuffled_task,
            raw_candidate_budget=64,
            expected_policy_origin_residual_id=origin,
        )
        shuffle_ok = _candidate_is_useful(restored_child, task, shuffled.selected_candidate_id, target)
        shuffle_success += int(shuffle_ok)

        more_compute_success += int(_more_compute_parent_success(parent, task, target))
        full_trace.append(
            {
                "task_id": task.residual_id,
                "target": target,
                "full_family": full.selected_operation_family,
                "remove_family": remove.selected_operation_family,
                "full_useful": full_ok,
                "remove_useful": remove_ok,
            }
        )

    total = len(tasks)
    full_rate = full_success / total
    remove_rate = remove_success / total
    wrong_rate = wrong_success / total
    shuffle_rate = shuffle_success / total
    more_compute_rate = more_compute_success / total

    if not (full_rate > remove_rate):
        raise SystemExit(f"FULL did not beat matched-budget REMOVE: full={full_rate} remove={remove_rate}")
    if not (full_rate > wrong_rate and full_rate > shuffle_rate):
        raise SystemExit(
            f"FULL did not beat WRONG/SHUFFLE: full={full_rate} wrong={wrong_rate} shuffle={shuffle_rate}"
        )
    if full_success != 24:
        raise SystemExit(f"contextual child did not solve every frozen-rule hidden task: {full_success}/24")
    if remove_success != 12:
        raise SystemExit(f"unexpected parent matched-budget baseline: {remove_success}/24")
    if more_compute_success != 24:
        raise SystemExit(f"MORE_COMPUTE parent did not close the bounded search gap: {more_compute_success}/24")

    # Re-run the exact hidden tasks from a second checkpoint reconstruction.
    restored_again = restore_body(checkpoint_dict(body)).morphology
    replay = [
        generate_contextual_body_candidate(
            restored_again,
            task,
            raw_candidate_budget=64,
            expected_policy_origin_residual_id=origin,
        ).selected_candidate_id
        for task in tasks
    ]
    original = [row["task_id"] + "::" + generate_contextual_body_candidate(
        restored_child,
        task,
        raw_candidate_budget=64,
        expected_policy_origin_residual_id=origin,
    ).selected_candidate_id for row, task in zip(full_trace, tasks)]
    replay_tagged = [task.residual_id + "::" + candidate_id for task, candidate_id in zip(tasks, replay)]
    restart_equal = original == replay_tagged
    if not restart_equal:
        raise SystemExit("cycle5 hidden-task candidate selection changed after cold BODY restart")

    material = {
        "parent_core_body_hash": parent_hash,
        "candidate_core_body_hash": current_candidate_hash,
        "hidden_task_count": total,
        "full_success": full_success,
        "remove_success": remove_success,
        "wrong_success": wrong_success,
        "shuffle_success": shuffle_success,
        "more_compute_success": more_compute_success,
        "restart_equal": restart_equal,
        "selected_budget_full": 1,
        "selected_budget_remove": 1,
        "selected_budget_more_compute": 2,
        "raw_budget_full": 64,
        "raw_budget_remove": 64,
        "official_benchmark_used": False,
    }
    outcome_hash = _sha(json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    receipt = {
        "schema": "arte.native_recursive_research_cycle5/v1",
        "generation": 5,
        "parent_evidence_head_sha": args.parent_head,
        "candidate_freeze_head_sha": args.candidate_freeze_head,
        "parent_core_body_hash": parent_hash,
        "candidate_core_body_hash": current_candidate_hash,
        "problem_detector": "INHERITED_L3_POLICY_CONTEXT_BLIND",
        "hidden_task_count": total,
        "hidden_task_balance": {"human_dependency_dominant": 12, "novelty_dominant": 12},
        "checkpoint_restore_before_hidden_tasks": True,
        "full_useful_candidate_count": full_success,
        "remove_useful_candidate_count": remove_success,
        "wrong_useful_candidate_count": wrong_success,
        "shuffle_useful_candidate_count": shuffle_success,
        "more_compute_parent_useful_candidate_count": more_compute_success,
        "full_useful_rate": full_rate,
        "remove_useful_rate": remove_rate,
        "wrong_useful_rate": wrong_rate,
        "shuffle_useful_rate": shuffle_rate,
        "more_compute_parent_useful_rate": more_compute_rate,
        "matched_selected_candidate_budget": 1,
        "matched_raw_candidate_generation_budget": 64,
        "restart_hidden_selection_equal": restart_equal,
        "current_outcomes_consumed_by_generation": False,
        "bounded_positive_descendant_generative_effect": full_rate > remove_rate,
        "effect_type": "CONTEXTUAL_SEARCH_EFFICIENCY_NOT_NEW_REPRESENTATIONAL_REACHABILITY",
        "more_compute_closes_gap": more_compute_rate == full_rate,
        "eligible_claim": "BOUNDED_DESCENDANT_CONTEXTUAL_SEARCH_EFFICIENCY_GAIN",
        "broad_capability_improvement_established": False,
        "new_representational_reachability_established": False,
        "recursive_acceleration_established": False,
        "official_benchmark_used": False,
        "external_claim_authority": False,
        "AGI": False,
        "ASI": False,
        "outcome_receipt_sha256": outcome_hash,
    }
    Path(args.output).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
