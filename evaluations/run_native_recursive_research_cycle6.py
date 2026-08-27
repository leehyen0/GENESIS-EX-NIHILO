from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.executable_morphology import (
    ExperienceArchive,
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
from arte_cognition.native_representation_genesis import (
    NativeRepresentationGenesisEngine,
    PRIMITIVE_FAMILIES,
    RepresentationSupportExample,
    apply_representation_primitive,
    executable_representation_programs,
    infer_representation_family,
)
from arte_cognition.self_evolving_body_checkpoint import SelfEvolvingResearchBody, checkpoint_dict, restore_body


@dataclass(frozen=True)
class HiddenTask:
    task_id: str
    family: str
    artifact_type: str
    support: Tuple[RepresentationSupportExample, ...]
    query: Tuple[int, int]
    hidden_output: int


def _sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.check_output(("git",) + args, text=True).strip()


def _core_hash(paths: Sequence[str]) -> str:
    rows = []
    for path in paths:
        data = Path(path).read_bytes()
        rows.append((path, hashlib.sha256(data).hexdigest()))
    return _sha256(rows)


def parent_genome() -> MorphologyGenome:
    return MorphologyGenome(
        organs=(
            OrganSpec("source", OrganKind.SOURCE, produces=("raw_observation",), implementation_ref="bootstrap://source"),
            OrganSpec("generator", OrganKind.GENERATOR, implementation_ref="bootstrap://generator"),
            OrganSpec("mutator", OrganKind.MUTATOR, implementation_ref="bootstrap://mutator"),
            OrganSpec("governor", OrganKind.GOVERNOR),
            OrganSpec("archive", OrganKind.ARCHIVE),
        ),
        edges=(),
        event_order=(),
    )


def task_residual(task: HiddenTask) -> MorphologyResidual:
    return MorphologyResidual(
        residual_id=task.task_id,
        pressure=PressureVector(transfer_failure=1.0, theory_blindspot=0.4),
        same_frozen_phenotype_different_outcome=True,
        more_compute_still_aliased=True,
        missing_artifact_types=(task.artifact_type,),
        source_refs=("hidden-native-support://typed-pairs",),
    )


def _random_pair(rng: random.Random) -> Tuple[int, int]:
    return (rng.randint(0, 31), rng.randint(0, 31))


def _make_task(rng: random.Random, index: int, family: str) -> HiddenTask:
    support = None
    for _ in range(1000):
        rows = tuple(
            RepresentationSupportExample(pair := _random_pair(rng), apply_representation_primitive(family, pair))
            for _ in range(4)
        )
        try:
            if infer_representation_family(rows) == family:
                support = rows
                break
        except ValueError:
            continue
    if support is None:
        raise RuntimeError("could not generate identifiable representation support")

    query = None
    for _ in range(1000):
        pair = _random_pair(rng)
        outputs = [apply_representation_primitive(name, pair) for name in PRIMITIVE_FAMILIES]
        if len(set(outputs)) == len(outputs):
            query = pair
            break
    if query is None:
        raise RuntimeError("could not generate discriminating representation query")

    return HiddenTask(
        task_id=f"cycle6-hidden-{index:03d}",
        family=family,
        artifact_type=f"latent_artifact_{index:03d}",
        support=support,
        query=query,
        hidden_output=apply_representation_primitive(family, query),
    )


def _hidden_tasks(seed: int, count: int) -> Tuple[HiddenTask, ...]:
    if count % len(PRIMITIVE_FAMILIES) != 0:
        raise ValueError("hidden task count must be balanced across primitive families")
    rng = random.Random(seed)
    families = list(PRIMITIVE_FAMILIES) * (count // len(PRIMITIVE_FAMILIES))
    rng.shuffle(families)
    return tuple(_make_task(rng, index, family) for index, family in enumerate(families))


def _apply_and_restore(parent: MorphologyGenome, candidate) -> Tuple[MorphologyGenome, object]:
    child = MorphologyMutator().apply(parent, candidate.mutation)
    body = SelfEvolvingResearchBody(
        runtime=PersistentCognitiveRuntime(),
        morphology=child,
        mutation_strategy=MutationStrategyState(),
        mutation_program_state=MutationProgramDevelopmentState(),
        experience_archive=ExperienceArchive(),
    )
    restored = restore_body(checkpoint_dict(body))
    return child, restored


def _parent_representation_reachability(parent: MorphologyGenome, residual: MorphologyResidual) -> Tuple[int, int]:
    base_rows = NativeMetaMorphologyGenesisEngine(candidate_budget=4096).generate(parent, (residual,))
    l1_count = 0
    native_repr_count = 0
    for row in base_rows:
        if row.mutation.level == MutationLevel.REPRESENTATION_MEMORY_TOOL:
            l1_count += 1
        try:
            descendant = MorphologyMutator().apply(parent, row.mutation)
        except ValueError:
            continue
        if any(str(organ.implementation_ref).startswith("native-repr://") for organ in descendant.organs):
            native_repr_count += 1
    return l1_count, native_repr_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-head", required=True)
    parser.add_argument("--candidate-head", required=True)
    parser.add_argument("--precommit", required=True)
    parser.add_argument("--cycle5-receipt", required=True)
    parser.add_argument("--hidden-seed-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    precommit = _read_json(args.precommit)
    cycle5 = _read_json(args.cycle5_receipt)
    if int(precommit.get("generation", -1)) != 6:
        raise SystemExit("cycle6 precommit generation mismatch")
    if precommit.get("parent_evidence_head_sha") != args.parent_head:
        raise SystemExit("cycle6 parent head mismatch")
    if cycle5.get("outcome_receipt_sha256") != precommit.get("parent_outcome_receipt_sha256"):
        raise SystemExit("cycle6 parent receipt mismatch")
    if cycle5.get("candidate_core_body_hash") != precommit.get("parent_core_body_sha256"):
        raise SystemExit("cycle6 parent core BODY mismatch")
    _git("cat-file", "-e", f"{args.parent_head}^{{commit}}")
    _git("cat-file", "-e", f"{args.candidate_head}^{{commit}}")

    seed = int(Path(args.hidden_seed_file).read_text(encoding="utf-8").strip())
    task_count = int(precommit["resource_contract"]["hidden_task_count"])
    tasks = _hidden_tasks(seed, task_count)
    parent = parent_genome()
    engine = NativeRepresentationGenesisEngine(candidate_budget=1)

    full = 0
    wrong = 0
    shuffle = 0
    restart_equal = True
    checkpoint_equal = True
    parent_l1_total = 0
    parent_native_repr_total = 0
    generated_l1_total = 0

    for index, task in enumerate(tasks):
        residual = task_residual(task)
        l1_count, repr_count = _parent_representation_reachability(parent, residual)
        parent_l1_total += l1_count
        parent_native_repr_total += repr_count

        candidate = engine.generate(parent, residual, task.support)[0]
        if candidate.mutation.level != MutationLevel.REPRESENTATION_MEMORY_TOOL:
            raise SystemExit("FULL did not emit L1 representation mutation")
        generated_l1_total += 1
        child, restored = _apply_and_restore(parent, candidate)
        checkpoint_equal = checkpoint_equal and restored.morphology.fingerprint() == child.fingerprint()
        program = executable_representation_programs(
            restored.morphology,
            expected_artifact_type=task.artifact_type,
            expected_residual_id=task.task_id,
        )[0]
        prediction = program.execute(task.query)
        full += int(prediction == task.hidden_output)

        restarted = restore_body(checkpoint_dict(restored))
        restart_program = executable_representation_programs(
            restarted.morphology,
            expected_artifact_type=task.artifact_type,
            expected_residual_id=task.task_id,
        )[0]
        restart_equal = restart_equal and (
            restart_program.fingerprint() == program.fingerprint()
            and restart_program.execute(task.query) == prediction
        )

        wrong_family = next(name for name in PRIMITIVE_FAMILIES if name != task.family)
        wrong_candidate = engine.generate(parent, residual, task.support, force_family=wrong_family)[0]
        _, wrong_restored = _apply_and_restore(parent, wrong_candidate)
        wrong_program = executable_representation_programs(
            wrong_restored.morphology,
            expected_artifact_type=task.artifact_type,
            expected_residual_id=task.task_id,
        )[0]
        wrong += int(wrong_program.execute(task.query) == task.hidden_output)

        donor = next(row for row in tasks if row.family != task.family and row.task_id != task.task_id)
        shuffled_candidate = engine.generate(parent, residual, donor.support)[0]
        _, shuffled_restored = _apply_and_restore(parent, shuffled_candidate)
        shuffled_program = executable_representation_programs(
            shuffled_restored.morphology,
            expected_artifact_type=task.artifact_type,
            expected_residual_id=task.task_id,
        )[0]
        shuffle += int(shuffled_program.execute(task.query) == task.hidden_output)

    balanced = {family: sum(1 for task in tasks if task.family == family) for family in PRIMITIVE_FAMILIES}
    parent_more_compute_unreachable = parent_l1_total == 0 and parent_native_repr_total == 0
    pass_contract = bool(
        full == task_count
        and generated_l1_total == task_count
        and parent_more_compute_unreachable
        and wrong < full
        and shuffle < full
        and restart_equal
        and checkpoint_equal
    )
    if not pass_contract:
        raise SystemExit(
            f"cycle6 contract failed full={full} wrong={wrong} shuffle={shuffle} "
            f"parent_l1={parent_l1_total} parent_repr={parent_native_repr_total} restart={restart_equal}"
        )

    candidate_core = _core_hash(
        (
            "arte_cognition/native_representation_genesis.py",
            "arte_cognition/morphology_genesis.py",
            "arte_cognition/native_recursive_research.py",
            "arte_cognition/self_evolving_body_checkpoint.py",
        )
    )
    material = {
        "generation": 6,
        "parent_head": args.parent_head,
        "candidate_head": args.candidate_head,
        "hidden_seed_sha256": hashlib.sha256(str(seed).encode()).hexdigest(),
        "full": full,
        "wrong": wrong,
        "shuffle": shuffle,
        "parent_l1": parent_l1_total,
        "parent_native_repr": parent_native_repr_total,
        "candidate_core": candidate_core,
    }
    receipt_hash = _sha256(material)
    receipt = {
        "schema": "arte.native_recursive_research_cycle6/v1",
        "generation": 6,
        "problem_detector": "REPRESENTATION_MUTATION_FAMILY_UNREACHABLE",
        "parent_evidence_head_sha": args.parent_head,
        "candidate_implementation_head_sha": args.candidate_head,
        "parent_core_body_hash": precommit["parent_core_body_sha256"],
        "candidate_core_body_hash": candidate_core,
        "hidden_task_count": task_count,
        "hidden_task_balance": balanced,
        "support_examples_per_task": int(precommit["resource_contract"]["support_examples_per_task"]),
        "full_hidden_useful_count": full,
        "full_hidden_useful_rate": full / task_count,
        "wrong_hidden_useful_count": wrong,
        "wrong_hidden_useful_rate": wrong / task_count,
        "shuffle_hidden_useful_count": shuffle,
        "shuffle_hidden_useful_rate": shuffle / task_count,
        "parent_more_compute_candidate_budget": int(precommit["resource_contract"]["remove_parent_candidate_budget"]),
        "parent_more_compute_l1_candidate_count": parent_l1_total,
        "parent_more_compute_native_representation_count": parent_native_repr_total,
        "old_language_more_compute_resistant": parent_more_compute_unreachable,
        "generated_l1_representation_count": generated_l1_total,
        "checkpoint_restore_before_hidden_query": True,
        "checkpoint_morphology_equal": checkpoint_equal,
        "restart_reconstruction_equal": restart_equal,
        "current_outcomes_consumed_by_generation": False,
        "new_representational_reachability_established": True,
        "bounded_positive_descendant_generative_effect": True,
        "effect_type": "MORE_COMPUTE_RESISTANT_EXECUTABLE_REPRESENTATION_REACHABILITY",
        "eligible_claim": "BOUNDED_MORE_COMPUTE_RESISTANT_REPRESENTATION_REACHABILITY",
        "broad_capability_improvement_established": False,
        "recursive_acceleration_established": False,
        "official_benchmark_used": False,
        "external_claim_authority": False,
        "next_problem_detectors": ["REPRESENTATION_GENESIS_SELF_MODIFICATION_UNPROVEN"],
        "outcome_receipt_sha256": receipt_hash,
        "AGI": False,
        "ASI": False,
    }
    Path(args.output).write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
