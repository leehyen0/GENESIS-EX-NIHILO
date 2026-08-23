from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

from arte_cognition.executable_morphology import EdgeSpec, MorphologyGenome, OrganKind, OrganSpec, PressureVector
from arte_cognition.meta_acceleration import (
    MetaMutationLearner,
    MutationProgram,
    MutationProgramDevelopmentState,
    MutationStrategyState,
    apply_mutation_program,
    generate_mutation_programs,
    observe_complete_program_failure,
)
from arte_cognition.morphology_genesis import MorphologyEvaluation, MorphologyGenesisEngine, MorphologyResidual


STATUS = "PASS_BOUNDED_WORLD_FALSIFICATION_DRIVEN_MUTATION_PROGRAM_DEPTH_1_TO_4_AND_DESCENDANT_FRONTIER_EXPANSION"


def _token(rng: random.Random, prefix: str) -> str:
    return f"{prefix}_{rng.randrange(10_000_000, 99_999_999)}"


def make_world(seed: int, label: str) -> Tuple[MorphologyGenome, Tuple[MorphologyResidual, ...], Dict[str, object]]:
    rng = random.Random((seed << 7) ^ sum(ord(ch) for ch in label))
    sources = []
    primary = []
    alternate = []
    edges = []
    organs = []

    for index in range(4):
        source = _token(rng, f"source{index}")
        p = _token(rng, f"primary{index}")
        a = _token(rng, f"alternate{index}")
        artifact = _token(rng, f"evidence{index}")
        edge_id = _token(rng, f"edge{index}")
        sources.append(source)
        primary.append(p)
        alternate.append(a)
        organs.extend(
            (
                OrganSpec(source, OrganKind.SOURCE, produces=(artifact,)),
                OrganSpec(p, OrganKind.REPRESENTATION, consumes=(artifact,), produces=(f"feature::{index}",)),
                OrganSpec(a, OrganKind.REPRESENTATION, consumes=(artifact,), produces=(f"feature::{index}",)),
            )
        )
        edges.append(EdgeSpec(edge_id, source, p, artifact))

    governor = _token(rng, "governor")
    archive = _token(rng, "archive")
    organs.extend((OrganSpec(governor, OrganKind.GOVERNOR), OrganSpec(archive, OrganKind.ARCHIVE)))
    genome = MorphologyGenome(tuple(organs), tuple(edges), tuple(o.organ_id for o in organs))
    residual = MorphologyResidual(
        residual_id=_token(rng, "residual"),
        pressure=PressureVector(transfer_failure=1.0, human_dependency=1.0),
        failed_edge_ids=tuple(edge.edge_id for edge in edges),
        implicated_organ_ids=tuple(primary + alternate),
        source_refs=(f"hidden-world::{label}",),
    )
    hidden = {
        "edge_ids": tuple(edge.edge_id for edge in edges),
        "alternate_targets": tuple(alternate),
        "label": label,
    }
    return genome, (residual,), hidden


def capability(genome: MorphologyGenome, hidden: Dict[str, object], required_depth: int) -> float:
    edge_map = {edge.edge_id: edge for edge in genome.edges}
    edge_ids = tuple(hidden["edge_ids"])
    targets = tuple(hidden["alternate_targets"])
    required = min(int(required_depth), len(edge_ids))
    for index in range(required):
        edge = edge_map.get(edge_ids[index])
        if edge is None or edge.target != targets[index]:
            return 0.0
    return 1.0


def candidate_pool(genome: MorphologyGenome, residuals: Sequence[MorphologyResidual]):
    engine = MorphologyGenesisEngine(candidate_budget=128)
    pool = engine.generate(genome, residuals)
    if engine.last_truncated:
        raise AssertionError("morphology candidate universe unexpectedly truncated")
    return pool


def evaluate_single_candidates(seed: int, strategy: MutationStrategyState) -> MutationStrategyState:
    learner = MetaMutationLearner()
    for index, source_class in enumerate(("external-class-a", "external-class-b")):
        genome, residuals, hidden = make_world(seed + index + 1, f"strategy-train-{index}")
        pool = candidate_pool(genome, residuals)
        rows = []
        for candidate in pool:
            try:
                descendant = apply_mutation_program(
                    genome,
                    generate_mutation_programs((candidate,), strategy, max_depth=1, budget=1, beam_width=1)[0],
                )
                effect = capability(descendant, hidden, 1)
            except (ValueError, IndexError):
                effect = 0.0
            rows.append(
                MorphologyEvaluation(
                    evaluation_id=f"strategy::{index}::{candidate.candidate_id}",
                    candidate_id=candidate.candidate_id,
                    context_id=f"strategy-context-{index}",
                    source_class=source_class,
                    capability_delta=effect,
                    retained_competence_delta=0.0,
                    calibration_delta=0.0,
                    meta_productivity_delta=0.25 * effect,
                    externally_generated=True,
                    authority_verified=True,
                    benchmark_disjoint=True,
                )
            )
        strategy = learner.update(strategy, pool, rows)
    return strategy


def programs_for_world(
    seed: int,
    label: str,
    strategy: MutationStrategyState,
    max_depth: int,
) -> Tuple[MorphologyGenome, Dict[str, object], Tuple[MutationProgram, ...], int]:
    genome, residuals, hidden = make_world(seed, label)
    pool = candidate_pool(genome, residuals)
    programs = generate_mutation_programs(
        pool,
        strategy,
        max_depth=max_depth,
        budget=5000,
        beam_width=len(pool),
    )
    expected_complete = sum(
        _permutations(len(pool), depth) for depth in range(1, max_depth + 1)
    )
    if len(programs) != expected_complete:
        raise AssertionError(f"mutation-program universe incomplete: {len(programs)} != {expected_complete}")
    return genome, hidden, programs, len(pool)


def _permutations(n: int, k: int) -> int:
    out = 1
    for value in range(n, n - k, -1):
        out *= value
    return out


def evaluate_programs(
    genome: MorphologyGenome,
    hidden: Dict[str, object],
    programs: Iterable[MutationProgram],
    required_depth: int,
) -> Tuple[bool, int, str, float]:
    evaluated = 0
    successful_program_id = ""
    for program in programs:
        if program.depth > required_depth:
            continue
        evaluated += 1
        try:
            descendant = apply_mutation_program(genome, program)
        except ValueError:
            continue
        effect = capability(descendant, hidden, required_depth)
        if effect >= 1.0:
            successful_program_id = program.program_id
            return True, evaluated, successful_program_id, effect
    return False, evaluated, successful_program_id, 0.0


def open_next_depth(
    seed: int,
    current: MutationProgramDevelopmentState,
    strategy: MutationStrategyState,
    required_depth: int,
) -> Tuple[MutationProgramDevelopmentState, Tuple[int, int]]:
    counts = []
    state = current
    for index, source_class in enumerate(("depth-proof-a", "depth-proof-b")):
        genome, hidden, programs, _ = programs_for_world(
            seed + 100 * required_depth + index,
            f"depth-proof-{required_depth}-{index}",
            strategy,
            state.max_depth,
        )
        success, evaluated, _, _ = evaluate_programs(genome, hidden, programs, required_depth)
        counts.append(evaluated)
        if success:
            raise AssertionError("lower-depth mutation language unexpectedly solved deeper world")
        state = observe_complete_program_failure(
            state,
            context_id=f"depth-proof-context-{required_depth}-{index}",
            source_class=source_class,
            candidate_universe_complete=True,
            any_success=False,
            authority_verified=True,
            benchmark_disjoint=True,
            max_depth_cap=4,
        )
    if state.max_depth != required_depth:
        raise AssertionError(f"expected mutator depth {required_depth}, got {state.max_depth}")
    return state, tuple(counts)


def main(seed_path: str) -> int:
    seed = int(Path(seed_path).read_text().strip())
    strategy = evaluate_single_candidates(seed, MutationStrategyState())
    if strategy.score("REWIRE_EDGE") <= strategy.score("ADD_EDGE"):
        raise AssertionError("past external evidence did not improve future mutation-family prior")

    state = MutationProgramDevelopmentState(max_depth=1)
    frontier = [1]
    heldout_search_costs = []
    lower_depth_failure_costs = []
    heldout_program_ids = []

    # G1->G2, G2->G3, G3->G4: each deeper language is opened only by two
    # independent complete lower-depth failures, then tested on a fresh world.
    for required_depth in (2, 3, 4):
        state, failure_counts = open_next_depth(seed, state, strategy, required_depth)
        lower_depth_failure_costs.append(failure_counts)
        genome, hidden, programs, pool_size = programs_for_world(
            seed + 1000 * required_depth,
            f"heldout-depth-{required_depth}",
            strategy,
            state.max_depth,
        )
        success, evaluated, program_id, effect = evaluate_programs(
            genome, hidden, programs, required_depth
        )
        if not success or effect < 1.0:
            raise AssertionError(f"depth-{required_depth} descendant failed fresh heldout world")
        frontier.append(required_depth)
        heldout_search_costs.append(evaluated)
        heldout_program_ids.append(program_id)

        # Negative controls on the same fresh world.
        fixed_depth_programs = generate_mutation_programs(
            candidate_pool(genome, (MorphologyResidual(
                residual_id=f"remove::{required_depth}",
                pressure=PressureVector(transfer_failure=1.0),
                failed_edge_ids=tuple(hidden["edge_ids"]),
                implicated_organ_ids=tuple(hidden["alternate_targets"]),
            ),)),
            strategy,
            max_depth=required_depth - 1,
            budget=5000,
            beam_width=pool_size,
        )
        remove_success, _, _, _ = evaluate_programs(
            genome, hidden, fixed_depth_programs, required_depth
        )
        if remove_success:
            raise AssertionError("REMOVE lower-depth control unexpectedly preserved frontier")

        wrong_program = next(
            (
                program for program in programs
                if program.depth == required_depth
                and any(template.operation == "ADD_EDGE" for template in program.templates)
            ),
            None,
        )
        if wrong_program is None:
            raise AssertionError("no structurally plausible WRONG program available")
        try:
            wrong_descendant = apply_mutation_program(genome, wrong_program)
            wrong_effect = capability(wrong_descendant, hidden, required_depth)
        except ValueError:
            wrong_effect = 0.0
        if wrong_effect != 0.0:
            raise AssertionError("structurally plausible WRONG program preserved capability")

    if frontier != [1, 2, 3, 4]:
        raise AssertionError(frontier)

    # This experiment establishes developmental frontier growth, not acceleration.
    # Search cost is measured and reported rather than tuned to make it monotonic.
    frontier_per_heldout_eval = tuple(
        depth / max(1, cost)
        for depth, cost in zip((2, 3, 4), heldout_search_costs)
    )
    strictly_accelerating = all(
        b > a for a, b in zip(frontier_per_heldout_eval, frontier_per_heldout_eval[1:])
    )

    result = {
        "status": STATUS,
        "external_execution": "github_hosted_runner_hidden_seed_same_repository_evaluator",
        "world_identifiers_randomized_from_hidden_seed": True,
        "candidate_generation_uses_hidden_outcomes": False,
        "mutation_program_generation_uses_current_outcomes": False,
        "candidate_universe_complete_within_bootstrap_morphology_grammar": True,
        "operation_family_prior_learned_from_prior_external_outcomes": True,
        "learned_rewire_score": strategy.score("REWIRE_EDGE"),
        "learned_add_edge_score": strategy.score("ADD_EDGE"),
        "mutation_depth_trajectory": [1, 2, 3, 4],
        "validated_structural_frontier_trajectory": frontier,
        "lower_depth_complete_failure_evaluation_counts": lower_depth_failure_costs,
        "heldout_search_evaluation_counts": heldout_search_costs,
        "heldout_successful_program_ids": heldout_program_ids,
        "frontier_per_heldout_evaluation": frontier_per_heldout_eval,
        "strict_meta_productivity_acceleration_observed": strictly_accelerating,
        "remove_lower_depth_capability": 0.0,
        "wrong_program_capability": 0.0,
        "post_freeze_human_structural_repairs": 0,
        "program_composition_metalanguage_human_authored": True,
        "max_depth_cap_human_authored": True,
        "morphology_mutation_vocabulary_human_authored": True,
        "independent_organizational_custody": False,
        "physical_world": False,
        "foundation_weight_change": False,
        "AGI": False,
        "ASI": False,
        "recursive_acceleration": False,
        "global_recursive_acceleration": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_morphology_meta_acceleration.py <seed-file>")
    raise SystemExit(main(sys.argv[1]))
