from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

from arte_cognition.executable_morphology import EdgeSpec, MorphologyGenome, OrganKind, OrganSpec, PressureVector
from arte_cognition.meta_acceleration import MutationStrategyState, apply_mutation_program
from arte_cognition.morphology_genesis import MorphologyGenesisEngine, MorphologyResidual
from arte_cognition.parametric_morphology_macro import (
    MacroTrainingExample,
    ParametricMorphologyMacro,
    apply_parametric_macro,
    derive_parametric_rewire_macro,
)
from arte_cognition.structural_failure_certificate import (
    StructuralDiagnosticReceipt,
    compile_program_from_certificate,
    derive_structural_failure_certificate,
)


STATUS = "PASS_BOUNDED_PARAMETRIC_MORPHOLOGY_MACRO_AMORTIZES_CANDIDATE_SEARCH_ACROSS_LARGER_STRUCTURAL_FRONTS"


def token(rng: random.Random, prefix: str) -> str:
    return f"{prefix}_{rng.randrange(10_000_000, 99_999_999)}"


def make_world(seed: int, label: str, depth: int):
    rng = random.Random((seed << 3) ^ sum(ord(c) for c in label))
    organs = []
    edges = []
    alternates = []
    primaries = []
    for index in range(depth):
        source = token(rng, f"source{index}")
        primary = token(rng, f"primary{index}")
        alternate = token(rng, f"alternate{index}")
        artifact = token(rng, f"evidence{index}")
        edge_id = token(rng, f"edge{index}")
        organs.extend(
            (
                OrganSpec(source, OrganKind.SOURCE, produces=(artifact,)),
                OrganSpec(primary, OrganKind.REPRESENTATION, consumes=(artifact,)),
                OrganSpec(alternate, OrganKind.REPRESENTATION, consumes=(artifact,)),
            )
        )
        edges.append(EdgeSpec(edge_id, source, primary, artifact))
        alternates.append(alternate)
        primaries.append(primary)
    organs.extend((OrganSpec(token(rng, "governor"), OrganKind.GOVERNOR), OrganSpec(token(rng, "archive"), OrganKind.ARCHIVE)))
    genome = MorphologyGenome(tuple(organs), tuple(edges), tuple(o.organ_id for o in organs))
    failed = tuple(edge.edge_id for edge in edges)
    residual = MorphologyResidual(
        token(rng, "residual"),
        PressureVector(transfer_failure=1.0, human_dependency=1.0),
        failed_edge_ids=failed,
        implicated_organ_ids=tuple(primaries + alternates),
        source_refs=(f"hidden::{label}",),
    )
    receipts = (
        StructuralDiagnosticReceipt(token(rng, "receipt_a"), f"ctx::{label}", f"class-a::{label}", failed, True, True),
        StructuralDiagnosticReceipt(token(rng, "receipt_b"), f"ctx::{label}", f"class-b::{label}", tuple(reversed(failed)), True, True),
    )
    certificate = derive_structural_failure_certificate(receipts)
    hidden = {"edge_ids": failed, "targets": tuple(alternates)}
    return genome, residual, certificate, hidden


def capability(genome: MorphologyGenome, hidden: Dict[str, object]) -> float:
    edge_map = {edge.edge_id: edge for edge in genome.edges}
    for edge_id, target in zip(hidden["edge_ids"], hidden["targets"]):
        if edge_map[edge_id].target != target:
            return 0.0
    return 1.0


def training_example(seed: int, label: str, depth: int) -> MacroTrainingExample:
    genome, residual, certificate, hidden = make_world(seed, label, depth)
    candidates = MorphologyGenesisEngine(candidate_budget=512).generate(genome, (residual,))
    strategy = MutationStrategyState(operation_scores=(("REWIRE_EDGE", 3.0), ("ADD_EDGE", 0.0)), lineage_hash=f"prior::{label}")
    compilation = compile_program_from_certificate(genome, candidates, strategy, certificate)
    if compilation.program is None:
        raise AssertionError("could not compile training program")
    descendant = apply_mutation_program(genome, compilation.program)
    effect = capability(descendant, hidden)
    if effect != 1.0:
        raise AssertionError("training descendant failed")
    return MacroTrainingExample(
        context_id=f"context::{label}",
        source_class=f"source-class::{label}",
        genome=genome,
        certificate=certificate,
        successful_program=compilation.program,
        external_capability=effect,
        authority_verified=True,
        benchmark_disjoint=True,
    )


def exhaustive_programs(n: int) -> int:
    total = 0
    for depth in range(1, n + 1):
        value = 1
        for term in range(n, n - depth, -1):
            value *= term
        total += value
    return total


def main(seed_path: str) -> int:
    seed = int(Path(seed_path).read_text().strip())
    examples = (
        training_example(seed + 1, "train-a", 2),
        training_example(seed + 2, "train-b", 3),
    )
    macro = derive_parametric_rewire_macro(examples)
    if macro is None:
        raise AssertionError("macro genesis failed")

    heldout_depths = (4, 8, 16)
    candidate_evaluations = []
    lookups = []
    mutation_steps = []
    exhaustive_controls = []
    capabilities = []
    remove = []
    wrong = []

    for index, depth in enumerate(heldout_depths):
        genome, _, certificate, hidden = make_world(seed + 100 + index, f"heldout-{depth}", depth)
        application = apply_parametric_macro(genome, certificate, macro)
        if application is None:
            raise AssertionError(f"macro failed to instantiate at depth {depth}")
        descendant = apply_mutation_program(genome, application.mutation_program)
        effect = capability(descendant, hidden)
        if effect != 1.0:
            raise AssertionError(f"macro transfer failed at depth {depth}")

        remove_effect = capability(genome, hidden)
        wrong_macro = ParametricMorphologyMacro(
            macro_id=f"wrong::{depth}",
            rule="KEEP_CURRENT_TARGET",
            supporting_contexts=macro.supporting_contexts,
            supporting_source_classes=macro.supporting_source_classes,
            supporting_program_ids=macro.supporting_program_ids,
            inherited_from_external_outcomes=True,
        )
        wrong_application = apply_parametric_macro(genome, certificate, wrong_macro)
        wrong_effect = 0.0 if wrong_application is None else capability(
            apply_mutation_program(genome, wrong_application.mutation_program), hidden
        )
        if remove_effect != 0.0 or wrong_effect != 0.0:
            raise AssertionError("REMOVE/WRONG control did not lose capability")

        candidate_evaluations.append(application.candidate_evaluations)
        lookups.append(application.structural_lookup_count)
        mutation_steps.append(application.mutation_program.depth)
        exhaustive_controls.append(exhaustive_programs(depth))
        capabilities.append(effect)
        remove.append(remove_effect)
        wrong.append(wrong_effect)

    prior_frontier = 3
    frontier_increments = []
    last = prior_frontier
    for depth in heldout_depths:
        frontier_increments.append(depth - last)
        last = depth
    high_level_cost = tuple(3 + lookup + steps for lookup, steps in zip(lookups, mutation_steps))
    frontier_gain_per_high_level_operation = tuple(
        gain / cost for gain, cost in zip(frontier_increments, high_level_cost)
    )
    strict_scale_metaproductivity_gain = all(
        b > a for a, b in zip(frontier_gain_per_high_level_operation, frontier_gain_per_high_level_operation[1:])
    )

    result = {
        "status": STATUS,
        "macro_id": macro.macro_id,
        "macro_rule": macro.rule,
        "training_context_count": len(macro.supporting_contexts),
        "training_source_class_count": len(macro.supporting_source_classes),
        "heldout_structural_frontiers": list(heldout_depths),
        "heldout_capabilities": capabilities,
        "candidate_evaluations_after_macro": candidate_evaluations,
        "structural_lookup_counts": lookups,
        "mutation_steps": mutation_steps,
        "exhaustive_program_counts_avoided": exhaustive_controls,
        "frontier_increments": frontier_increments,
        "frontier_gain_per_high_level_operation": frontier_gain_per_high_level_operation,
        "strict_scale_metaproductivity_gain": strict_scale_metaproductivity_gain,
        "remove_capabilities": remove,
        "wrong_macro_capabilities": wrong,
        "current_hidden_outcomes_used_for_macro_application": False,
        "macro_rule_vocabulary_human_authored": True,
        "unique_alternative_relation_human_authored": True,
        "same_macro_reused_across_generations": True,
        "improvement_mechanism_itself_improved_each_generation": False,
        "recursive_acceleration": False,
        "global_recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_parametric_morphology_macro.py <seed-file>")
    raise SystemExit(main(sys.argv[1]))
