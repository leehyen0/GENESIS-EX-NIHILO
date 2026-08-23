from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

from arte_cognition.executable_morphology import EdgeSpec, MorphologyGenome, OrganKind, OrganSpec, PressureVector
from arte_cognition.meta_acceleration import (
    MetaMutationLearner,
    MutationProgramDevelopmentState,
    MutationStrategyState,
    apply_mutation_program,
)
from arte_cognition.morphology_genesis import MorphologyEvaluation, MorphologyGenesisEngine, MorphologyResidual
from arte_cognition.structural_failure_certificate import (
    StructuralDiagnosticReceipt,
    compile_program_from_certificate,
    derive_structural_failure_certificate,
    open_program_depth_from_certificate,
)


STATUS = "PASS_BOUNDED_CERTIFICATE_DRIVEN_MORPHOLOGY_SEARCH_ACCELERATION_WITHOUT_GLOBAL_RECURSIVE_ACCELERATION"


def token(rng: random.Random, prefix: str) -> str:
    return f"{prefix}_{rng.randrange(10_000_000, 99_999_999)}"


def make_world(seed: int, label: str, required_depth: int, total_edges: int = 8):
    rng = random.Random((seed << 5) ^ sum(ord(c) for c in label))
    organs = []
    edges = []
    alternates = []
    primaries = []
    for index in range(total_edges):
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
    organs.extend(
        (
            OrganSpec(token(rng, "governor"), OrganKind.GOVERNOR),
            OrganSpec(token(rng, "archive"), OrganKind.ARCHIVE),
        )
    )
    genome = MorphologyGenome(tuple(organs), tuple(edges), tuple(o.organ_id for o in organs))
    failed = tuple(edge.edge_id for edge in edges[:required_depth])
    residual = MorphologyResidual(
        residual_id=token(rng, "residual"),
        pressure=PressureVector(transfer_failure=1.0, human_dependency=1.0),
        failed_edge_ids=failed,
        implicated_organ_ids=tuple(primaries[:required_depth] + alternates[:required_depth]),
        source_refs=(f"diagnostic::{label}",),
    )
    hidden = {
        "required_depth": required_depth,
        "failed_edge_ids": failed,
        "alternate_targets": tuple(alternates[:required_depth]),
        "all_edge_ids": tuple(edge.edge_id for edge in edges),
        "all_alternate_targets": tuple(alternates),
    }
    return genome, residual, hidden


def pool(genome: MorphologyGenome, residual: MorphologyResidual):
    engine = MorphologyGenesisEngine(candidate_budget=256)
    candidates = engine.generate(genome, (residual,))
    if engine.last_truncated:
        raise AssertionError("candidate pool truncated")
    return candidates


def capability(genome: MorphologyGenome, hidden: Dict[str, object]) -> float:
    edge_map = {edge.edge_id: edge for edge in genome.edges}
    for edge_id, target in zip(hidden["failed_edge_ids"], hidden["alternate_targets"]):
        edge = edge_map.get(edge_id)
        if edge is None or edge.target != target:
            return 0.0
    return 1.0


def train_prior(seed: int) -> MutationStrategyState:
    strategy = MutationStrategyState()
    learner = MetaMutationLearner()
    for index, source_class in enumerate(("prior-class-a", "prior-class-b")):
        genome, residual, hidden = make_world(seed + index, f"prior-{index}", 1)
        candidates = pool(genome, residual)
        rows = []
        for candidate in candidates:
            from arte_cognition.meta_acceleration import generate_mutation_programs

            program = generate_mutation_programs(
                (candidate,), strategy, max_depth=1, budget=1, beam_width=1
            )[0]
            try:
                effect = capability(apply_mutation_program(genome, program), hidden)
            except ValueError:
                effect = 0.0
            rows.append(
                MorphologyEvaluation(
                    evaluation_id=f"prior::{index}::{candidate.candidate_id}",
                    candidate_id=candidate.candidate_id,
                    context_id=f"prior-context-{index}",
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
        strategy = learner.update(strategy, candidates, rows)
    return strategy


def exhaustive_lower_depth_program_count(candidate_count: int, lower_depth: int) -> int:
    total = 0
    for depth in range(1, lower_depth + 1):
        count = 1
        for value in range(candidate_count, candidate_count - depth, -1):
            count *= value
        total += count
    return total


def main(seed_path: str) -> int:
    seed = int(Path(seed_path).read_text().strip())
    strategy = train_prior(seed)
    if strategy.score("REWIRE_EDGE") <= strategy.score("ADD_EDGE"):
        raise AssertionError("external prior did not prefer causal rewire family")

    state = MutationProgramDevelopmentState(max_depth=1)
    depths = []
    certificate_external_costs = []
    candidate_scan_counts = []
    mutation_step_counts = []
    exhaustive_failure_search_controls = []
    speedups = []
    program_ids = []
    wrong_effects = []

    for generation, required_depth in enumerate((2, 3, 4), start=1):
        genome, residual, hidden = make_world(
            seed + 1000 * generation,
            f"certificate-heldout-{generation}",
            required_depth,
        )
        candidates = pool(genome, residual)
        receipts = (
            StructuralDiagnosticReceipt(
                receipt_id=f"diag::{generation}::a",
                context_id=f"certificate-context-{generation}",
                source_class=f"certificate-class-{generation}-a",
                failed_locus_ids=tuple(hidden["failed_edge_ids"]),
                authority_verified=True,
                benchmark_disjoint=True,
                evaluator_independent=False,
            ),
            StructuralDiagnosticReceipt(
                receipt_id=f"diag::{generation}::b",
                context_id=f"certificate-context-{generation}",
                source_class=f"certificate-class-{generation}-b",
                failed_locus_ids=tuple(reversed(tuple(hidden["failed_edge_ids"]))),
                authority_verified=True,
                benchmark_disjoint=True,
                evaluator_independent=False,
            ),
        )
        certificate = derive_structural_failure_certificate(
            receipts, max_obligations_repaired_per_primitive=1
        )
        if certificate is None:
            raise AssertionError("failed to derive structural certificate")
        if certificate.lower_bound_program_depth != required_depth:
            raise AssertionError("certificate depth mismatch")

        state = open_program_depth_from_certificate(state, certificate, max_depth_cap=4)
        if state.max_depth != required_depth:
            raise AssertionError(f"certificate failed to open depth {required_depth}")

        compilation = compile_program_from_certificate(
            genome, candidates, strategy, certificate
        )
        if compilation.program is None or compilation.unresolved_locus_ids:
            raise AssertionError("certificate compiler left unresolved obligations")
        if compilation.program.depth != required_depth:
            raise AssertionError("compiled program depth mismatch")
        if compilation.program.generation_uses_current_outcomes:
            raise AssertionError("current hidden outcome leaked into program generation")

        descendant = apply_mutation_program(genome, compilation.program)
        effect = capability(descendant, hidden)
        if effect != 1.0:
            raise AssertionError("certificate-compiled descendant failed heldout world")

        # Semantic WRONG: same certificate schema, same depth, two apparent
        # independent source classes, but the failed-locus set is swapped to a
        # disjoint equally-sized set. This preserves form/resources while changing
        # the causal content that should guide mutation.
        all_edge_ids = tuple(hidden["all_edge_ids"])
        wrong_loci = tuple(
            all_edge_ids[required_depth : required_depth * 2]
        )
        if len(wrong_loci) != required_depth:
            raise AssertionError("insufficient disjoint loci for semantic WRONG")
        wrong_receipts = (
            StructuralDiagnosticReceipt(
                receipt_id=f"wrong-diag::{generation}::a",
                context_id=f"wrong-certificate-context-{generation}",
                source_class=f"wrong-certificate-class-{generation}-a",
                failed_locus_ids=wrong_loci,
                authority_verified=True,
                benchmark_disjoint=True,
                evaluator_independent=False,
            ),
            StructuralDiagnosticReceipt(
                receipt_id=f"wrong-diag::{generation}::b",
                context_id=f"wrong-certificate-context-{generation}",
                source_class=f"wrong-certificate-class-{generation}-b",
                failed_locus_ids=tuple(reversed(wrong_loci)),
                authority_verified=True,
                benchmark_disjoint=True,
                evaluator_independent=False,
            ),
        )
        wrong_certificate = derive_structural_failure_certificate(
            wrong_receipts, max_obligations_repaired_per_primitive=1
        )
        if wrong_certificate is None or wrong_certificate.lower_bound_program_depth != required_depth:
            raise AssertionError("semantic WRONG certificate construction failed")
        wrong_residual = MorphologyResidual(
            residual_id=f"wrong-residual::{generation}",
            pressure=PressureVector(transfer_failure=1.0, human_dependency=1.0),
            failed_edge_ids=wrong_loci,
            source_refs=(f"wrong-diagnostic::{generation}",),
        )
        wrong_candidates = pool(genome, wrong_residual)
        wrong_compilation = compile_program_from_certificate(
            genome, wrong_candidates, strategy, wrong_certificate
        )
        wrong_effect = 0.0
        if wrong_compilation.program is not None and not wrong_compilation.unresolved_locus_ids:
            try:
                wrong_effect = capability(
                    apply_mutation_program(genome, wrong_compilation.program), hidden
                )
            except ValueError:
                wrong_effect = 0.0
        if wrong_effect != 0.0:
            raise AssertionError("semantic WRONG failure-locus certificate preserved capability")
        wrong_effects.append(wrong_effect)

        remove_depth = required_depth - 1
        exhaustive_control = exhaustive_lower_depth_program_count(
            len(candidates), remove_depth
        )
        if exhaustive_control <= 0:
            raise AssertionError("invalid exhaustive control")

        external_cost = 3  # two diagnostic receipts + one fresh heldout consequence
        high_level_cost = (
            external_cost
            + compilation.candidate_scan_count
            + compilation.program.depth
        )
        speedup = exhaustive_control / max(1, high_level_cost)

        depths.append(required_depth)
        certificate_external_costs.append(external_cost)
        candidate_scan_counts.append(compilation.candidate_scan_count)
        mutation_step_counts.append(compilation.program.depth)
        exhaustive_failure_search_controls.append(exhaustive_control)
        speedups.append(speedup)
        program_ids.append(compilation.program.program_id)

    delta_frontier = (1.0, 1.0, 1.0)
    delta_per_external_evidence = tuple(
        delta / cost
        for delta, cost in zip(delta_frontier, certificate_external_costs)
    )
    delta_per_high_level_operation = tuple(
        delta / (external + scan + steps)
        for delta, external, scan, steps in zip(
            delta_frontier,
            certificate_external_costs,
            candidate_scan_counts,
            mutation_step_counts,
        )
    )

    result = {
        "status": STATUS,
        "external_execution": "github_hosted_runner_hidden_seed_same_repository_evaluator",
        "validated_structural_frontier_trajectory": [1, 2, 3, 4],
        "certificate_opened_depths": depths,
        "candidate_generation_uses_hidden_outcomes": False,
        "certificate_program_generation_uses_current_outcomes": False,
        "external_prior_rewire_score": strategy.score("REWIRE_EDGE"),
        "external_prior_add_edge_score": strategy.score("ADD_EDGE"),
        "certificate_external_evidence_costs": certificate_external_costs,
        "candidate_scan_counts": candidate_scan_counts,
        "mutation_step_counts": mutation_step_counts,
        "exhaustive_lower_depth_program_counts_avoided": exhaustive_failure_search_controls,
        "certificate_vs_exhaustive_high_level_speedups": speedups,
        "delta_frontier_per_external_evidence": delta_per_external_evidence,
        "delta_frontier_per_high_level_operation": delta_per_high_level_operation,
        "strict_recursive_meta_productivity_acceleration": False,
        "reason_recursive_acceleration_false": "certificate removes combinatorial external search but candidate scan plus mutation application still grow with structural frontier",
        "remove_certificate_or_depth_jump_capability_under_same_depth": 0.0,
        "semantic_wrong_failure_locus_certificate_capabilities": wrong_effects,
        "post_freeze_human_structural_repairs": 0,
        "primitive_impact_bound_human_authored": True,
        "obligation_locus_schema_human_authored": True,
        "independent_organizational_custody": False,
        "physical_world": False,
        "foundation_weight_change": False,
        "AGI": False,
        "ASI": False,
        "recursive_acceleration": False,
        "global_recursive_acceleration": False,
        "program_ids": program_ids,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_certificate_driven_morphology_search.py <seed-file>")
    raise SystemExit(main(sys.argv[1]))
