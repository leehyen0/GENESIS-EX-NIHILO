from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

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


STATUS = "PASS_BOUNDED_PARAMETRIC_MORPHOLOGY_MACRO_CROSS_SCALE_TRANSFER"


def token(rng: random.Random, prefix: str) -> str:
    return f"{prefix}_{rng.randrange(10_000_000, 99_999_999)}"


def make_world(seed: int, label: str, required_depth: int, *, distractor_count: int = 0):
    rng = random.Random((seed << 9) ^ sum(ord(ch) for ch in label))
    organs = []
    edges = []
    true_alternates = []
    true_edge_ids = []
    distractor_edge_ids = []

    total = required_depth + distractor_count
    for index in range(total):
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
        if index < required_depth:
            true_edge_ids.append(edge_id)
            true_alternates.append(alternate)
        else:
            distractor_edge_ids.append(edge_id)

    organs.extend(
        (
            OrganSpec(token(rng, "governor"), OrganKind.GOVERNOR),
            OrganSpec(token(rng, "archive"), OrganKind.ARCHIVE),
        )
    )
    genome = MorphologyGenome(tuple(organs), tuple(edges), tuple(organ.organ_id for organ in organs))
    residual = MorphologyResidual(
        residual_id=token(rng, "residual"),
        pressure=PressureVector(transfer_failure=1.0, human_dependency=1.0),
        failed_edge_ids=tuple(true_edge_ids),
        source_refs=(f"hidden::{label}",),
    )
    hidden = {
        "true_edge_ids": tuple(true_edge_ids),
        "true_alternates": tuple(true_alternates),
        "distractor_edge_ids": tuple(distractor_edge_ids),
    }
    return genome, residual, hidden


def certificate_for(label: str, failed_locus_ids: Sequence[str]):
    failed = tuple(failed_locus_ids)
    receipts = (
        StructuralDiagnosticReceipt(
            receipt_id=f"{label}::receipt-a",
            context_id=f"{label}::context",
            source_class=f"{label}::class-a",
            failed_locus_ids=failed,
            authority_verified=True,
            benchmark_disjoint=True,
            evaluator_independent=False,
        ),
        StructuralDiagnosticReceipt(
            receipt_id=f"{label}::receipt-b",
            context_id=f"{label}::context",
            source_class=f"{label}::class-b",
            failed_locus_ids=tuple(reversed(failed)),
            authority_verified=True,
            benchmark_disjoint=True,
            evaluator_independent=False,
        ),
    )
    certificate = derive_structural_failure_certificate(receipts, max_obligations_repaired_per_primitive=1)
    if certificate is None:
        raise AssertionError("certificate derivation failed")
    return certificate


def capability(genome: MorphologyGenome, hidden: Dict[str, object]) -> float:
    edges = {edge.edge_id: edge for edge in genome.edges}
    for edge_id, target in zip(hidden["true_edge_ids"], hidden["true_alternates"]):
        edge = edges.get(edge_id)
        if edge is None or edge.target != target:
            return 0.0
    return 1.0


def candidates(genome: MorphologyGenome, residual: MorphologyResidual):
    engine = MorphologyGenesisEngine(candidate_budget=4096)
    pool = engine.generate(genome, (residual,))
    if engine.last_truncated:
        raise AssertionError("candidate universe truncated")
    return pool


def compile_training_example(seed: int, label: str, depth: int) -> MacroTrainingExample:
    genome, residual, hidden = make_world(seed, label, depth)
    certificate = certificate_for(label, hidden["true_edge_ids"])
    pool = candidates(genome, residual)
    strategy = MutationStrategyState(
        operation_scores=(("REWIRE_EDGE", 3.0), ("ADD_EDGE", 0.0)),
        lineage_hash=f"prior::{label}",
    )
    compilation = compile_program_from_certificate(genome, pool, strategy, certificate)
    if compilation.program is None or compilation.unresolved_locus_ids:
        raise AssertionError("training program compilation failed")
    descendant = apply_mutation_program(genome, compilation.program)
    effect = capability(descendant, hidden)
    if effect != 1.0:
        raise AssertionError("training external consequence failed")
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


def main(seed_path: str) -> int:
    seed = int(Path(seed_path).read_text().strip())

    training = (
        compile_training_example(seed + 11, "train-small", 2),
        compile_training_example(seed + 29, "train-medium", 3),
    )
    macro = derive_parametric_rewire_macro(training)
    if macro is None:
        raise AssertionError("cross-context successful programs failed to induce macro")
    if macro.current_outcomes_required_for_application:
        raise AssertionError("macro application improperly requires current outcome")

    depths = (4, 8, 16, 32)
    capability_trajectory = []
    candidate_evaluation_counts = []
    structural_lookup_counts = []
    emitted_mutation_steps = []
    certificate_compiler_scan_controls = []
    semantic_wrong_capabilities = []
    remove_capabilities = []

    for generation, depth in enumerate(depths, start=1):
        genome, residual, hidden = make_world(
            seed + 1000 * generation,
            f"heldout-{generation}",
            depth,
            distractor_count=depth,
        )
        certificate = certificate_for(f"heldout-{generation}", hidden["true_edge_ids"])
        application = apply_parametric_macro(genome, certificate, macro)
        if application is None:
            raise AssertionError("macro failed to apply to heldout morphology")
        descendant = apply_mutation_program(genome, application.mutation_program)
        effect = capability(descendant, hidden)
        if effect != 1.0:
            raise AssertionError("macro descendant failed fresh heldout consequence")

        # Same-form semantic WRONG: certify an equally sized, disjoint distractor set.
        wrong_certificate = certificate_for(
            f"wrong-{generation}",
            hidden["distractor_edge_ids"],
        )
        wrong_application = apply_parametric_macro(genome, wrong_certificate, macro)
        wrong_effect = 0.0
        if wrong_application is not None:
            wrong_descendant = apply_mutation_program(genome, wrong_application.mutation_program)
            wrong_effect = capability(wrong_descendant, hidden)
        if wrong_effect != 0.0:
            raise AssertionError("semantic WRONG certificate preserved capability")

        if capability(genome, hidden) != 0.0:
            raise AssertionError("REMOVE macro control unexpectedly capable")

        # Frozen certificate compiler control: same correct structural diagnosis,
        # but it must scan the generated candidate population to assemble the program.
        pool = candidates(genome, residual)
        strategy = MutationStrategyState(
            operation_scores=(("REWIRE_EDGE", 3.0), ("ADD_EDGE", 0.0)),
            lineage_hash="frozen-control-prior",
        )
        compilation = compile_program_from_certificate(genome, pool, strategy, certificate)
        if compilation.program is None or compilation.unresolved_locus_ids:
            raise AssertionError("certificate compiler control failed")
        control_descendant = apply_mutation_program(genome, compilation.program)
        if capability(control_descendant, hidden) != 1.0:
            raise AssertionError("certificate compiler control lost capability")

        capability_trajectory.append(effect)
        candidate_evaluation_counts.append(application.candidate_evaluations)
        structural_lookup_counts.append(application.structural_lookup_count)
        emitted_mutation_steps.append(application.mutation_program.depth)
        certificate_compiler_scan_controls.append(compilation.candidate_scan_count)
        semantic_wrong_capabilities.append(wrong_effect)
        remove_capabilities.append(0.0)

    if any(count != 0 for count in candidate_evaluation_counts):
        raise AssertionError("macro transfer evaluated candidate consequences")
    if structural_lookup_counts != list(depths):
        raise AssertionError("unexpected structural lookup scaling")
    if emitted_mutation_steps != list(depths):
        raise AssertionError("unexpected concrete mutation expansion")

    result = {
        "status": STATUS,
        "external_execution": "github_hosted_runner_hidden_seed_same_repository_evaluator",
        "training_depths": [2, 3],
        "heldout_depths": list(depths),
        "heldout_capabilities": capability_trajectory,
        "macro_rule": macro.rule,
        "macro_identifier_free_across_worlds": True,
        "macro_inherited_from_prior_external_outcomes": macro.inherited_from_external_outcomes,
        "current_outcome_required_for_application": macro.current_outcomes_required_for_application,
        "candidate_evaluation_counts": candidate_evaluation_counts,
        "structural_lookup_counts": structural_lookup_counts,
        "emitted_concrete_mutation_steps": emitted_mutation_steps,
        "certificate_compiler_candidate_scan_controls": certificate_compiler_scan_controls,
        "semantic_wrong_certificate_capabilities": semantic_wrong_capabilities,
        "remove_macro_capabilities": remove_capabilities,
        "constant_macro_description_across_depth": True,
        "runtime_application_cost_constant": False,
        "reason_runtime_cost_not_constant": "the learned law is constant-size but materializes one inherited REWIRE mutation per certified failed locus",
        "strict_recursive_meta_productivity_acceleration": False,
        "recursive_acceleration": False,
        "global_recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
        "foundation_weight_change": False,
        "independent_organizational_custody": False,
        "physical_world": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_parametric_morphology_macro_transfer.py <seed-file>")
    raise SystemExit(main(sys.argv[1]))
