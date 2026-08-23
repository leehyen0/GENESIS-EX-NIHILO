from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

from arte_cognition.executable_morphology import EdgeSpec, MorphologyGenome, MutationLevel, OrganKind, OrganSpec, PressureVector
from arte_cognition.meta_acceleration import MutationProgram, MutationTemplate, apply_mutation_program
from arte_cognition.morphology_genesis import MorphologyGenesisEngine, MorphologyResidual
from arte_cognition.morphology_rewrite_schema_genesis import (
    EDGE_FIELDS,
    GeneratedRewriteSchema,
    RewriteSchemaTrainingExample,
    apply_generated_rewrite_schema,
    generate_rewrite_schemas,
)
from arte_cognition.parametric_morphology_macro import ParametricMorphologyMacro, apply_parametric_macro
from arte_cognition.structural_failure_certificate import StructuralDiagnosticReceipt, derive_structural_failure_certificate


STATUS = "PASS_BOUNDED_WORLD_SELECTED_RELATIONAL_MORPHOLOGY_REWRITE_SCHEMA_GENESIS"
KINDS = (
    OrganKind.REPRESENTATION,
    OrganKind.MEMORY,
    OrganKind.TOOL,
    OrganKind.PERCEPTOR,
    OrganKind.GENERATOR,
)


def token(rng: random.Random, prefix: str) -> str:
    return f"{prefix}_{rng.randrange(10_000_000, 99_999_999)}"


def make_world(seed: int, label: str, depth: int, old_kind: OrganKind):
    rng = random.Random((seed << 11) ^ sum(ord(ch) for ch in label))
    organs = []
    edges = []
    correct = []
    distractors = []
    for index in range(depth):
        artifact = token(rng, f"artifact{index}")
        source = token(rng, f"source{index}")
        old = token(rng, f"old{index}")
        good = token(rng, f"good{index}")
        bad = token(rng, f"bad{index}")
        edge_id = token(rng, f"edge{index}")
        distractor_kind = next(kind for kind in KINDS if kind != old_kind)
        organs.extend(
            (
                OrganSpec(source, OrganKind.SOURCE, produces=(artifact,)),
                OrganSpec(old, old_kind, consumes=(artifact,), produces=(token(rng, "oldout"),)),
                OrganSpec(good, old_kind, consumes=(artifact,), produces=(token(rng, "goodout"),)),
                OrganSpec(bad, distractor_kind, consumes=(artifact,), produces=(token(rng, "badout"),)),
            )
        )
        edges.append(EdgeSpec(edge_id, source, old, artifact))
        correct.append(good)
        distractors.append(bad)
    organs.extend((OrganSpec(token(rng, "governor"), OrganKind.GOVERNOR), OrganSpec(token(rng, "archive"), OrganKind.ARCHIVE)))
    genome = MorphologyGenome(tuple(organs), tuple(edges), tuple(organ.organ_id for organ in organs))
    residual = MorphologyResidual(
        residual_id=token(rng, "residual"),
        pressure=PressureVector(transfer_failure=1.0, human_dependency=1.0, theory_blindspot=1.0),
        failed_edge_ids=tuple(edge.edge_id for edge in edges),
        source_refs=(f"hidden::{label}",),
    )
    hidden = {
        "edge_ids": tuple(edge.edge_id for edge in edges),
        "correct_targets": tuple(correct),
        "distractor_targets": tuple(distractors),
        "old_kind": old_kind.value,
    }
    return genome, residual, hidden


def certificate(label: str, loci: Sequence[str]):
    receipts = (
        StructuralDiagnosticReceipt(f"{label}::a", label, f"{label}::class-a", tuple(loci), True, True),
        StructuralDiagnosticReceipt(f"{label}::b", label, f"{label}::class-b", tuple(reversed(tuple(loci))), True, True),
    )
    out = derive_structural_failure_certificate(receipts)
    if out is None:
        raise AssertionError("certificate derivation failed")
    return out


def capability(genome: MorphologyGenome, hidden: Dict[str, object]) -> float:
    edge_map = {edge.edge_id: edge for edge in genome.edges}
    return float(
        all(
            edge_map[edge_id].target == target
            for edge_id, target in zip(hidden["edge_ids"], hidden["correct_targets"])
        )
    )


def candidate_pool(genome: MorphologyGenome, residual: MorphologyResidual):
    engine = MorphologyGenesisEngine(candidate_budget=4096)
    pool = engine.generate(genome, (residual,))
    if engine.last_truncated:
        raise AssertionError("candidate universe truncated")
    return pool


def externally_select_successful_program(
    genome: MorphologyGenome,
    residual: MorphologyResidual,
    hidden: Dict[str, object],
    label: str,
) -> Tuple[MutationProgram, int]:
    """External evaluator selects from a frozen outcome-free candidate population."""
    pool = candidate_pool(genome, residual)
    correct = dict(zip(hidden["edge_ids"], hidden["correct_targets"]))
    selected = []
    evaluations = 0
    for locus in hidden["edge_ids"]:
        matches = []
        for candidate in pool:
            if candidate.operation_family != "REWIRE_EDGE":
                continue
            payload = dict(candidate.mutation.payload)
            if str(payload.get("edge_id", "")) != locus:
                continue
            replacement = payload.get("edge")
            if not isinstance(replacement, dict):
                continue
            evaluations += 1
            # The consequence/ground truth is evaluator-owned and used only to
            # select a prior successful training repair, never heldout generation.
            if str(replacement.get("target", "")) == correct[locus]:
                matches.append(candidate)
        if len(matches) != 1:
            raise AssertionError(f"external selector expected one successful local rewire at {locus}")
        mutation = matches[0].mutation
        selected.append(
            MutationTemplate(
                operation=mutation.operation,
                level=mutation.level,
                payload=dict(mutation.payload),
                rationale=(f"external-world-selected::{label}",),
                source_candidate_id=matches[0].candidate_id,
            )
        )
    program = MutationProgram(
        program_id=f"WORLD_SELECTED_PROGRAM::{label}",
        templates=tuple(selected),
        inherited_strategy_hash=f"external-selection::{label}",
        generation_uses_current_outcomes=False,
    )
    if capability(apply_mutation_program(genome, program), hidden) != 1.0:
        raise AssertionError("world-selected training program did not solve training world")
    return program, evaluations


def main(seed_path: str) -> int:
    seed = int(Path(seed_path).read_text().strip())
    rng = random.Random(seed)
    training_kinds = rng.sample(list(KINDS), 2)
    examples = []
    training_candidate_evaluations = []

    for index, (depth, kind) in enumerate(zip((2, 3), training_kinds)):
        label = f"train-{index}"
        genome, residual, hidden = make_world(seed + 100 + index, label, depth, kind)
        cert = certificate(label, hidden["edge_ids"])
        program, evaluations = externally_select_successful_program(genome, residual, hidden, label)
        examples.append(
            RewriteSchemaTrainingExample(
                context_id=label,
                source_class=f"external-training-class-{index}",
                genome=genome,
                certificate=cert,
                successful_program=program,
                external_capability=1.0,
                authority_verified=True,
                benchmark_disjoint=True,
            )
        )
        training_candidate_evaluations.append(evaluations)

    schemas = generate_rewrite_schemas(tuple(examples))
    if not schemas:
        raise AssertionError("no rewrite schema generated from prior external successes")
    schema = schemas[0]
    if "SAME_KIND_AS_OLD_TARGET" not in schema.target_predicates:
        raise AssertionError("generated schema failed to discover relative kind relation")
    if "CONSUMES_EDGE_ARTIFACT" not in schema.target_predicates:
        raise AssertionError("generated schema failed to discover artifact-consumption relation")

    predecessor = ParametricMorphologyMacro(
        macro_id="frozen-predecessor",
        rule="FOR_EACH_CERTIFIED_FAILED_EDGE_REWIRE_TO_UNIQUE_COMPATIBLE_ALTERNATIVE",
        supporting_contexts=("old-a", "old-b"),
        supporting_source_classes=("old-class-a", "old-class-b"),
        training_program_ids=("old-p-a", "old-p-b"),
    )

    heldout_depths = (4, 8, 16)
    heldout_capabilities = []
    predecessor_capabilities = []
    predecessor_applicable = []
    heldout_outcome_evaluations_for_generation = []
    structural_checks = []
    wrong_capabilities = []
    remove_capabilities = []
    heldout_kinds = []

    remaining_kinds = [kind for kind in KINDS if kind not in training_kinds] or list(KINDS)
    for generation, depth in enumerate(heldout_depths, start=1):
        kind = remaining_kinds[(generation - 1) % len(remaining_kinds)]
        heldout_kinds.append(kind.value)
        label = f"heldout-{generation}"
        genome, residual, hidden = make_world(seed + 1000 * generation, label, depth, kind)
        cert = certificate(label, hidden["edge_ids"])

        old_application = apply_parametric_macro(genome, cert, predecessor)
        predecessor_applicable.append(old_application is not None)
        old_effect = 0.0
        if old_application is not None:
            old_effect = capability(apply_mutation_program(genome, old_application.mutation_program), hidden)
        predecessor_capabilities.append(old_effect)
        if old_effect != 0.0:
            raise AssertionError("frozen unique-compatible predecessor unexpectedly solved ambiguous world")

        application = apply_generated_rewrite_schema(genome, cert, schema)
        if application is None:
            raise AssertionError("generated relational schema failed fresh heldout transfer")
        if application.outcome_evaluations != 0:
            raise AssertionError("heldout outcomes leaked into schema application")
        descendant = apply_mutation_program(genome, application.mutation_program)
        effect = capability(descendant, hidden)
        if effect != 1.0:
            raise AssertionError("generated schema descendant failed heldout world")
        heldout_capabilities.append(effect)
        heldout_outcome_evaluations_for_generation.append(application.outcome_evaluations)
        structural_checks.append(application.structural_candidate_checks)
        remove_capabilities.append(capability(genome, hidden))

        wrong = GeneratedRewriteSchema(
            schema_id=f"SEMANTIC_WRONG::{generation}",
            operation=schema.operation,
            target_predicates=("CONSUMES_EDGE_ARTIFACT", "DIFFERENT_KIND_FROM_OLD_TARGET"),
            preserved_edge_fields=EDGE_FIELDS,
            supporting_contexts=schema.supporting_contexts,
            supporting_source_classes=schema.supporting_source_classes,
            supporting_program_ids=schema.supporting_program_ids,
        )
        wrong_application = apply_generated_rewrite_schema(genome, cert, wrong)
        wrong_effect = 0.0
        if wrong_application is not None:
            wrong_effect = capability(apply_mutation_program(genome, wrong_application.mutation_program), hidden)
        wrong_capabilities.append(wrong_effect)
        if wrong_effect != 0.0:
            raise AssertionError("semantic wrong relational schema retained capability")

    result = {
        "status": STATUS,
        "external_execution": "github_hosted_runner_hidden_seed_same_repository_evaluator",
        "training_depths": [2, 3],
        "training_old_target_kinds": [kind.value for kind in training_kinds],
        "training_candidate_outcome_evaluations": training_candidate_evaluations,
        "generated_schema_id": schema.schema_id,
        "generated_target_predicates": list(schema.target_predicates),
        "generated_operation": schema.operation,
        "heldout_depths": list(heldout_depths),
        "heldout_old_target_kinds": heldout_kinds,
        "heldout_capabilities": heldout_capabilities,
        "heldout_outcome_evaluations_for_schema_generation_or_application": heldout_outcome_evaluations_for_generation,
        "frozen_unique_compatible_predecessor_applicable": predecessor_applicable,
        "frozen_unique_compatible_predecessor_capabilities": predecessor_capabilities,
        "semantic_wrong_relational_schema_capabilities": wrong_capabilities,
        "remove_generated_schema_capabilities": remove_capabilities,
        "structural_candidate_checks": structural_checks,
        "current_hidden_outcomes_used_to_generate_schema": False,
        "relation_atom_vocabulary_human_authored": True,
        "unique_match_interpreter_human_authored": True,
        "rewire_operation_semantics_human_authored": True,
        "rewrite_schema_itself_generated_from_prior_external_successes": True,
        "post_freeze_human_structural_repairs": 0,
        "strict_recursive_meta_productivity_acceleration": False,
        "recursive_acceleration": False,
        "global_recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
        "foundation_weight_change": False,
        "physical_world": False,
        "independent_organizational_custody": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_generated_morphology_rewrite_schema.py <seed-file>")
    raise SystemExit(main(sys.argv[1]))
