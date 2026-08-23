from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, Sequence

from arte_cognition.executable_morphology import EdgeSpec, MorphologyGenome, MutationLevel, OrganKind, OrganSpec, PressureVector
from arte_cognition.meta_acceleration import MutationProgram, MutationTemplate, apply_mutation_program
from arte_cognition.morphology_genesis import MorphologyGenesisEngine, MorphologyResidual
from arte_cognition.morphology_rewrite_schema_genesis import RewriteSchemaTrainingExample, generate_rewrite_schemas
from arte_cognition.reflective_relation_genesis import (
    FieldRef,
    ReflectiveRewriteSchema,
    ReflectiveTrainingExample,
    RelationExpression,
    apply_reflective_rewrite_schema,
    generate_reflective_rewrite_schemas,
)
from arte_cognition.relation_access_plan_genesis import apply_relation_access_plan, compile_relation_access_plan
from arte_cognition.structural_failure_certificate import StructuralDiagnosticReceipt, derive_structural_failure_certificate


STATUS = "PASS_BOUNDED_GENERATED_RELATION_ACCESS_PLAN_AND_SCALING_TRANSFER"
KINDS = (
    OrganKind.REPRESENTATION,
    OrganKind.MEMORY,
    OrganKind.TOOL,
    OrganKind.PERCEPTOR,
    OrganKind.GENERATOR,
)
STEP = 2.875


def token(rng: random.Random, prefix: str) -> str:
    return f"{prefix}_{rng.randrange(10_000_000, 99_999_999)}"


def make_world(seed: int, label: str, depth: int, kind: OrganKind, base_priority: float):
    rng = random.Random((seed << 5) ^ sum(ord(ch) for ch in label))
    organs = []
    edges = []
    good_targets = []
    bad_targets = []
    for index in range(depth):
        artifact = token(rng, f"artifact{index}")
        source = token(rng, f"source{index}")
        old = token(rng, f"old{index}")
        good = token(rng, f"good{index}")
        bad = token(rng, f"bad{index}")
        edge_id = token(rng, f"edge{index}")
        priority = base_priority + STEP * index
        shared_output = (token(rng, f"sharedout{index}"),)
        shared_impl = token(rng, f"impl{index}")
        source_cost = priority + 0.5
        organs.extend(
            (
                OrganSpec(source, OrganKind.SOURCE, produces=(artifact,), cost_hint=source_cost),
                OrganSpec(old, kind, consumes=(artifact,), produces=shared_output, implementation_ref=shared_impl, version=2, cost_hint=priority + STEP),
                OrganSpec(good, kind, consumes=(artifact,), produces=shared_output, implementation_ref=shared_impl, version=2, cost_hint=priority),
                OrganSpec(bad, kind, consumes=(artifact,), produces=shared_output, implementation_ref=shared_impl, version=2, cost_hint=source_cost),
            )
        )
        edges.append(EdgeSpec(edge_id, source, old, artifact, priority=priority))
        good_targets.append(good)
        bad_targets.append(bad)
    organs.extend((OrganSpec(token(rng, "governor"), OrganKind.GOVERNOR), OrganSpec(token(rng, "archive"), OrganKind.ARCHIVE)))
    genome = MorphologyGenome(tuple(organs), tuple(edges), tuple(organ.organ_id for organ in organs))
    residual = MorphologyResidual(
        residual_id=token(rng, "residual"),
        pressure=PressureVector(transfer_failure=1.0, efficiency_pressure=1.0, theory_blindspot=1.0),
        failed_edge_ids=tuple(edge.edge_id for edge in edges),
        source_refs=(f"access-plan::{label}",),
    )
    hidden = {
        "edge_ids": tuple(edge.edge_id for edge in edges),
        "good_targets": tuple(good_targets),
        "bad_targets": tuple(bad_targets),
        "kind": kind.value,
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


def capability(genome: MorphologyGenome, hidden: Dict[str, object], key: str = "good_targets") -> float:
    edge_map = {edge.edge_id: edge for edge in genome.edges}
    return float(all(edge_map[locus].target == target for locus, target in zip(hidden["edge_ids"], hidden[key])))


def externally_select_program(genome: MorphologyGenome, residual: MorphologyResidual, hidden: Dict[str, object], label: str):
    engine = MorphologyGenesisEngine(candidate_budget=4096)
    pool = engine.generate(genome, (residual,))
    if engine.last_truncated:
        raise AssertionError("candidate universe truncated")
    good_by_locus = dict(zip(hidden["edge_ids"], hidden["good_targets"]))
    templates = []
    external_evaluations = 0
    for locus in hidden["edge_ids"]:
        successful = []
        for candidate in pool:
            if candidate.operation_family != "REWIRE_EDGE":
                continue
            payload = dict(candidate.mutation.payload)
            if str(payload.get("edge_id", "")) != locus:
                continue
            replacement = payload.get("edge")
            if not isinstance(replacement, dict):
                continue
            external_evaluations += 1
            if str(replacement.get("target", "")) == good_by_locus[locus]:
                successful.append(candidate)
        if len(successful) != 1:
            raise AssertionError("external training outcome did not isolate one local rewire")
        mutation = successful[0].mutation
        templates.append(
            MutationTemplate(
                operation=mutation.operation,
                level=mutation.level,
                payload=dict(mutation.payload),
                rationale=(f"external-success::{label}",),
                source_candidate_id=successful[0].candidate_id,
            )
        )
    program = MutationProgram(
        program_id=f"ACCESS_PLAN_TRAIN_PROGRAM::{label}",
        templates=tuple(templates),
        inherited_strategy_hash=f"external::{label}",
        generation_uses_current_outcomes=False,
    )
    if capability(apply_mutation_program(genome, program), hidden) != 1.0:
        raise AssertionError("training program failed")
    return program, external_evaluations


def semantic_wrong_schema(schema: ReflectiveRewriteSchema) -> ReflectiveRewriteSchema:
    relations = []
    replaced = 0
    for relation in schema.relations:
        if relation.token() == "EQ(candidate.cost_hint,edge.priority)":
            relations.append(RelationExpression("EQ", FieldRef("candidate", "cost_hint"), FieldRef("source", "cost_hint")))
            replaced += 1
        else:
            relations.append(relation)
    if replaced != 1:
        raise AssertionError("expected one cost relation to wrong-swap")
    return ReflectiveRewriteSchema(
        schema_id="ACCESS_PLAN_WRONG_SCHEMA",
        operation=schema.operation,
        relations=tuple(relations),
        supporting_contexts=schema.supporting_contexts,
        supporting_source_classes=schema.supporting_source_classes,
        supporting_program_ids=schema.supporting_program_ids,
    )


def main(seed_path: str) -> int:
    seed = int(Path(seed_path).read_text().strip())
    rng = random.Random(seed)
    training_kinds = rng.sample(list(KINDS), 2)
    reflective_examples = []
    named_examples = []
    training_external_evaluations = []

    for index, (depth, kind) in enumerate(zip((2, 3), training_kinds)):
        base = 17.0 + rng.random() * 41.0 + index * 91.0
        label = f"train-{index}"
        genome, residual, hidden = make_world(seed + 31 + index, label, depth, kind, base)
        cert = certificate(label, hidden["edge_ids"])
        program, evaluations = externally_select_program(genome, residual, hidden, label)
        reflective_examples.append(ReflectiveTrainingExample(label, f"external-class-{index}", genome, cert, program, 1.0, True, True))
        named_examples.append(RewriteSchemaTrainingExample(label, f"external-class-{index}", genome, cert, program, 1.0, True, True))
        training_external_evaluations.append(evaluations)

    if generate_rewrite_schemas(tuple(named_examples)):
        raise AssertionError("named relation predecessor unexpectedly expressive")
    schemas = generate_reflective_rewrite_schemas(tuple(reflective_examples))
    if not schemas:
        raise AssertionError("reflective schema generation failed")
    schema = schemas[0]
    if len(schema.relations) < 2 or "EQ(candidate.cost_hint,edge.priority)" not in {r.token() for r in schema.relations}:
        raise AssertionError("reflective schema does not contain the required causally tested structure")
    plan = compile_relation_access_plan(schema)
    if plan is None:
        raise AssertionError("generated reflective schema could not compile to an access plan")
    if plan.current_outcomes_required_for_compilation:
        raise AssertionError("current outcomes leaked into access-plan compilation")

    wrong_schema = semantic_wrong_schema(schema)
    wrong_plan = compile_relation_access_plan(wrong_schema)
    if wrong_plan is None:
        raise AssertionError("semantic wrong schema should remain structurally compilable")

    heldout_depths = (4, 8, 16, 32)
    remaining_kinds = [kind for kind in KINDS if kind not in training_kinds] or list(KINDS)
    scan_checks = []
    indexed_total_work = []
    index_build_reads = []
    index_lookups = []
    intersections = []
    speedups = []
    scan_capabilities = []
    indexed_capabilities = []
    wrong_capabilities = []
    remove_access_plan_capabilities = []
    heldout_outcome_evaluations = []

    for generation, depth in enumerate(heldout_depths, start=1):
        kind = remaining_kinds[(generation - 1) % len(remaining_kinds)]
        base = 500.0 + generation * 137.0 + rng.random() * 53.0
        label = f"heldout-{generation}"
        genome, _, hidden = make_world(seed + 2000 * generation, label, depth, kind, base)
        cert = certificate(label, hidden["edge_ids"])

        scan = apply_reflective_rewrite_schema(genome, cert, schema)
        indexed = apply_relation_access_plan(genome, cert, schema, plan)
        if scan is None or indexed is None:
            raise AssertionError("scan or indexed treatment became inapplicable")
        scan_descendant = apply_mutation_program(genome, scan.mutation_program)
        indexed_descendant = apply_mutation_program(genome, indexed.mutation_program)
        scan_effect = capability(scan_descendant, hidden)
        indexed_effect = capability(indexed_descendant, hidden)
        if scan_effect != 1.0 or indexed_effect != 1.0:
            raise AssertionError("capability was not preserved by generated access plan")

        scan_work = scan.candidate_relation_checks
        indexed_work = indexed.index_build_field_reads + indexed.index_lookup_count + indexed.candidate_intersection_count
        if indexed_work >= scan_work:
            raise AssertionError("generated access plan did not reduce total structural work")
        scan_checks.append(scan_work)
        indexed_total_work.append(indexed_work)
        index_build_reads.append(indexed.index_build_field_reads)
        index_lookups.append(indexed.index_lookup_count)
        intersections.append(indexed.candidate_intersection_count)
        speedups.append(scan_work / indexed_work)
        scan_capabilities.append(scan_effect)
        indexed_capabilities.append(indexed_effect)
        remove_access_plan_capabilities.append(scan_effect)
        heldout_outcome_evaluations.append(indexed.outcome_evaluations)

        wrong_applied = apply_relation_access_plan(genome, cert, wrong_schema, wrong_plan)
        if wrong_applied is None:
            raise AssertionError("semantic wrong access plan unexpectedly inapplicable")
        wrong_descendant = apply_mutation_program(genome, wrong_applied.mutation_program)
        wrong_effect = capability(wrong_descendant, hidden)
        if wrong_effect != 0.0 or capability(wrong_descendant, hidden, "bad_targets") != 1.0:
            raise AssertionError("semantic wrong plan did not lose target capability")
        wrong_capabilities.append(wrong_effect)

        if apply_relation_access_plan(genome, cert, wrong_schema, plan) is not None:
            raise AssertionError("schema/plan mismatch did not fail closed")

    if not all(b > a for a, b in zip(speedups, speedups[1:])):
        raise AssertionError("access-plan speedup did not strictly improve with morphology scale")

    result = {
        "status": STATUS,
        "training_depths": [2, 3],
        "training_external_candidate_evaluations": training_external_evaluations,
        "generated_relation_tokens": [relation.token() for relation in schema.relations],
        "generated_access_plan_id": plan.plan_id,
        "generated_access_clauses": [clause.token() for clause in plan.clauses],
        "heldout_depths": list(heldout_depths),
        "scan_relation_checks": scan_checks,
        "indexed_total_structural_work": indexed_total_work,
        "index_build_field_reads": index_build_reads,
        "index_lookup_counts": index_lookups,
        "candidate_intersection_counts": intersections,
        "scan_to_indexed_speedup": speedups,
        "strict_speedup_growth_with_scale": True,
        "scan_capabilities": scan_capabilities,
        "indexed_capabilities": indexed_capabilities,
        "remove_access_plan_capabilities": remove_access_plan_capabilities,
        "semantic_wrong_capabilities": wrong_capabilities,
        "heldout_outcome_evaluations_for_access_plan": heldout_outcome_evaluations,
        "current_hidden_outcomes_used_to_compile_access_plan": False,
        "access_plan_compiled_from_generated_relation_structure": True,
        "index_compiler_rules_human_authored": True,
        "generic_relation_operators_human_authored": True,
        "same_frontier_cheaper_execution_not_frontier_growth": True,
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
        raise SystemExit("usage: run_relation_access_plan_genesis.py <seed-file>")
    raise SystemExit(main(sys.argv[1]))
