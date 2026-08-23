from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

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
from arte_cognition.structural_failure_certificate import StructuralDiagnosticReceipt, derive_structural_failure_certificate


STATUS = "PASS_BOUNDED_REFLECTIVE_FIELD_RELATION_GENESIS_BEYOND_NAMED_ATOMS"
KINDS = (
    OrganKind.REPRESENTATION,
    OrganKind.MEMORY,
    OrganKind.TOOL,
    OrganKind.PERCEPTOR,
    OrganKind.GENERATOR,
)
STEP = 2.75


def token(rng: random.Random, prefix: str) -> str:
    return f"{prefix}_{rng.randrange(10_000_000, 99_999_999)}"


def make_world(seed: int, label: str, depth: int, kind: OrganKind, base_priority: float):
    rng = random.Random((seed << 7) ^ sum(ord(ch) for ch in label))
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
                # Anti-spurious pressure: old[i].cost collides with edge[i+1].priority,
                # so cost equality alone cannot identify a local target.
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
        pressure=PressureVector(transfer_failure=1.0, theory_blindspot=1.0),
        failed_edge_ids=tuple(edge.edge_id for edge in edges),
        source_refs=(f"reflective::{label}",),
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


def capability(genome: MorphologyGenome, hidden: Dict[str, object], target_key: str = "good_targets") -> float:
    edge_map = {edge.edge_id: edge for edge in genome.edges}
    return float(all(edge_map[locus].target == target for locus, target in zip(hidden["edge_ids"], hidden[target_key])))


def externally_select_program(genome: MorphologyGenome, residual: MorphologyResidual, hidden: Dict[str, object], label: str):
    engine = MorphologyGenesisEngine(candidate_budget=4096)
    pool = engine.generate(genome, (residual,))
    if engine.last_truncated:
        raise AssertionError("candidate universe truncated")
    good_by_locus = dict(zip(hidden["edge_ids"], hidden["good_targets"]))
    templates = []
    evaluations = 0
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
            evaluations += 1
            if str(replacement.get("target", "")) == good_by_locus[locus]:
                successful.append(candidate)
        if len(successful) != 1:
            raise AssertionError("external consequence did not identify exactly one successful local mutation")
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
        program_id=f"REFLECTIVE_TRAIN_PROGRAM::{label}",
        templates=tuple(templates),
        inherited_strategy_hash=f"external::{label}",
        generation_uses_current_outcomes=False,
    )
    if capability(apply_mutation_program(genome, program), hidden) != 1.0:
        raise AssertionError("training program failed")
    return program, evaluations


def main(seed_path: str) -> int:
    seed = int(Path(seed_path).read_text().strip())
    rng = random.Random(seed)
    training_kinds = rng.sample(list(KINDS), 2)
    reflective_examples = []
    named_examples = []
    training_evaluations = []

    for index, (depth, kind) in enumerate(zip((2, 3), training_kinds)):
        base = 20.0 + rng.random() * 50.0 + index * 80.0
        label = f"train-{index}"
        genome, residual, hidden = make_world(seed + 10 + index, label, depth, kind, base)
        cert = certificate(label, hidden["edge_ids"])
        program, evaluations = externally_select_program(genome, residual, hidden, label)
        reflective_examples.append(ReflectiveTrainingExample(label, f"external-class-{index}", genome, cert, program, 1.0, True, True))
        named_examples.append(RewriteSchemaTrainingExample(label, f"external-class-{index}", genome, cert, program, 1.0, True, True))
        training_evaluations.append(evaluations)

    named_predecessor = generate_rewrite_schemas(tuple(named_examples))
    if named_predecessor:
        raise AssertionError("named relation-atom predecessor unexpectedly expressed reflective-only distinction")

    schemas = generate_reflective_rewrite_schemas(tuple(reflective_examples))
    if not schemas:
        raise AssertionError("reflective relation generator produced no schema")
    schema = schemas[0]
    relation_tokens = [relation.token() for relation in schema.relations]
    required = {
        "EQ(candidate.cost_hint,edge.priority)",
        "IN(edge.artifact_type,candidate.consumes)",
    }
    if not required.issubset(set(relation_tokens)):
        raise AssertionError(f"reflective generator missed required raw-field relations: {required - set(relation_tokens)}")

    heldout_depths = (4, 8, 16)
    remaining_kinds = [kind for kind in KINDS if kind not in training_kinds] or list(KINDS)
    treatment = []
    remove = []
    wrong = []
    checks = []
    heldout_kinds = []
    heldout_bases = []
    outcome_evaluations = []

    wrong_schema = ReflectiveRewriteSchema(
        schema_id="REFLECTIVE_WRONG",
        operation="REWIRE_EDGE",
        relations=(
            RelationExpression("EQ", FieldRef("candidate", "cost_hint"), FieldRef("source", "cost_hint")),
            RelationExpression("IN", FieldRef("edge", "artifact_type"), FieldRef("candidate", "consumes")),
        ),
        supporting_contexts=schema.supporting_contexts,
        supporting_source_classes=schema.supporting_source_classes,
        supporting_program_ids=schema.supporting_program_ids,
    )

    for generation, depth in enumerate(heldout_depths, start=1):
        kind = remaining_kinds[(generation - 1) % len(remaining_kinds)]
        base = 300.0 + generation * 110.0 + rng.random() * 37.0
        heldout_bases.append(base)
        heldout_kinds.append(kind.value)
        label = f"heldout-{generation}"
        genome, _, hidden = make_world(seed + 1000 * generation, label, depth, kind, base)
        cert = certificate(label, hidden["edge_ids"])
        application = apply_reflective_rewrite_schema(genome, cert, schema)
        if application is None:
            raise AssertionError("reflective schema failed heldout transfer")
        if application.outcome_evaluations != 0:
            raise AssertionError("heldout outcomes leaked into reflective schema application")
        descendant = apply_mutation_program(genome, application.mutation_program)
        effect = capability(descendant, hidden)
        if effect != 1.0:
            raise AssertionError("reflective treatment failed")
        treatment.append(effect)
        remove.append(capability(genome, hidden))
        outcome_evaluations.append(application.outcome_evaluations)
        checks.append(application.candidate_relation_checks)

        wrong_application = apply_reflective_rewrite_schema(genome, cert, wrong_schema)
        wrong_effect = 0.0
        if wrong_application is not None:
            wrong_descendant = apply_mutation_program(genome, wrong_application.mutation_program)
            wrong_effect = capability(wrong_descendant, hidden)
            if capability(wrong_descendant, hidden, "bad_targets") != 1.0:
                raise AssertionError("semantic wrong relation did not select intended local distractor")
        if wrong_effect != 0.0:
            raise AssertionError("semantic wrong reflective relation retained target capability")
        wrong.append(wrong_effect)

    result = {
        "status": STATUS,
        "training_depths": [2, 3],
        "training_kinds": [kind.value for kind in training_kinds],
        "training_candidate_outcome_evaluations": training_evaluations,
        "named_relation_atom_predecessor_candidate_count": len(named_predecessor),
        "generated_schema_id": schema.schema_id,
        "generated_relation_tokens": relation_tokens,
        "heldout_depths": list(heldout_depths),
        "heldout_kinds": heldout_kinds,
        "heldout_priority_bases": heldout_bases,
        "heldout_capabilities": treatment,
        "remove_capabilities": remove,
        "semantic_wrong_capabilities": wrong,
        "heldout_outcome_evaluations_for_generation_or_application": outcome_evaluations,
        "candidate_relation_checks": checks,
        "anti_spurious_cross_locus_collision_present": True,
        "concrete_identifier_values_embedded_in_schema": False,
        "concrete_priority_values_embedded_in_schema": False,
        "named_domain_relation_atoms_supplied": False,
        "dataclass_field_reflection_human_authored": True,
        "generic_relation_operator_vocabulary": ["EQ", "IN"],
        "generic_relation_operator_vocabulary_human_authored": True,
        "rewire_semantics_human_authored": True,
        "current_hidden_outcomes_used_to_generate_schema": False,
        "post_freeze_human_structural_repairs": 0,
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
        raise SystemExit("usage: run_reflective_relation_genesis.py <seed-file>")
    raise SystemExit(main(sys.argv[1]))
