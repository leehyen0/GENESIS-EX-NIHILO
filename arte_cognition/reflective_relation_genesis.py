from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from itertools import combinations
from typing import Dict, Mapping, Optional, Sequence, Tuple
import hashlib
import json

from .executable_morphology import MorphologyGenome, MutationLevel
from .meta_acceleration import MutationProgram, MutationTemplate
from .structural_failure_certificate import StructuralFailureCertificate


IDENTIFIER_FIELDS = {"organ_id", "edge_id"}
CONTEXT_ROLES = ("old_target", "edge", "source")
GENERIC_RELATION_OPERATORS = ("EQ", "IN")


@dataclass(frozen=True, order=True)
class FieldRef:
    role: str
    field_name: str

    def token(self) -> str:
        return f"{self.role}.{self.field_name}"


@dataclass(frozen=True, order=True)
class RelationExpression:
    operator: str
    left: FieldRef
    right: FieldRef

    def token(self) -> str:
        return f"{self.operator}({self.left.token()},{self.right.token()})"


@dataclass(frozen=True)
class ReflectiveTrainingExample:
    context_id: str
    source_class: str
    genome: MorphologyGenome
    certificate: StructuralFailureCertificate
    successful_program: MutationProgram
    external_capability: float
    authority_verified: bool
    benchmark_disjoint: bool


@dataclass(frozen=True)
class ReflectiveRewriteSchema:
    schema_id: str
    operation: str
    relations: Tuple[RelationExpression, ...]
    supporting_contexts: Tuple[str, ...]
    supporting_source_classes: Tuple[str, ...]
    supporting_program_ids: Tuple[str, ...]
    current_outcomes_required_for_application: bool = False


@dataclass(frozen=True)
class ReflectiveSchemaApplication:
    schema_id: str
    certificate_id: str
    mutation_program: MutationProgram
    candidate_relation_checks: int
    outcome_evaluations: int = 0


def _edge_map(genome: MorphologyGenome):
    return {edge.edge_id: edge for edge in genome.edges}


def _role_objects(genome: MorphologyGenome, edge_id: str, candidate_id: str):
    edges = _edge_map(genome)
    organs = genome.organ_map()
    edge = edges.get(edge_id)
    if edge is None or candidate_id not in organs:
        return None
    old_target = organs.get(edge.target)
    source = organs.get(edge.source)
    if old_target is None or source is None:
        return None
    return {
        "candidate": organs[candidate_id],
        "old_target": old_target,
        "edge": edge,
        "source": source,
    }


def _field_refs_for(role: str, obj: object) -> Tuple[FieldRef, ...]:
    if not is_dataclass(obj):
        return ()
    return tuple(
        FieldRef(role, item.name)
        for item in fields(obj)
        if item.name not in IDENTIFIER_FIELDS
    )


def _value(objects: Mapping[str, object], ref: FieldRef):
    obj = objects.get(ref.role)
    if obj is None or not hasattr(obj, ref.field_name):
        raise KeyError(ref.token())
    return getattr(obj, ref.field_name)


def _is_container(value: object) -> bool:
    return isinstance(value, (tuple, list, set, frozenset))


def relation_holds(
    genome: MorphologyGenome,
    edge_id: str,
    candidate_id: str,
    expression: RelationExpression,
) -> bool:
    objects = _role_objects(genome, edge_id, candidate_id)
    if objects is None:
        return False
    try:
        left = _value(objects, expression.left)
        right = _value(objects, expression.right)
    except KeyError:
        return False
    if expression.operator == "EQ":
        return left == right
    if expression.operator == "IN":
        if not _is_container(right):
            return False
        try:
            return left in right
        except TypeError:
            return False
    raise ValueError(f"unsupported generic relation operator: {expression.operator}")


def _candidate_targets(genome: MorphologyGenome, edge_id: str) -> Tuple[str, ...]:
    edge = _edge_map(genome).get(edge_id)
    if edge is None:
        return ()
    return tuple(
        sorted(
            organ.organ_id
            for organ in genome.organs
            if organ.enabled and organ.organ_id not in {edge.source, edge.target}
        )
    )


def _extract_rows(example: ReflectiveTrainingExample):
    edges = _edge_map(example.genome)
    required = set(example.certificate.failed_locus_ids)
    rows = []
    seen = set()
    for template in example.successful_program.templates:
        if template.operation != "REWIRE_EDGE":
            return ()
        payload = dict(template.payload)
        locus = str(payload.get("edge_id", ""))
        replacement = payload.get("edge")
        if not locus or locus in seen or locus not in required or locus not in edges:
            return ()
        if not isinstance(replacement, Mapping):
            return ()
        target = str(replacement.get("target", ""))
        if not target or target == edges[locus].target:
            return ()
        rows.append((example.genome, locus, target))
        seen.add(locus)
    if seen != required:
        return ()
    return tuple(rows)


def _enumerate_generic_expressions(genome: MorphologyGenome, edge_id: str, target_id: str) -> Tuple[RelationExpression, ...]:
    objects = _role_objects(genome, edge_id, target_id)
    if objects is None:
        return ()
    candidate_refs = _field_refs_for("candidate", objects["candidate"])
    context_refs = tuple(
        ref
        for role in CONTEXT_ROLES
        for ref in _field_refs_for(role, objects[role])
    )
    expressions = []
    for left in candidate_refs:
        for right in context_refs:
            expressions.append(RelationExpression("EQ", left, right))
            try:
                right_value = _value(objects, right)
            except KeyError:
                continue
            if _is_container(right_value):
                expressions.append(RelationExpression("IN", left, right))
    return tuple(sorted(set(expressions), key=lambda item: item.token()))


def _uniquely_selects(
    genome: MorphologyGenome,
    edge_id: str,
    target_id: str,
    relations: Sequence[RelationExpression],
) -> bool:
    selected = tuple(
        candidate_id
        for candidate_id in _candidate_targets(genome, edge_id)
        if all(relation_holds(genome, edge_id, candidate_id, relation) for relation in relations)
    )
    return selected == (target_id,)


def generate_reflective_rewrite_schemas(
    examples: Sequence[ReflectiveTrainingExample],
    *,
    min_contexts: int = 2,
    min_source_classes: int = 2,
    max_relation_count: int = 3,
) -> Tuple[ReflectiveRewriteSchema, ...]:
    usable = [
        example
        for example in examples
        if example.external_capability > 0.0
        and example.authority_verified
        and example.benchmark_disjoint
        and example.certificate.authority_verified
    ]
    contexts = tuple(sorted({example.context_id for example in usable}))
    source_classes = tuple(sorted({example.source_class for example in usable if example.source_class and example.source_class != "UNVERIFIED"}))
    if len(contexts) < max(1, int(min_contexts)) or len(source_classes) < max(1, int(min_source_classes)):
        return ()

    rows = []
    program_ids = []
    for example in usable:
        extracted = _extract_rows(example)
        if not extracted:
            return ()
        rows.extend(extracted)
        program_ids.append(example.successful_program.program_id)
    if not rows:
        return ()

    first_genome, first_locus, first_target = rows[0]
    universe = _enumerate_generic_expressions(first_genome, first_locus, first_target)
    common_discriminating = []
    for expression in universe:
        if not all(relation_holds(genome, locus, target, expression) for genome, locus, target in rows):
            continue
        # Do not retain a relation that is true for every alternative in every
        # training row. It must contribute observable structural discrimination.
        if not any(
            any(
                not relation_holds(genome, locus, candidate_id, expression)
                for candidate_id in _candidate_targets(genome, locus)
                if candidate_id != target
            )
            for genome, locus, target in rows
        ):
            continue
        common_discriminating.append(expression)

    valid = []
    max_size = min(max(1, int(max_relation_count)), len(common_discriminating))
    for size in range(1, max_size + 1):
        for subset in combinations(common_discriminating, size):
            if all(_uniquely_selects(genome, locus, target, subset) for genome, locus, target in rows):
                valid.append(tuple(subset))
        if valid:
            break
    if not valid:
        return ()

    # Behavioral quotient on training selections, then stable lexical choice.
    by_signature: Dict[Tuple[Tuple[str, ...], ...], Tuple[RelationExpression, ...]] = {}
    for relations in valid:
        signature = tuple(
            tuple(
                candidate_id
                for candidate_id in _candidate_targets(genome, locus)
                if all(relation_holds(genome, locus, candidate_id, relation) for relation in relations)
            )
            for genome, locus, _ in rows
        )
        incumbent = by_signature.get(signature)
        if incumbent is None or tuple(item.token() for item in relations) < tuple(item.token() for item in incumbent):
            by_signature[signature] = relations

    schemas = []
    for relations in sorted(by_signature.values(), key=lambda rels: tuple(item.token() for item in rels)):
        payload = {
            "operation": "REWIRE_EDGE",
            "relations": [relation.token() for relation in relations],
            "contexts": contexts,
            "classes": source_classes,
            "programs": sorted(set(program_ids)),
        }
        schema_id = "REFLECTIVE_REWRITE_SCHEMA::" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:20]
        schemas.append(
            ReflectiveRewriteSchema(
                schema_id=schema_id,
                operation="REWIRE_EDGE",
                relations=relations,
                supporting_contexts=contexts,
                supporting_source_classes=source_classes,
                supporting_program_ids=tuple(sorted(set(program_ids))),
            )
        )
    return tuple(schemas)


def apply_reflective_rewrite_schema(
    genome: MorphologyGenome,
    certificate: StructuralFailureCertificate,
    schema: ReflectiveRewriteSchema,
) -> Optional[ReflectiveSchemaApplication]:
    if not certificate.authority_verified or schema.operation != "REWIRE_EDGE":
        return None
    edges = _edge_map(genome)
    templates = []
    checks = 0
    for locus in certificate.failed_locus_ids:
        edge = edges.get(locus)
        if edge is None:
            return None
        candidates = _candidate_targets(genome, locus)
        selected = []
        for candidate_id in candidates:
            checks += len(schema.relations)
            if all(relation_holds(genome, locus, candidate_id, relation) for relation in schema.relations):
                selected.append(candidate_id)
        if len(selected) != 1:
            return None
        replacement = {
            "edge_id": edge.edge_id,
            "source": edge.source,
            "target": selected[0],
            "artifact_type": edge.artifact_type,
            "authority_required": edge.authority_required,
            "gate": edge.gate,
            "priority": edge.priority,
        }
        templates.append(
            MutationTemplate(
                operation="REWIRE_EDGE",
                level=MutationLevel.TOPOLOGY,
                payload={"edge_id": edge.edge_id, "edge": replacement},
                rationale=(f"reflective-schema::{schema.schema_id}", f"certificate::{certificate.certificate_id}"),
                source_candidate_id="",
            )
        )
    raw = json.dumps(
        {
            "schema": schema.schema_id,
            "certificate": certificate.certificate_id,
            "templates": [template.fingerprint() for template in templates],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    program = MutationProgram(
        program_id="REFLECTIVE_REWRITE_PROGRAM::" + hashlib.sha256(raw).hexdigest()[:20],
        templates=tuple(templates),
        inherited_strategy_hash=schema.schema_id,
        generation_uses_current_outcomes=False,
    )
    return ReflectiveSchemaApplication(
        schema_id=schema.schema_id,
        certificate_id=certificate.certificate_id,
        mutation_program=program,
        candidate_relation_checks=checks,
        outcome_evaluations=0,
    )
