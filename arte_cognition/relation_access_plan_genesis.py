from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple
import hashlib
import json

from .executable_morphology import MorphologyGenome, MutationLevel
from .meta_acceleration import MutationProgram, MutationTemplate
from .reflective_relation_genesis import (
    FieldRef,
    ReflectiveRewriteSchema,
    RelationExpression,
)
from .structural_failure_certificate import StructuralFailureCertificate


@dataclass(frozen=True)
class AccessClause:
    mode: str
    candidate_field: str
    context_ref: FieldRef
    source_relation_token: str

    def token(self) -> str:
        return f"{self.mode}::{self.candidate_field}::{self.context_ref.token()}"


@dataclass(frozen=True)
class RelationAccessPlan:
    plan_id: str
    schema_id: str
    clauses: Tuple[AccessClause, ...]
    generated_from_schema_structure: bool = True
    current_outcomes_required_for_compilation: bool = False


@dataclass(frozen=True)
class AccessPlanApplication:
    plan_id: str
    schema_id: str
    certificate_id: str
    mutation_program: MutationProgram
    index_build_field_reads: int
    index_lookup_count: int
    candidate_intersection_count: int
    outcome_evaluations: int = 0


def _freeze(value: object):
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_freeze(item) for item in value))
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _context_objects(genome: MorphologyGenome, edge_id: str):
    edge_map = {edge.edge_id: edge for edge in genome.edges}
    organs = genome.organ_map()
    edge = edge_map.get(edge_id)
    if edge is None:
        return None
    old_target = organs.get(edge.target)
    source = organs.get(edge.source)
    if old_target is None or source is None:
        return None
    return {"edge": edge, "old_target": old_target, "source": source}


def _context_value(objects: Mapping[str, object], ref: FieldRef):
    obj = objects.get(ref.role)
    if obj is None or not hasattr(obj, ref.field_name):
        raise KeyError(ref.token())
    return getattr(obj, ref.field_name)


def compile_relation_access_plan(schema: ReflectiveRewriteSchema) -> Optional[RelationAccessPlan]:
    """Compile a generated relational schema into structural index lookups.

    This compiler is deliberately bounded. It does not invent new relation
    operators; it converts supported generated EQ/IN relations into access paths.
    Unsupported relations fail closed rather than silently falling back to scan.
    """
    clauses = []
    for relation in schema.relations:
        if relation.operator == "EQ" and relation.left.role == "candidate" and relation.right.role != "candidate":
            clauses.append(
                AccessClause(
                    mode="EQ_INDEX",
                    candidate_field=relation.left.field_name,
                    context_ref=relation.right,
                    source_relation_token=relation.token(),
                )
            )
            continue
        if relation.operator == "EQ" and relation.right.role == "candidate" and relation.left.role != "candidate":
            clauses.append(
                AccessClause(
                    mode="EQ_INDEX",
                    candidate_field=relation.right.field_name,
                    context_ref=relation.left,
                    source_relation_token=relation.token(),
                )
            )
            continue
        if relation.operator == "IN" and relation.right.role == "candidate" and relation.left.role != "candidate":
            clauses.append(
                AccessClause(
                    mode="MEMBER_INDEX",
                    candidate_field=relation.right.field_name,
                    context_ref=relation.left,
                    source_relation_token=relation.token(),
                )
            )
            continue
        return None
    if not clauses:
        return None
    payload = {
        "schema": schema.schema_id,
        "clauses": [clause.token() for clause in clauses],
    }
    plan_id = "RELATION_ACCESS_PLAN::" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return RelationAccessPlan(
        plan_id=plan_id,
        schema_id=schema.schema_id,
        clauses=tuple(clauses),
    )


def _build_indexes(genome: MorphologyGenome, clauses: Sequence[AccessClause]):
    enabled = tuple(organ for organ in genome.organs if organ.enabled)
    eq_fields = tuple(sorted({clause.candidate_field for clause in clauses if clause.mode == "EQ_INDEX"}))
    member_fields = tuple(sorted({clause.candidate_field for clause in clauses if clause.mode == "MEMBER_INDEX"}))
    eq_indexes: Dict[str, Dict[object, Tuple[str, ...]]] = {}
    member_indexes: Dict[str, Dict[object, Tuple[str, ...]]] = {}
    field_reads = 0

    for field_name in eq_fields:
        scratch: Dict[object, list[str]] = {}
        for organ in enabled:
            if not hasattr(organ, field_name):
                continue
            field_reads += 1
            key = _freeze(getattr(organ, field_name))
            scratch.setdefault(key, []).append(organ.organ_id)
        eq_indexes[field_name] = {key: tuple(sorted(ids)) for key, ids in scratch.items()}

    for field_name in member_fields:
        scratch: Dict[object, list[str]] = {}
        for organ in enabled:
            if not hasattr(organ, field_name):
                continue
            field_reads += 1
            value = getattr(organ, field_name)
            if not isinstance(value, (tuple, list, set, frozenset)):
                continue
            for member in value:
                scratch.setdefault(_freeze(member), []).append(organ.organ_id)
        member_indexes[field_name] = {key: tuple(sorted(set(ids))) for key, ids in scratch.items()}

    return eq_indexes, member_indexes, field_reads


def apply_relation_access_plan(
    genome: MorphologyGenome,
    certificate: StructuralFailureCertificate,
    schema: ReflectiveRewriteSchema,
    plan: RelationAccessPlan,
) -> Optional[AccessPlanApplication]:
    if not certificate.authority_verified:
        return None
    if plan.schema_id != schema.schema_id:
        return None
    recompiled = compile_relation_access_plan(schema)
    if recompiled is None or recompiled.clauses != plan.clauses:
        return None

    edge_map = {edge.edge_id: edge for edge in genome.edges}
    eq_indexes, member_indexes, field_reads = _build_indexes(genome, plan.clauses)
    templates = []
    lookups = 0
    intersections = 0

    for locus in certificate.failed_locus_ids:
        edge = edge_map.get(locus)
        objects = _context_objects(genome, locus)
        if edge is None or objects is None:
            return None
        candidate_sets = []
        for clause in plan.clauses:
            try:
                context_value = _freeze(_context_value(objects, clause.context_ref))
            except KeyError:
                return None
            lookups += 1
            if clause.mode == "EQ_INDEX":
                candidate_sets.append(set(eq_indexes.get(clause.candidate_field, {}).get(context_value, ())))
            elif clause.mode == "MEMBER_INDEX":
                candidate_sets.append(set(member_indexes.get(clause.candidate_field, {}).get(context_value, ())))
            else:
                return None
        if not candidate_sets:
            return None
        selected = set(candidate_sets[0])
        for other in candidate_sets[1:]:
            intersections += 1
            selected.intersection_update(other)
        selected.discard(edge.source)
        selected.discard(edge.target)
        if len(selected) != 1:
            return None
        target = next(iter(selected))
        replacement = {
            "edge_id": edge.edge_id,
            "source": edge.source,
            "target": target,
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
                rationale=(f"access-plan::{plan.plan_id}", f"schema::{schema.schema_id}"),
                source_candidate_id="",
            )
        )

    raw = json.dumps(
        {
            "plan": plan.plan_id,
            "certificate": certificate.certificate_id,
            "templates": [template.fingerprint() for template in templates],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    program = MutationProgram(
        program_id="ACCESS_PLAN_MUTATION_PROGRAM::" + hashlib.sha256(raw).hexdigest()[:20],
        templates=tuple(templates),
        inherited_strategy_hash=plan.plan_id,
        generation_uses_current_outcomes=False,
    )
    return AccessPlanApplication(
        plan_id=plan.plan_id,
        schema_id=schema.schema_id,
        certificate_id=certificate.certificate_id,
        mutation_program=program,
        index_build_field_reads=field_reads,
        index_lookup_count=lookups,
        candidate_intersection_count=intersections,
        outcome_evaluations=0,
    )
