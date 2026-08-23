from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Mapping, Optional, Sequence, Tuple
import hashlib
import json

from .executable_morphology import MorphologyGenome, MutationLevel
from .meta_acceleration import MutationProgram, MutationTemplate
from .structural_failure_certificate import StructuralFailureCertificate


# Bootstrap relation atoms. The winning conjunction is not named or selected by an
# evaluator. It is synthesized from structural before/after regularities across
# prior externally validated repair programs. Expanding this atom vocabulary is a
# later frontier; this module does not claim unrestricted rewrite-language genesis.
RELATION_ATOMS: Tuple[str, ...] = (
    "ENABLED",
    "CONSUMES_EDGE_ARTIFACT",
    "SAME_KIND_AS_OLD_TARGET",
    "DIFFERENT_KIND_FROM_OLD_TARGET",
    "SAME_CONSUMES_SIGNATURE_AS_OLD_TARGET",
    "SAME_PRODUCES_SIGNATURE_AS_OLD_TARGET",
    "SAME_IMPLEMENTATION_REF_AS_OLD_TARGET",
    "SAME_VERSION_AS_OLD_TARGET",
)

EDGE_FIELDS: Tuple[str, ...] = (
    "source",
    "artifact_type",
    "authority_required",
    "gate",
    "priority",
)


@dataclass(frozen=True)
class RewriteSchemaTrainingExample:
    context_id: str
    source_class: str
    genome: MorphologyGenome
    certificate: StructuralFailureCertificate
    successful_program: MutationProgram
    external_capability: float
    authority_verified: bool
    benchmark_disjoint: bool


@dataclass(frozen=True)
class GeneratedRewriteSchema:
    schema_id: str
    operation: str
    target_predicates: Tuple[str, ...]
    preserved_edge_fields: Tuple[str, ...]
    supporting_contexts: Tuple[str, ...]
    supporting_source_classes: Tuple[str, ...]
    supporting_program_ids: Tuple[str, ...]
    relation_atom_vocabulary: Tuple[str, ...] = RELATION_ATOMS
    generated_from_prior_external_successes: bool = True
    current_outcomes_required_for_application: bool = False


@dataclass(frozen=True)
class RewriteSchemaApplication:
    schema_id: str
    certificate_id: str
    mutation_program: MutationProgram
    matched_locus_ids: Tuple[str, ...]
    structural_candidate_checks: int
    outcome_evaluations: int = 0


def _edge_map(genome: MorphologyGenome):
    return {edge.edge_id: edge for edge in genome.edges}


def _organ_atom_truth(
    genome: MorphologyGenome,
    edge_id: str,
    candidate_id: str,
    atom: str,
) -> bool:
    edges = _edge_map(genome)
    edge = edges.get(edge_id)
    organs = genome.organ_map()
    if edge is None or candidate_id not in organs or edge.target not in organs:
        return False
    candidate = organs[candidate_id]
    old_target = organs[edge.target]
    if atom == "ENABLED":
        return bool(candidate.enabled)
    if atom == "CONSUMES_EDGE_ARTIFACT":
        return edge.artifact_type in candidate.consumes
    if atom == "SAME_KIND_AS_OLD_TARGET":
        return candidate.kind == old_target.kind
    if atom == "DIFFERENT_KIND_FROM_OLD_TARGET":
        return candidate.kind != old_target.kind
    if atom == "SAME_CONSUMES_SIGNATURE_AS_OLD_TARGET":
        return tuple(candidate.consumes) == tuple(old_target.consumes)
    if atom == "SAME_PRODUCES_SIGNATURE_AS_OLD_TARGET":
        return tuple(candidate.produces) == tuple(old_target.produces)
    if atom == "SAME_IMPLEMENTATION_REF_AS_OLD_TARGET":
        return candidate.implementation_ref == old_target.implementation_ref
    if atom == "SAME_VERSION_AS_OLD_TARGET":
        return candidate.version == old_target.version
    raise ValueError(f"unknown relation atom: {atom}")


def _candidate_targets(genome: MorphologyGenome, edge_id: str) -> Tuple[str, ...]:
    edge = _edge_map(genome).get(edge_id)
    if edge is None:
        return ()
    return tuple(
        sorted(
            organ.organ_id
            for organ in genome.organs
            if organ.organ_id not in {edge.source, edge.target}
        )
    )


def _select_targets(
    genome: MorphologyGenome,
    edge_id: str,
    predicates: Sequence[str],
) -> Tuple[str, ...]:
    return tuple(
        candidate_id
        for candidate_id in _candidate_targets(genome, edge_id)
        if all(_organ_atom_truth(genome, edge_id, candidate_id, atom) for atom in predicates)
    )


def _training_rows(example: RewriteSchemaTrainingExample):
    edges = _edge_map(example.genome)
    certificate_loci = set(example.certificate.failed_locus_ids)
    rows = []
    seen = set()
    for template in example.successful_program.templates:
        if template.operation != "REWIRE_EDGE":
            return ()
        payload = dict(template.payload)
        locus = str(payload.get("edge_id", ""))
        replacement = payload.get("edge")
        if not locus or locus in seen or locus not in certificate_loci:
            return ()
        if not isinstance(replacement, Mapping) or locus not in edges:
            return ()
        old = edges[locus]
        if str(replacement.get("edge_id", "")) != old.edge_id:
            return ()
        new_target = str(replacement.get("target", ""))
        if not new_target or new_target == old.target:
            return ()
        # This bounded schema family only learns target rewrites. All other edge
        # semantics must be invariant across the validated examples.
        for field in EDGE_FIELDS:
            old_value = getattr(old, field)
            new_value = replacement.get(field)
            if new_value != old_value:
                return ()
        rows.append((example.genome, locus, new_target))
        seen.add(locus)
    if seen != certificate_loci:
        return ()
    return tuple(rows)


def _predicate_signature(
    rows: Sequence[Tuple[MorphologyGenome, str, str]],
    predicates: Sequence[str],
) -> Tuple[Tuple[str, ...], ...]:
    return tuple(_select_targets(genome, locus, predicates) for genome, locus, _ in rows)


def generate_rewrite_schemas(
    examples: Sequence[RewriteSchemaTrainingExample],
    *,
    min_contexts: int = 2,
    min_source_classes: int = 2,
    max_predicate_count: int = 4,
) -> Tuple[GeneratedRewriteSchema, ...]:
    usable = [
        example
        for example in examples
        if example.external_capability > 0.0
        and example.authority_verified
        and example.benchmark_disjoint
        and example.certificate.authority_verified
    ]
    contexts = tuple(sorted({example.context_id for example in usable}))
    source_classes = tuple(
        sorted(
            {
                example.source_class
                for example in usable
                if example.source_class and example.source_class != "UNVERIFIED"
            }
        )
    )
    if len(contexts) < max(1, int(min_contexts)):
        return ()
    if len(source_classes) < max(1, int(min_source_classes)):
        return ()

    rows = []
    program_ids = []
    for example in usable:
        extracted = _training_rows(example)
        if not extracted:
            return ()
        rows.extend(extracted)
        program_ids.append(example.successful_program.program_id)
    if not rows:
        return ()

    # Keep only relation atoms true of every externally successful target, then
    # search the smallest conjunction that uniquely reconstructs every target.
    common_true_atoms = tuple(
        atom
        for atom in RELATION_ATOMS
        if all(_organ_atom_truth(genome, locus, target, atom) for genome, locus, target in rows)
    )
    valid: Dict[Tuple[Tuple[str, ...], ...], Tuple[str, ...]] = {}
    max_size = min(max(1, int(max_predicate_count)), len(common_true_atoms))
    for size in range(1, max_size + 1):
        for subset in combinations(common_true_atoms, size):
            selections = _predicate_signature(rows, subset)
            if not all(
                selected == (target,)
                for selected, (_, _, target) in zip(selections, rows)
            ):
                continue
            # Behavioral quotient: different syntactic conjunctions producing the
            # same target partition are one schema phenotype. Prefer shortest then
            # lexicographic representation without consulting task outcomes.
            valid.setdefault(selections, tuple(subset))
        if valid:
            break
    if not valid:
        return ()

    schemas = []
    for predicates in sorted(valid.values(), key=lambda p: (len(p), p)):
        payload = {
            "operation": "REWIRE_EDGE",
            "predicates": list(predicates),
            "preserved": list(EDGE_FIELDS),
            "contexts": contexts,
            "classes": source_classes,
            "programs": sorted(set(program_ids)),
        }
        schema_id = "GENERATED_REWRITE_SCHEMA::" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:20]
        schemas.append(
            GeneratedRewriteSchema(
                schema_id=schema_id,
                operation="REWIRE_EDGE",
                target_predicates=tuple(predicates),
                preserved_edge_fields=EDGE_FIELDS,
                supporting_contexts=contexts,
                supporting_source_classes=source_classes,
                supporting_program_ids=tuple(sorted(set(program_ids))),
            )
        )
    return tuple(schemas)


def apply_generated_rewrite_schema(
    genome: MorphologyGenome,
    certificate: StructuralFailureCertificate,
    schema: GeneratedRewriteSchema,
) -> Optional[RewriteSchemaApplication]:
    if not certificate.authority_verified:
        return None
    if schema.operation != "REWIRE_EDGE":
        return None
    if tuple(schema.preserved_edge_fields) != EDGE_FIELDS:
        return None
    if any(atom not in RELATION_ATOMS for atom in schema.target_predicates):
        return None

    edges = _edge_map(genome)
    templates = []
    checks = 0
    for locus in certificate.failed_locus_ids:
        edge = edges.get(locus)
        if edge is None:
            return None
        candidates = _candidate_targets(genome, locus)
        checks += len(candidates)
        selected = tuple(
            candidate_id
            for candidate_id in candidates
            if all(
                _organ_atom_truth(genome, locus, candidate_id, atom)
                for atom in schema.target_predicates
            )
        )
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
                rationale=(f"generated-schema::{schema.schema_id}", f"certificate::{certificate.certificate_id}"),
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
        program_id="GENERATED_REWRITE_PROGRAM::" + hashlib.sha256(raw).hexdigest()[:20],
        templates=tuple(templates),
        inherited_strategy_hash=schema.schema_id,
        generation_uses_current_outcomes=False,
    )
    return RewriteSchemaApplication(
        schema_id=schema.schema_id,
        certificate_id=certificate.certificate_id,
        mutation_program=program,
        matched_locus_ids=tuple(certificate.failed_locus_ids),
        structural_candidate_checks=checks,
        outcome_evaluations=0,
    )
