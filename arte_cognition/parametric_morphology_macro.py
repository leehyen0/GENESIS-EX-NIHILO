from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple
import hashlib
import json

from .executable_morphology import (
    MorphologyGenome,
    MorphologyMutation,
    MorphologyMutator,
    MutationLevel,
)
from .meta_acceleration import MutationProgram, MutationTemplate
from .structural_failure_certificate import StructuralFailureCertificate


@dataclass(frozen=True)
class MacroTrainingExample:
    context_id: str
    source_class: str
    genome: MorphologyGenome
    certificate: StructuralFailureCertificate
    successful_program: MutationProgram
    external_capability: float
    authority_verified: bool
    benchmark_disjoint: bool


@dataclass(frozen=True)
class ParametricMorphologyMacro:
    macro_id: str
    rule: str
    supporting_contexts: Tuple[str, ...]
    supporting_source_classes: Tuple[str, ...]
    supporting_program_ids: Tuple[str, ...]
    inherited_from_external_outcomes: bool
    current_outcomes_required_for_application: bool = False


@dataclass(frozen=True)
class MacroApplication:
    macro_id: str
    certificate_id: str
    descendant_fingerprint: str
    rewritten_locus_ids: Tuple[str, ...]
    structural_lookup_count: int
    candidate_evaluations: int
    mutation_program: MutationProgram


def _edge_map(genome: MorphologyGenome):
    return {edge.edge_id: edge for edge in genome.edges}


def _compatible_alternatives(genome: MorphologyGenome, edge_id: str) -> Tuple[str, ...]:
    edges = _edge_map(genome)
    edge = edges.get(edge_id)
    if edge is None:
        return ()
    by_id = genome.organ_map()
    source = by_id.get(edge.source)
    if source is None or edge.artifact_type not in source.produces:
        return ()
    return tuple(
        sorted(
            organ.organ_id
            for organ in genome.organs
            if organ.enabled
            and organ.organ_id not in {edge.source, edge.target}
            and edge.artifact_type in organ.consumes
        )
    )


def _program_matches_unique_alternative_rule(
    genome: MorphologyGenome,
    certificate: StructuralFailureCertificate,
    program: MutationProgram,
) -> bool:
    templates = tuple(program.templates)
    if len(templates) != len(certificate.failed_locus_ids):
        return False
    by_locus: Dict[str, MutationTemplate] = {}
    for template in templates:
        if template.operation != "REWIRE_EDGE":
            return False
        payload = dict(template.payload)
        locus = str(payload.get("edge_id", ""))
        edge_payload = payload.get("edge")
        if not locus or not isinstance(edge_payload, Mapping):
            return False
        by_locus[locus] = template
    if set(by_locus) != set(certificate.failed_locus_ids):
        return False
    for locus in certificate.failed_locus_ids:
        alternatives = _compatible_alternatives(genome, locus)
        if len(alternatives) != 1:
            return False
        target = str(dict(by_locus[locus].payload["edge"])["target"])
        if target != alternatives[0]:
            return False
    return True


def derive_parametric_rewire_macro(
    examples: Sequence[MacroTrainingExample],
    *,
    min_contexts: int = 2,
    min_source_classes: int = 2,
) -> Optional[ParametricMorphologyMacro]:
    usable = [
        example
        for example in examples
        if example.authority_verified
        and example.benchmark_disjoint
        and example.external_capability > 0.0
        and example.certificate.authority_verified
        and _program_matches_unique_alternative_rule(
            example.genome,
            example.certificate,
            example.successful_program,
        )
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
        return None
    if len(source_classes) < max(1, int(min_source_classes)):
        return None
    rule = "FOR_EACH_CERTIFIED_FAILED_EDGE_REWIRE_TO_UNIQUE_COMPATIBLE_ALTERNATIVE"
    program_ids = tuple(sorted({example.successful_program.program_id for example in usable}))
    payload = {
        "rule": rule,
        "contexts": contexts,
        "classes": source_classes,
        "programs": program_ids,
    }
    macro_id = "PARAMETRIC_MORPH_MACRO::" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return ParametricMorphologyMacro(
        macro_id=macro_id,
        rule=rule,
        supporting_contexts=contexts,
        supporting_source_classes=source_classes,
        supporting_program_ids=program_ids,
        inherited_from_external_outcomes=True,
        current_outcomes_required_for_application=False,
    )


def apply_parametric_macro(
    genome: MorphologyGenome,
    certificate: StructuralFailureCertificate,
    macro: ParametricMorphologyMacro,
) -> Optional[MacroApplication]:
    if not certificate.authority_verified:
        return None
    if macro.rule != "FOR_EACH_CERTIFIED_FAILED_EDGE_REWIRE_TO_UNIQUE_COMPATIBLE_ALTERNATIVE":
        return None

    edge_map = _edge_map(genome)
    templates = []
    lookup_count = 0
    for locus in certificate.failed_locus_ids:
        edge = edge_map.get(locus)
        if edge is None:
            return None
        alternatives = _compatible_alternatives(genome, locus)
        lookup_count += 1
        if len(alternatives) != 1:
            return None
        replacement = {
            "edge_id": edge.edge_id,
            "source": edge.source,
            "target": alternatives[0],
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
                rationale=(f"macro::{macro.macro_id}", f"certificate::{certificate.certificate_id}"),
                source_candidate_id="",
            )
        )

    raw = "|".join(template.fingerprint() for template in templates) + "|" + macro.macro_id + "|" + certificate.certificate_id
    program = MutationProgram(
        program_id="PARAMETRIC_MUTATION_PROGRAM::" + hashlib.sha256(raw.encode()).hexdigest()[:20],
        templates=tuple(templates),
        inherited_strategy_hash=macro.macro_id,
        generation_uses_current_outcomes=False,
    )
    current = genome
    mutator = MorphologyMutator()
    for index, template in enumerate(program.templates):
        mutation = MorphologyMutation(
            mutation_id=f"{program.program_id}::STEP::{index}",
            level=template.level,
            operation=template.operation,
            payload=dict(template.payload),
            parent_body_hash=current.fingerprint(),
            rationale=template.rationale,
            reversible=True,
        )
        current = mutator.apply(current, mutation)

    return MacroApplication(
        macro_id=macro.macro_id,
        certificate_id=certificate.certificate_id,
        descendant_fingerprint=current.fingerprint(),
        rewritten_locus_ids=tuple(certificate.failed_locus_ids),
        structural_lookup_count=lookup_count,
        candidate_evaluations=0,
        mutation_program=program,
    )
