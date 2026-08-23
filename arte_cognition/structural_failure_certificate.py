from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, Mapping, Optional, Sequence, Tuple
import hashlib
import json

from .executable_morphology import MorphologyGenome
from .meta_acceleration import (
    MutationProgram,
    MutationProgramDevelopmentState,
    MutationStrategyState,
    MutationTemplate,
)
from .morphology_genesis import MorphologyCandidate


@dataclass(frozen=True)
class StructuralDiagnosticReceipt:
    receipt_id: str
    context_id: str
    source_class: str
    failed_locus_ids: Tuple[str, ...]
    authority_verified: bool
    benchmark_disjoint: bool
    evaluator_independent: bool = False


@dataclass(frozen=True)
class StructuralFailureCertificate:
    certificate_id: str
    context_id: str
    failed_locus_ids: Tuple[str, ...]
    supporting_receipt_ids: Tuple[str, ...]
    independent_source_classes: Tuple[str, ...]
    max_obligations_repaired_per_primitive: int
    lower_bound_program_depth: int
    evaluator_independent: bool
    authority_verified: bool


@dataclass(frozen=True)
class CertificateCompilation:
    program: Optional[MutationProgram]
    selected_candidate_ids: Tuple[str, ...]
    unresolved_locus_ids: Tuple[str, ...]
    candidate_scan_count: int
    certificate_id: str


def derive_structural_failure_certificate(
    receipts: Sequence[StructuralDiagnosticReceipt],
    *,
    max_obligations_repaired_per_primitive: int = 1,
    min_independent_classes: int = 2,
) -> Optional[StructuralFailureCertificate]:
    """Derive an exact within-context structural lower-bound certificate.

    Two or more verifier classes must agree on the same failed-locus set. The
    primitive impact bound is explicit and remains a bootstrap assumption until a
    later world-derived impact-language frontier replaces it.
    """
    usable = [
        receipt
        for receipt in receipts
        if receipt.authority_verified
        and receipt.benchmark_disjoint
        and receipt.source_class
        and receipt.source_class != "UNVERIFIED"
    ]
    if len(usable) < max(1, int(min_independent_classes)):
        return None
    contexts = {receipt.context_id for receipt in usable}
    if len(contexts) != 1:
        return None
    failed_sets = {tuple(sorted(set(receipt.failed_locus_ids))) for receipt in usable}
    if len(failed_sets) != 1:
        return None
    source_classes = tuple(sorted({receipt.source_class for receipt in usable}))
    if len(source_classes) < max(1, int(min_independent_classes)):
        return None
    failed = next(iter(failed_sets))
    if not failed:
        return None
    max_coverage = max(1, int(max_obligations_repaired_per_primitive))
    lower_bound = int(ceil(len(failed) / max_coverage))
    payload = {
        "context": next(iter(contexts)),
        "failed": failed,
        "receipts": sorted(receipt.receipt_id for receipt in usable),
        "classes": source_classes,
        "max_coverage": max_coverage,
    }
    certificate_id = "STRUCTURAL_FAILURE_CERT::" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return StructuralFailureCertificate(
        certificate_id=certificate_id,
        context_id=next(iter(contexts)),
        failed_locus_ids=failed,
        supporting_receipt_ids=tuple(sorted(receipt.receipt_id for receipt in usable)),
        independent_source_classes=source_classes,
        max_obligations_repaired_per_primitive=max_coverage,
        lower_bound_program_depth=lower_bound,
        evaluator_independent=all(receipt.evaluator_independent for receipt in usable),
        authority_verified=True,
    )


def open_program_depth_from_certificate(
    state: MutationProgramDevelopmentState,
    certificate: StructuralFailureCertificate,
    *,
    max_depth_cap: int = 64,
) -> MutationProgramDevelopmentState:
    if not certificate.authority_verified:
        return state
    desired = min(max(1, int(max_depth_cap)), max(state.max_depth, certificate.lower_bound_program_depth))
    if desired == state.max_depth:
        return state
    payload = {
        "parent": state.lineage_hash,
        "certificate": certificate.certificate_id,
        "old_depth": state.max_depth,
        "new_depth": desired,
    }
    lineage_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return MutationProgramDevelopmentState(
        max_depth=desired,
        complete_failure_receipts=(),
        lineage_hash=lineage_hash,
    )


def _candidate_locus(candidate: MorphologyCandidate) -> Optional[str]:
    payload = dict(candidate.mutation.payload)
    if candidate.operation_family == "REWIRE_EDGE":
        value = payload.get("edge_id")
        return str(value) if value is not None else None
    if candidate.operation_family == "REMOVE_EDGE":
        value = payload.get("edge_id")
        return str(value) if value is not None else None
    edge = payload.get("edge")
    if isinstance(edge, Mapping):
        value = edge.get("edge_id")
        if value is not None:
            return str(value)
    return None


def compile_program_from_certificate(
    genome: MorphologyGenome,
    candidates: Sequence[MorphologyCandidate],
    strategy: MutationStrategyState,
    certificate: StructuralFailureCertificate,
) -> CertificateCompilation:
    """Compile one candidate mutation per independent structural obligation.

    Selection uses only the frozen certificate, candidate structure and an inherited
    strategy prior. No current candidate outcome is consumed.
    """
    if not certificate.authority_verified:
        return CertificateCompilation(None, (), certificate.failed_locus_ids, 0, certificate.certificate_id)

    by_locus: Dict[str, list[MorphologyCandidate]] = {}
    scan_count = 0
    for candidate in candidates:
        scan_count += 1
        locus = _candidate_locus(candidate)
        if locus is not None:
            by_locus.setdefault(locus, []).append(candidate)

    selected = []
    unresolved = []
    for locus in certificate.failed_locus_ids:
        options = by_locus.get(locus, [])
        options = sorted(
            options,
            key=lambda candidate: (
                -strategy.score(candidate.operation_family),
                candidate.operation_family,
                candidate.candidate_id,
            ),
        )
        if not options:
            unresolved.append(locus)
            continue
        selected.append(options[0])

    if unresolved:
        return CertificateCompilation(
            None,
            tuple(candidate.candidate_id for candidate in selected),
            tuple(unresolved),
            scan_count,
            certificate.certificate_id,
        )

    templates = tuple(
        MutationTemplate(
            operation=candidate.mutation.operation,
            level=candidate.mutation.level,
            payload=dict(candidate.mutation.payload),
            rationale=tuple(candidate.mutation.rationale) + (f"certificate::{certificate.certificate_id}",),
            source_candidate_id=candidate.candidate_id,
        )
        for candidate in selected
    )
    raw = "|".join(template.fingerprint() for template in templates) + "|" + certificate.certificate_id + "|" + strategy.lineage_hash
    program = MutationProgram(
        program_id="CERT_MUTATION_PROGRAM::" + hashlib.sha256(raw.encode()).hexdigest()[:20],
        templates=templates,
        inherited_strategy_hash=strategy.lineage_hash,
        generation_uses_current_outcomes=False,
    )
    return CertificateCompilation(
        program,
        tuple(candidate.candidate_id for candidate in selected),
        (),
        scan_count,
        certificate.certificate_id,
    )
