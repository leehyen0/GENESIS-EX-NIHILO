from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple
import hashlib
import json

from .executable_morphology import (
    EdgeSpec,
    MorphologyGenome,
    MorphologyMutation,
    MorphologyMutator,
    MutationLevel,
    OrganKind,
    OrganSpec,
    PressureVector,
)


@dataclass(frozen=True)
class MorphologyResidual:
    residual_id: str
    pressure: PressureVector
    same_frozen_phenotype_different_outcome: bool = False
    more_compute_still_aliased: bool = False
    failed_edge_ids: Tuple[str, ...] = ()
    implicated_organ_ids: Tuple[str, ...] = ()
    missing_artifact_types: Tuple[str, ...] = ()
    source_refs: Tuple[str, ...] = ()

    @property
    def structural_split_authorized_for_shadow(self) -> bool:
        return bool(
            self.same_frozen_phenotype_different_outcome
            and self.more_compute_still_aliased
        )


@dataclass(frozen=True)
class MorphologyCandidate:
    candidate_id: str
    mutation: MorphologyMutation
    descendant_fingerprint: str
    origin_residual_ids: Tuple[str, ...]
    operation_family: str
    generation_uses_outcomes: bool = False


@dataclass(frozen=True)
class MorphologyEvaluation:
    evaluation_id: str
    candidate_id: str
    context_id: str
    source_class: str
    capability_delta: float
    retained_competence_delta: float
    calibration_delta: float
    meta_productivity_delta: float
    externally_generated: bool
    authority_verified: bool
    benchmark_disjoint: bool


@dataclass(frozen=True)
class MorphologyPolicy:
    allowed_candidate_ids: Tuple[str, ...]
    min_contexts: int
    min_independent_classes: int


class MorphologyGenesisEngine:
    """Generate structural descendants from typed pressure, before outcome access.

    The engine deliberately has no generation-number ladder. It derives candidate
    rewires from type compatibility, local schedule changes from the inherited event
    order, and organ splits from an actual frozen-phenotype alias collision. The
    mutation families themselves remain bootstrap metalanguage and are therefore a
    later object of mutation rather than being described as unrestricted genesis.
    """

    def __init__(self, candidate_budget: int = 256) -> None:
        self.candidate_budget = max(1, int(candidate_budget))
        self.last_raw_candidate_count = 0
        self.last_unique_candidate_count = 0
        self.last_truncated = False

    @staticmethod
    def _candidate_id(parent_hash: str, mutation: MorphologyMutation, descendant: MorphologyGenome) -> str:
        payload = {
            "parent": parent_hash,
            "mutation_id": mutation.mutation_id,
            "level": int(mutation.level),
            "operation": mutation.operation,
            "payload": dict(mutation.payload),
            "descendant": descendant.fingerprint(),
        }
        return "MORPH_CANDIDATE::" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()[:20]

    @staticmethod
    def _mutation_id(parent_hash: str, operation: str, payload: Mapping[str, object], residual_ids: Sequence[str]) -> str:
        raw = json.dumps(
            {
                "parent": parent_hash,
                "operation": operation,
                "payload": dict(payload),
                "residuals": sorted(str(x) for x in residual_ids),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        return "MORPH_MUTATION::" + hashlib.sha256(raw).hexdigest()[:20]

    @staticmethod
    def _typed_rewire_payloads(genome: MorphologyGenome, residual: MorphologyResidual) -> Iterable[Tuple[str, Mapping[str, object]]]:
        organs = genome.organ_map()
        edges = {edge.edge_id: edge for edge in genome.edges}
        failed = set(residual.failed_edge_ids)
        for edge in genome.edges:
            if failed and edge.edge_id not in failed:
                continue
            source = organs.get(edge.source)
            if source is None:
                continue
            for target in genome.organs:
                if target.organ_id == edge.target or target.organ_id == edge.source:
                    continue
                if edge.artifact_type not in source.produces or edge.artifact_type not in target.consumes:
                    continue
                replacement = {
                    "edge_id": edge.edge_id,
                    "source": edge.source,
                    "target": target.organ_id,
                    "artifact_type": edge.artifact_type,
                    "authority_required": edge.authority_required,
                    "gate": edge.gate,
                    "priority": edge.priority,
                }
                yield "REWIRE_EDGE", {"edge_id": edge.edge_id, "edge": replacement}

        existing = {(edge.source, edge.target, edge.artifact_type) for edge in genome.edges}
        for source in genome.organs:
            for target in genome.organs:
                if source.organ_id == target.organ_id:
                    continue
                shared = sorted(set(source.produces).intersection(target.consumes))
                for artifact_type in shared:
                    key = (source.organ_id, target.organ_id, artifact_type)
                    if key in existing:
                        continue
                    digest = hashlib.sha256("|".join(key).encode()).hexdigest()[:12]
                    edge = {
                        "edge_id": f"GEN_EDGE::{digest}",
                        "source": source.organ_id,
                        "target": target.organ_id,
                        "artifact_type": artifact_type,
                        "authority_required": artifact_type.startswith("authorized_"),
                        "gate": "PRESSURE_DERIVED",
                        "priority": 1.0,
                    }
                    yield "ADD_EDGE", {"edge": edge}

    @staticmethod
    def _schedule_payloads(genome: MorphologyGenome, residual: MorphologyResidual) -> Iterable[Tuple[str, Mapping[str, object]]]:
        order = list(genome.event_order)
        if len(order) < 2 or residual.pressure.efficiency_pressure <= 0.0:
            return
        implicated = set(residual.implicated_organ_ids)
        for index in range(len(order) - 1):
            left, right = order[index], order[index + 1]
            if implicated and left not in implicated and right not in implicated:
                continue
            candidate = list(order)
            candidate[index], candidate[index + 1] = candidate[index + 1], candidate[index]
            yield "SET_EVENT_ORDER", {"event_order": candidate}

    @staticmethod
    def _split_payloads(genome: MorphologyGenome, residual: MorphologyResidual) -> Iterable[Tuple[str, Mapping[str, object]]]:
        if not residual.structural_split_authorized_for_shadow:
            return
        implicated = tuple(residual.implicated_organ_ids) or tuple(o.organ_id for o in genome.organs)
        by_id = genome.organ_map()
        for organ_id in implicated:
            organ = by_id.get(organ_id)
            if organ is None or organ.kind in {OrganKind.GOVERNOR, OrganKind.ARCHIVE}:
                continue
            suffix = hashlib.sha256(f"{residual.residual_id}|{organ_id}".encode()).hexdigest()[:10]
            split = {
                "organ_id": f"{organ_id}::SPLIT::{suffix}",
                "kind": organ.kind.value,
                "consumes": list(organ.consumes),
                "produces": list(organ.produces),
                "implementation_ref": organ.implementation_ref,
                "version": organ.version + 1,
                "cost_hint": organ.cost_hint,
                "provenance": list(organ.provenance) + [f"split-pressure::{residual.residual_id}"],
                "enabled": True,
            }
            yield "ADD_ORGAN", {"organ": split}

    def generate(self, genome: MorphologyGenome, residuals: Sequence[MorphologyResidual]) -> Tuple[MorphologyCandidate, ...]:
        parent_hash = genome.fingerprint()
        mutator = MorphologyMutator()
        raw = []
        for residual in residuals:
            families = []
            if residual.pressure.transfer_failure > 0.0 or residual.pressure.calibration_failure > 0.0 or residual.failed_edge_ids:
                families.append(self._typed_rewire_payloads(genome, residual))
            if residual.pressure.efficiency_pressure > 0.0:
                families.append(self._schedule_payloads(genome, residual))
            if residual.structural_split_authorized_for_shadow:
                families.append(self._split_payloads(genome, residual))

            for family in families:
                for operation, payload in family:
                    level = MutationLevel.TOPOLOGY
                    mutation_id = self._mutation_id(parent_hash, operation, payload, (residual.residual_id,))
                    mutation = MorphologyMutation(
                        mutation_id=mutation_id,
                        level=level,
                        operation=operation,
                        payload=payload,
                        parent_body_hash=parent_hash,
                        rationale=(f"pressure::{residual.residual_id}",),
                    )
                    try:
                        descendant = mutator.apply(genome, mutation)
                    except ValueError:
                        continue
                    raw.append(
                        MorphologyCandidate(
                            candidate_id=self._candidate_id(parent_hash, mutation, descendant),
                            mutation=mutation,
                            descendant_fingerprint=descendant.fingerprint(),
                            origin_residual_ids=(residual.residual_id,),
                            operation_family=operation,
                            generation_uses_outcomes=False,
                        )
                    )

        self.last_raw_candidate_count = len(raw)
        unique: Dict[str, MorphologyCandidate] = {}
        for candidate in sorted(raw, key=lambda item: item.candidate_id):
            unique.setdefault(candidate.descendant_fingerprint, candidate)
        self.last_unique_candidate_count = len(unique)
        self.last_truncated = len(unique) > self.candidate_budget
        return tuple(sorted(unique.values(), key=lambda item: item.candidate_id)[: self.candidate_budget])


def derive_morphology_policy(
    candidates: Sequence[MorphologyCandidate],
    evaluations: Sequence[MorphologyEvaluation],
    min_contexts: int = 2,
    min_independent_classes: int = 2,
) -> MorphologyPolicy:
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    allowed = []
    for candidate_id in sorted(candidate_ids):
        rows = [
            row for row in evaluations
            if row.candidate_id == candidate_id
            and row.externally_generated
            and row.authority_verified
            and row.benchmark_disjoint
            and row.capability_delta > 0.0
            and row.retained_competence_delta >= 0.0
            and row.calibration_delta >= 0.0
            and row.meta_productivity_delta >= 0.0
        ]
        contexts = {row.context_id for row in rows}
        classes = {row.source_class for row in rows if row.source_class and row.source_class != "UNVERIFIED"}
        if len(contexts) >= max(1, int(min_contexts)) and len(classes) >= max(1, int(min_independent_classes)):
            allowed.append(candidate_id)
    return MorphologyPolicy(tuple(allowed), max(1, int(min_contexts)), max(1, int(min_independent_classes)))


def select_authorized_morphology_candidate(
    candidates: Sequence[MorphologyCandidate], policy: MorphologyPolicy
) -> Optional[MorphologyCandidate]:
    allowed = set(policy.allowed_candidate_ids)
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        if candidate.candidate_id in allowed:
            return candidate
    return None
