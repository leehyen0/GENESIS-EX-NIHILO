from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple
import hashlib


class ExposureClass(str, Enum):
    PUBLIC_TRAINING = "PUBLIC_TRAINING"
    PUBLIC_DEV = "PUBLIC_DEV"
    FROZEN_HELDOUT = "FROZEN_HELDOUT"
    PRIVATE_EXTERNAL = "PRIVATE_EXTERNAL"
    SOURCE_DISJOINT_TRANSFER = "SOURCE_DISJOINT_TRANSFER"


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    uri: str
    content_hash: str
    source_family: str
    acquired_at: str
    independent_from_evaluator: bool = False

    @staticmethod
    def from_content(source_id: str, uri: str, content: bytes, source_family: str, acquired_at: str, independent_from_evaluator: bool = False) -> "SourceRecord":
        return SourceRecord(str(source_id), str(uri), hashlib.sha256(content).hexdigest(), str(source_family), str(acquired_at), bool(independent_from_evaluator))


@dataclass(frozen=True)
class PreReadCommitment:
    commitment_id: str
    source_request_id: str
    predictions: Tuple[str, ...]
    questions: Tuple[str, ...]
    uncertainty: float


@dataclass(frozen=True)
class KnowledgeClaim:
    claim_id: str
    source_id: str
    statement_hash: str
    relation_tags: Tuple[str, ...]
    confidence: float
    contradiction_refs: Tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceSourceRequest:
    request_id: str
    pressure_kind: str
    missing_distinction: str
    excluded_source_families: Tuple[str, ...]
    desired_evidence_type: str
    source_disjoint_required: bool = True


@dataclass(frozen=True)
class KnowledgeAssimilationEpisode:
    episode_id: str
    source_ids: Tuple[str, ...]
    commitment_id: str
    claim_ids: Tuple[str, ...]
    target_task_ref: str
    behavior_changed: bool
    source_disjoint_transfer: bool
    removal_effect: Optional[float]
    wrong_swap_effect: Optional[float]
    delayed_reconstruction_equal: Optional[bool]
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkEpisode:
    benchmark_family: str
    task_id: str
    exposure: ExposureClass
    task_content_hash: str
    pre_body_hash: str
    post_body_hash: str
    score: float
    calibrated_score: Optional[float]
    compute_cost: float
    evidence_cost: float
    human_structural_intervention: float
    evaluator_independent: bool
    answer_or_patch_seen_before_freeze: bool = False


class KnowledgeAssimilationGate:
    """Promote external knowledge only when it causes transferable behavior change."""

    @staticmethod
    def promoteable(episode: KnowledgeAssimilationEpisode, sources: Sequence[SourceRecord], commitment: Optional[PreReadCommitment]) -> bool:
        if commitment is None or commitment.commitment_id != episode.commitment_id:
            return False
        source_ids = {source.source_id for source in sources}
        if not episode.source_ids or not set(episode.source_ids).issubset(source_ids):
            return False
        if not episode.behavior_changed or not episode.source_disjoint_transfer:
            return False
        if episode.removal_effect is None or float(episode.removal_effect) <= 0.0:
            return False
        if episode.wrong_swap_effect is None or float(episode.wrong_swap_effect) <= 0.0:
            return False
        if episode.delayed_reconstruction_equal is not True:
            return False
        return True


class BenchmarkAuthorityGate:
    """Separate development experience from authoritative heldout evidence."""

    @staticmethod
    def authoritative_for_promotion(episode: BenchmarkEpisode) -> bool:
        if episode.answer_or_patch_seen_before_freeze:
            return False
        if not episode.evaluator_independent:
            return False
        if episode.exposure in {ExposureClass.PUBLIC_TRAINING, ExposureClass.PUBLIC_DEV}:
            return False
        return episode.exposure in {ExposureClass.FROZEN_HELDOUT, ExposureClass.PRIVATE_EXTERNAL, ExposureClass.SOURCE_DISJOINT_TRANSFER}


def request_source_from_identifiability_deficit(missing_distinction: str, prior_source_families: Sequence[str], desired_evidence_type: str) -> EvidenceSourceRequest:
    payload = "|".join((str(missing_distinction), ",".join(sorted(set(str(v) for v in prior_source_families))), str(desired_evidence_type)))
    request_id = "SOURCE_REQUEST::" + hashlib.sha256(payload.encode()).hexdigest()[:16]
    return EvidenceSourceRequest(request_id=request_id, pressure_kind="IDENTIFIABILITY_DEFICIT", missing_distinction=str(missing_distinction), excluded_source_families=tuple(sorted(set(str(v) for v in prior_source_families))), desired_evidence_type=str(desired_evidence_type), source_disjoint_required=True)
