from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import hashlib
import json

from .canonical_body_checkpoint import checkpoint_dict as canonical_checkpoint_dict
from .canonical_body_checkpoint import restore_runtime as restore_canonical_runtime
from .cognitive_runtime import PersistentCognitiveRuntime
from .executable_morphology import (
    EdgeSpec,
    ExperienceArchive,
    ExperienceUnit,
    MorphologyGenome,
    OrganKind,
    OrganSpec,
)
from .meta_acceleration import MutationProgramDevelopmentState, MutationStrategyState
from .raw_observation_authority import RawObservationVerifier
from .world_coupling import WorldReceiptVerifier


SELF_EVOLVING_BODY_SCHEMA = "arte.self_evolving_research_body/v1"
_REQUIRED = (
    "canonical_runtime",
    "morphology_genome",
    "mutation_strategy",
    "mutation_program_development",
    "experience_archive",
)


@dataclass
class SelfEvolvingResearchBody:
    runtime: PersistentCognitiveRuntime
    morphology: MorphologyGenome
    mutation_strategy: MutationStrategyState
    mutation_program_state: MutationProgramDevelopmentState
    experience_archive: ExperienceArchive


def _genome_dict(genome: MorphologyGenome) -> Dict[str, Any]:
    return {
        "organs": [
            {
                "organ_id": organ.organ_id,
                "kind": organ.kind.value,
                "consumes": list(organ.consumes),
                "produces": list(organ.produces),
                "implementation_ref": organ.implementation_ref,
                "version": organ.version,
                "cost_hint": organ.cost_hint,
                "provenance": list(organ.provenance),
                "enabled": organ.enabled,
            }
            for organ in genome.organs
        ],
        "edges": [
            {
                "edge_id": edge.edge_id,
                "source": edge.source,
                "target": edge.target,
                "artifact_type": edge.artifact_type,
                "authority_required": edge.authority_required,
                "gate": edge.gate,
                "priority": edge.priority,
            }
            for edge in genome.edges
        ],
        "event_order": list(genome.event_order),
        "constitution_epoch": genome.constitution_epoch,
        "fingerprint": genome.fingerprint(),
    }


def _restore_genome(payload: Dict[str, Any]) -> MorphologyGenome:
    organs = tuple(
        OrganSpec(
            organ_id=str(row["organ_id"]),
            kind=OrganKind(str(row["kind"])),
            consumes=tuple(str(value) for value in row.get("consumes", ())),
            produces=tuple(str(value) for value in row.get("produces", ())),
            implementation_ref=str(row.get("implementation_ref", "")),
            version=int(row.get("version", 1)),
            cost_hint=float(row.get("cost_hint", 1.0)),
            provenance=tuple(str(value) for value in row.get("provenance", ())),
            enabled=bool(row.get("enabled", True)),
        )
        for row in payload.get("organs", ())
    )
    edges = tuple(
        EdgeSpec(
            edge_id=str(row["edge_id"]),
            source=str(row["source"]),
            target=str(row["target"]),
            artifact_type=str(row["artifact_type"]),
            authority_required=bool(row.get("authority_required", False)),
            gate=str(row.get("gate", "ALWAYS")),
            priority=float(row.get("priority", 1.0)),
        )
        for row in payload.get("edges", ())
    )
    genome = MorphologyGenome(
        organs=organs,
        edges=edges,
        event_order=tuple(str(value) for value in payload.get("event_order", ())),
        constitution_epoch=int(payload.get("constitution_epoch", 0)),
    )
    errors = genome.validate()
    if errors:
        raise ValueError("invalid restored morphology: " + ",".join(errors))
    expected = str(payload.get("fingerprint", ""))
    if not expected or expected != genome.fingerprint():
        raise ValueError("morphology fingerprint mismatch")
    return genome


def _strategy_dict(state: MutationStrategyState) -> Dict[str, Any]:
    return {
        "operation_scores": [[key, value] for key, value in state.operation_scores],
        "operation_support": [[key, value] for key, value in state.operation_support],
        "fossilized_operations": list(state.fossilized_operations),
        "lineage_hash": state.lineage_hash,
    }


def _restore_strategy(payload: Dict[str, Any]) -> MutationStrategyState:
    return MutationStrategyState(
        operation_scores=tuple((str(key), float(value)) for key, value in payload.get("operation_scores", ())),
        operation_support=tuple((str(key), int(value)) for key, value in payload.get("operation_support", ())),
        fossilized_operations=tuple(str(value) for value in payload.get("fossilized_operations", ())),
        lineage_hash=str(payload.get("lineage_hash", "")),
    )


def _program_state_dict(state: MutationProgramDevelopmentState) -> Dict[str, Any]:
    return {
        "max_depth": state.max_depth,
        "complete_failure_receipts": [[context, source] for context, source in state.complete_failure_receipts],
        "lineage_hash": state.lineage_hash,
    }


def _restore_program_state(payload: Dict[str, Any]) -> MutationProgramDevelopmentState:
    return MutationProgramDevelopmentState(
        max_depth=max(1, int(payload.get("max_depth", 1))),
        complete_failure_receipts=tuple(
            (str(context), str(source))
            for context, source in payload.get("complete_failure_receipts", ())
        ),
        lineage_hash=str(payload.get("lineage_hash", "")),
    )


def _experience_dict(episode: ExperienceUnit) -> Dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "pre_body_hash": episode.pre_body_hash,
        "source_refs": list(episode.source_refs),
        "task_ref": episode.task_ref,
        "benchmark_family": episode.benchmark_family,
        "precommitted_hypotheses": list(episode.precommitted_hypotheses),
        "selected_goal_id": episode.selected_goal_id,
        "action_trace_hash": episode.action_trace_hash,
        "outcome_summary": episode.outcome_summary,
        "success": episode.success,
        "uncertainty_before": episode.uncertainty_before,
        "uncertainty_after": episode.uncertainty_after,
        "mutation_ids": list(episode.mutation_ids),
        "removal_effect": episode.removal_effect,
        "wrong_swap_effect": episode.wrong_swap_effect,
        "heldout_effect": episode.heldout_effect,
        "delayed_replay_equal": episode.delayed_replay_equal,
        "descendant_body_hash": episode.descendant_body_hash,
        "notes": list(episode.notes),
    }


def _restore_experience(payload: Dict[str, Any]) -> ExperienceUnit:
    return ExperienceUnit(
        episode_id=str(payload["episode_id"]),
        pre_body_hash=str(payload.get("pre_body_hash", "")),
        source_refs=tuple(str(value) for value in payload.get("source_refs", ())),
        task_ref=str(payload.get("task_ref", "")),
        benchmark_family=str(payload.get("benchmark_family", "")),
        precommitted_hypotheses=tuple(str(value) for value in payload.get("precommitted_hypotheses", ())),
        selected_goal_id=str(payload.get("selected_goal_id", "")),
        action_trace_hash=str(payload.get("action_trace_hash", "")),
        outcome_summary=str(payload.get("outcome_summary", "")),
        success=bool(payload.get("success", False)),
        uncertainty_before=float(payload.get("uncertainty_before", 0.0)),
        uncertainty_after=float(payload.get("uncertainty_after", 0.0)),
        mutation_ids=tuple(str(value) for value in payload.get("mutation_ids", ())),
        removal_effect=None if payload.get("removal_effect") is None else float(payload["removal_effect"]),
        wrong_swap_effect=None if payload.get("wrong_swap_effect") is None else float(payload["wrong_swap_effect"]),
        heldout_effect=None if payload.get("heldout_effect") is None else float(payload["heldout_effect"]),
        delayed_replay_equal=payload.get("delayed_replay_equal"),
        descendant_body_hash=str(payload.get("descendant_body_hash", "")),
        notes=tuple(str(value) for value in payload.get("notes", ())),
    )


def _archive_dict(archive: ExperienceArchive) -> Dict[str, Any]:
    return {
        "episodes": [_experience_dict(archive.episodes[key]) for key in sorted(archive.episodes)],
        "fossils": dict(sorted(archive.fossils.items())),
    }


def _restore_archive(payload: Dict[str, Any]) -> ExperienceArchive:
    archive = ExperienceArchive()
    for row in payload.get("episodes", ()):
        episode = _restore_experience(dict(row))
        if not archive.append(episode):
            raise ValueError("duplicate experience episode in checkpoint")
    for key, value in dict(payload.get("fossils", {})).items():
        archive.fossilize(str(key), str(value))
    return archive


def _without_integrity(payload: Dict[str, Any]) -> Dict[str, Any]:
    clone = dict(payload)
    envelope = dict(clone.get("self_evolving_body", {}))
    envelope.pop("integrity_sha256", None)
    clone["self_evolving_body"] = envelope
    return clone


def integrity_sha256(payload: Dict[str, Any]) -> str:
    material = json.dumps(
        _without_integrity(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def checkpoint_dict(body: SelfEvolvingResearchBody) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "canonical_runtime": canonical_checkpoint_dict(body.runtime),
        "morphology_genome": _genome_dict(body.morphology),
        "mutation_strategy": _strategy_dict(body.mutation_strategy),
        "mutation_program_development": _program_state_dict(body.mutation_program_state),
        "experience_archive": _archive_dict(body.experience_archive),
    }
    payload["self_evolving_body"] = {
        "schema": SELF_EVOLVING_BODY_SCHEMA,
        "required_namespaces": list(_REQUIRED),
        "integrity_sha256": "",
        "authority_note": (
            "morphology, strategy, development and experience are inherited state; "
            "external verifier/certificate authority is not checkpointed and must be re-established"
        ),
    }
    payload["self_evolving_body"]["integrity_sha256"] = integrity_sha256(payload)
    return payload


def checkpoint_json(body: SelfEvolvingResearchBody) -> str:
    return json.dumps(checkpoint_dict(body), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def restore_body(
    payload: Dict[str, Any],
    world_verifier: Optional[WorldReceiptVerifier] = None,
    raw_observation_verifier: Optional[RawObservationVerifier] = None,
) -> SelfEvolvingResearchBody:
    envelope = dict(payload.get("self_evolving_body", {}))
    if envelope.get("schema") != SELF_EVOLVING_BODY_SCHEMA:
        raise ValueError("unsupported self-evolving BODY checkpoint schema")
    required = tuple(str(value) for value in envelope.get("required_namespaces", ()))
    if required != _REQUIRED:
        raise ValueError("self-evolving BODY required-namespace contract mismatch")
    missing = [name for name in _REQUIRED if name not in payload]
    if missing:
        raise ValueError("self-evolving BODY missing namespaces: " + ",".join(missing))
    expected = str(envelope.get("integrity_sha256", ""))
    if not expected or expected != integrity_sha256(payload):
        raise ValueError("self-evolving BODY checkpoint integrity mismatch")

    runtime = restore_canonical_runtime(
        dict(payload["canonical_runtime"]),
        world_verifier=world_verifier,
        raw_observation_verifier=raw_observation_verifier,
    )
    morphology = _restore_genome(dict(payload["morphology_genome"]))
    strategy = _restore_strategy(dict(payload["mutation_strategy"]))
    program_state = _restore_program_state(dict(payload["mutation_program_development"]))
    archive = _restore_archive(dict(payload["experience_archive"]))
    return SelfEvolvingResearchBody(runtime, morphology, strategy, program_state, archive)
