from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import hashlib
import json

from .body_checkpoint import checkpoint_dict as base_checkpoint_dict
from .body_checkpoint import restore_runtime as restore_base_runtime
from .cognitive_runtime import PersistentCognitiveRuntime
from .epistemic_depth_runtime import (
    EpistemicallyDeepPersistentCognitiveRuntime,
    epistemic_checkpoint_dict,
    restore_epistemic_runtime,
)
from .primitive_genesis_runtime import (
    PRIMITIVE_DEVELOPMENT_SCHEMA,
    WorldDrivenPrimitiveRuntime,
    primitive_checkpoint_dict,
    restore_world_driven_primitive_runtime,
)
from .raw_observation_authority import RawObservationVerifier
from .world_coupling import WorldReceiptVerifier


CANONICAL_BODY_SCHEMA = "arte.canonical_developmental_body/v1"
KIND_BASE = "PERSISTENT_COGNITIVE_RUNTIME"
KIND_EPISTEMIC = "EPISTEMICALLY_DEEP_RUNTIME"
KIND_PRIMITIVE = "WORLD_DRIVEN_PRIMITIVE_RUNTIME"

_REQUIRED_NAMESPACES = {
    KIND_BASE: ("policy", "topology", "world_coupling", "memory"),
    KIND_EPISTEMIC: (
        "policy",
        "topology",
        "world_coupling",
        "memory",
        "epistemic_depth_schema",
        "world_model_ecology",
    ),
    KIND_PRIMITIVE: (
        "policy",
        "topology",
        "world_coupling",
        "memory",
        "epistemic_depth_schema",
        "world_model_ecology",
        "primitive_development_schema",
        "raw_observation_receipts",
        "primitive_genesis_policy",
    ),
}


def _runtime_kind(runtime: PersistentCognitiveRuntime) -> str:
    # Most-specific-first is deliberate. Primitive BODY is also an epistemic BODY.
    if isinstance(runtime, WorldDrivenPrimitiveRuntime):
        return KIND_PRIMITIVE
    if isinstance(runtime, EpistemicallyDeepPersistentCognitiveRuntime):
        return KIND_EPISTEMIC
    if isinstance(runtime, PersistentCognitiveRuntime):
        return KIND_BASE
    raise TypeError(f"unsupported BODY runtime type: {type(runtime)!r}")


def _payload_without_fingerprint(payload: Dict[str, Any]) -> Dict[str, Any]:
    clone = dict(payload)
    envelope = dict(clone.get("canonical_body", {}))
    envelope.pop("integrity_sha256", None)
    clone["canonical_body"] = envelope
    return clone


def integrity_sha256(payload: Dict[str, Any]) -> str:
    """Unkeyed integrity hash for accidental-truncation detection, not authority."""
    material = json.dumps(
        _payload_without_fingerprint(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _required_namespaces(kind: str) -> Tuple[str, ...]:
    if kind not in _REQUIRED_NAMESPACES:
        raise ValueError(f"unsupported canonical runtime kind: {kind}")
    return _REQUIRED_NAMESPACES[kind]


def _assert_required_namespaces(payload: Dict[str, Any], kind: str) -> None:
    missing = [name for name in _required_namespaces(kind) if name not in payload]
    if missing:
        raise ValueError(
            "canonical BODY checkpoint is missing required state namespaces: "
            + ", ".join(missing)
        )


def checkpoint_dict(runtime: PersistentCognitiveRuntime) -> Dict[str, Any]:
    """Serialize the deepest developmental state owned by this exact BODY.

    Calling this one function on a G5-G7 runtime cannot silently fall back to the
    shallower epistemic/base checkpoint contracts. The runtime type selects the
    most-specific serializer, then the canonical envelope records which state
    namespaces are mandatory for future reconstruction.
    """
    kind = _runtime_kind(runtime)
    if kind == KIND_PRIMITIVE:
        payload = primitive_checkpoint_dict(runtime)
    elif kind == KIND_EPISTEMIC:
        payload = epistemic_checkpoint_dict(runtime)
    else:
        payload = base_checkpoint_dict(runtime)

    required = _required_namespaces(kind)
    _assert_required_namespaces(payload, kind)
    payload["canonical_body"] = {
        "schema": CANONICAL_BODY_SCHEMA,
        "runtime_kind": kind,
        "required_namespaces": list(required),
        "integrity_sha256": "",
        "authority_note": (
            "integrity hash is unkeyed and non-authoritative; external evidence "
            "must still be reverified after restore"
        ),
    }
    payload["canonical_body"]["integrity_sha256"] = integrity_sha256(payload)
    return payload


def checkpoint_json(runtime: PersistentCognitiveRuntime) -> str:
    return json.dumps(
        checkpoint_dict(runtime),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _inferred_kind(payload: Dict[str, Any]) -> str:
    if payload.get("primitive_development_schema") is not None:
        return KIND_PRIMITIVE
    if payload.get("epistemic_depth_schema") is not None:
        return KIND_EPISTEMIC
    return KIND_BASE


def _validate_canonical_envelope(payload: Dict[str, Any]) -> str:
    envelope = payload.get("canonical_body")
    inferred = _inferred_kind(payload)
    if envelope is None:
        # Legacy checkpoints remain readable. They receive no canonical-integrity
        # claim, but restore still chooses the deepest schema actually present.
        return inferred
    if envelope.get("schema") != CANONICAL_BODY_SCHEMA:
        raise ValueError("unsupported canonical BODY checkpoint schema")
    declared = str(envelope.get("runtime_kind", ""))
    _required_namespaces(declared)
    if declared != inferred:
        raise ValueError(
            f"canonical BODY runtime downcast/schema mismatch: declared={declared}, inferred={inferred}"
        )
    declared_required = tuple(str(value) for value in envelope.get("required_namespaces", ()))
    expected_required = _required_namespaces(declared)
    if declared_required != expected_required:
        raise ValueError("canonical BODY required-namespace contract mismatch")
    _assert_required_namespaces(payload, declared)
    expected_hash = str(envelope.get("integrity_sha256", ""))
    if not expected_hash or expected_hash != integrity_sha256(payload):
        raise ValueError("canonical BODY checkpoint integrity mismatch")
    return declared


def restore_runtime(
    payload: Dict[str, Any],
    world_verifier: Optional[WorldReceiptVerifier] = None,
    raw_observation_verifier: Optional[RawObservationVerifier] = None,
) -> PersistentCognitiveRuntime:
    """Restore the deepest runtime represented by the checkpoint, fail-closed."""
    kind = _validate_canonical_envelope(payload)
    if kind == KIND_PRIMITIVE:
        runtime = restore_world_driven_primitive_runtime(
            payload,
            world_verifier=world_verifier,
            raw_observation_verifier=raw_observation_verifier,
        )
    elif kind == KIND_EPISTEMIC:
        runtime = restore_epistemic_runtime(payload, world_verifier=world_verifier)
    else:
        runtime = restore_base_runtime(payload, world_verifier=world_verifier)

    restored_kind = _runtime_kind(runtime)
    if restored_kind != kind:
        raise ValueError(
            f"canonical BODY restore type mismatch: expected={kind}, restored={restored_kind}"
        )
    return runtime


def restore_json(
    text: str,
    world_verifier: Optional[WorldReceiptVerifier] = None,
    raw_observation_verifier: Optional[RawObservationVerifier] = None,
) -> PersistentCognitiveRuntime:
    return restore_runtime(
        json.loads(text),
        world_verifier=world_verifier,
        raw_observation_verifier=raw_observation_verifier,
    )
