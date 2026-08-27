from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple
import hashlib
import json

from .executable_morphology import (
    MorphologyGenome,
    MorphologyMutation,
    MorphologyMutator,
    MutationLevel,
    OrganKind,
)
from .morphology_genesis import MorphologyCandidate, MorphologyResidual


PRIMITIVE_FAMILIES = ("DIFFERENCE", "XOR", "ORDER_CANONICAL")


def _sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def apply_representation_primitive(family: str, pair: Tuple[int, int]) -> int:
    left, right = int(pair[0]), int(pair[1])
    family = str(family).upper()
    if family == "DIFFERENCE":
        return left - right
    if family == "XOR":
        if left < 0 or right < 0:
            raise ValueError("XOR_REQUIRES_NONNEGATIVE_INPUT")
        return left ^ right
    if family == "ORDER_CANONICAL":
        lo, hi = sorted((left, right))
        return lo * 257 + hi
    raise ValueError("UNKNOWN_NATIVE_REPRESENTATION_PRIMITIVE")


@dataclass(frozen=True)
class RepresentationSupportExample:
    input_pair: Tuple[int, int]
    output_value: int


@dataclass(frozen=True)
class NativeRepresentationProgram:
    family: str
    artifact_type: str
    origin_residual_id: str
    implementation_ref: str

    def fingerprint(self) -> str:
        return _sha256(
            {
                "family": self.family,
                "artifact_type": self.artifact_type,
                "origin_residual_id": self.origin_residual_id,
                "implementation_ref": self.implementation_ref,
            }
        )

    def execute(self, pair: Tuple[int, int]) -> int:
        return apply_representation_primitive(self.family, pair)


def infer_representation_family(examples: Sequence[RepresentationSupportExample]) -> str:
    if not examples:
        raise ValueError("REPRESENTATION_SUPPORT_REQUIRED")
    matches = []
    for family in PRIMITIVE_FAMILIES:
        if all(
            apply_representation_primitive(family, row.input_pair) == int(row.output_value)
            for row in examples
        ):
            matches.append(family)
    if len(matches) != 1:
        raise ValueError("REPRESENTATION_PRIMITIVE_NOT_IDENTIFIABLE")
    return matches[0]


def representation_ref(family: str, artifact_type: str, residual_id: str) -> str:
    family = str(family).upper()
    if family not in PRIMITIVE_FAMILIES:
        raise ValueError("UNKNOWN_NATIVE_REPRESENTATION_PRIMITIVE")
    if not artifact_type or "/" in artifact_type:
        raise ValueError("INVALID_REPRESENTATION_ARTIFACT_TYPE")
    if not residual_id or "/" in residual_id:
        raise ValueError("INVALID_REPRESENTATION_RESIDUAL_ID")
    suffix = _sha256({"family": family, "artifact": artifact_type, "residual": residual_id})[:16]
    return f"native-repr://{family.lower()}/{artifact_type}/{residual_id}/{suffix}"


def compile_representation_ref(
    implementation_ref: str,
    *,
    expected_artifact_type: str | None = None,
    expected_residual_id: str | None = None,
) -> NativeRepresentationProgram:
    prefix = "native-repr://"
    if not str(implementation_ref).startswith(prefix):
        raise ValueError("NOT_NATIVE_REPRESENTATION_REF")
    parts = str(implementation_ref)[len(prefix):].split("/")
    if len(parts) != 4:
        raise ValueError("MALFORMED_NATIVE_REPRESENTATION_REF")
    family_raw, artifact_type, residual_id, suffix = parts
    family = family_raw.upper()
    expected = representation_ref(family, artifact_type, residual_id).rsplit("/", 1)[-1]
    if suffix != expected:
        raise ValueError("NATIVE_REPRESENTATION_REF_HASH_MISMATCH")
    if expected_artifact_type is not None and artifact_type != str(expected_artifact_type):
        raise ValueError("NATIVE_REPRESENTATION_ARTIFACT_MISMATCH")
    if expected_residual_id is not None and residual_id != str(expected_residual_id):
        raise ValueError("NATIVE_REPRESENTATION_RESIDUAL_MISMATCH")
    return NativeRepresentationProgram(family, artifact_type, residual_id, str(implementation_ref))


def executable_representation_programs(
    genome: MorphologyGenome,
    *,
    expected_artifact_type: str | None = None,
    expected_residual_id: str | None = None,
) -> Tuple[NativeRepresentationProgram, ...]:
    programs = []
    for organ in sorted(genome.organs, key=lambda row: row.organ_id):
        if not str(organ.implementation_ref).startswith("native-repr://"):
            continue
        if organ.kind != OrganKind.REPRESENTATION:
            raise ValueError("NATIVE_REPRESENTATION_WRONG_ORGAN_KIND")
        program = compile_representation_ref(
            organ.implementation_ref,
            expected_artifact_type=expected_artifact_type,
            expected_residual_id=expected_residual_id,
        )
        if tuple(organ.consumes) != ("raw_observation",):
            raise ValueError("NATIVE_REPRESENTATION_INPUT_TYPE_MISMATCH")
        if tuple(organ.produces) != (program.artifact_type,):
            raise ValueError("NATIVE_REPRESENTATION_OUTPUT_TYPE_MISMATCH")
        programs.append(program)
    return tuple(programs)


class NativeRepresentationGenesisEngine:
    """Generate an executable L1 representation only after old-language alias evidence.

    The engine consumes pre-outcome support examples and a typed missing-artifact
    requirement. It cannot use hidden query outcomes. A unique primitive must be
    identifiable from support before a mutation is emitted.
    """

    def __init__(self, candidate_budget: int = 1) -> None:
        self.candidate_budget = max(1, int(candidate_budget))

    def generate(
        self,
        genome: MorphologyGenome,
        residual: MorphologyResidual,
        support_examples: Sequence[RepresentationSupportExample],
        *,
        force_family: str | None = None,
    ) -> Tuple[MorphologyCandidate, ...]:
        if not residual.same_frozen_phenotype_different_outcome or not residual.more_compute_still_aliased:
            raise ValueError("REPRESENTATION_ESCAPE_NOT_AUTHORIZED")
        if len(residual.missing_artifact_types) != 1:
            raise ValueError("REPRESENTATION_ESCAPE_REQUIRES_ONE_MISSING_ARTIFACT")
        artifact_type = str(residual.missing_artifact_types[0])
        inferred = infer_representation_family(support_examples)
        family = inferred if force_family is None else str(force_family).upper()
        if family not in PRIMITIVE_FAMILIES:
            raise ValueError("UNKNOWN_NATIVE_REPRESENTATION_PRIMITIVE")

        parent_hash = genome.fingerprint()
        ref = representation_ref(family, artifact_type, residual.residual_id)
        suffix = _sha256(
            {
                "parent": parent_hash,
                "residual": residual.residual_id,
                "artifact": artifact_type,
                "family": family,
                "support": [
                    [list(row.input_pair), int(row.output_value)] for row in support_examples
                ],
            }
        )[:16]
        organ_id = f"representation::{artifact_type}::{suffix}"
        organ = {
            "organ_id": organ_id,
            "kind": OrganKind.REPRESENTATION.value,
            "consumes": ["raw_observation"],
            "produces": [artifact_type],
            "implementation_ref": ref,
            "version": 1,
            "cost_hint": 1.0,
            "provenance": [
                f"native-representation-pressure::{residual.residual_id}",
                f"old-language-more-compute-aliased::{residual.more_compute_still_aliased}",
                f"support-fingerprint::{_sha256([[list(row.input_pair), int(row.output_value)] for row in support_examples])}",
            ],
            "enabled": True,
        }
        mutation = MorphologyMutation(
            mutation_id="NATIVE_L1_REPRESENTATION::" + suffix,
            level=MutationLevel.REPRESENTATION_MEMORY_TOOL,
            operation="ADD_ORGAN",
            payload={"organ": organ},
            parent_body_hash=parent_hash,
            rationale=(
                f"representation-pressure::{residual.residual_id}",
                f"missing-artifact::{artifact_type}",
                f"inferred-family::{inferred}",
                "pre-outcome-support-only",
            ),
            reversible=True,
        )
        descendant = MorphologyMutator().apply(genome, mutation)
        candidate = MorphologyCandidate(
            candidate_id="NATIVE_L1_REPRESENTATION_CANDIDATE::" + _sha256(
                {"mutation": mutation.mutation_id, "descendant": descendant.fingerprint()}
            )[:20],
            mutation=mutation,
            descendant_fingerprint=descendant.fingerprint(),
            origin_residual_ids=(residual.residual_id,),
            operation_family="ADD_REPRESENTATION_OPERATOR",
            generation_uses_outcomes=False,
        )
        return (candidate,)[: self.candidate_budget]
