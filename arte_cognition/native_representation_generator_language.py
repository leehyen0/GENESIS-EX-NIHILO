from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple
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
from .native_representation_genesis import RepresentationSupportExample


EXPRESSION_OPS = ("ADD", "SUB", "XOR", "MIN", "MAX")
RHS_SOURCES = ("x", "y")


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _apply_op(op: str, left: int, right: int) -> int:
    op = str(op).upper()
    left, right = int(left), int(right)
    if op == "ADD":
        return left + right
    if op == "SUB":
        return left - right
    if op == "XOR":
        if left < 0 or right < 0:
            raise ValueError("XOR_REQUIRES_NONNEGATIVE_INPUT")
        return left ^ right
    if op == "MIN":
        return min(left, right)
    if op == "MAX":
        return max(left, right)
    raise ValueError("UNKNOWN_EXPRESSION_OP")


@dataclass(frozen=True, order=True)
class ExpressionSpec:
    first_op: str
    second_op: str
    second_rhs: str

    def __post_init__(self) -> None:
        if self.first_op not in EXPRESSION_OPS or self.second_op not in EXPRESSION_OPS:
            raise ValueError("INVALID_EXPRESSION_OP")
        if self.second_rhs not in RHS_SOURCES:
            raise ValueError("INVALID_EXPRESSION_RHS")

    def token(self) -> str:
        return f"{self.first_op.lower()}~{self.second_op.lower()}~{self.second_rhs}"

    @classmethod
    def from_token(cls, token: str) -> "ExpressionSpec":
        parts = str(token).split("~")
        if len(parts) != 3:
            raise ValueError("MALFORMED_EXPRESSION_TOKEN")
        return cls(parts[0].upper(), parts[1].upper(), parts[2])

    def execute(self, pair: Tuple[int, int]) -> int:
        x, y = int(pair[0]), int(pair[1])
        first = _apply_op(self.first_op, x, y)
        rhs = x if self.second_rhs == "x" else y
        return _apply_op(self.second_op, first, rhs)

    def fingerprint(self) -> str:
        return _sha((self.first_op, self.second_op, self.second_rhs))


def expression_language() -> Tuple[ExpressionSpec, ...]:
    return tuple(
        ExpressionSpec(first, second, rhs)
        for first in EXPRESSION_OPS
        for second in EXPRESSION_OPS
        for rhs in RHS_SOURCES
    )


def infer_expression_spec(examples: Sequence[RepresentationSupportExample]) -> ExpressionSpec:
    if not examples:
        raise ValueError("EXPRESSION_SUPPORT_REQUIRED")
    matches = []
    for spec in expression_language():
        try:
            valid = all(spec.execute(row.input_pair) == int(row.output_value) for row in examples)
        except ValueError:
            valid = False
        if valid:
            matches.append(spec)
    if len(matches) != 1:
        raise ValueError("EXPRESSION_PROGRAM_NOT_IDENTIFIABLE")
    return matches[0]


@dataclass(frozen=True)
class RepresentationGeneratorPolicy:
    origin_residual_id: str
    max_binary_ops: int
    implementation_ref: str

    def fingerprint(self) -> str:
        return _sha((self.origin_residual_id, self.max_binary_ops, self.implementation_ref))


def generator_policy_ref(origin_residual_id: str, max_binary_ops: int = 2) -> str:
    if not origin_residual_id or "/" in origin_residual_id:
        raise ValueError("INVALID_GENERATOR_POLICY_ORIGIN")
    if int(max_binary_ops) != 2:
        raise ValueError("UNSUPPORTED_EXPRESSION_DEPTH")
    suffix = _sha((origin_residual_id, int(max_binary_ops), EXPRESSION_OPS, RHS_SOURCES))[:16]
    return f"native-repr-gen://compose2/{origin_residual_id}/{suffix}"


def compile_generator_policy(ref: str, *, expected_origin_residual_id: str | None = None) -> RepresentationGeneratorPolicy:
    prefix = "native-repr-gen://compose2/"
    if not str(ref).startswith(prefix):
        raise ValueError("NOT_NATIVE_REPRESENTATION_GENERATOR_POLICY")
    parts = str(ref)[len(prefix):].split("/")
    if len(parts) != 2:
        raise ValueError("MALFORMED_REPRESENTATION_GENERATOR_POLICY")
    origin, suffix = parts
    expected = generator_policy_ref(origin, 2).rsplit("/", 1)[-1]
    if suffix != expected:
        raise ValueError("REPRESENTATION_GENERATOR_POLICY_HASH_MISMATCH")
    if expected_origin_residual_id is not None and origin != str(expected_origin_residual_id):
        raise ValueError("REPRESENTATION_GENERATOR_POLICY_ORIGIN_MISMATCH")
    return RepresentationGeneratorPolicy(origin, 2, str(ref))


def generator_policies(genome: MorphologyGenome, *, expected_origin_residual_id: str | None = None) -> Tuple[RepresentationGeneratorPolicy, ...]:
    rows = []
    for organ in sorted(genome.organs, key=lambda row: row.organ_id):
        if not str(organ.implementation_ref).startswith("native-repr-gen://"):
            continue
        if organ.kind != OrganKind.GENERATOR:
            raise ValueError("REPRESENTATION_GENERATOR_POLICY_WRONG_ORGAN_KIND")
        rows.append(compile_generator_policy(organ.implementation_ref, expected_origin_residual_id=expected_origin_residual_id))
    return tuple(rows)


def derive_generator_language_mutation(
    genome: MorphologyGenome,
    *,
    origin_residual_id: str,
    failure_fossil: str,
) -> MorphologyMutation:
    generators = [organ for organ in genome.organs if organ.kind == OrganKind.GENERATOR and organ.enabled]
    if len(generators) != 1:
        raise ValueError("GENERATOR_LANGUAGE_MUTATION_REQUIRES_ONE_ACTIVE_GENERATOR")
    parent = generators[0]
    ref = generator_policy_ref(origin_residual_id, 2)
    replacement = {
        "organ_id": parent.organ_id,
        "kind": parent.kind.value,
        "consumes": list(parent.consumes),
        "produces": list(parent.produces),
        "implementation_ref": ref,
        "version": parent.version + 1,
        "cost_hint": parent.cost_hint,
        "provenance": list(parent.provenance)
        + [
            f"representation-generator-language-expansion::{origin_residual_id}",
            f"failure-fossil::{failure_fossil}",
            "fixed-family-parent-inadequacy::cycle6",
        ],
        "enabled": parent.enabled,
    }
    mutation_id = "NATIVE_REPR_GENERATOR_LANGUAGE::" + _sha(
        {"parent": genome.fingerprint(), "replacement": replacement, "origin": origin_residual_id}
    )[:20]
    return MorphologyMutation(
        mutation_id=mutation_id,
        level=MutationLevel.GENERATOR_MUTATOR,
        operation="REPLACE_ORGAN",
        payload={"organ": replacement},
        parent_body_hash=genome.fingerprint(),
        rationale=(
            f"failure-fossil::{failure_fossil}",
            f"generator-language-origin::{origin_residual_id}",
            "expand-fixed-family-selector-to-compositional-expression-search",
        ),
        reversible=True,
    )


@dataclass(frozen=True)
class ExpressionRepresentationProgram:
    spec: ExpressionSpec
    artifact_type: str
    origin_residual_id: str
    implementation_ref: str

    def execute(self, pair: Tuple[int, int]) -> int:
        return self.spec.execute(pair)

    def fingerprint(self) -> str:
        return _sha((self.spec.token(), self.artifact_type, self.origin_residual_id, self.implementation_ref))


def expression_representation_ref(spec: ExpressionSpec, artifact_type: str, residual_id: str) -> str:
    if not artifact_type or "/" in artifact_type or not residual_id or "/" in residual_id:
        raise ValueError("INVALID_EXPRESSION_REPRESENTATION_IDENTITY")
    suffix = _sha((spec.token(), artifact_type, residual_id))[:16]
    return f"native-repr-expr://{spec.token()}/{artifact_type}/{residual_id}/{suffix}"


def compile_expression_representation_ref(
    ref: str,
    *,
    expected_artifact_type: str | None = None,
    expected_residual_id: str | None = None,
) -> ExpressionRepresentationProgram:
    prefix = "native-repr-expr://"
    if not str(ref).startswith(prefix):
        raise ValueError("NOT_NATIVE_EXPRESSION_REPRESENTATION")
    parts = str(ref)[len(prefix):].split("/")
    if len(parts) != 4:
        raise ValueError("MALFORMED_NATIVE_EXPRESSION_REPRESENTATION")
    token, artifact, residual, suffix = parts
    spec = ExpressionSpec.from_token(token)
    expected = expression_representation_ref(spec, artifact, residual).rsplit("/", 1)[-1]
    if suffix != expected:
        raise ValueError("NATIVE_EXPRESSION_REPRESENTATION_HASH_MISMATCH")
    if expected_artifact_type is not None and artifact != str(expected_artifact_type):
        raise ValueError("NATIVE_EXPRESSION_ARTIFACT_MISMATCH")
    if expected_residual_id is not None and residual != str(expected_residual_id):
        raise ValueError("NATIVE_EXPRESSION_RESIDUAL_MISMATCH")
    return ExpressionRepresentationProgram(spec, artifact, residual, str(ref))


def expression_representation_programs(
    genome: MorphologyGenome,
    *,
    expected_artifact_type: str | None = None,
    expected_residual_id: str | None = None,
) -> Tuple[ExpressionRepresentationProgram, ...]:
    out = []
    for organ in sorted(genome.organs, key=lambda row: row.organ_id):
        if not str(organ.implementation_ref).startswith("native-repr-expr://"):
            continue
        if organ.kind != OrganKind.REPRESENTATION:
            raise ValueError("NATIVE_EXPRESSION_WRONG_ORGAN_KIND")
        program = compile_expression_representation_ref(
            organ.implementation_ref,
            expected_artifact_type=expected_artifact_type,
            expected_residual_id=expected_residual_id,
        )
        if tuple(organ.consumes) != ("raw_observation",) or tuple(organ.produces) != (program.artifact_type,):
            raise ValueError("NATIVE_EXPRESSION_TYPE_BINDING_MISMATCH")
        out.append(program)
    return tuple(out)


class CompositionalRepresentationGenesisEngine:
    def __init__(self, candidate_budget: int = 1) -> None:
        self.candidate_budget = max(1, int(candidate_budget))

    def generate(
        self,
        generator_genome: MorphologyGenome,
        residual: MorphologyResidual,
        support_examples: Sequence[RepresentationSupportExample],
        *,
        expected_generator_origin_residual_id: str,
        force_spec: ExpressionSpec | None = None,
    ) -> Tuple[MorphologyCandidate, ...]:
        policies = generator_policies(
            generator_genome,
            expected_origin_residual_id=expected_generator_origin_residual_id,
        )
        if len(policies) != 1:
            raise ValueError("COMPOSITIONAL_GENESIS_REQUIRES_INHERITED_GENERATOR_POLICY")
        if len(residual.missing_artifact_types) != 1:
            raise ValueError("COMPOSITIONAL_GENESIS_REQUIRES_ONE_MISSING_ARTIFACT")
        inferred = infer_expression_spec(support_examples)
        spec = inferred if force_spec is None else force_spec
        artifact = str(residual.missing_artifact_types[0])
        ref = expression_representation_ref(spec, artifact, residual.residual_id)
        suffix = _sha(
            {
                "generator_policy": policies[0].fingerprint(),
                "spec": spec.token(),
                "artifact": artifact,
                "residual": residual.residual_id,
                "support": [[list(row.input_pair), int(row.output_value)] for row in support_examples],
            }
        )[:16]
        organ = {
            "organ_id": f"representation::expr::{artifact}::{suffix}",
            "kind": OrganKind.REPRESENTATION.value,
            "consumes": ["raw_observation"],
            "produces": [artifact],
            "implementation_ref": ref,
            "version": 1,
            "cost_hint": 1.0,
            "provenance": [
                f"inherited-generator-policy::{policies[0].fingerprint()}",
                f"expression-search-support::{_sha([[list(row.input_pair), int(row.output_value)] for row in support_examples])}",
                f"expression-spec::{spec.token()}",
            ],
            "enabled": True,
        }
        mutation = MorphologyMutation(
            mutation_id="NATIVE_EXPR_REPRESENTATION::" + suffix,
            level=MutationLevel.REPRESENTATION_MEMORY_TOOL,
            operation="ADD_ORGAN",
            payload={"organ": organ},
            parent_body_hash=generator_genome.fingerprint(),
            rationale=(
                f"generator-policy::{policies[0].fingerprint()}",
                f"support-inferred-expression::{inferred.token()}",
                "hidden-query-outcome-not-consumed",
            ),
            reversible=True,
        )
        child = MorphologyMutator().apply(generator_genome, mutation)
        candidate = MorphologyCandidate(
            candidate_id="NATIVE_EXPR_REPRESENTATION_CANDIDATE::" + _sha(
                {"mutation": mutation.mutation_id, "child": child.fingerprint()}
            )[:20],
            mutation=mutation,
            descendant_fingerprint=child.fingerprint(),
            origin_residual_ids=(residual.residual_id,),
            operation_family="SYNTHESIZE_REPRESENTATION_EXPRESSION",
            generation_uses_outcomes=False,
        )
        return (candidate,)[: self.candidate_budget]
