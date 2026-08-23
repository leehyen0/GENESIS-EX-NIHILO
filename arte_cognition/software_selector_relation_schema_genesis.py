from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import itertools
import json
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .software_call_binding_representation import (
    CallBindingSchema,
    UpstreamSelectorRepresentation,
    select_upstream_patch_with_representation,
)
from .software_selector_representation_program_genesis import remap_failure_line_after_normalization
from .software_upstream_failure_locus_genesis import UpstreamPatchCandidate
from .software_upstream_patch_discrimination import UpstreamPatchSelector
from .world_coupling import WorldOutcomePair


SCHEMA_MARKER = "selector_binding_schema="
SCHEMA_RELATIONS_MARKER = "selector_binding_schema_relations="


@dataclass(frozen=True)
class BindingSchemaInexpressivityAssessment:
    status: str
    inexpressive_contexts: Tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class GeneratedBindingSchema:
    """A source-derived relation schema, not a member of a fixed primitive name list."""

    relation_signatures: Tuple[str, ...]

    @property
    def schema_id(self) -> str:
        payload = json.dumps(self.relation_signatures, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        return f"SELECTOR_BINDING_SCHEMA::{digest}"


@dataclass(frozen=True)
class BindingSchemaProposal:
    schema: GeneratedBindingSchema
    proposal: InterventionProposal


@dataclass(frozen=True)
class BindingSchemaPolicy:
    status: str
    schema_id: Optional[str]
    supporting_contexts: Tuple[str, ...]
    candidate_schema_count: int
    reason: str


def assess_binding_schema_inexpressivity(
    contexts: Sequence[Tuple[str, int, int]],
    min_contexts: int = 2,
) -> BindingSchemaInexpressivityAssessment:
    inexpressive = tuple(sorted(
        str(context_id)
        for context_id, frontier_count, selected_count in contexts
        if int(frontier_count) > 0 and int(selected_count) == 0
    ))
    if len(inexpressive) < max(1, int(min_contexts)):
        return BindingSchemaInexpressivityAssessment(
            status="INSUFFICIENT_BINDING_SCHEMA_INEXPRESSIVITY",
            inexpressive_contexts=inexpressive,
            reason="schema genesis requires repeated nonempty frontiers unreachable by the inherited representation program",
        )
    return BindingSchemaInexpressivityAssessment(
        status="BINDING_SCHEMA_INEXPRESSIVE_OPEN_RELATION_GENESIS",
        inexpressive_contexts=inexpressive,
        reason="repeated failure persists while the inherited repair frontier remains nonempty",
    )


def _module_bindings(tree: ast.Module) -> Dict[str, ast.AST]:
    """Return only simple static module bindings; dynamic expressions stay outside authority."""
    bindings: Dict[str, ast.AST] = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = stmt.value
        if isinstance(value, (ast.Name, ast.Dict, ast.Tuple, ast.List)):
            bindings[target.id] = value
    return bindings


def _resolved_name(name: str, bindings: Dict[str, ast.AST]) -> str:
    seen = set()
    current = str(name)
    while current not in seen:
        seen.add(current)
        value = bindings.get(current)
        if not isinstance(value, ast.Name):
            return current
        current = value.id
    return current


def _relation_signature(producer: ast.AST, consumer_parent: ast.AST, consumer_field: str) -> str:
    """Describe a relation through AST types/field role, erasing concrete identifiers and literals."""
    return f"{type(producer).__name__}->{type(consumer_parent).__name__}.{consumer_field}"


def _literal_dict_items(node: ast.Dict) -> Optional[Tuple[Tuple[str, ast.AST], ...]]:
    items = []
    seen = set()
    for key, value in zip(node.keys, node.values):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            return None
        name = str(key.value)
        if name in seen:
            return None
        seen.add(name)
        items.append((name, value))
    return tuple(items)


def discover_binding_relation_signatures(
    source: str,
    target_function_name: str,
) -> Tuple[str, ...]:
    """Generate relation candidates from the source graph without consulting execution outcomes."""
    tree = ast.parse(str(source))
    bindings = _module_bindings(tree)
    signatures = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if isinstance(node.func, ast.Name) and node.func.id in bindings:
            producer = bindings[node.func.id]
            if (
                isinstance(producer, ast.Name)
                and _resolved_name(node.func.id, bindings) == str(target_function_name)
            ):
                signatures.add(_relation_signature(producer, node, "func"))

        for keyword in node.keywords:
            if keyword.arg is not None or not isinstance(keyword.value, ast.Name):
                continue
            producer = bindings.get(keyword.value.id)
            if isinstance(producer, ast.Dict) and _literal_dict_items(producer) is not None:
                signatures.add(_relation_signature(producer, node, "keywords**"))

        for arg in node.args:
            if not isinstance(arg, ast.Starred) or not isinstance(arg.value, ast.Name):
                continue
            producer = bindings.get(arg.value.id)
            if isinstance(producer, (ast.Tuple, ast.List)):
                signatures.add(_relation_signature(producer, node, "args*"))

    return tuple(sorted(signatures))


def generate_binding_schema_candidates(
    assessment: BindingSchemaInexpressivityAssessment,
    sources: Sequence[str],
    target_function_name: str,
    max_relations: int = 3,
) -> Tuple[GeneratedBindingSchema, ...]:
    """Generate schema subsets from relations reproduced across contexts, not a named primitive catalogue."""
    if assessment.status != "BINDING_SCHEMA_INEXPRESSIVE_OPEN_RELATION_GENESIS" or not sources:
        return ()
    relation_sets = [
        set(discover_binding_relation_signatures(source, target_function_name))
        for source in sources
    ]
    common = set.intersection(*relation_sets) if relation_sets else set()
    ordered = tuple(sorted(common))
    if not ordered:
        return ()
    limit = max(1, min(int(max_relations), len(ordered)))
    candidates = []
    for size in range(1, limit + 1):
        for combo in itertools.combinations(ordered, size):
            candidates.append(GeneratedBindingSchema(tuple(combo)))
    return tuple(candidates)


def propose_binding_schema(schema: GeneratedBindingSchema) -> BindingSchemaProposal:
    digest = hashlib.sha256(schema.schema_id.encode("utf-8")).hexdigest()[:20]
    proposal = InterventionProposal(
        experiment_id=f"SOFTWARE_SELECTOR_BINDING_SCHEMA::{digest}",
        axis_id=f"AXIS::SOFTWARE_SELECTOR_BINDING_SCHEMA::{digest}",
        manipulated_variable=schema.schema_id,
        held_fixed=(),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="INHERITED_REPRESENTATION_PROGRAM_INEXPRESSIVE",
        predicted_high_side="SOURCE_DERIVED_BINDING_SCHEMA_ENABLES_PRE_OUTCOME_SELECTION",
        reason=(
            "outcome-independent source-derived relation schema; "
            f"{SCHEMA_MARKER}{schema.schema_id} "
            f"{SCHEMA_RELATIONS_MARKER}{'|'.join(schema.relation_signatures)}"
        ),
        status="PROPOSAL_ONLY",
    )
    return BindingSchemaProposal(schema=schema, proposal=proposal)


def normalize_source_with_binding_schema(
    source: str,
    schema: GeneratedBindingSchema,
    target_function_name: str,
) -> str:
    """Interpret generated relation signatures with a generic static-binding inliner."""
    tree = ast.parse(str(source))
    bindings = _module_bindings(tree)
    enabled = set(schema.relation_signatures)

    class Transformer(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call):
            self.generic_visit(node)

            if isinstance(node.func, ast.Name) and node.func.id in bindings:
                producer = bindings[node.func.id]
                signature = _relation_signature(producer, node, "func")
                if (
                    signature in enabled
                    and isinstance(producer, ast.Name)
                    and _resolved_name(node.func.id, bindings) == str(target_function_name)
                ):
                    node.func = ast.copy_location(
                        ast.Name(id=str(target_function_name), ctx=ast.Load()),
                        node.func,
                    )

            expanded_keywords = []
            occupied = {item.arg for item in node.keywords if item.arg is not None}
            for keyword in node.keywords:
                if keyword.arg is not None or not isinstance(keyword.value, ast.Name):
                    expanded_keywords.append(keyword)
                    continue
                producer = bindings.get(keyword.value.id)
                if not isinstance(producer, ast.Dict):
                    expanded_keywords.append(keyword)
                    continue
                signature = _relation_signature(producer, node, "keywords**")
                items = _literal_dict_items(producer)
                if signature not in enabled or items is None:
                    expanded_keywords.append(keyword)
                    continue
                names = [name for name, _ in items]
                if any(name in occupied for name in names):
                    expanded_keywords.append(keyword)
                    continue
                for name, value in items:
                    occupied.add(name)
                    expanded_keywords.append(ast.keyword(arg=name, value=value))
            node.keywords = expanded_keywords

            expanded_args = []
            for arg in node.args:
                if not isinstance(arg, ast.Starred) or not isinstance(arg.value, ast.Name):
                    expanded_args.append(arg)
                    continue
                producer = bindings.get(arg.value.id)
                if not isinstance(producer, (ast.Tuple, ast.List)):
                    expanded_args.append(arg)
                    continue
                signature = _relation_signature(producer, node, "args*")
                if signature not in enabled:
                    expanded_args.append(arg)
                    continue
                expanded_args.extend(producer.elts)
            node.args = expanded_args
            return node

    tree = Transformer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def select_upstream_patch_with_binding_schema(
    selector: UpstreamPatchSelector,
    representation: UpstreamSelectorRepresentation,
    schema: GeneratedBindingSchema,
    candidates: Sequence[UpstreamPatchCandidate],
    source: str,
    failure_line: int,
    call_schema: CallBindingSchema,
) -> Optional[UpstreamPatchCandidate]:
    """Normalize a shadow copy using the generated schema, then preserve original candidate identity."""
    normalized_source = normalize_source_with_binding_schema(
        source, schema, call_schema.function_name
    )
    normalized_failure_line = remap_failure_line_after_normalization(
        source, failure_line, normalized_source
    )
    if normalized_failure_line is None:
        return None

    normalized_candidates = []
    original_by_id = {}
    for candidate in candidates:
        try:
            patched = normalize_source_with_binding_schema(
                candidate.patched_source, schema, call_schema.function_name
            )
        except (SyntaxError, ValueError):
            continue
        shadow = UpstreamPatchCandidate(
            program_id=candidate.program_id,
            candidate_id=candidate.candidate_id,
            patched_source=patched,
            operation_count=candidate.operation_count,
            oracle_fingerprint_sha256=candidate.oracle_fingerprint_sha256,
        )
        normalized_candidates.append(shadow)
        original_by_id[candidate.candidate_id] = candidate

    chosen = select_upstream_patch_with_representation(
        selector,
        representation,
        tuple(normalized_candidates),
        normalized_source,
        normalized_failure_line,
        call_schema,
    )
    if chosen is None:
        return None
    return original_by_id.get(chosen.candidate_id)


def _parse_schema_id(proposal: InterventionProposal) -> Optional[str]:
    reason = str(proposal.reason)
    if SCHEMA_MARKER not in reason:
        return None
    return reason.split(SCHEMA_MARKER, 1)[1].strip().split()[0].rstrip(",;)") or None


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def derive_binding_schema_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> BindingSchemaPolicy:
    schema_by_experiment: Dict[str, str] = {}
    all_schemas = set()
    for proposal in proposals:
        schema_id = _parse_schema_id(proposal)
        if schema_id:
            schema_by_experiment[proposal.experiment_id] = schema_id
            all_schemas.add(schema_id)

    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if pair.experiment_id not in schema_by_experiment or not _authoritative(pair):
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(
            pair.independence_class_id, pair
        )

    required = max(1, int(min_independent_classes))
    support: Dict[str, Dict[str, float]] = {}
    for (experiment_id, context_id), by_class in grouped.items():
        if len(by_class) < required:
            continue
        score = sum(max(0.0, float(pair.effect)) for pair in by_class.values()) / len(by_class)
        if score >= float(strong_effect_threshold):
            schema_id = schema_by_experiment[experiment_id]
            support.setdefault(schema_id, {})[context_id] = score

    eligible = []
    for schema_id, contexts in support.items():
        if len(contexts) < max(1, int(min_contexts)):
            continue
        eligible.append((
            -len(contexts),
            -sum(contexts.values()) / len(contexts),
            schema_id,
            tuple(sorted(contexts)),
        ))
    eligible.sort()

    if not eligible:
        return BindingSchemaPolicy(
            status="NO_REPRODUCED_BINDING_SCHEMA",
            schema_id=None,
            supporting_contexts=(),
            candidate_schema_count=len(all_schemas),
            reason="no source-derived binding schema has repeated verifier-derived support",
        )
    chosen = eligible[0]
    return BindingSchemaPolicy(
        status="REPRODUCED_BINDING_SCHEMA",
        schema_id=chosen[2],
        supporting_contexts=chosen[3],
        candidate_schema_count=len(all_schemas),
        reason="source-derived relation schema retained by repeated external executable outcomes",
    )


def select_authorized_binding_schema(
    schemas: Sequence[GeneratedBindingSchema],
    policy: BindingSchemaPolicy,
) -> Optional[GeneratedBindingSchema]:
    if policy.status != "REPRODUCED_BINDING_SCHEMA" or not policy.schema_id:
        return None
    return next((item for item in schemas if item.schema_id == policy.schema_id), None)


class BindingSchemaOrgan:
    def __init__(self, body) -> None:
        self.body = body

    def remember(self, proposals: Sequence[BindingSchemaProposal]) -> None:
        for item in proposals:
            self.body.memory.remember_experiment(item.proposal)

    def policy(self) -> BindingSchemaPolicy:
        return derive_binding_schema_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )
