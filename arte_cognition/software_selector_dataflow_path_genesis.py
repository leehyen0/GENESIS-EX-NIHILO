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


PATH_MARKER = "selector_dataflow_path_schema="
PATHS_MARKER = "selector_dataflow_paths="


@dataclass(frozen=True)
class DataflowPathInexpressivityAssessment:
    status: str
    inexpressive_contexts: Tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class GeneratedDataflowPathSchema:
    path_signatures: Tuple[str, ...]

    @property
    def schema_id(self) -> str:
        material = json.dumps(self.path_signatures, separators=(",", ":"))
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
        return f"SELECTOR_DATAFLOW_PATH_SCHEMA::{digest}"


@dataclass(frozen=True)
class DataflowPathProposal:
    schema: GeneratedDataflowPathSchema
    proposal: InterventionProposal


@dataclass(frozen=True)
class DataflowPathPolicy:
    status: str
    schema_id: Optional[str]
    supporting_contexts: Tuple[str, ...]
    candidate_schema_count: int
    reason: str


@dataclass(frozen=True)
class _ResolvedValue:
    node: ast.AST
    path: Tuple[str, ...]


def assess_dataflow_path_inexpressivity(
    contexts: Sequence[Tuple[str, int, int]],
    min_contexts: int = 2,
) -> DataflowPathInexpressivityAssessment:
    inexpressive = tuple(sorted(
        str(context_id)
        for context_id, frontier_count, selected_count in contexts
        if int(frontier_count) > 0 and int(selected_count) == 0
    ))
    if len(inexpressive) < max(1, int(min_contexts)):
        return DataflowPathInexpressivityAssessment(
            status="INSUFFICIENT_DATAFLOW_PATH_INEXPRESSIVITY",
            inexpressive_contexts=inexpressive,
            reason="path genesis requires repeated nonempty repair frontiers unreachable by the inherited static relation schema",
        )
    return DataflowPathInexpressivityAssessment(
        status="DATAFLOW_PATH_INEXPRESSIVE_OPEN_PATH_GENESIS",
        inexpressive_contexts=inexpressive,
        reason="repeated source-local static relation failure leaves a live repair frontier and opens deeper def-use path generation",
    )


def _module_bindings(tree: ast.Module) -> Dict[str, ast.AST]:
    bindings: Dict[str, ast.AST] = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            bindings[target.id] = stmt.value
    return bindings


def _local_functions(tree: ast.Module) -> Dict[str, ast.FunctionDef]:
    return {
        stmt.name: stmt
        for stmt in tree.body
        if isinstance(stmt, ast.FunctionDef)
    }


def _single_return(function: ast.FunctionDef) -> Optional[ast.AST]:
    body = [stmt for stmt in function.body if not isinstance(stmt, ast.Pass)]
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return None
    return body[0].value


def _resolve_value(
    expr: ast.AST,
    bindings: Dict[str, ast.AST],
    functions: Dict[str, ast.FunctionDef],
    *,
    seen_names: Tuple[str, ...] = (),
    depth: int = 0,
    max_depth: int = 6,
) -> Optional[_ResolvedValue]:
    if depth > int(max_depth):
        return None
    if isinstance(expr, (ast.Dict, ast.Tuple, ast.List)):
        return _ResolvedValue(expr, (f"VALUE:{type(expr).__name__}",))
    if isinstance(expr, ast.Name):
        if expr.id in seen_names:
            return None
        bound = bindings.get(expr.id)
        if bound is None:
            return _ResolvedValue(expr, ("UNBOUND_NAME",))
        resolved = _resolve_value(
            bound,
            bindings,
            functions,
            seen_names=seen_names + (expr.id,),
            depth=depth + 1,
            max_depth=max_depth,
        )
        if resolved is None:
            return None
        return _ResolvedValue(resolved.node, ("MODULE_BINDING",) + resolved.path)
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and not expr.args
        and not expr.keywords
        and expr.func.id in functions
    ):
        returned = _single_return(functions[expr.func.id])
        if returned is None:
            return None
        resolved = _resolve_value(
            returned,
            bindings,
            functions,
            seen_names=seen_names,
            depth=depth + 1,
            max_depth=max_depth,
        )
        if resolved is None:
            return None
        return _ResolvedValue(
            resolved.node,
            ("LOCAL_ZEROARG_CALL", "FUNCTION_RETURN") + resolved.path,
        )
    return None


def _resolve_target_callable_path(
    func: ast.AST,
    bindings: Dict[str, ast.AST],
    target_function_name: str,
    max_depth: int = 6,
) -> Optional[Tuple[str, ...]]:
    current = func
    path = []
    seen = set()
    for _ in range(max(1, int(max_depth))):
        if not isinstance(current, ast.Name):
            return None
        if current.id == str(target_function_name):
            return tuple(path + ["TARGET_CALLABLE"])
        if current.id in seen:
            return None
        seen.add(current.id)
        bound = bindings.get(current.id)
        if not isinstance(bound, ast.Name):
            return None
        path.append("MODULE_NAME_ALIAS")
        current = bound
    return None


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


def discover_dataflow_path_signatures(
    source: str,
    target_function_name: str,
) -> Tuple[str, ...]:
    """Derive bounded def-use path signatures from source only, without execution labels."""
    tree = ast.parse(str(source))
    bindings = _module_bindings(tree)
    functions = _local_functions(tree)
    signatures = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callable_path = _resolve_target_callable_path(node.func, bindings, target_function_name)
        if callable_path:
            signatures.add("->".join(callable_path + ("CONSUMER:Call.func",)))
        for keyword in node.keywords:
            if keyword.arg is not None or not isinstance(keyword.value, ast.Name):
                continue
            resolved = _resolve_value(keyword.value, bindings, functions)
            if resolved is None or not isinstance(resolved.node, ast.Dict):
                continue
            if _literal_dict_items(resolved.node) is None:
                continue
            signatures.add("->".join(resolved.path + ("CONSUMER:Call.keywords**",)))
        for arg in node.args:
            if not isinstance(arg, ast.Starred) or not isinstance(arg.value, ast.Name):
                continue
            resolved = _resolve_value(arg.value, bindings, functions)
            if resolved is None or not isinstance(resolved.node, (ast.Tuple, ast.List)):
                continue
            signatures.add("->".join(resolved.path + ("CONSUMER:Call.args*",)))
    return tuple(sorted(signatures))


def generate_dataflow_path_schema_candidates(
    assessment: DataflowPathInexpressivityAssessment,
    sources: Sequence[str],
    target_function_name: str,
    max_paths: int = 3,
) -> Tuple[GeneratedDataflowPathSchema, ...]:
    if assessment.status != "DATAFLOW_PATH_INEXPRESSIVE_OPEN_PATH_GENESIS" or not sources:
        return ()
    observed = [
        set(discover_dataflow_path_signatures(source, target_function_name))
        for source in sources
    ]
    common = set.intersection(*observed) if observed else set()
    ordered = tuple(sorted(common))
    if not ordered:
        return ()
    limit = max(1, min(int(max_paths), len(ordered)))
    candidates = []
    for size in range(1, limit + 1):
        for combo in itertools.combinations(ordered, size):
            candidates.append(GeneratedDataflowPathSchema(tuple(combo)))
    return tuple(candidates)


def propose_dataflow_path_schema(schema: GeneratedDataflowPathSchema) -> DataflowPathProposal:
    digest = hashlib.sha256(schema.schema_id.encode("utf-8")).hexdigest()[:20]
    proposal = InterventionProposal(
        experiment_id=f"SOFTWARE_SELECTOR_DATAFLOW_PATH::{digest}",
        axis_id=f"AXIS::SOFTWARE_SELECTOR_DATAFLOW_PATH::{digest}",
        manipulated_variable=schema.schema_id,
        held_fixed=(),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="STATIC_RELATION_SCHEMA_INEXPRESSIVE",
        predicted_high_side="SOURCE_DERIVED_DATAFLOW_PATH_ENABLES_PRE_OUTCOME_SELECTION",
        reason=(
            "outcome-independent bounded source-derived def-use path schema; "
            f"{PATH_MARKER}{schema.schema_id} "
            f"{PATHS_MARKER}{'|'.join(schema.path_signatures)}"
        ),
        status="PROPOSAL_ONLY",
    )
    return DataflowPathProposal(schema=schema, proposal=proposal)


def normalize_source_with_dataflow_path_schema(
    source: str,
    schema: GeneratedDataflowPathSchema,
    target_function_name: str,
) -> str:
    tree = ast.parse(str(source))
    bindings = _module_bindings(tree)
    functions = _local_functions(tree)
    enabled = set(schema.path_signatures)

    class Transformer(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call):
            self.generic_visit(node)
            callable_path = _resolve_target_callable_path(node.func, bindings, target_function_name)
            if callable_path:
                signature = "->".join(callable_path + ("CONSUMER:Call.func",))
                if signature in enabled:
                    node.func = ast.copy_location(
                        ast.Name(id=str(target_function_name), ctx=ast.Load()), node.func
                    )

            occupied = {item.arg for item in node.keywords if item.arg is not None}
            keywords = []
            for keyword in node.keywords:
                if keyword.arg is not None or not isinstance(keyword.value, ast.Name):
                    keywords.append(keyword)
                    continue
                resolved = _resolve_value(keyword.value, bindings, functions)
                if resolved is None or not isinstance(resolved.node, ast.Dict):
                    keywords.append(keyword)
                    continue
                signature = "->".join(resolved.path + ("CONSUMER:Call.keywords**",))
                items = _literal_dict_items(resolved.node)
                if signature not in enabled or items is None:
                    keywords.append(keyword)
                    continue
                names = [name for name, _ in items]
                if any(name in occupied for name in names):
                    keywords.append(keyword)
                    continue
                for name, value in items:
                    occupied.add(name)
                    keywords.append(ast.keyword(arg=name, value=value))
            node.keywords = keywords

            args = []
            for arg in node.args:
                if not isinstance(arg, ast.Starred) or not isinstance(arg.value, ast.Name):
                    args.append(arg)
                    continue
                resolved = _resolve_value(arg.value, bindings, functions)
                if resolved is None or not isinstance(resolved.node, (ast.Tuple, ast.List)):
                    args.append(arg)
                    continue
                signature = "->".join(resolved.path + ("CONSUMER:Call.args*",))
                if signature not in enabled:
                    args.append(arg)
                    continue
                args.extend(resolved.node.elts)
            node.args = args
            return node

    tree = Transformer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def select_upstream_patch_with_dataflow_path_schema(
    selector: UpstreamPatchSelector,
    representation: UpstreamSelectorRepresentation,
    schema: GeneratedDataflowPathSchema,
    candidates: Sequence[UpstreamPatchCandidate],
    source: str,
    failure_line: int,
    call_schema: CallBindingSchema,
) -> Optional[UpstreamPatchCandidate]:
    normalized_source = normalize_source_with_dataflow_path_schema(
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
            patched = normalize_source_with_dataflow_path_schema(
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
    if PATH_MARKER not in reason:
        return None
    return reason.split(PATH_MARKER, 1)[1].strip().split()[0].rstrip(",;)") or None


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def derive_dataflow_path_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> DataflowPathPolicy:
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
        return DataflowPathPolicy(
            status="NO_REPRODUCED_DATAFLOW_PATH_SCHEMA",
            schema_id=None,
            supporting_contexts=(),
            candidate_schema_count=len(all_schemas),
            reason="no generated path schema has repeated verifier-derived support",
        )
    chosen = eligible[0]
    return DataflowPathPolicy(
        status="REPRODUCED_DATAFLOW_PATH_SCHEMA",
        schema_id=chosen[2],
        supporting_contexts=chosen[3],
        candidate_schema_count=len(all_schemas),
        reason="source-derived def-use path retained by repeated external executable outcomes",
    )


def select_authorized_dataflow_path_schema(
    schemas: Sequence[GeneratedDataflowPathSchema],
    policy: DataflowPathPolicy,
) -> Optional[GeneratedDataflowPathSchema]:
    if policy.status != "REPRODUCED_DATAFLOW_PATH_SCHEMA" or not policy.schema_id:
        return None
    return next((schema for schema in schemas if schema.schema_id == policy.schema_id), None)


class DataflowPathOrgan:
    def __init__(self, body) -> None:
        self.body = body

    def remember(self, proposals: Sequence[DataflowPathProposal]) -> None:
        for item in proposals:
            self.body.memory.remember_experiment(item.proposal)

    def policy(self) -> DataflowPathPolicy:
        return derive_dataflow_path_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )
