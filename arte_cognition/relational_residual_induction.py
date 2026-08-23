from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import hashlib

from .world_coupling import WorldOutcomePair


@dataclass(frozen=True, order=True)
class RelationalEdge:
    source: str
    relation: str
    target: str
    source_kind: str
    target_kind: str

    @property
    def abstract_step(self) -> str:
        return f"{self.source_kind}-[{self.relation}]->{self.target_kind}"


@dataclass(frozen=True)
class RelationalContext:
    context_id: str
    edges: Tuple[RelationalEdge, ...]
    source_anchor: str
    target_anchor: str
    domain: str = "UNSPECIFIED"


@dataclass(frozen=True)
class GeneratedRelationalPathSchema:
    steps: Tuple[str, ...]

    @property
    def schema_id(self) -> str:
        raw = "|".join(self.steps).encode()
        return "RELATIONAL_PATH_SCHEMA::" + hashlib.sha256(raw).hexdigest()[:20]


@dataclass(frozen=True)
class RelationalResidualAssessment:
    status: str
    context_ids: Tuple[str, ...]
    domains: Tuple[str, ...]
    repeated_failure: bool


@dataclass(frozen=True)
class RelationalPathPolicy:
    allowed_schema_ids: Tuple[str, ...]
    supporting_contexts: Tuple[Tuple[str, Tuple[str, ...]], ...]
    min_independent_classes: int
    min_contexts: int


class RelationalResidualInducer:
    """Generate identifier-invariant relation paths from repeated structural residuals.

    The inducer knows only directed typed edges. It does not know Python AST node
    classes, causal primitive names, world outcomes, or domain-specific repair
    semantics. Domain adapters may expose a structural graph, but candidate path
    schemas are generated solely from graph topology before outcome evidence is
    consulted.

    This is deliberately bounded: edge labels/kinds and graph adapters are still
    human-authored, and path enumeration is limited by max_depth. It therefore
    supports cross-domain structural induction, not unrestricted meta-language
    genesis.
    """

    def __init__(self, max_depth: int = 6, candidate_budget: int = 64) -> None:
        self.max_depth = max(1, int(max_depth))
        self.candidate_budget = max(1, int(candidate_budget))

    @staticmethod
    def assess_repeated_residual(
        contexts: Sequence[RelationalContext],
        selected_counts: Sequence[int],
        min_contexts: int = 2,
    ) -> RelationalResidualAssessment:
        ids = tuple(context.context_id for context in contexts)
        domains = tuple(sorted({context.domain for context in contexts}))
        repeated = (
            len(contexts) >= max(1, int(min_contexts))
            and len(selected_counts) == len(contexts)
            and all(int(value) == 0 for value in selected_counts)
        )
        return RelationalResidualAssessment(
            status=(
                "RELATIONAL_RESIDUAL_OPEN_INDUCTION"
                if repeated
                else "RELATIONAL_RESIDUAL_NOT_ESTABLISHED"
            ),
            context_ids=ids,
            domains=domains,
            repeated_failure=repeated,
        )

    def _paths(self, context: RelationalContext) -> Tuple[Tuple[str, ...], ...]:
        adjacency: Dict[str, List[RelationalEdge]] = {}
        for edge in context.edges:
            adjacency.setdefault(edge.source, []).append(edge)
        for edges in adjacency.values():
            edges.sort(key=lambda item: (
                item.relation,
                item.source_kind,
                item.target_kind,
                item.target,
            ))

        found: Dict[Tuple[str, ...], None] = {}

        def walk(node: str, visited: Tuple[str, ...], steps: Tuple[str, ...]) -> None:
            if len(steps) >= self.max_depth:
                return
            for edge in adjacency.get(node, ()):  # deterministic order above
                if edge.target in visited:
                    continue
                next_steps = steps + (edge.abstract_step,)
                if edge.target == context.target_anchor:
                    found.setdefault(next_steps, None)
                    if len(found) >= self.candidate_budget:
                        return
                walk(edge.target, visited + (edge.target,), next_steps)
                if len(found) >= self.candidate_budget:
                    return

        walk(context.source_anchor, (context.source_anchor,), ())
        return tuple(sorted(found))

    def generate_candidates(
        self,
        assessment: RelationalResidualAssessment,
        contexts: Sequence[RelationalContext],
    ) -> Tuple[GeneratedRelationalPathSchema, ...]:
        if assessment.status != "RELATIONAL_RESIDUAL_OPEN_INDUCTION" or not contexts:
            return ()
        per_context = [set(self._paths(context)) for context in contexts]
        if not per_context or any(not paths for paths in per_context):
            return ()
        common = set.intersection(*per_context)
        ordered = sorted(common, key=lambda path: (len(path), path))
        return tuple(GeneratedRelationalPathSchema(path) for path in ordered[: self.candidate_budget])

    def matches(self, schema: GeneratedRelationalPathSchema, context: RelationalContext) -> bool:
        return schema.steps in set(self._paths(context))


def derive_relational_path_policy(
    schemas: Sequence[GeneratedRelationalPathSchema],
    pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int = 2,
    min_contexts: int = 2,
) -> RelationalPathPolicy:
    min_classes = max(1, int(min_independent_classes))
    min_ctx = max(1, int(min_contexts))
    allowed: List[str] = []
    support_rows: List[Tuple[str, Tuple[str, ...]]] = []

    for schema in schemas:
        by_context: Dict[str, set[str]] = {}
        for pair in pairs:
            if pair.experiment_id != schema.schema_id:
                continue
            if not (
                pair.matched_budget
                and pair.externally_generated
                and pair.authority_verified
                and pair.independence_class_id != "UNVERIFIED"
                and pair.effect > 0.0
            ):
                continue
            by_context.setdefault(pair.context_id, set()).add(pair.independence_class_id)
        ready_contexts = tuple(sorted(
            context_id
            for context_id, classes in by_context.items()
            if len(classes) >= min_classes
        ))
        if len(ready_contexts) >= min_ctx:
            allowed.append(schema.schema_id)
            support_rows.append((schema.schema_id, ready_contexts))

    return RelationalPathPolicy(
        allowed_schema_ids=tuple(sorted(allowed)),
        supporting_contexts=tuple(sorted(support_rows)),
        min_independent_classes=min_classes,
        min_contexts=min_ctx,
    )


def select_authorized_relational_path_schema(
    schemas: Sequence[GeneratedRelationalPathSchema],
    policy: RelationalPathPolicy,
) -> Optional[GeneratedRelationalPathSchema]:
    allowed = set(policy.allowed_schema_ids)
    for schema in sorted(schemas, key=lambda item: item.schema_id):
        if schema.schema_id in allowed:
            return schema
    return None


def make_context(
    context_id: str,
    edges: Iterable[RelationalEdge],
    source_anchor: str,
    target_anchor: str,
    domain: str,
) -> RelationalContext:
    return RelationalContext(
        context_id=str(context_id),
        edges=tuple(sorted(edges)),
        source_anchor=str(source_anchor),
        target_anchor=str(target_anchor),
        domain=str(domain),
    )
