from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import hashlib

from .world_coupling import WorldOutcomePair


Snapshot = Tuple[Tuple[str, float], ...]
Timeline = Tuple[Snapshot, ...]


@dataclass(frozen=True)
class OpaqueInterventionContrast:
    contrast_id: str
    source_node: str
    low_timeline: Timeline
    high_timeline: Timeline


@dataclass(frozen=True)
class OpaqueInterventionalWorld:
    context_id: str
    domain: str
    source_anchor: str
    target_anchor: str
    contrasts: Tuple[OpaqueInterventionContrast, ...]


@dataclass(frozen=True, order=True)
class LatentRelationEdge:
    source: str
    relation_token: str
    target: str


@dataclass(frozen=True)
class GeneratedLatentPathSchema:
    relation_tokens: Tuple[str, ...]

    @property
    def schema_id(self) -> str:
        raw = "|".join(self.relation_tokens).encode()
        return "LATENT_RELATION_PATH::" + hashlib.sha256(raw).hexdigest()[:20]


@dataclass(frozen=True)
class LatentOntologyResidualAssessment:
    status: str
    context_ids: Tuple[str, ...]
    repeated_failure: bool


@dataclass(frozen=True)
class LatentPathPolicy:
    allowed_schema_ids: Tuple[str, ...]
    supporting_contexts: Tuple[Tuple[str, Tuple[str, ...]], ...]
    min_independent_classes: int
    min_contexts: int


class WorldDerivedLatentRelationInducer:
    """Induce an opaque relation vocabulary from interventional trace contrasts.

    Unlike the typed relational predecessor, callers do not supply semantic edge
    labels such as PRODUCES, BOUND_AS, EMITS, or MEDIATED_BY, and callers do not
    supply node kinds. A relation token is synthesized from a repeated local
    intervention-response fingerprint and is therefore identifier- and
    domain-name invariant.

    This is still bounded. The representation of matched low/high intervention
    traces, the lag window, effect threshold, sign fingerprint and graph-path
    enumeration are authored. It is not unrestricted ontology or operator genesis.
    """

    def __init__(
        self,
        lag: int = 1,
        min_effect: float = 0.5,
        min_repeats: int = 2,
        max_path_depth: int = 6,
        candidate_budget: int = 64,
    ) -> None:
        self.lag = max(1, int(lag))
        self.min_effect = max(0.0, float(min_effect))
        self.min_repeats = max(1, int(min_repeats))
        self.max_path_depth = max(1, int(max_path_depth))
        self.candidate_budget = max(1, int(candidate_budget))

    @staticmethod
    def assess_residual(
        worlds: Sequence[OpaqueInterventionalWorld],
        predecessor_selected_counts: Sequence[int],
        min_contexts: int = 2,
    ) -> LatentOntologyResidualAssessment:
        repeated = (
            len(worlds) >= max(1, int(min_contexts))
            and len(predecessor_selected_counts) == len(worlds)
            and all(int(value) == 0 for value in predecessor_selected_counts)
        )
        return LatentOntologyResidualAssessment(
            status=(
                "OPAQUE_RELATION_ONTOLOGY_RESIDUAL_OPEN"
                if repeated
                else "OPAQUE_RELATION_ONTOLOGY_RESIDUAL_NOT_ESTABLISHED"
            ),
            context_ids=tuple(world.context_id for world in worlds),
            repeated_failure=repeated,
        )

    @staticmethod
    def _snapshot(snapshot: Snapshot) -> Dict[str, float]:
        return {str(node): float(value) for node, value in snapshot}

    @staticmethod
    def _nodes(contrast: OpaqueInterventionContrast) -> Tuple[str, ...]:
        nodes = {
            str(node)
            for timeline in (contrast.low_timeline, contrast.high_timeline)
            for snapshot in timeline
            for node, _ in snapshot
        }
        return tuple(sorted(nodes))

    def _fingerprint(self, effect: float) -> str:
        sign = "POS" if effect > 0.0 else "NEG"
        raw = f"lag={self.lag}|sign={sign}".encode()
        return "LATENT_REL::" + hashlib.sha256(raw).hexdigest()[:16]

    def infer_edges(self, world: OpaqueInterventionalWorld) -> Tuple[LatentRelationEdge, ...]:
        support: Dict[Tuple[str, str, str], int] = {}
        conflicts: Dict[Tuple[str, str], set[str]] = {}

        for contrast in world.contrasts:
            if len(contrast.low_timeline) <= self.lag or len(contrast.high_timeline) <= self.lag:
                continue
            low = self._snapshot(contrast.low_timeline[self.lag])
            high = self._snapshot(contrast.high_timeline[self.lag])
            for target in self._nodes(contrast):
                if target == contrast.source_node:
                    continue
                effect = float(high.get(target, 0.0)) - float(low.get(target, 0.0))
                if abs(effect) < self.min_effect:
                    continue
                token = self._fingerprint(effect)
                support[(contrast.source_node, target, token)] = support.get(
                    (contrast.source_node, target, token), 0
                ) + 1
                conflicts.setdefault((contrast.source_node, target), set()).add(token)

        edges = []
        for (source, target, token), count in support.items():
            if count < self.min_repeats:
                continue
            if len(conflicts.get((source, target), ())) != 1:
                continue
            edges.append(LatentRelationEdge(source, token, target))
        return tuple(sorted(edges))

    def _paths(self, world: OpaqueInterventionalWorld) -> Tuple[Tuple[str, ...], ...]:
        adjacency: Dict[str, List[LatentRelationEdge]] = {}
        for edge in self.infer_edges(world):
            adjacency.setdefault(edge.source, []).append(edge)
        for edges in adjacency.values():
            edges.sort(key=lambda edge: (edge.relation_token, edge.target))

        found: Dict[Tuple[str, ...], None] = {}

        def walk(node: str, visited: Tuple[str, ...], tokens: Tuple[str, ...]) -> None:
            if len(tokens) >= self.max_path_depth or len(found) >= self.candidate_budget:
                return
            for edge in adjacency.get(node, ()):
                if edge.target in visited:
                    continue
                nxt = tokens + (edge.relation_token,)
                if edge.target == world.target_anchor:
                    found.setdefault(nxt, None)
                walk(edge.target, visited + (edge.target,), nxt)
                if len(found) >= self.candidate_budget:
                    return

        walk(world.source_anchor, (world.source_anchor,), ())
        return tuple(sorted(found, key=lambda row: (len(row), row)))

    def generate_candidates(
        self,
        assessment: LatentOntologyResidualAssessment,
        worlds: Sequence[OpaqueInterventionalWorld],
    ) -> Tuple[GeneratedLatentPathSchema, ...]:
        if assessment.status != "OPAQUE_RELATION_ONTOLOGY_RESIDUAL_OPEN" or not worlds:
            return ()
        per_world = [set(self._paths(world)) for world in worlds]
        if not per_world or any(not paths for paths in per_world):
            return ()
        common = set.intersection(*per_world)
        ordered = sorted(common, key=lambda row: (len(row), row))[: self.candidate_budget]
        return tuple(GeneratedLatentPathSchema(tokens) for tokens in ordered)

    def matches(self, schema: GeneratedLatentPathSchema, world: OpaqueInterventionalWorld) -> bool:
        return schema.relation_tokens in set(self._paths(world))


def derive_latent_path_policy(
    schemas: Sequence[GeneratedLatentPathSchema],
    pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int = 2,
    min_contexts: int = 2,
) -> LatentPathPolicy:
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
        ready = tuple(sorted(
            context_id
            for context_id, classes in by_context.items()
            if len(classes) >= min_classes
        ))
        if len(ready) >= min_ctx:
            allowed.append(schema.schema_id)
            support_rows.append((schema.schema_id, ready))

    return LatentPathPolicy(
        allowed_schema_ids=tuple(sorted(allowed)),
        supporting_contexts=tuple(sorted(support_rows)),
        min_independent_classes=min_classes,
        min_contexts=min_ctx,
    )


def select_authorized_latent_path(
    schemas: Sequence[GeneratedLatentPathSchema],
    policy: LatentPathPolicy,
) -> Optional[GeneratedLatentPathSchema]:
    allowed = set(policy.allowed_schema_ids)
    for schema in sorted(schemas, key=lambda item: item.schema_id):
        if schema.schema_id in allowed:
            return schema
    return None


def snapshot(values: Mapping[str, float]) -> Snapshot:
    return tuple(sorted((str(node), float(value)) for node, value in values.items()))


def contrast(
    contrast_id: str,
    source_node: str,
    low_timeline: Iterable[Mapping[str, float]],
    high_timeline: Iterable[Mapping[str, float]],
) -> OpaqueInterventionContrast:
    return OpaqueInterventionContrast(
        contrast_id=str(contrast_id),
        source_node=str(source_node),
        low_timeline=tuple(snapshot(row) for row in low_timeline),
        high_timeline=tuple(snapshot(row) for row in high_timeline),
    )
