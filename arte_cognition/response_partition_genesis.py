from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .latent_relation_ontology_genesis import OpaqueInterventionalWorld
from .world_coupling import WorldOutcomePair


@dataclass(frozen=True, order=True)
class EmpiricalProfileEdge:
    source: str
    target: str
    normalized_profile: Tuple[float, ...]
    support: int

    @property
    def profile_token(self) -> str:
        raw = "|".join(f"{value:+.6f}" for value in self.normalized_profile).encode()
        return "RESPONSE_PROFILE::" + hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True)
class GeneratedResponsePartition:
    profile_tokens: Tuple[str, ...]
    profile_shapes: Tuple[Tuple[float, ...], ...]

    @property
    def partition_id(self) -> str:
        rows = [
            token + "=" + ",".join(f"{value:+.6f}" for value in shape)
            for token, shape in zip(self.profile_tokens, self.profile_shapes)
        ]
        return "RESPONSE_PARTITION::" + hashlib.sha256("|".join(rows).encode()).hexdigest()[:20]


@dataclass(frozen=True)
class GeneratedResponsePathSchema:
    partition_id: str
    profile_tokens: Tuple[str, ...]

    @property
    def schema_id(self) -> str:
        raw = (self.partition_id + "|" + "|".join(self.profile_tokens)).encode()
        return "RESPONSE_PATH::" + hashlib.sha256(raw).hexdigest()[:20]


@dataclass(frozen=True)
class ResponsePartitionResidualAssessment:
    status: str
    context_ids: Tuple[str, ...]
    predecessor_ambiguities: Tuple[int, ...]


@dataclass(frozen=True)
class ResponsePartitionPolicy:
    allowed_schema_ids: Tuple[str, ...]
    supporting_contexts: Tuple[Tuple[str, Tuple[str, ...]], ...]
    min_independent_classes: int
    min_contexts: int


class WorldDerivedResponsePartitionInducer:
    """Generate relation distinctions from complete normalized response shapes.

    The predecessor observation-basis organ can discover which temporal offsets
    matter, but it reduces each edge to peak-lag plus effect sign. Distinct causal
    response curves can therefore collapse to one relation token. This inducer
    removes that named partition: for every repeated source-target intervention it
    computes the full mean effect curve over all observed lags, divides by its own
    maximum absolute effect, and uses the resulting scale-free signed shape as the
    relation identity. Repeated shapes form a generated response partition.

    Still bounded: matched contrasts, arithmetic mean, max-absolute normalization,
    floating-point rounding, minimum repeat/effect support, path enumeration, and
    authority rules remain authored. This is not unrestricted perceptual ontology
    or arbitrary concept genesis.
    """

    def __init__(
        self,
        min_repeats: int = 2,
        min_peak_effect: float = 1e-9,
        precision: int = 6,
        max_path_depth: int = 8,
        candidate_budget: int = 64,
    ) -> None:
        self.min_repeats = max(1, int(min_repeats))
        self.min_peak_effect = max(0.0, float(min_peak_effect))
        self.precision = max(1, int(precision))
        self.max_path_depth = max(1, int(max_path_depth))
        self.candidate_budget = max(1, int(candidate_budget))

    @staticmethod
    def assess_residual(
        worlds: Sequence[OpaqueInterventionalWorld],
        predecessor_ambiguities: Sequence[int],
        min_contexts: int = 2,
    ) -> ResponsePartitionResidualAssessment:
        ambiguities = tuple(int(value) for value in predecessor_ambiguities)
        opened = (
            len(worlds) >= max(1, int(min_contexts))
            and len(ambiguities) == len(worlds)
            and all(value >= 2 for value in ambiguities)
        )
        return ResponsePartitionResidualAssessment(
            status=(
                "PEAK_SIGN_PARTITION_AMBIGUOUS_OPEN_RESPONSE_PARTITION"
                if opened
                else "PEAK_SIGN_PARTITION_RESIDUAL_NOT_ESTABLISHED"
            ),
            context_ids=tuple(world.context_id for world in worlds),
            predecessor_ambiguities=ambiguities,
        )

    @staticmethod
    def _snapshot(row: Iterable[Tuple[str, float]]) -> Dict[str, float]:
        return {str(node): float(value) for node, value in row}

    @staticmethod
    def _nodes(world: OpaqueInterventionalWorld) -> Tuple[str, ...]:
        result = {world.source_anchor, world.target_anchor}
        for contrast in world.contrasts:
            result.add(contrast.source_node)
            for timeline in (contrast.low_timeline, contrast.high_timeline):
                for snapshot in timeline:
                    result.update(str(node) for node, _ in snapshot)
        return tuple(sorted(result))

    def infer_edges(self, world: OpaqueInterventionalWorld) -> Tuple[EmpiricalProfileEdge, ...]:
        nodes = self._nodes(world)
        by_source: Dict[str, List[object]] = {}
        for contrast in world.contrasts:
            by_source.setdefault(contrast.source_node, []).append(contrast)

        edges: List[EmpiricalProfileEdge] = []
        for source, contrasts in sorted(by_source.items()):
            if len(contrasts) < self.min_repeats:
                continue
            max_lag = min(
                min(len(row.low_timeline), len(row.high_timeline)) - 1
                for row in contrasts
            )
            if max_lag < 1:
                continue
            for target in nodes:
                if target == source:
                    continue
                means: List[float] = []
                for lag in range(1, max_lag + 1):
                    values = []
                    for row in contrasts:
                        low = self._snapshot(row.low_timeline[lag])
                        high = self._snapshot(row.high_timeline[lag])
                        values.append(float(high.get(target, 0.0)) - float(low.get(target, 0.0)))
                    means.append(sum(values) / len(values))
                peak = max((abs(value) for value in means), default=0.0)
                if peak < self.min_peak_effect:
                    continue
                normalized = tuple(round(value / peak, self.precision) for value in means)
                edges.append(EmpiricalProfileEdge(
                    source=source,
                    target=target,
                    normalized_profile=normalized,
                    support=len(contrasts),
                ))
        return tuple(sorted(edges))

    def derive_partition(self, world: OpaqueInterventionalWorld) -> Optional[GeneratedResponsePartition]:
        edges = self.infer_edges(world)
        if not edges:
            return None
        by_token: Dict[str, Tuple[float, ...]] = {}
        for edge in edges:
            by_token.setdefault(edge.profile_token, edge.normalized_profile)
        ordered = sorted(by_token.items())
        return GeneratedResponsePartition(
            profile_tokens=tuple(token for token, _ in ordered),
            profile_shapes=tuple(shape for _, shape in ordered),
        )

    def _paths(self, world: OpaqueInterventionalWorld) -> Tuple[Tuple[str, ...], ...]:
        adjacency: Dict[str, List[EmpiricalProfileEdge]] = {}
        for edge in self.infer_edges(world):
            adjacency.setdefault(edge.source, []).append(edge)
        for values in adjacency.values():
            values.sort(key=lambda edge: (edge.profile_token, edge.target))

        found: Dict[Tuple[str, ...], None] = {}

        def walk(node: str, visited: Tuple[str, ...], tokens: Tuple[str, ...]) -> None:
            if len(tokens) >= self.max_path_depth or len(found) >= self.candidate_budget:
                return
            for edge in adjacency.get(node, ()):
                if edge.target in visited:
                    continue
                next_tokens = tokens + (edge.profile_token,)
                if edge.target == world.target_anchor:
                    found.setdefault(next_tokens, None)
                walk(edge.target, visited + (edge.target,), next_tokens)
                if len(found) >= self.candidate_budget:
                    return

        walk(world.source_anchor, (world.source_anchor,), ())
        return tuple(sorted(found, key=lambda row: (len(row), row)))

    def generate_candidates(
        self,
        assessment: ResponsePartitionResidualAssessment,
        worlds: Sequence[OpaqueInterventionalWorld],
    ) -> Tuple[GeneratedResponsePathSchema, ...]:
        if assessment.status != "PEAK_SIGN_PARTITION_AMBIGUOUS_OPEN_RESPONSE_PARTITION" or not worlds:
            return ()
        partitions = [self.derive_partition(world) for world in worlds]
        if any(partition is None for partition in partitions):
            return ()
        assert all(partition is not None for partition in partitions)

        common_tokens = set(partitions[0].profile_tokens)
        shape_by_token = dict(zip(partitions[0].profile_tokens, partitions[0].profile_shapes))
        for partition in partitions[1:]:
            common_tokens &= set(partition.profile_tokens)
        if not common_tokens:
            return ()
        partition = GeneratedResponsePartition(
            profile_tokens=tuple(sorted(common_tokens)),
            profile_shapes=tuple(shape_by_token[token] for token in sorted(common_tokens)),
        )

        path_sets = [set(self._paths(world)) for world in worlds]
        if any(not paths for paths in path_sets):
            return ()
        common_paths = set.intersection(*path_sets)
        ordered = sorted(common_paths, key=lambda row: (len(row), row))[: self.candidate_budget]
        return tuple(
            GeneratedResponsePathSchema(partition.partition_id, tokens)
            for tokens in ordered
        )

    def matches(self, schema: GeneratedResponsePathSchema, world: OpaqueInterventionalWorld) -> bool:
        return schema.profile_tokens in set(self._paths(world))


def derive_response_partition_policy(
    schemas: Sequence[GeneratedResponsePathSchema],
    pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int = 2,
    min_contexts: int = 2,
) -> ResponsePartitionPolicy:
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
            context_id for context_id, classes in by_context.items()
            if len(classes) >= min_classes
        ))
        if len(ready) >= min_ctx:
            allowed.append(schema.schema_id)
            support_rows.append((schema.schema_id, ready))
    return ResponsePartitionPolicy(
        allowed_schema_ids=tuple(sorted(allowed)),
        supporting_contexts=tuple(sorted(support_rows)),
        min_independent_classes=min_classes,
        min_contexts=min_ctx,
    )


def select_authorized_response_schema(
    schemas: Sequence[GeneratedResponsePathSchema],
    policy: ResponsePartitionPolicy,
) -> Optional[GeneratedResponsePathSchema]:
    allowed = set(policy.allowed_schema_ids)
    for schema in sorted(schemas, key=lambda item: item.schema_id):
        if schema.schema_id in allowed:
            return schema
    return None
