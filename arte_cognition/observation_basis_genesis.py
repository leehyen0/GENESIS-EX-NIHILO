from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .latent_relation_ontology_genesis import OpaqueInterventionalWorld
from .world_coupling import WorldOutcomePair


@dataclass(frozen=True, order=True)
class EmpiricalResponseEdge:
    source: str
    target: str
    lag: int
    sign: int
    normalized_peak: float
    support: int

    @property
    def profile_token(self) -> str:
        sign_name = "POS" if self.sign > 0 else "NEG"
        raw = f"peak_lag={self.lag}|sign={sign_name}".encode()
        return "OBS_REL::" + hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True)
class GeneratedObservationBasis:
    lag_offsets: Tuple[int, ...]
    profile_tokens: Tuple[str, ...]

    @property
    def basis_id(self) -> str:
        raw = (
            "lags=" + ",".join(map(str, self.lag_offsets))
            + "|profiles=" + "|".join(self.profile_tokens)
        ).encode()
        return "OBSERVATION_BASIS::" + hashlib.sha256(raw).hexdigest()[:20]


@dataclass(frozen=True)
class GeneratedObservationPathSchema:
    basis_id: str
    profile_tokens: Tuple[str, ...]

    @property
    def schema_id(self) -> str:
        raw = (self.basis_id + "|" + "|".join(self.profile_tokens)).encode()
        return "OBSERVATION_PATH::" + hashlib.sha256(raw).hexdigest()[:20]


@dataclass(frozen=True)
class ObservationBasisResidualAssessment:
    status: str
    context_ids: Tuple[str, ...]
    predecessor_selected_counts: Tuple[int, ...]


@dataclass(frozen=True)
class ObservationBasisPolicy:
    allowed_schema_ids: Tuple[str, ...]
    supporting_contexts: Tuple[Tuple[str, Tuple[str, ...]], ...]
    min_independent_classes: int
    min_contexts: int


class WorldDerivedObservationBasisInducer:
    """Generate temporal observation coordinates from raw intervention contrasts.

    The predecessor ontology used one caller-selected lag when constructing a
    relation token. This inducer instead scans every available post-intervention
    offset in the raw low/high timelines and derives each repeated source-target
    relation from its strongest stable response lag. The union of those empirical
    peak lags becomes a generated observation basis; no semantic relation label,
    node kind, or preselected lag is supplied to candidate generation.

    The mechanism remains bounded. Matched low/high trace structure, arithmetic
    differencing, strongest-peak selection, minimum repeat support, the tie rule,
    path enumeration, and authority contract are still authored. This is therefore
    bounded observation-basis genesis, not unrestricted sensory/meta-language
    invention.
    """

    def __init__(
        self,
        min_repeats: int = 2,
        min_peak_effect: float = 1e-9,
        max_path_depth: int = 8,
        candidate_budget: int = 64,
    ) -> None:
        self.min_repeats = max(1, int(min_repeats))
        self.min_peak_effect = max(0.0, float(min_peak_effect))
        self.max_path_depth = max(1, int(max_path_depth))
        self.candidate_budget = max(1, int(candidate_budget))

    @staticmethod
    def assess_residual(
        worlds: Sequence[OpaqueInterventionalWorld],
        predecessor_selected_counts: Sequence[int],
        min_contexts: int = 2,
    ) -> ObservationBasisResidualAssessment:
        counts = tuple(int(value) for value in predecessor_selected_counts)
        opened = (
            len(worlds) >= max(1, int(min_contexts))
            and len(counts) == len(worlds)
            and all(value == 0 for value in counts)
        )
        return ObservationBasisResidualAssessment(
            status=(
                "FIXED_OBSERVATION_BASIS_RESIDUAL_OPEN"
                if opened
                else "FIXED_OBSERVATION_BASIS_RESIDUAL_NOT_ESTABLISHED"
            ),
            context_ids=tuple(world.context_id for world in worlds),
            predecessor_selected_counts=counts,
        )

    @staticmethod
    def _snapshot(row: Iterable[Tuple[str, float]]) -> Dict[str, float]:
        return {str(node): float(value) for node, value in row}

    @staticmethod
    def _all_nodes(world: OpaqueInterventionalWorld) -> Tuple[str, ...]:
        nodes = {world.source_anchor, world.target_anchor}
        for contrast in world.contrasts:
            nodes.add(contrast.source_node)
            for timeline in (contrast.low_timeline, contrast.high_timeline):
                for snapshot in timeline:
                    nodes.update(str(node) for node, _ in snapshot)
        return tuple(sorted(nodes))

    def infer_edges(self, world: OpaqueInterventionalWorld) -> Tuple[EmpiricalResponseEdge, ...]:
        nodes = self._all_nodes(world)
        effects: Dict[Tuple[str, str, int], List[float]] = {}

        for contrast in world.contrasts:
            max_lag = min(len(contrast.low_timeline), len(contrast.high_timeline)) - 1
            for lag in range(1, max_lag + 1):
                low = self._snapshot(contrast.low_timeline[lag])
                high = self._snapshot(contrast.high_timeline[lag])
                for target in nodes:
                    if target == contrast.source_node:
                        continue
                    effect = float(high.get(target, 0.0)) - float(low.get(target, 0.0))
                    effects.setdefault((contrast.source_node, target, lag), []).append(effect)

        by_pair: Dict[Tuple[str, str], List[Tuple[float, int, int, float]]] = {}
        for (source, target, lag), values in effects.items():
            nonzero = [value for value in values if abs(value) >= self.min_peak_effect]
            if len(nonzero) < self.min_repeats:
                continue
            mean_effect = sum(nonzero) / len(nonzero)
            if abs(mean_effect) < self.min_peak_effect:
                continue
            # Prefer larger repeated response; deterministic lower-lag tie break.
            by_pair.setdefault((source, target), []).append(
                (abs(mean_effect), -lag, len(nonzero), mean_effect)
            )

        edges: List[EmpiricalResponseEdge] = []
        for (source, target), candidates in sorted(by_pair.items()):
            candidates.sort(reverse=True)
            peak_abs, neg_lag, support, mean_effect = candidates[0]
            lag = -neg_lag
            sign = 1 if mean_effect > 0.0 else -1
            edges.append(EmpiricalResponseEdge(
                source=source,
                target=target,
                lag=lag,
                sign=sign,
                normalized_peak=1.0,
                support=support,
            ))
        return tuple(sorted(edges))

    def derive_basis(self, world: OpaqueInterventionalWorld) -> Optional[GeneratedObservationBasis]:
        edges = self.infer_edges(world)
        if not edges:
            return None
        lags = tuple(sorted({edge.lag for edge in edges}))
        tokens = tuple(sorted({edge.profile_token for edge in edges}))
        return GeneratedObservationBasis(lag_offsets=lags, profile_tokens=tokens)

    def _paths(self, world: OpaqueInterventionalWorld) -> Tuple[Tuple[str, ...], ...]:
        adjacency: Dict[str, List[EmpiricalResponseEdge]] = {}
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
        assessment: ObservationBasisResidualAssessment,
        worlds: Sequence[OpaqueInterventionalWorld],
    ) -> Tuple[GeneratedObservationPathSchema, ...]:
        if assessment.status != "FIXED_OBSERVATION_BASIS_RESIDUAL_OPEN" or not worlds:
            return ()

        bases = [self.derive_basis(world) for world in worlds]
        if any(basis is None for basis in bases):
            return ()
        assert all(basis is not None for basis in bases)

        common_lags = set(bases[0].lag_offsets)
        common_profiles = set(bases[0].profile_tokens)
        for basis in bases[1:]:
            common_lags &= set(basis.lag_offsets)
            common_profiles &= set(basis.profile_tokens)
        if not common_lags or not common_profiles:
            return ()

        common_basis = GeneratedObservationBasis(
            lag_offsets=tuple(sorted(common_lags)),
            profile_tokens=tuple(sorted(common_profiles)),
        )
        per_world_paths = [set(self._paths(world)) for world in worlds]
        if any(not paths for paths in per_world_paths):
            return ()
        common_paths = set.intersection(*per_world_paths)
        ordered = sorted(common_paths, key=lambda row: (len(row), row))[: self.candidate_budget]
        return tuple(
            GeneratedObservationPathSchema(common_basis.basis_id, tokens)
            for tokens in ordered
        )

    def matches(
        self,
        schema: GeneratedObservationPathSchema,
        world: OpaqueInterventionalWorld,
    ) -> bool:
        basis = self.derive_basis(world)
        if basis is None:
            return False
        # Exact path semantics matter; the basis identifier need not be identical
        # if a heldout world contains extra irrelevant response coordinates.
        return schema.profile_tokens in set(self._paths(world))


def derive_observation_basis_policy(
    schemas: Sequence[GeneratedObservationPathSchema],
    pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int = 2,
    min_contexts: int = 2,
) -> ObservationBasisPolicy:
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

    return ObservationBasisPolicy(
        allowed_schema_ids=tuple(sorted(allowed)),
        supporting_contexts=tuple(sorted(support_rows)),
        min_independent_classes=min_classes,
        min_contexts=min_ctx,
    )


def select_authorized_observation_schema(
    schemas: Sequence[GeneratedObservationPathSchema],
    policy: ObservationBasisPolicy,
) -> Optional[GeneratedObservationPathSchema]:
    allowed = set(policy.allowed_schema_ids)
    for schema in sorted(schemas, key=lambda item: item.schema_id):
        if schema.schema_id in allowed:
            return schema
    return None
