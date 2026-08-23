from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import hashlib

from .latent_relation_ontology_genesis import OpaqueInterventionalWorld
from .world_coupling import WorldOutcomePair

OrderConstraint = Tuple[int, int, int]
SignConstraint = Tuple[int, int]


@dataclass(frozen=True, order=True)
class EmpiricalOrderEdge:
    source: str
    target: str
    order_constraints: Tuple[OrderConstraint, ...]
    sign_constraints: Tuple[SignConstraint, ...]

    @property
    def criterion_token(self) -> str:
        payload = repr((self.order_constraints, self.sign_constraints)).encode()
        return "EQUIV_CRITERION::" + hashlib.sha256(payload).hexdigest()[:18]


@dataclass(frozen=True)
class GeneratedEquivalenceCriterion:
    edge_tokens: Tuple[str, ...]
    constraints: Tuple[Tuple[Tuple[OrderConstraint, ...], Tuple[SignConstraint, ...]], ...]

    @property
    def criterion_id(self) -> str:
        raw = repr(self.constraints).encode()
        return "EQUIVALENCE_PATH::" + hashlib.sha256(raw).hexdigest()[:20]


@dataclass(frozen=True)
class EquivalenceResidualAssessment:
    status: str
    context_ids: Tuple[str, ...]
    repeated_failure: bool


@dataclass(frozen=True)
class EquivalencePolicy:
    allowed_criterion_ids: Tuple[str, ...]
    supporting_contexts: Tuple[Tuple[str, Tuple[str, ...]], ...]
    min_independent_classes: int
    min_contexts: int


class WorldDerivedEquivalenceCriterionInducer:
    """Generate a measurement-invariant relation criterion from raw response order.

    The predecessor identifies relations by exact normalized numeric response shape.
    This inducer does not receive a named SCALE/OFFSET/POWER transformation. Instead,
    it generates every pairwise lag-order and sign constraint that is actually
    realized by repeated raw intervention-response curves. Cross-context intersection
    then discovers the invariant quotient that survives distinct measurement systems.

    Still bounded: arithmetic differencing, pairwise comparison, tolerance, repeated
    support, path enumeration and external authority semantics remain authored.
    """

    def __init__(self, min_repeats: int = 2, tolerance: float = 1e-9, max_path_depth: int = 8) -> None:
        self.min_repeats = max(1, int(min_repeats))
        self.tolerance = max(0.0, float(tolerance))
        self.max_path_depth = max(1, int(max_path_depth))

    @staticmethod
    def assess_residual(
        worlds: Sequence[OpaqueInterventionalWorld],
        predecessor_candidate_counts: Sequence[int],
        min_contexts: int = 2,
    ) -> EquivalenceResidualAssessment:
        repeated = (
            len(worlds) >= max(1, int(min_contexts))
            and len(predecessor_candidate_counts) == len(worlds)
            and all(int(v) == 0 for v in predecessor_candidate_counts)
        )
        return EquivalenceResidualAssessment(
            status=(
                "FIXED_NUMERIC_EQUIVALENCE_RESIDUAL_OPEN"
                if repeated else "FIXED_NUMERIC_EQUIVALENCE_RESIDUAL_NOT_ESTABLISHED"
            ),
            context_ids=tuple(w.context_id for w in worlds),
            repeated_failure=repeated,
        )

    @staticmethod
    def _snapshot(snapshot) -> Dict[str, float]:
        return {str(k): float(v) for k, v in snapshot}

    def _curves(self, world: OpaqueInterventionalWorld) -> Dict[Tuple[str, str], Tuple[float, ...]]:
        rows: Dict[Tuple[str, str], List[Tuple[float, ...]]] = {}
        max_lag = 0
        for contrast in world.contrasts:
            max_lag = max(max_lag, min(len(contrast.low_timeline), len(contrast.high_timeline)) - 1)
        for contrast in world.contrasts:
            length = min(len(contrast.low_timeline), len(contrast.high_timeline))
            if length <= 1:
                continue
            nodes = {
                str(node)
                for timeline in (contrast.low_timeline, contrast.high_timeline)
                for snap in timeline
                for node, _ in snap
            }
            for target in sorted(nodes):
                if target == contrast.source_node:
                    continue
                curve = []
                for lag in range(1, max_lag + 1):
                    if lag >= length:
                        curve.append(0.0)
                        continue
                    low = self._snapshot(contrast.low_timeline[lag])
                    high = self._snapshot(contrast.high_timeline[lag])
                    curve.append(float(high.get(target, 0.0)) - float(low.get(target, 0.0)))
                if any(abs(v) > self.tolerance for v in curve):
                    rows.setdefault((contrast.source_node, target), []).append(tuple(curve))

        out: Dict[Tuple[str, str], Tuple[float, ...]] = {}
        for key, values in rows.items():
            if len(values) < self.min_repeats:
                continue
            width = max(len(v) for v in values)
            mean = tuple(
                sum(v[i] if i < len(v) else 0.0 for v in values) / len(values)
                for i in range(width)
            )
            out[key] = mean
        return out

    def _constraints(self, curve: Sequence[float]) -> Tuple[Tuple[OrderConstraint, ...], Tuple[SignConstraint, ...]]:
        orders: List[OrderConstraint] = []
        for i in range(len(curve)):
            for j in range(i + 1, len(curve)):
                d = float(curve[i]) - float(curve[j])
                relation = 0 if abs(d) <= self.tolerance else (1 if d > 0.0 else -1)
                orders.append((i + 1, j + 1, relation))
        signs = []
        for i, value in enumerate(curve, start=1):
            sign = 0 if abs(float(value)) <= self.tolerance else (1 if value > 0.0 else -1)
            signs.append((i, sign))
        return tuple(orders), tuple(signs)

    def infer_edges(self, world: OpaqueInterventionalWorld) -> Tuple[EmpiricalOrderEdge, ...]:
        edges = []
        for (source, target), curve in self._curves(world).items():
            order_constraints, sign_constraints = self._constraints(curve)
            edges.append(EmpiricalOrderEdge(source, target, order_constraints, sign_constraints))
        return tuple(sorted(edges))

    def _paths(self, world: OpaqueInterventionalWorld) -> Tuple[GeneratedEquivalenceCriterion, ...]:
        adjacency: Dict[str, List[EmpiricalOrderEdge]] = {}
        for edge in self.infer_edges(world):
            adjacency.setdefault(edge.source, []).append(edge)
        for items in adjacency.values():
            items.sort(key=lambda e: (e.criterion_token, e.target))
        found: Dict[Tuple[Tuple[Tuple[OrderConstraint, ...], Tuple[SignConstraint, ...]], ...], None] = {}

        def walk(node: str, visited: Tuple[str, ...], constraints) -> None:
            if len(constraints) >= self.max_path_depth:
                return
            for edge in adjacency.get(node, ()):
                if edge.target in visited:
                    continue
                nxt = constraints + ((edge.order_constraints, edge.sign_constraints),)
                if edge.target == world.target_anchor:
                    found.setdefault(nxt, None)
                walk(edge.target, visited + (edge.target,), nxt)

        walk(world.source_anchor, (world.source_anchor,), ())
        return tuple(
            GeneratedEquivalenceCriterion(
                edge_tokens=tuple(
                    "EQUIV_CRITERION::" + hashlib.sha256(repr(c).encode()).hexdigest()[:18]
                    for c in constraints
                ),
                constraints=constraints,
            )
            for constraints in sorted(found, key=repr)
        )

    def generate_candidates(
        self,
        assessment: EquivalenceResidualAssessment,
        worlds: Sequence[OpaqueInterventionalWorld],
    ) -> Tuple[GeneratedEquivalenceCriterion, ...]:
        if assessment.status != "FIXED_NUMERIC_EQUIVALENCE_RESIDUAL_OPEN" or not worlds:
            return ()
        per_world = [{candidate.constraints: candidate for candidate in self._paths(world)} for world in worlds]
        if any(not row for row in per_world):
            return ()
        common = set.intersection(*(set(row) for row in per_world))
        first = per_world[0]
        return tuple(first[key] for key in sorted(common, key=repr))

    def matches(self, criterion: GeneratedEquivalenceCriterion, world: OpaqueInterventionalWorld) -> bool:
        return criterion.constraints in {candidate.constraints for candidate in self._paths(world)}


def derive_equivalence_policy(
    criteria: Sequence[GeneratedEquivalenceCriterion],
    pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int = 2,
    min_contexts: int = 2,
) -> EquivalencePolicy:
    allowed = []
    support = []
    for criterion in criteria:
        by_context: Dict[str, set[str]] = {}
        for pair in pairs:
            if pair.experiment_id != criterion.criterion_id:
                continue
            if not (
                pair.matched_budget and pair.externally_generated and pair.authority_verified
                and pair.independence_class_id != "UNVERIFIED" and pair.effect > 0.0
            ):
                continue
            by_context.setdefault(pair.context_id, set()).add(pair.independence_class_id)
        ready = tuple(sorted(k for k, v in by_context.items() if len(v) >= min_independent_classes))
        if len(ready) >= min_contexts:
            allowed.append(criterion.criterion_id)
            support.append((criterion.criterion_id, ready))
    return EquivalencePolicy(tuple(sorted(allowed)), tuple(sorted(support)), min_independent_classes, min_contexts)


def select_authorized_equivalence(
    criteria: Sequence[GeneratedEquivalenceCriterion], policy: EquivalencePolicy
) -> Optional[GeneratedEquivalenceCriterion]:
    allowed = set(policy.allowed_criterion_ids)
    for criterion in sorted(criteria, key=lambda c: c.criterion_id):
        if criterion.criterion_id in allowed:
            return criterion
    return None
