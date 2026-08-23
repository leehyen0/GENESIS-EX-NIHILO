from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import gcd
from typing import Dict, List, Optional, Sequence, Tuple
import hashlib

from .latent_relation_ontology_genesis import OpaqueInterventionalWorld
from .world_coupling import WorldOutcomePair

Coefficients = Tuple[int, ...]


@dataclass(frozen=True)
class ComparisonResidualAssessment:
    status: str
    context_ids: Tuple[str, ...]
    repeated_ambiguity: bool


@dataclass(frozen=True)
class GeneratedComparisonSchema:
    coefficients: Coefficients
    path_signs: Tuple[int, ...]

    @property
    def schema_id(self) -> str:
        raw = repr((self.coefficients, self.path_signs)).encode()
        return "COMPARISON_PATH::" + hashlib.sha256(raw).hexdigest()[:20]


@dataclass(frozen=True)
class ComparisonPolicy:
    allowed_schema_ids: Tuple[str, ...]
    min_independent_classes: int
    min_contexts: int


class WorldDerivedComparisonFunctionalInducer:
    """Generate a comparison functional when order/sign invariants collapse paths.

    No named curvature/ratio/acceleration feature is supplied. The bounded search
    enumerates primitive zero-sum integer coefficient vectors over observed lag
    coordinates and retains only functionals that *structurally* split a repeated
    predecessor ambiguity in every training context. World outcomes are not used
    during generation. External consequences later decide which split is useful.
    """

    def __init__(self, coefficient_bound: int = 2, min_repeats: int = 2, tolerance: float = 1e-9) -> None:
        self.coefficient_bound = max(1, int(coefficient_bound))
        self.min_repeats = max(1, int(min_repeats))
        self.tolerance = max(0.0, float(tolerance))

    @staticmethod
    def assess_residual(
        worlds: Sequence[OpaqueInterventionalWorld],
        predecessor_unique_counts: Sequence[int],
        predecessor_concrete_multiplicities: Sequence[int],
        min_contexts: int = 2,
    ) -> ComparisonResidualAssessment:
        repeated = (
            len(worlds) >= max(1, int(min_contexts))
            and len(predecessor_unique_counts) == len(worlds)
            and len(predecessor_concrete_multiplicities) == len(worlds)
            and all(int(v) == 1 for v in predecessor_unique_counts)
            and all(int(v) >= 2 for v in predecessor_concrete_multiplicities)
        )
        return ComparisonResidualAssessment(
            status=("ORDER_INVARIANT_AMBIGUITY_OPEN_COMPARISON_FUNCTIONAL" if repeated
                    else "ORDER_INVARIANT_AMBIGUITY_NOT_ESTABLISHED"),
            context_ids=tuple(w.context_id for w in worlds),
            repeated_ambiguity=repeated,
        )

    def _curves(self, world: OpaqueInterventionalWorld) -> Dict[Tuple[str, str], Tuple[float, ...]]:
        rows: Dict[Tuple[str, str], List[Tuple[float, ...]]] = {}
        width = max(
            (min(len(c.low_timeline), len(c.high_timeline)) - 1 for c in world.contrasts),
            default=0,
        )
        for c in world.contrasts:
            n = min(len(c.low_timeline), len(c.high_timeline))
            nodes = {
                str(node)
                for timeline in (c.low_timeline, c.high_timeline)
                for snapshot in timeline
                for node, _ in snapshot
            }
            for target in nodes:
                if target == c.source_node:
                    continue
                curve = []
                for lag in range(1, width + 1):
                    if lag >= n:
                        curve.append(0.0)
                        continue
                    low = dict(c.low_timeline[lag]); high = dict(c.high_timeline[lag])
                    curve.append(float(high.get(target, 0.0)) - float(low.get(target, 0.0)))
                if any(abs(v) > self.tolerance for v in curve):
                    rows.setdefault((c.source_node, target), []).append(tuple(curve))
        result = {}
        for key, values in rows.items():
            if len(values) < self.min_repeats:
                continue
            result[key] = tuple(sum(v[i] for v in values) / len(values) for i in range(width))
        return result

    def _primitive_functionals(self, width: int) -> Tuple[Coefficients, ...]:
        if width <= 1:
            return ()
        found = []
        values = range(-self.coefficient_bound, self.coefficient_bound + 1)
        for coeffs in product(values, repeat=width):
            if not any(coeffs) or sum(coeffs) != 0:
                continue
            divisor = 0
            for value in coeffs:
                divisor = gcd(divisor, abs(value))
            if divisor != 1:
                continue
            first = next(value for value in coeffs if value)
            if first < 0:
                continue
            found.append(tuple(int(v) for v in coeffs))
        return tuple(sorted(found))

    def _path_signatures(self, world: OpaqueInterventionalWorld, coeffs: Coefficients) -> Tuple[Tuple[int, ...], ...]:
        adjacency: Dict[str, List[Tuple[str, int]]] = {}
        for (source, target), curve in self._curves(world).items():
            if len(curve) != len(coeffs):
                continue
            value = sum(float(c) * float(x) for c, x in zip(coeffs, curve))
            sign = 0 if abs(value) <= self.tolerance else (1 if value > 0 else -1)
            adjacency.setdefault(source, []).append((target, sign))
        found = []
        def walk(node: str, visited: Tuple[str, ...], signs: Tuple[int, ...]) -> None:
            for target, sign in adjacency.get(node, ()):
                if target in visited:
                    continue
                nxt = signs + (sign,)
                if target == world.target_anchor:
                    found.append(nxt)
                walk(target, visited + (target,), nxt)
        walk(world.source_anchor, (world.source_anchor,), ())
        return tuple(sorted(set(found)))

    def generate_candidates(
        self,
        assessment: ComparisonResidualAssessment,
        worlds: Sequence[OpaqueInterventionalWorld],
    ) -> Tuple[GeneratedComparisonSchema, ...]:
        if assessment.status != "ORDER_INVARIANT_AMBIGUITY_OPEN_COMPARISON_FUNCTIONAL" or not worlds:
            return ()
        first_curves = self._curves(worlds[0])
        width = len(next(iter(first_curves.values()), ()))
        candidates = []
        for coeffs in self._primitive_functionals(width):
            per_world = [self._path_signatures(world, coeffs) for world in worlds]
            if any(len(signatures) < 2 for signatures in per_world):
                continue
            common = set(per_world[0])
            for signatures in per_world[1:]:
                common.intersection_update(signatures)
            if len(common) < 2:
                continue
            for signs in sorted(common):
                candidates.append(GeneratedComparisonSchema(coeffs, signs))
        return tuple(sorted(candidates, key=lambda c: c.schema_id))

    def matches(self, schema: GeneratedComparisonSchema, world: OpaqueInterventionalWorld) -> bool:
        return schema.path_signs in self._path_signatures(world, schema.coefficients)


def derive_comparison_policy(
    schemas: Sequence[GeneratedComparisonSchema],
    pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int = 2,
    min_contexts: int = 2,
) -> ComparisonPolicy:
    allowed = []
    for schema in schemas:
        by_context: Dict[str, set[str]] = {}
        for pair in pairs:
            if pair.experiment_id != schema.schema_id:
                continue
            if not (pair.matched_budget and pair.externally_generated and pair.authority_verified
                    and pair.independence_class_id != "UNVERIFIED" and pair.effect > 0.0):
                continue
            by_context.setdefault(pair.context_id, set()).add(pair.independence_class_id)
        ready = [ctx for ctx, classes in by_context.items() if len(classes) >= min_independent_classes]
        if len(ready) >= min_contexts:
            allowed.append(schema.schema_id)
    return ComparisonPolicy(tuple(sorted(allowed)), min_independent_classes, min_contexts)


def select_authorized_comparison(
    schemas: Sequence[GeneratedComparisonSchema], policy: ComparisonPolicy
) -> Optional[GeneratedComparisonSchema]:
    allowed = set(policy.allowed_schema_ids)
    for schema in sorted(schemas, key=lambda s: s.schema_id):
        if schema.schema_id in allowed:
            return schema
    return None
