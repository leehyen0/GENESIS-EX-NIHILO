from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Dict, List, Optional, Sequence, Tuple
import hashlib

from .latent_relation_ontology_genesis import OpaqueInterventionalWorld
from .world_coupling import WorldOutcomePair

Exponent = Tuple[int, ...]
Functional = Tuple[Exponent, Exponent]


@dataclass(frozen=True)
class PolynomialResidualAssessment:
    status: str
    context_ids: Tuple[str, ...]
    complete_linear_failure: bool


@dataclass(frozen=True)
class GeneratedPolynomialRelationSchema:
    positive_monomial: Exponent
    negative_monomial: Exponent
    path_signs: Tuple[int, ...]

    @property
    def schema_id(self) -> str:
        raw = repr((self.positive_monomial, self.negative_monomial, self.path_signs)).encode()
        return "POLYNOMIAL_RELATION_PATH::" + hashlib.sha256(raw).hexdigest()[:20]


@dataclass(frozen=True)
class PolynomialRelationPolicy:
    allowed_schema_ids: Tuple[str, ...]
    min_independent_classes: int
    min_contexts: int


class WorldDerivedPolynomialRelationInducer:
    """Lift raw response curves only after the bounded linear grammar is exhausted.

    No named ratio, curvature, geometric-convexity, or domain feature is supplied.
    The bounded lift enumerates homogeneous degree-d monomials over observed lag
    coordinates and differences between them. Only relations that split the same
    repeated predecessor ambiguity in every training context survive generation.
    Outcomes are absent from this stage.

    This is still bounded: monomial lifting, degree, arithmetic multiplication,
    subtraction, sign discretization and graph traversal are authored.
    """

    def __init__(self, degree: int = 2, min_repeats: int = 2, tolerance: float = 1e-9) -> None:
        self.degree = max(2, int(degree))
        self.min_repeats = max(1, int(min_repeats))
        self.tolerance = max(0.0, float(tolerance))

    @staticmethod
    def assess_residual(
        worlds: Sequence[OpaqueInterventionalWorld],
        predecessor_candidate_counts: Sequence[int],
        min_contexts: int = 2,
    ) -> PolynomialResidualAssessment:
        complete = (
            len(worlds) >= max(1, int(min_contexts))
            and len(predecessor_candidate_counts) == len(worlds)
            and all(int(v) == 0 for v in predecessor_candidate_counts)
        )
        return PolynomialResidualAssessment(
            status=("LINEAR_COMPARISON_GRAMMAR_EXHAUSTED_OPEN_POLYNOMIAL_RELATION"
                    if complete else "LINEAR_COMPARISON_GRAMMAR_FAILURE_NOT_ESTABLISHED"),
            context_ids=tuple(w.context_id for w in worlds),
            complete_linear_failure=complete,
        )

    def _curves(self, world: OpaqueInterventionalWorld) -> Dict[Tuple[str, str], Tuple[float, ...]]:
        rows: Dict[Tuple[str, str], List[Tuple[float, ...]]] = {}
        width = max((min(len(c.low_timeline), len(c.high_timeline)) - 1 for c in world.contrasts), default=0)
        for contrast in world.contrasts:
            n = min(len(contrast.low_timeline), len(contrast.high_timeline))
            nodes = {
                str(node)
                for timeline in (contrast.low_timeline, contrast.high_timeline)
                for snap in timeline
                for node, _ in snap
            }
            for target in nodes:
                if target == contrast.source_node:
                    continue
                curve=[]
                for lag in range(1,width+1):
                    if lag >= n:
                        curve.append(0.0); continue
                    low=dict(contrast.low_timeline[lag]); high=dict(contrast.high_timeline[lag])
                    curve.append(float(high.get(target,0.0))-float(low.get(target,0.0)))
                if any(abs(v)>self.tolerance for v in curve):
                    rows.setdefault((contrast.source_node,target),[]).append(tuple(curve))
        out={}
        for key, values in rows.items():
            if len(values)<self.min_repeats: continue
            out[key]=tuple(sum(v[i] for v in values)/len(values) for i in range(width))
        return out

    def _monomials(self, width: int) -> Tuple[Exponent, ...]:
        found=[]
        for indices in combinations_with_replacement(range(width), self.degree):
            exp=[0]*width
            for index in indices: exp[index]+=1
            found.append(tuple(exp))
        return tuple(sorted(set(found)))

    @staticmethod
    def _eval_monomial(curve: Sequence[float], exponent: Exponent) -> float:
        value=1.0
        for x,power in zip(curve,exponent):
            if power: value*=float(x)**int(power)
        return value

    def _functionals(self, width: int) -> Tuple[Functional, ...]:
        monomials=self._monomials(width)
        return tuple((a,b) for i,a in enumerate(monomials) for b in monomials[i+1:])

    def _path_signatures(self, world: OpaqueInterventionalWorld, functional: Functional) -> Tuple[Tuple[int,...],...]:
        pos,neg=functional
        adjacency: Dict[str,List[Tuple[str,int]]]={}
        for (source,target),curve in self._curves(world).items():
            value=self._eval_monomial(curve,pos)-self._eval_monomial(curve,neg)
            sign=0 if abs(value)<=self.tolerance else (1 if value>0 else -1)
            adjacency.setdefault(source,[]).append((target,sign))
        paths=[]
        def walk(node,visited,signs):
            for target,sign in adjacency.get(node,()):
                if target in visited: continue
                nxt=signs+(sign,)
                if target==world.target_anchor: paths.append(nxt)
                walk(target,visited+(target,),nxt)
        walk(world.source_anchor,(world.source_anchor,),())
        return tuple(sorted(set(paths)))

    def generate_candidates(
        self,
        assessment: PolynomialResidualAssessment,
        worlds: Sequence[OpaqueInterventionalWorld],
    ) -> Tuple[GeneratedPolynomialRelationSchema,...]:
        if assessment.status!="LINEAR_COMPARISON_GRAMMAR_EXHAUSTED_OPEN_POLYNOMIAL_RELATION" or not worlds:
            return ()
        width=len(next(iter(self._curves(worlds[0]).values()),()))
        generated=[]
        for functional in self._functionals(width):
            per_world=[self._path_signatures(world,functional) for world in worlds]
            if any(len(signatures)<2 for signatures in per_world): continue
            common=set(per_world[0])
            for signatures in per_world[1:]: common.intersection_update(signatures)
            if len(common)<2: continue
            for signs in sorted(common):
                generated.append(GeneratedPolynomialRelationSchema(functional[0],functional[1],signs))
        # Quotient exact behavioral duplicates across the training worlds. The
        # representative is deterministic and outcome-free.
        quotient={}
        for schema in sorted(generated,key=lambda s:s.schema_id):
            signature=tuple(self._path_signatures(w,(schema.positive_monomial,schema.negative_monomial)) for w in worlds)+(schema.path_signs,)
            quotient.setdefault(signature,schema)
        return tuple(sorted(quotient.values(),key=lambda s:s.schema_id))

    def matches(self,schema:GeneratedPolynomialRelationSchema,world:OpaqueInterventionalWorld)->bool:
        return schema.path_signs in self._path_signatures(world,(schema.positive_monomial,schema.negative_monomial))


def derive_polynomial_relation_policy(
    schemas: Sequence[GeneratedPolynomialRelationSchema],
    pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int = 2,
    min_contexts: int = 2,
) -> PolynomialRelationPolicy:
    allowed=[]
    for schema in schemas:
        by_context: Dict[str,set[str]]={}
        for pair in pairs:
            if pair.experiment_id!=schema.schema_id: continue
            if not (pair.matched_budget and pair.externally_generated and pair.authority_verified
                    and pair.independence_class_id!="UNVERIFIED" and pair.effect>0.0): continue
            by_context.setdefault(pair.context_id,set()).add(pair.independence_class_id)
        ready=[ctx for ctx,classes in by_context.items() if len(classes)>=min_independent_classes]
        if len(ready)>=min_contexts: allowed.append(schema.schema_id)
    return PolynomialRelationPolicy(tuple(sorted(allowed)),min_independent_classes,min_contexts)


def select_authorized_polynomial_relation(
    schemas:Sequence[GeneratedPolynomialRelationSchema],policy:PolynomialRelationPolicy
)->Optional[GeneratedPolynomialRelationSchema]:
    allowed=set(policy.allowed_schema_ids)
    for schema in sorted(schemas,key=lambda s:s.schema_id):
        if schema.schema_id in allowed: return schema
    return None
