from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True, order=True)
class Fact:
    subject: str
    relation: str
    object: str


@dataclass(frozen=True)
class OperatorSpec:
    relation_opposites: Mapping[str, str] = field(default_factory=dict)
    object_complements: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PossibilityCandidate:
    mode: str
    facts: Tuple[Fact, ...]
    provenance: Tuple[str, ...]
    asserted: bool = False
    query_targets: Tuple[str, ...] = ()

    @property
    def signature(self) -> Tuple:
        return (
            self.mode,
            tuple(sorted(self.facts)),
            tuple(sorted(self.query_targets)),
            self.asserted,
        )


class PossibilitySpaceGenerator:
    """Generate bounded structural alternatives without turning them into evidence.

    The generator only applies transformations for which the caller supplied an
    operator meaning. `IMAGINARY` generates latent query targets rather than
    asserted facts, preserving possibility without smuggling speculation into the
    evidence layer.
    """

    def expand(
        self,
        facts: Sequence[Fact],
        modal_basis: Iterable[str],
        spec: OperatorSpec | None = None,
        budget: int = 32,
    ) -> List[PossibilityCandidate]:
        spec = spec or OperatorSpec()
        basis = {str(x).upper() for x in modal_basis}
        base = tuple(sorted(facts))
        out: List[PossibilityCandidate] = []

        def add(candidate: PossibilityCandidate) -> None:
            if len(out) >= max(1, int(budget)):
                return
            if candidate.signature not in {x.signature for x in out}:
                out.append(candidate)

        if "EXPLICIT" in basis:
            add(PossibilityCandidate("EXPLICIT", base, ("observed/current representation",), asserted=False))

        if "OPPOSITE" in basis:
            for i, fact in enumerate(base):
                opposite = spec.relation_opposites.get(fact.relation)
                if not opposite:
                    continue
                changed = list(base)
                changed[i] = Fact(fact.subject, opposite, fact.object)
                add(PossibilityCandidate(
                    "OPPOSITE",
                    tuple(sorted(changed)),
                    (f"relation opposite: {fact.relation}->{opposite}",),
                ))

        if "COMPLEMENT" in basis:
            for i, fact in enumerate(base):
                complement = spec.object_complements.get(fact.object)
                if not complement:
                    continue
                changed = list(base)
                changed[i] = Fact(fact.subject, fact.relation, complement)
                add(PossibilityCandidate(
                    "COMPLEMENT",
                    tuple(sorted(changed)),
                    (f"object complement: {fact.object}->{complement}",),
                ))

        if "ABSENCE" in basis:
            for i, fact in enumerate(base):
                changed = tuple(x for j, x in enumerate(base) if j != i)
                add(PossibilityCandidate(
                    "ABSENCE",
                    changed,
                    (f"remove fact: {fact.subject}/{fact.relation}/{fact.object}",),
                ))

        if "COUNTERFACTUAL" in basis:
            for i, fact in enumerate(base):
                opposite = spec.relation_opposites.get(fact.relation)
                if not opposite:
                    continue
                changed = list(base)
                changed[i] = Fact(fact.subject, opposite, fact.object)
                add(PossibilityCandidate(
                    "COUNTERFACTUAL",
                    tuple(sorted(changed)),
                    (f"do({fact.subject}.{fact.relation}:={opposite})",),
                ))

        if "IMAGINARY" in basis:
            for fact in base:
                target = f"LATENT::{fact.subject}::{fact.relation}::{fact.object}"
                add(PossibilityCandidate(
                    "IMAGINARY",
                    base,
                    ("latent distinction requested; not asserted as world fact",),
                    asserted=False,
                    query_targets=(target,),
                ))

        return out
