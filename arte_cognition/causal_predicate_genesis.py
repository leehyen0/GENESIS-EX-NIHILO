from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
import hashlib
import itertools

from .causal_model_genesis import InterventionDescriptor
from .world_model_ecology import CausalWorldModel, ModelEvidence


@dataclass(frozen=True, order=True)
class PredicateLiteral:
    atom: str
    value: bool

    def render(self) -> str:
        return self.atom if self.value else f"!{self.atom}"


@dataclass(frozen=True)
class PredicateTerm:
    literals: Tuple[PredicateLiteral, ...]

    def render(self) -> str:
        return " & ".join(literal.render() for literal in self.literals)


@dataclass(frozen=True)
class ActivationPredicate:
    terms: Tuple[PredicateTerm, ...]

    def render(self) -> str:
        return " | ".join(f"({term.render()})" for term in self.terms)


@dataclass(frozen=True)
class GeneratedPredicateModel:
    cause: str
    sign: str
    predicate: ActivationPredicate
    model: CausalWorldModel
    equivalent_predicates: Tuple[str, ...] = ()


class BooleanCausalPredicateGenesisEngine:
    """Synthesize causal activation logic outside the fixed conjunction grammar.

    The atom vocabulary is observable intervention semantics, while the activation
    predicate is searched as bounded DNF. This can discover negation/disjunction
    such as `(DELAY & !CONTEXT) | (!DELAY & CONTEXT)` without a hardcoded XOR
    causal operator. The Boolean metalanguage is still human-authored and bounded.
    """

    def __init__(
        self,
        model_budget: int = 512,
        max_literals_per_term: int = 3,
        max_terms: int = 2,
    ) -> None:
        self.model_budget = max(1, int(model_budget))
        self.max_literals_per_term = max(1, int(max_literals_per_term))
        self.max_terms = max(1, int(max_terms))

    @staticmethod
    def _atoms(variables: Sequence[str], cause: str) -> Tuple[str, ...]:
        atoms = ["DELAY", "CONTEXT"]
        for variable in sorted({str(v) for v in variables if str(v)}):
            if variable != cause:
                atoms.append(f"TARGET:{variable}")
                atoms.append(f"BLOCKED:{variable}")
        return tuple(atoms)

    @staticmethod
    def _atom_value(atom: str, d: InterventionDescriptor) -> bool:
        if atom == "DELAY":
            return int(d.delay_steps) > 0
        if atom == "CONTEXT":
            return bool(d.context_shift)
        if atom.startswith("TARGET:"):
            return atom.split(":", 1)[1] in d.targets
        if atom.startswith("BLOCKED:"):
            return atom.split(":", 1)[1] in d.blocked
        return False

    @classmethod
    def _predicate_value(cls, predicate: ActivationPredicate, d: InterventionDescriptor) -> bool:
        for term in predicate.terms:
            if all(cls._atom_value(literal.atom, d) == literal.value for literal in term.literals):
                return True
        return False

    @staticmethod
    def _effect(sign: str) -> str:
        return "POSITIVE_EFFECT" if sign == "POS" else "NEGATIVE_EFFECT"

    @classmethod
    def predict(
        cls,
        cause: str,
        sign: str,
        predicate: ActivationPredicate,
        d: InterventionDescriptor,
    ) -> str:
        cause_active = cause in d.targets and cause not in d.blocked
        active = cause_active and cls._predicate_value(predicate, d)
        return cls._effect(sign) if active else "NO_EFFECT"

    @staticmethod
    def _term_key(term: PredicateTerm) -> Tuple[str, ...]:
        return tuple(literal.render() for literal in term.literals)

    def _terms(self, atoms: Sequence[str]) -> List[PredicateTerm]:
        out: Dict[Tuple[str, ...], PredicateTerm] = {}
        atoms = tuple(atoms)
        for width in range(1, min(self.max_literals_per_term, len(atoms)) + 1):
            for chosen_atoms in itertools.combinations(atoms, width):
                for values in itertools.product((False, True), repeat=width):
                    literals = tuple(sorted(
                        (PredicateLiteral(atom, bool(value)) for atom, value in zip(chosen_atoms, values)),
                        key=lambda literal: (literal.atom, literal.value),
                    ))
                    term = PredicateTerm(literals)
                    out[self._term_key(term)] = term
        return list(out.values())

    def _predicates(self, atoms: Sequence[str]) -> Iterable[ActivationPredicate]:
        terms = self._terms(atoms)
        # Single conjunctions plus disjunctions of bounded conjunctions.
        for term in terms:
            yield ActivationPredicate((term,))
        if self.max_terms >= 2:
            for left, right in itertools.combinations(terms, 2):
                # Remove trivially absorbed disjuncts, e.g. A | (A & B).
                lset = set(left.literals)
                rset = set(right.literals)
                if lset.issubset(rset) or rset.issubset(lset):
                    continue
                ordered = tuple(sorted((left, right), key=self._term_key))
                yield ActivationPredicate(ordered)

    @staticmethod
    def _model_id(cause: str, sign: str, predicate: ActivationPredicate) -> str:
        raw = f"{cause}|{sign}|{predicate.render()}".encode()
        return "GENPRED::" + hashlib.sha256(raw).hexdigest()[:16]

    @staticmethod
    def _compatible(model: CausalWorldModel, evidence: Sequence[ModelEvidence]) -> bool:
        for item in evidence:
            if not item.authoritative:
                continue
            prediction = model.prediction_for(item.intervention_id)
            if prediction is not None and prediction != item.observed_outcome:
                return False
        return True

    @staticmethod
    def _signature(model: CausalWorldModel) -> Tuple[Tuple[str, str], ...]:
        return tuple(sorted(model.predictions))

    @staticmethod
    def _complexity(item: Tuple[str, str, ActivationPredicate, CausalWorldModel]):
        cause, sign, predicate, model = item
        literal_count = sum(len(term.literals) for term in predicate.terms)
        return (len(predicate.terms), literal_count, predicate.render(), cause, sign, model.model_id)

    def generate_novel(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
        residual_evidence: Sequence[ModelEvidence],
        existing_models: Sequence[CausalWorldModel],
    ) -> List[GeneratedPredicateModel]:
        variables = tuple(sorted({str(v) for v in variables if str(v)}))
        descriptors = tuple(descriptors)
        existing_signatures = {self._signature(model) for model in existing_models}
        raw: List[Tuple[str, str, ActivationPredicate, CausalWorldModel]] = []

        for cause in variables:
            atoms = self._atoms(variables, cause)
            for sign in ("POS", "NEG"):
                for predicate in self._predicates(atoms):
                    predictions = tuple(
                        (d.intervention_id, self.predict(cause, sign, predicate, d))
                        for d in descriptors
                    )
                    model = CausalWorldModel(
                        model_id=self._model_id(cause, sign, predicate),
                        prior=1.0,
                        predictions=predictions,
                        origin="GENERATED_PREDICATE",
                        family="SYNTHESIZED_ACTIVATION_PREDICATE",
                        structure=(
                            f"CAUSE({cause})",
                            f"GATE({predicate.render()})",
                            f"SIGN({sign})",
                        ),
                        generation=3,
                        parent_model_ids=tuple(sorted(
                            model.model_id for model in existing_models
                            if model.origin == "GENERATED_COMPOSITIONAL"
                        )),
                    )
                    signature = self._signature(model)
                    if signature in existing_signatures:
                        continue
                    if not self._compatible(model, residual_evidence):
                        continue
                    raw.append((cause, sign, predicate, model))

        by_signature: Dict[Tuple[Tuple[str, str], ...], List[Tuple[str, str, ActivationPredicate, CausalWorldModel]]] = {}
        for item in raw:
            by_signature.setdefault(self._signature(item[3]), []).append(item)

        out: List[GeneratedPredicateModel] = []
        for signature, group in sorted(by_signature.items(), key=lambda item: item[0]):
            ordered = sorted(group, key=self._complexity)
            cause, sign, predicate, model = ordered[0]
            equivalents = tuple(item[2].render() for item in ordered[1:])
            model = CausalWorldModel(
                model_id=model.model_id,
                prior=model.prior,
                predictions=model.predictions,
                origin=model.origin,
                family=model.family,
                structure=model.structure,
                generation=model.generation,
                parent_model_ids=model.parent_model_ids,
                equivalent_structures=equivalents,
            )
            out.append(GeneratedPredicateModel(cause, sign, predicate, model, equivalents))
            if len(out) >= self.model_budget:
                break
        return out
