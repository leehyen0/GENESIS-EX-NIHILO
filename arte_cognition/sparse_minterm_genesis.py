from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
import hashlib
import itertools

from .causal_model_genesis import InterventionDescriptor
from .causal_predicate_genesis import BooleanCausalPredicateGenesisEngine
from .world_model_ecology import CausalWorldModel, ModelEvidence


@dataclass(frozen=True)
class ExactMinterm:
    assignments: Tuple[Tuple[str, bool], ...]

    def render(self) -> str:
        return " & ".join(atom if value else f"!{atom}" for atom, value in self.assignments)


@dataclass(frozen=True)
class GeneratedSparseMintermModel:
    cause: str
    sign: str
    minterms: Tuple[ExactMinterm, ...]
    model: CausalWorldModel
    equivalent_structures: Tuple[str, ...] = ()


class SparseMintermCausalGenesisEngine:
    """Falsification-driven G4 expansion of the bounded Boolean grammar.

    G3 searches DNF with at most two conjunction terms and at most three literals
    per term. G4 does not invent a new logical metalanguage; it opens only after G3
    is externally falsified and searches sparse unions of *exact* intervention-state
    minterms. This raises representational complexity enough to express three
    disjoint activation islands while keeping the search finite and auditable.
    """

    def __init__(self, model_budget: int = 4096, max_minterms: int = 3) -> None:
        self.model_budget = max(1, int(model_budget))
        self.max_minterms = max(1, int(max_minterms))
        self.last_unique_signature_count = 0
        self.last_truncated = False

    @staticmethod
    def _atoms(variables: Sequence[str], cause: str) -> Tuple[str, ...]:
        return BooleanCausalPredicateGenesisEngine._atoms(variables, cause)

    @staticmethod
    def _atom_value(atom: str, d: InterventionDescriptor) -> bool:
        return BooleanCausalPredicateGenesisEngine._atom_value(atom, d)

    @classmethod
    def _matches(cls, minterm: ExactMinterm, d: InterventionDescriptor) -> bool:
        return all(cls._atom_value(atom, d) == value for atom, value in minterm.assignments)

    @staticmethod
    def _effect(sign: str) -> str:
        return "POSITIVE_EFFECT" if sign == "POS" else "NEGATIVE_EFFECT"

    @classmethod
    def predict(
        cls,
        cause: str,
        sign: str,
        minterms: Sequence[ExactMinterm],
        d: InterventionDescriptor,
    ) -> str:
        if cause not in d.targets or cause in d.blocked:
            return "NO_EFFECT"
        return cls._effect(sign) if any(cls._matches(term, d) for term in minterms) else "NO_EFFECT"

    @staticmethod
    def _model_id(cause: str, sign: str, minterms: Sequence[ExactMinterm]) -> str:
        rendered = "|".join(sorted(term.render() for term in minterms))
        digest = hashlib.sha256(f"{cause}|{sign}|{rendered}".encode()).hexdigest()[:16]
        return f"GENMINTERM::{digest}"

    @staticmethod
    def _signature(model: CausalWorldModel) -> Tuple[Tuple[str, str], ...]:
        return tuple(sorted(model.predictions))

    @staticmethod
    def _compatible(model: CausalWorldModel, evidence: Sequence[ModelEvidence]) -> bool:
        for item in evidence:
            if not item.authoritative:
                continue
            prediction = model.prediction_for(item.intervention_id)
            if prediction is not None and prediction != item.observed_outcome:
                return False
        return True

    @classmethod
    def _exact_minterms(cls, atoms: Sequence[str]) -> Tuple[ExactMinterm, ...]:
        atoms = tuple(atoms)
        return tuple(
            ExactMinterm(tuple((atom, bool(value)) for atom, value in zip(atoms, values)))
            for values in itertools.product((False, True), repeat=len(atoms))
        )

    def generate_novel(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
        residual_evidence: Sequence[ModelEvidence],
        existing_models: Sequence[CausalWorldModel],
    ) -> List[GeneratedSparseMintermModel]:
        variables = tuple(sorted({str(value) for value in variables if str(value)}))
        descriptors = tuple(descriptors)
        existing_signatures = {self._signature(model) for model in existing_models}
        parents = tuple(sorted(
            model.model_id for model in existing_models
            if int(model.generation) == 3 and model.origin == "GENERATED_PREDICATE"
        ))
        raw: List[Tuple[str, str, Tuple[ExactMinterm, ...], CausalWorldModel]] = []

        for cause in variables:
            exact = self._exact_minterms(self._atoms(variables, cause))
            for width in range(1, min(self.max_minterms, len(exact)) + 1):
                for chosen in itertools.combinations(exact, width):
                    for sign in ("POS", "NEG"):
                        predictions = tuple(
                            (d.intervention_id, self.predict(cause, sign, chosen, d))
                            for d in descriptors
                        )
                        structure = (
                            f"CAUSE({cause})",
                            *(f"MINTERM({term.render()})" for term in chosen),
                            f"SIGN({sign})",
                        )
                        model = CausalWorldModel(
                            model_id=self._model_id(cause, sign, chosen),
                            prior=1.0,
                            predictions=predictions,
                            origin="GENERATED_SPARSE_MINTERM",
                            family="SPARSE_EXACT_MINTERM_GATE",
                            structure=tuple(structure),
                            generation=4,
                            parent_model_ids=parents,
                        )
                        signature = self._signature(model)
                        if signature in existing_signatures:
                            continue
                        if not self._compatible(model, residual_evidence):
                            continue
                        raw.append((cause, sign, tuple(chosen), model))

        by_signature: Dict[Tuple[Tuple[str, str], ...], List[Tuple[str, str, Tuple[ExactMinterm, ...], CausalWorldModel]]] = {}
        for item in raw:
            by_signature.setdefault(self._signature(item[3]), []).append(item)
        self.last_unique_signature_count = len(by_signature)
        self.last_truncated = self.last_unique_signature_count > self.model_budget

        out: List[GeneratedSparseMintermModel] = []
        for _, group in sorted(by_signature.items(), key=lambda item: item[0]):
            ordered = sorted(
                group,
                key=lambda item: (
                    len(item[2]),
                    tuple(term.render() for term in item[2]),
                    item[0],
                    item[1],
                    item[3].model_id,
                ),
            )
            cause, sign, minterms, model = ordered[0]
            equivalents = tuple(
                " | ".join(term.render() for term in item[2]) for item in ordered[1:]
            )
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
            out.append(GeneratedSparseMintermModel(cause, sign, minterms, model, equivalents))
            if len(out) >= self.model_budget:
                break
        return out
