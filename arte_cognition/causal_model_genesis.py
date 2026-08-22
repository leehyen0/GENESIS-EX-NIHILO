from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import hashlib

from .adaptive_cognition import QueryCandidate
from .world_model_ecology import CausalWorldModel, ModelEvidence


EFFECTS = ("POSITIVE_EFFECT", "NEGATIVE_EFFECT")


@dataclass(frozen=True)
class InterventionDescriptor:
    """Observable intervention semantics available to causal-model search.

    The descriptor says what the BODY can manipulate or block; it does not contain
    the hidden outcome. Cost remains metadata for epistemic intervention ranking.
    """

    intervention_id: str
    targets: Tuple[str, ...]
    blocked: Tuple[str, ...] = ()
    delay_steps: int = 0
    context_shift: bool = False
    cost: float = 1.0


@dataclass(frozen=True)
class GeneratedCausalModel:
    model: CausalWorldModel
    equivalent_structures: Tuple[str, ...] = ()


class CausalModelGenesisEngine:
    """Generate bounded causal structures after the current model class fails.

    Candidate structures are generated from observable variable/intervention
    descriptors, never from a hidden answer key. Models that induce the same
    intervention prediction signature are quotiented before entering the live
    ecology so syntactic multiplicity cannot masquerade as independent support.
    """

    def __init__(self, model_budget: int = 48) -> None:
        self.model_budget = max(1, int(model_budget))

    @staticmethod
    def _effect(sign: str) -> str:
        return "POSITIVE_EFFECT" if sign == "POS" else "NEGATIVE_EFFECT"

    @staticmethod
    def _model_id(family: str, structure: Tuple[str, ...], sign: str) -> str:
        raw = "|".join((family, sign, *structure)).encode()
        digest = hashlib.sha256(raw).hexdigest()[:12]
        return f"GENMODEL::{family}::{digest}"

    @staticmethod
    def _predict_direct(cause: str, sign: str, d: InterventionDescriptor) -> str:
        return CausalModelGenesisEngine._effect(sign) if cause in d.targets and cause not in d.blocked else "NO_EFFECT"

    @staticmethod
    def _predict_mediated(cause: str, mediator: str, sign: str, d: InterventionDescriptor) -> str:
        # do(cause) acts through mediator; blocking mediator cuts the path.
        if mediator in d.blocked:
            return "NO_EFFECT"
        if cause in d.targets or mediator in d.targets:
            return CausalModelGenesisEngine._effect(sign)
        return "NO_EFFECT"

    @staticmethod
    def _predict_interaction(a: str, b: str, sign: str, d: InterventionDescriptor) -> str:
        live_targets = set(d.targets) - set(d.blocked)
        return CausalModelGenesisEngine._effect(sign) if {a, b}.issubset(live_targets) else "NO_EFFECT"

    @staticmethod
    def _predict_temporal(cause: str, sign: str, d: InterventionDescriptor) -> str:
        if cause in d.targets and cause not in d.blocked and int(d.delay_steps) > 0:
            return CausalModelGenesisEngine._effect(sign)
        return "NO_EFFECT"

    @staticmethod
    def _predict_latent_common(a: str, b: str, sign: str, d: InterventionDescriptor) -> str:
        # Direct manipulation of either observed child does not move the other via
        # a latent common cause. A context-shift probe can expose the shared cause.
        if d.context_shift and not d.targets:
            return CausalModelGenesisEngine._effect(sign)
        return "NO_EFFECT"

    def _candidate(
        self,
        family: str,
        structure: Tuple[str, ...],
        sign: str,
        descriptors: Sequence[InterventionDescriptor],
    ) -> CausalWorldModel:
        predictions: List[Tuple[str, str]] = []
        for d in descriptors:
            if family == "DIRECT":
                outcome = self._predict_direct(structure[0], sign, d)
            elif family == "MEDIATED":
                outcome = self._predict_mediated(structure[0], structure[1], sign, d)
            elif family == "INTERACTION":
                outcome = self._predict_interaction(structure[0], structure[1], sign, d)
            elif family == "TEMPORAL":
                outcome = self._predict_temporal(structure[0], sign, d)
            elif family == "LATENT_COMMON_CAUSE":
                outcome = self._predict_latent_common(structure[0], structure[1], sign, d)
            else:
                outcome = "NO_EFFECT"
            predictions.append((d.intervention_id, outcome))

        if family == "DIRECT":
            graph = (f"{structure[0]}->OUTCOME",)
        elif family == "MEDIATED":
            graph = (f"{structure[0]}->{structure[1]}", f"{structure[1]}->OUTCOME")
        elif family == "INTERACTION":
            graph = (f"{structure[0]}*{structure[1]}->OUTCOME",)
        elif family == "TEMPORAL":
            graph = (f"{structure[0]}_t->OUTCOME_t+1",)
        else:
            latent = f"LATENT::{structure[0]}|{structure[1]}"
            graph = (f"{latent}->{structure[0]}", f"{latent}->{structure[1]}", f"{latent}->OUTCOME")

        return CausalWorldModel(
            model_id=self._model_id(family, structure, sign),
            prior=1.0,
            predictions=tuple(predictions),
            origin="GENERATED",
            family=family,
            structure=tuple(graph),
            generation=1,
            parent_model_ids=(),
        )

    @staticmethod
    def _compatible(model: CausalWorldModel, evidence: Sequence[ModelEvidence]) -> bool:
        authoritative = [item for item in evidence if item.authoritative]
        if not authoritative:
            return True
        for item in authoritative:
            prediction = model.prediction_for(item.intervention_id)
            if prediction is not None and prediction != item.observed_outcome:
                return False
        return True

    @staticmethod
    def _signature(model: CausalWorldModel) -> Tuple[Tuple[str, str], ...]:
        return tuple(sorted(model.predictions))

    @staticmethod
    def _complexity(model: CausalWorldModel) -> Tuple[int, int, str]:
        # Prefer fewer graph clauses, then simpler family ordering, then stable id.
        family_rank = {
            "DIRECT": 0,
            "TEMPORAL": 1,
            "MEDIATED": 2,
            "INTERACTION": 3,
            "LATENT_COMMON_CAUSE": 4,
        }
        return (len(model.structure), family_rank.get(model.family, 99), model.model_id)

    def generate(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
        residual_evidence: Sequence[ModelEvidence] = (),
    ) -> List[GeneratedCausalModel]:
        variables = tuple(sorted({str(v) for v in variables if str(v)}))
        descriptors = tuple(descriptors)
        raw: List[CausalWorldModel] = []
        for sign in ("POS", "NEG"):
            for cause in variables:
                raw.append(self._candidate("DIRECT", (cause,), sign, descriptors))
                raw.append(self._candidate("TEMPORAL", (cause,), sign, descriptors))
            for i, a in enumerate(variables):
                for b in variables[i + 1 :]:
                    raw.append(self._candidate("INTERACTION", (a, b), sign, descriptors))
                    raw.append(self._candidate("LATENT_COMMON_CAUSE", (a, b), sign, descriptors))
                    raw.append(self._candidate("MEDIATED", (a, b), sign, descriptors))
                    raw.append(self._candidate("MEDIATED", (b, a), sign, descriptors))

        compatible = [model for model in raw if self._compatible(model, residual_evidence)]
        by_signature: Dict[Tuple[Tuple[str, str], ...], List[CausalWorldModel]] = {}
        for model in compatible:
            by_signature.setdefault(self._signature(model), []).append(model)

        out: List[GeneratedCausalModel] = []
        for signature, group in sorted(by_signature.items(), key=lambda item: item[0]):
            ordered = sorted(group, key=self._complexity)
            representative = ordered[0]
            equivalents = tuple(
                f"{model.family}:{'|'.join(model.structure)}"
                for model in ordered[1:]
            )
            # Preserve the shadow-equivalence lineage on the representative.
            representative = CausalWorldModel(
                model_id=representative.model_id,
                prior=representative.prior,
                predictions=representative.predictions,
                origin=representative.origin,
                family=representative.family,
                structure=representative.structure,
                generation=representative.generation,
                parent_model_ids=representative.parent_model_ids,
                equivalent_structures=equivalents,
            )
            out.append(GeneratedCausalModel(representative, equivalents))
            if len(out) >= self.model_budget:
                break
        return out

    @staticmethod
    def query_candidates(
        descriptors: Sequence[InterventionDescriptor],
        models: Sequence[CausalWorldModel],
    ) -> List[QueryCandidate]:
        out: List[QueryCandidate] = []
        for descriptor in descriptors:
            distinguishes = {
                model.model_id: model.prediction_for(descriptor.intervention_id) or "__UNKNOWN__"
                for model in models
            }
            out.append(QueryCandidate(
                query_id=descriptor.intervention_id,
                distinguishes=distinguishes,
                cost=max(1e-9, float(descriptor.cost)),
                intervention=True,
                source_class="GENERATED_CAUSAL_MODEL_DISCRIMINATION",
            ))
        return out
