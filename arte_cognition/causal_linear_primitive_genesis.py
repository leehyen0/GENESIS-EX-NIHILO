from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Dict, List, Mapping, Sequence, Tuple
import hashlib
import itertools

from .causal_model_genesis import InterventionDescriptor
from .world_model_ecology import CausalWorldModel, ModelEvidence


@dataclass(frozen=True)
class GeneratedLinearPrimitive:
    coefficients: Tuple[Tuple[str, int], ...]
    threshold: float
    direction: str

    def render(self) -> str:
        expression = " + ".join(f"{weight}*RAW[{channel}]" for channel, weight in self.coefficients)
        return f"({expression}) {self.direction} {self.threshold:.12g}"


@dataclass(frozen=True)
class GeneratedLinearPrimitiveModel:
    cause: str
    sign: str
    primitive: GeneratedLinearPrimitive
    model: CausalWorldModel
    equivalent_primitives: Tuple[str, ...] = ()


class LinearFormPrimitiveGenesisEngine:
    """Synthesize a new scalar relation from multiple uninterpreted raw channels.

    This generation is opened only after single-channel threshold primitives have
    failed in the world. It does not choose from named SUM/DIFF operators. Instead
    it enumerates a bounded integer coefficient lattice, quotients scale/sign
    duplicates, constructs linear scores, and then enumerates threshold atoms over
    those scores. The coefficient/threshold shadow universe is outcome-independent;
    authoritative outcomes only filter it after construction.

    The integer-linear meta-grammar remains human-authored and bounded, so this is
    not unrestricted mathematical/operator genesis.
    """

    def __init__(
        self,
        model_budget: int = 8192,
        max_coefficient_abs: int = 2,
        min_active_channels: int = 2,
    ) -> None:
        self.model_budget = max(1, int(model_budget))
        self.max_coefficient_abs = max(1, int(max_coefficient_abs))
        self.min_active_channels = max(2, int(min_active_channels))
        self.last_linear_form_count = 0
        self.last_raw_candidate_count = 0
        self.last_unique_signature_count = 0
        self.last_truncated = False

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

    @staticmethod
    def _effect(sign: str) -> str:
        return "POSITIVE_EFFECT" if sign == "POS" else "NEGATIVE_EFFECT"

    @staticmethod
    def _thresholds(values: Sequence[float]) -> Tuple[float, ...]:
        unique = sorted({float(value) for value in values})
        return tuple((left + right) / 2.0 for left, right in zip(unique, unique[1:]))

    @staticmethod
    def _canonical_vector(vector: Sequence[int]) -> Tuple[int, ...] | None:
        vector = tuple(int(value) for value in vector)
        nonzero = [abs(value) for value in vector if value]
        if not nonzero:
            return None
        divisor = nonzero[0]
        for value in nonzero[1:]:
            divisor = gcd(divisor, value)
        reduced = tuple(value // divisor for value in vector)
        first = next(value for value in reduced if value)
        if first < 0:
            reduced = tuple(-value for value in reduced)
        return reduced

    def _coefficient_vectors(self, channel_count: int) -> Tuple[Tuple[int, ...], ...]:
        values = range(-self.max_coefficient_abs, self.max_coefficient_abs + 1)
        unique = set()
        for raw in itertools.product(values, repeat=channel_count):
            if sum(1 for value in raw if value) < self.min_active_channels:
                continue
            canonical = self._canonical_vector(raw)
            if canonical is not None:
                unique.add(canonical)
        return tuple(sorted(unique))

    @staticmethod
    def _score(
        coefficients: Sequence[Tuple[str, int]],
        raw_row: Mapping[str, float],
    ) -> float | None:
        total = 0.0
        for channel, weight in coefficients:
            if channel not in raw_row:
                return None
            total += float(weight) * float(raw_row[channel])
        return total

    @classmethod
    def predict(
        cls,
        cause: str,
        sign: str,
        primitive: GeneratedLinearPrimitive,
        descriptor: InterventionDescriptor,
        raw_observations: Mapping[str, Mapping[str, float]],
    ) -> str:
        if cause not in descriptor.targets or cause in descriptor.blocked:
            return "NO_EFFECT"
        row = raw_observations.get(descriptor.intervention_id, {})
        score = cls._score(primitive.coefficients, row)
        if score is None:
            return "NO_EFFECT"
        active = score >= primitive.threshold if primitive.direction == ">=" else score < primitive.threshold
        return cls._effect(sign) if active else "NO_EFFECT"

    @staticmethod
    def _model_id(cause: str, sign: str, primitive: GeneratedLinearPrimitive) -> str:
        coefficients = ";".join(f"{channel}:{weight}" for channel, weight in primitive.coefficients)
        raw = f"{cause}|{sign}|{coefficients}|{primitive.direction}|{primitive.threshold.hex()}".encode()
        return "GENLINEARPRIMITIVE::" + hashlib.sha256(raw).hexdigest()[:16]

    def generate_novel(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
        raw_observations: Mapping[str, Mapping[str, float]],
        residual_evidence: Sequence[ModelEvidence],
        existing_models: Sequence[CausalWorldModel],
    ) -> List[GeneratedLinearPrimitiveModel]:
        variables = tuple(sorted({str(value) for value in variables if str(value)}))
        descriptors = tuple(descriptors)
        descriptor_ids = {row.intervention_id for row in descriptors}
        channels = tuple(sorted({
            str(channel)
            for intervention_id, row in raw_observations.items()
            if intervention_id in descriptor_ids
            for channel in row
            if str(channel)
        }))
        if len(channels) < self.min_active_channels:
            self.last_linear_form_count = 0
            self.last_raw_candidate_count = 0
            self.last_unique_signature_count = 0
            self.last_truncated = False
            return []

        vectors = self._coefficient_vectors(len(channels))
        self.last_linear_form_count = len(vectors)
        existing_signatures = {self._signature(model) for model in existing_models}
        parents = tuple(sorted(
            model.model_id for model in existing_models
            if int(model.generation) == 5 and model.origin == "GENERATED_PRIMITIVE_THRESHOLD"
        ))

        raw: List[Tuple[str, str, GeneratedLinearPrimitive, CausalWorldModel]] = []
        for vector in vectors:
            coefficients = tuple((channel, weight) for channel, weight in zip(channels, vector) if weight)
            scores = []
            complete = True
            for descriptor in descriptors:
                score = self._score(coefficients, raw_observations.get(descriptor.intervention_id, {}))
                if score is None:
                    complete = False
                    break
                scores.append(score)
            if not complete or len(set(scores)) < 3:
                continue
            for threshold in self._thresholds(scores):
                for direction in (">=", "<"):
                    primitive = GeneratedLinearPrimitive(coefficients, float(threshold), direction)
                    for cause in variables:
                        for sign in ("POS", "NEG"):
                            predictions = tuple(
                                (
                                    descriptor.intervention_id,
                                    self.predict(cause, sign, primitive, descriptor, raw_observations),
                                )
                                for descriptor in descriptors
                            )
                            model = CausalWorldModel(
                                model_id=self._model_id(cause, sign, primitive),
                                prior=1.0,
                                predictions=predictions,
                                origin="GENERATED_LINEAR_PRIMITIVE",
                                family="RAW_LINEAR_FORM_THRESHOLD",
                                structure=(
                                    f"CAUSE({cause})",
                                    f"PRIMITIVE({primitive.render()})",
                                    f"SIGN({sign})",
                                ),
                                generation=6,
                                parent_model_ids=parents,
                            )
                            signature = self._signature(model)
                            if signature in existing_signatures:
                                continue
                            if not self._compatible(model, residual_evidence):
                                continue
                            raw.append((cause, sign, primitive, model))

        self.last_raw_candidate_count = len(raw)
        by_signature: Dict[
            Tuple[Tuple[str, str], ...],
            List[Tuple[str, str, GeneratedLinearPrimitive, CausalWorldModel]],
        ] = {}
        for item in raw:
            by_signature.setdefault(self._signature(item[3]), []).append(item)

        self.last_unique_signature_count = len(by_signature)
        self.last_truncated = self.last_unique_signature_count > self.model_budget

        out: List[GeneratedLinearPrimitiveModel] = []
        for _, group in sorted(by_signature.items(), key=lambda item: item[0]):
            ordered = sorted(
                group,
                key=lambda item: (
                    item[2].coefficients,
                    item[2].threshold,
                    item[2].direction,
                    item[0],
                    item[1],
                    item[3].model_id,
                ),
            )
            cause, sign, primitive, model = ordered[0]
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
            out.append(GeneratedLinearPrimitiveModel(cause, sign, primitive, model, equivalents))
            if len(out) >= self.model_budget:
                break
        return out
