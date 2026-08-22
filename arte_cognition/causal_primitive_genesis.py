from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple
import hashlib

from .causal_model_genesis import InterventionDescriptor
from .world_model_ecology import CausalWorldModel, ModelEvidence


@dataclass(frozen=True)
class GeneratedPrimitive:
    channel: str
    threshold: float
    direction: str

    def render(self) -> str:
        return f"RAW[{self.channel}] {self.direction} {self.threshold:.12g}"


@dataclass(frozen=True)
class GeneratedPrimitiveModel:
    cause: str
    sign: str
    primitive: GeneratedPrimitive
    model: CausalWorldModel
    equivalent_primitives: Tuple[str, ...] = ()


class RawThresholdPrimitiveGenesisEngine:
    """Generate a new causal atom from uninterpreted numeric world observations.

    G1-G4 can only reason over target/block/delay/context atoms. This engine is
    deliberately different: it receives numeric raw channels whose names carry no
    causal semantics and enumerates threshold atoms at midpoints between observed
    values. Candidate thresholds are generated without looking at outcomes.
    Authoritative outcome evidence is used only to filter the already-generated
    shadow universe.

    The mechanism is bounded and auditable. It is not unrestricted sensor or
    operator genesis: the meta-rule "numeric channel -> threshold atom" remains
    human-authored, while the channel identity, threshold and causal use are not.
    """

    def __init__(self, model_budget: int = 4096, min_distinct_values: int = 3) -> None:
        self.model_budget = max(1, int(model_budget))
        self.min_distinct_values = max(2, int(min_distinct_values))
        self.last_unique_signature_count = 0
        self.last_truncated = False
        self.last_raw_candidate_count = 0

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
    def _raw_value(
        raw_observations: Mapping[str, Mapping[str, float]],
        intervention_id: str,
        channel: str,
    ) -> float | None:
        row = raw_observations.get(intervention_id, {})
        value = row.get(channel)
        return None if value is None else float(value)

    @classmethod
    def predict(
        cls,
        cause: str,
        sign: str,
        primitive: GeneratedPrimitive,
        descriptor: InterventionDescriptor,
        raw_observations: Mapping[str, Mapping[str, float]],
    ) -> str:
        if cause not in descriptor.targets or cause in descriptor.blocked:
            return "NO_EFFECT"
        value = cls._raw_value(raw_observations, descriptor.intervention_id, primitive.channel)
        if value is None:
            return "NO_EFFECT"
        active = value >= primitive.threshold if primitive.direction == ">=" else value < primitive.threshold
        return cls._effect(sign) if active else "NO_EFFECT"

    @staticmethod
    def _model_id(cause: str, sign: str, primitive: GeneratedPrimitive) -> str:
        raw = f"{cause}|{sign}|{primitive.channel}|{primitive.direction}|{primitive.threshold.hex()}".encode()
        return "GENPRIMITIVE::" + hashlib.sha256(raw).hexdigest()[:16]

    @staticmethod
    def _thresholds(values: Sequence[float]) -> Tuple[float, ...]:
        unique = sorted({float(value) for value in values})
        return tuple((left + right) / 2.0 for left, right in zip(unique, unique[1:]))

    def generate_novel(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
        raw_observations: Mapping[str, Mapping[str, float]],
        residual_evidence: Sequence[ModelEvidence],
        existing_models: Sequence[CausalWorldModel],
    ) -> List[GeneratedPrimitiveModel]:
        variables = tuple(sorted({str(value) for value in variables if str(value)}))
        descriptors = tuple(descriptors)
        descriptor_ids = {row.intervention_id for row in descriptors}
        channels = sorted({
            str(channel)
            for intervention_id, row in raw_observations.items()
            if intervention_id in descriptor_ids
            for channel in row
            if str(channel)
        })
        existing_signatures = {self._signature(model) for model in existing_models}
        parents = tuple(sorted(
            model.model_id for model in existing_models
            if int(model.generation) == 4 and model.origin == "GENERATED_SPARSE_MINTERM"
        ))

        raw: List[Tuple[str, str, GeneratedPrimitive, CausalWorldModel]] = []
        for channel in channels:
            values = [
                value
                for descriptor in descriptors
                for value in [self._raw_value(raw_observations, descriptor.intervention_id, channel)]
                if value is not None
            ]
            if len(set(values)) < self.min_distinct_values:
                continue
            for threshold in self._thresholds(values):
                for direction in (">=", "<"):
                    primitive = GeneratedPrimitive(channel, float(threshold), direction)
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
                                origin="GENERATED_PRIMITIVE_THRESHOLD",
                                family="RAW_THRESHOLD_PRIMITIVE",
                                structure=(
                                    f"CAUSE({cause})",
                                    f"PRIMITIVE({primitive.render()})",
                                    f"SIGN({sign})",
                                ),
                                generation=5,
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
            List[Tuple[str, str, GeneratedPrimitive, CausalWorldModel]],
        ] = {}
        for item in raw:
            by_signature.setdefault(self._signature(item[3]), []).append(item)

        self.last_unique_signature_count = len(by_signature)
        self.last_truncated = self.last_unique_signature_count > self.model_budget

        out: List[GeneratedPrimitiveModel] = []
        for _, group in sorted(by_signature.items(), key=lambda item: item[0]):
            ordered = sorted(
                group,
                key=lambda item: (
                    item[2].channel,
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
            out.append(GeneratedPrimitiveModel(cause, sign, primitive, model, equivalents))
            if len(out) >= self.model_budget:
                break
        return out
