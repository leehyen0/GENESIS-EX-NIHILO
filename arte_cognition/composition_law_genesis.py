from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, List, Mapping, Sequence, Tuple
import hashlib

from .causal_model_genesis import InterventionDescriptor
from .world_model_ecology import CausalWorldModel, ModelEvidence


@dataclass(frozen=True)
class GeneratedCompositionLaw:
    """A generated finite binary operation plus an activation subset.

    State semantics are intentionally unnamed. The current bounded interpreter maps
    a raw scalar to one of three ordinal sign states, fixes the middle state as an
    identity element, and enumerates every commutative table compatible with that
    identity. The table entries themselves are generated rather than selected from
    ADD/SUB/MUL/ABS or another named arithmetic operator vocabulary.
    """

    state_count: int
    identity_state: int
    table: Tuple[int, ...]
    active_states: Tuple[int, ...]

    def combine(self, left: int, right: int) -> int:
        return int(self.table[int(left) * self.state_count + int(right)])

    @property
    def law_id(self) -> str:
        raw = repr((self.state_count, self.identity_state, self.table, self.active_states)).encode()
        return "COMPOSITION_LAW::" + hashlib.sha256(raw).hexdigest()[:20]

    def render(self) -> str:
        rows = tuple(
            self.table[index * self.state_count : (index + 1) * self.state_count]
            for index in range(self.state_count)
        )
        return f"TABLE{rows}|ACTIVE{self.active_states}"


@dataclass(frozen=True)
class GeneratedCompositionLawModel:
    cause: str
    sign: str
    left_channel: str
    right_channel: str
    law: GeneratedCompositionLaw
    model: CausalWorldModel
    equivalent_laws: Tuple[str, ...] = ()


class CompositionLawGenesisEngine:
    """Generate a bounded operation law after fixed symbolic algebra is falsified.

    Candidate generation sees raw observations and intervention structure, but no
    success outcomes. It enumerates a finite operation-table metalanguage, freezes
    prediction classes, and only then may authoritative evidence filter the shadow
    universe. This moves one level above a fixed named operator alphabet while
    retaining an explicit boundary: the three-state encoder, commutativity,
    identity constraint, table cardinality, and interpreter remain human-authored.
    """

    def __init__(
        self,
        model_budget: int = 4096,
        state_count: int = 3,
        identity_state: int = 1,
        tolerance: float = 1e-9,
        min_active_channels: int = 2,
    ) -> None:
        if int(state_count) != 3:
            raise ValueError("current bounded composition-law metalanguage requires exactly three states")
        if int(identity_state) != 1:
            raise ValueError("current bounded composition-law metalanguage fixes middle state as identity")
        self.model_budget = max(1, int(model_budget))
        self.state_count = 3
        self.identity_state = 1
        self.tolerance = max(0.0, float(tolerance))
        self.min_active_channels = max(2, int(min_active_channels))
        self.last_table_count = 0
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

    def encode_state(self, value: float) -> int:
        value = float(value)
        if value < -self.tolerance:
            return 0
        if value > self.tolerance:
            return 2
        return 1

    def _tables(self) -> Tuple[Tuple[int, ...], ...]:
        # state 1 is identity, commutativity is fixed. The remaining independent
        # cells are (0,0), (0,2), and (2,2): 3^3 = 27 generated operation tables.
        tables: List[Tuple[int, ...]] = []
        for a00, a02, a22 in product(range(3), repeat=3):
            matrix = [[0 for _ in range(3)] for _ in range(3)]
            for state in range(3):
                matrix[1][state] = state
                matrix[state][1] = state
            matrix[0][0] = a00
            matrix[0][2] = matrix[2][0] = a02
            matrix[2][2] = a22
            tables.append(tuple(value for row in matrix for value in row))
        self.last_table_count = len(tables)
        return tuple(tables)

    @staticmethod
    def _active_subsets() -> Tuple[Tuple[int, ...], ...]:
        out = []
        for mask in range(1, 1 << 3):
            states = tuple(state for state in range(3) if mask & (1 << state))
            if len(states) == 3:
                continue
            out.append(states)
        return tuple(out)

    def shadow_laws(self) -> Tuple[GeneratedCompositionLaw, ...]:
        return tuple(
            GeneratedCompositionLaw(3, 1, table, active_states)
            for table in self._tables()
            for active_states in self._active_subsets()
        )

    def predict(
        self,
        cause: str,
        sign: str,
        left_channel: str,
        right_channel: str,
        law: GeneratedCompositionLaw,
        descriptor: InterventionDescriptor,
        raw_observations: Mapping[str, Mapping[str, float]],
    ) -> str:
        if cause not in descriptor.targets or cause in descriptor.blocked:
            return "NO_EFFECT"
        row = raw_observations.get(descriptor.intervention_id, {})
        if left_channel not in row or right_channel not in row:
            return "NO_EFFECT"
        left = self.encode_state(float(row[left_channel]))
        right = self.encode_state(float(row[right_channel]))
        output = law.combine(left, right)
        return self._effect(sign) if output in set(law.active_states) else "NO_EFFECT"

    @staticmethod
    def _model_id(
        cause: str,
        sign: str,
        left_channel: str,
        right_channel: str,
        law: GeneratedCompositionLaw,
    ) -> str:
        raw = f"{cause}|{sign}|{left_channel}|{right_channel}|{law.law_id}".encode()
        return "GENCOMPOSITIONLAW::" + hashlib.sha256(raw).hexdigest()[:16]

    def generate_novel(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
        raw_observations: Mapping[str, Mapping[str, float]],
        residual_evidence: Sequence[ModelEvidence],
        existing_models: Sequence[CausalWorldModel],
    ) -> List[GeneratedCompositionLawModel]:
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
        channel_pairs = tuple(combinations(channels, 2))
        if len(channels) < self.min_active_channels or not channel_pairs:
            self.last_raw_candidate_count = 0
            self.last_unique_signature_count = 0
            self.last_truncated = False
            return []

        existing_signatures = {self._signature(model) for model in existing_models}
        parents = tuple(sorted(
            model.model_id for model in existing_models
            if int(model.generation) == 7 and model.origin == "GENERATED_SYMBOLIC_PRIMITIVE"
        ))
        laws = self.shadow_laws()
        raw_candidates: List[GeneratedCompositionLawModel] = []

        for left_channel, right_channel in channel_pairs:
            complete = all(
                left_channel in raw_observations.get(descriptor.intervention_id, {})
                and right_channel in raw_observations.get(descriptor.intervention_id, {})
                for descriptor in descriptors
            )
            if not complete:
                continue
            for law in laws:
                for cause in variables:
                    for sign in ("POS", "NEG"):
                        predictions = tuple(
                            (
                                descriptor.intervention_id,
                                self.predict(
                                    cause,
                                    sign,
                                    left_channel,
                                    right_channel,
                                    law,
                                    descriptor,
                                    raw_observations,
                                ),
                            )
                            for descriptor in descriptors
                        )
                        model = CausalWorldModel(
                            model_id=self._model_id(cause, sign, left_channel, right_channel, law),
                            prior=1.0,
                            predictions=predictions,
                            origin="GENERATED_COMPOSITION_LAW",
                            family="RAW_GENERATED_FINITE_ALGEBRA",
                            structure=(
                                f"CAUSE({cause})",
                                f"LEFT_CHANNEL({left_channel})",
                                f"RIGHT_CHANNEL({right_channel})",
                                f"LAW({law.render()})",
                                f"SIGN({sign})",
                            ),
                            generation=8,
                            parent_model_ids=parents,
                        )
                        signature = self._signature(model)
                        if signature in existing_signatures:
                            continue
                        if not self._compatible(model, residual_evidence):
                            continue
                        raw_candidates.append(GeneratedCompositionLawModel(
                            cause,
                            sign,
                            left_channel,
                            right_channel,
                            law,
                            model,
                        ))

        self.last_raw_candidate_count = len(raw_candidates)
        by_signature: Dict[Tuple[Tuple[str, str], ...], List[GeneratedCompositionLawModel]] = {}
        for item in raw_candidates:
            by_signature.setdefault(self._signature(item.model), []).append(item)
        self.last_unique_signature_count = len(by_signature)
        self.last_truncated = self.last_unique_signature_count > self.model_budget

        out: List[GeneratedCompositionLawModel] = []
        for _, group in sorted(by_signature.items(), key=lambda item: item[0]):
            ordered = sorted(
                group,
                key=lambda item: (
                    item.law.law_id,
                    item.left_channel,
                    item.right_channel,
                    item.cause,
                    item.sign,
                ),
            )
            chosen = ordered[0]
            equivalents = tuple(item.law.law_id for item in ordered[1:])
            model = CausalWorldModel(
                model_id=chosen.model.model_id,
                prior=chosen.model.prior,
                predictions=chosen.model.predictions,
                origin=chosen.model.origin,
                family=chosen.model.family,
                structure=chosen.model.structure,
                generation=chosen.model.generation,
                parent_model_ids=chosen.model.parent_model_ids,
                equivalent_structures=equivalents,
            )
            out.append(GeneratedCompositionLawModel(
                chosen.cause,
                chosen.sign,
                chosen.left_channel,
                chosen.right_channel,
                chosen.law,
                model,
                equivalents,
            ))
            if len(out) >= self.model_budget:
                break
        return out
