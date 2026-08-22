from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple
import hashlib
import itertools
import math

from .causal_model_genesis import InterventionDescriptor
from .world_model_ecology import CausalWorldModel, ModelEvidence


@dataclass(frozen=True)
class SymbolicExpression:
    op: str
    args: Tuple["SymbolicExpression", ...] = ()
    channel: str = ""

    def render(self) -> str:
        if self.op == "RAW":
            return f"RAW[{self.channel}]"
        if self.op == "ABS":
            return f"abs({self.args[0].render()})"
        if self.op == "NEG":
            return f"-({self.args[0].render()})"
        left, right = self.args
        token = {"ADD": "+", "SUB": "-", "MUL": "*"}[self.op]
        return f"({left.render()} {token} {right.render()})"

    def channels(self) -> Tuple[str, ...]:
        if self.op == "RAW":
            return (self.channel,)
        return tuple(sorted({ch for arg in self.args for ch in arg.channels()}))

    def complexity(self) -> int:
        if self.op == "RAW":
            return 1
        return 1 + sum(arg.complexity() for arg in self.args)


@dataclass(frozen=True)
class GeneratedSymbolicPrimitive:
    expression: SymbolicExpression
    threshold: float
    direction: str

    def render(self) -> str:
        return f"{self.expression.render()} {self.direction} {self.threshold:.12g}"


@dataclass(frozen=True)
class GeneratedSymbolicPrimitiveModel:
    cause: str
    sign: str
    primitive: GeneratedSymbolicPrimitive
    model: CausalWorldModel
    equivalent_primitives: Tuple[str, ...] = ()


class SymbolicPrimitiveGenesisEngine:
    """Search generic symbolic relations over raw observations.

    The evaluator does not select a named next family. A small authored operation
    alphabet is recursively composed into expression trees, prediction-equivalent
    expressions are quotiented, and threshold atoms are generated over expression
    values before outcomes are consulted. Authoritative evidence can only filter
    that pre-existing shadow universe.

    Search depth and the operation alphabet are bounded. This therefore proves
    generic compositional symbolic primitive search, not unrestricted invention of
    mathematical operators or arbitrary executable code.
    """

    COMMUTATIVE = {"ADD", "MUL"}

    def __init__(
        self,
        model_budget: int = 16384,
        expression_budget: int = 2048,
        max_depth: int = 2,
        operators: Sequence[str] = ("ADD", "SUB", "MUL", "ABS"),
        min_active_channels: int = 2,
    ) -> None:
        self.model_budget = max(1, int(model_budget))
        self.expression_budget = max(1, int(expression_budget))
        self.max_depth = max(1, int(max_depth))
        self.operators = tuple(op for op in operators if op in {"ADD", "SUB", "MUL", "ABS", "NEG"})
        self.min_active_channels = max(1, int(min_active_channels))
        self.last_expression_count = 0
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
        unique = sorted({float(value) for value in values if math.isfinite(float(value))})
        return tuple((left + right) / 2.0 for left, right in zip(unique, unique[1:]))

    @classmethod
    def _canonical_binary(cls, op: str, left: SymbolicExpression, right: SymbolicExpression) -> SymbolicExpression:
        if op in cls.COMMUTATIVE and right.render() < left.render():
            left, right = right, left
        return SymbolicExpression(op, (left, right))

    @staticmethod
    def _eval(expr: SymbolicExpression, row: Mapping[str, float]) -> float | None:
        if expr.op == "RAW":
            value = row.get(expr.channel)
            return None if value is None else float(value)
        if expr.op in {"ABS", "NEG"}:
            value = SymbolicPrimitiveGenesisEngine._eval(expr.args[0], row)
            if value is None:
                return None
            return abs(value) if expr.op == "ABS" else -value
        left = SymbolicPrimitiveGenesisEngine._eval(expr.args[0], row)
        right = SymbolicPrimitiveGenesisEngine._eval(expr.args[1], row)
        if left is None or right is None:
            return None
        if expr.op == "ADD":
            return left + right
        if expr.op == "SUB":
            return left - right
        if expr.op == "MUL":
            return left * right
        return None

    def _expressions(self, channels: Sequence[str]) -> Tuple[SymbolicExpression, ...]:
        base = tuple(SymbolicExpression("RAW", channel=channel) for channel in sorted(set(channels)))
        seen: Dict[str, SymbolicExpression] = {expr.render(): expr for expr in base}
        frontier = list(base)
        all_expr = list(base)
        for _depth in range(1, self.max_depth + 1):
            created: Dict[str, SymbolicExpression] = {}
            pool = tuple(all_expr)
            for op in self.operators:
                if op in {"ABS", "NEG"}:
                    for child in frontier:
                        expr = SymbolicExpression(op, (child,))
                        created.setdefault(expr.render(), expr)
                else:
                    for left, right in itertools.product(pool, repeat=2):
                        if op == "SUB" and left.render() == right.render():
                            continue
                        expr = self._canonical_binary(op, left, right)
                        created.setdefault(expr.render(), expr)
            next_frontier = []
            for key in sorted(created):
                if key in seen:
                    continue
                expr = created[key]
                seen[key] = expr
                all_expr.append(expr)
                next_frontier.append(expr)
                if len(all_expr) >= self.expression_budget:
                    return tuple(all_expr)
            frontier = next_frontier
            if not frontier:
                break
        return tuple(all_expr)

    @classmethod
    def predict(
        cls,
        cause: str,
        sign: str,
        primitive: GeneratedSymbolicPrimitive,
        descriptor: InterventionDescriptor,
        raw_observations: Mapping[str, Mapping[str, float]],
    ) -> str:
        if cause not in descriptor.targets or cause in descriptor.blocked:
            return "NO_EFFECT"
        value = cls._eval(primitive.expression, raw_observations.get(descriptor.intervention_id, {}))
        if value is None or not math.isfinite(value):
            return "NO_EFFECT"
        active = value >= primitive.threshold if primitive.direction == ">=" else value < primitive.threshold
        return cls._effect(sign) if active else "NO_EFFECT"

    @staticmethod
    def _model_id(cause: str, sign: str, primitive: GeneratedSymbolicPrimitive) -> str:
        raw = f"{cause}|{sign}|{primitive.expression.render()}|{primitive.direction}|{primitive.threshold.hex()}".encode()
        return "GENSYMBOLICPRIMITIVE::" + hashlib.sha256(raw).hexdigest()[:16]

    def generate_novel(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
        raw_observations: Mapping[str, Mapping[str, float]],
        residual_evidence: Sequence[ModelEvidence],
        existing_models: Sequence[CausalWorldModel],
    ) -> List[GeneratedSymbolicPrimitiveModel]:
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
        expressions = tuple(
            expr for expr in self._expressions(channels)
            if len(expr.channels()) >= self.min_active_channels
        )
        self.last_expression_count = len(expressions)
        existing_signatures = {self._signature(model) for model in existing_models}
        parents = tuple(sorted(
            model.model_id for model in existing_models
            if int(model.generation) == 6 and model.origin == "GENERATED_LINEAR_PRIMITIVE"
        ))

        raw_candidates: List[Tuple[str, str, GeneratedSymbolicPrimitive, CausalWorldModel]] = []
        for expression in expressions:
            values = []
            complete = True
            for descriptor in descriptors:
                value = self._eval(expression, raw_observations.get(descriptor.intervention_id, {}))
                if value is None or not math.isfinite(value):
                    complete = False
                    break
                values.append(value)
            if not complete or len(set(values)) < 3:
                continue
            for threshold in self._thresholds(values):
                for direction in (">=", "<"):
                    primitive = GeneratedSymbolicPrimitive(expression, float(threshold), direction)
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
                                origin="GENERATED_SYMBOLIC_PRIMITIVE",
                                family="RAW_SYMBOLIC_EXPRESSION_THRESHOLD",
                                structure=(
                                    f"CAUSE({cause})",
                                    f"PRIMITIVE({primitive.render()})",
                                    f"SIGN({sign})",
                                ),
                                generation=7,
                                parent_model_ids=parents,
                            )
                            signature = self._signature(model)
                            if signature in existing_signatures:
                                continue
                            if not self._compatible(model, residual_evidence):
                                continue
                            raw_candidates.append((cause, sign, primitive, model))

        self.last_raw_candidate_count = len(raw_candidates)
        by_signature: Dict[
            Tuple[Tuple[str, str], ...],
            List[Tuple[str, str, GeneratedSymbolicPrimitive, CausalWorldModel]],
        ] = {}
        for item in raw_candidates:
            by_signature.setdefault(self._signature(item[3]), []).append(item)

        self.last_unique_signature_count = len(by_signature)
        self.last_truncated = (
            self.last_unique_signature_count > self.model_budget
            or len(self._expressions(channels)) >= self.expression_budget
        )

        out: List[GeneratedSymbolicPrimitiveModel] = []
        for _, group in sorted(by_signature.items(), key=lambda item: item[0]):
            ordered = sorted(
                group,
                key=lambda item: (
                    item[2].expression.complexity(),
                    item[2].expression.render(),
                    item[2].threshold,
                    item[2].direction,
                    item[0],
                    item[1],
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
            out.append(GeneratedSymbolicPrimitiveModel(cause, sign, primitive, model, equivalents))
            if len(out) >= self.model_budget:
                break
        return out
