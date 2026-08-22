from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import hashlib
import itertools

from .causal_model_genesis import InterventionDescriptor
from .world_model_ecology import CausalWorldModel, ModelEvidence


@dataclass(frozen=True)
class CausalPrimitive:
    op: str
    arg: str = ""


@dataclass(frozen=True)
class CausalProgram:
    cause: str
    sign: str
    primitives: Tuple[CausalPrimitive, ...]

    @property
    def signature(self) -> Tuple[str, ...]:
        return (self.cause, self.sign) + tuple(f"{p.op}:{p.arg}" for p in self.primitives)


@dataclass(frozen=True)
class GeneratedCausalProgram:
    program: CausalProgram
    model: CausalWorldModel
    equivalent_programs: Tuple[str, ...] = ()


class CompositionalCausalProgramGenesisEngine:
    """Search a compositional causal grammar rather than a fixed family catalog.

    Human-authored primitives remain bounded, but candidate *structures* are built
    by composing them. This allows model-class expansion to create hypotheses such
    as cause->mediator with an additional temporal gate without adding a new named
    family for every combination.
    """

    def __init__(self, model_budget: int = 96, max_extra_primitives: int = 2) -> None:
        self.model_budget = max(1, int(model_budget))
        self.max_extra_primitives = max(0, int(max_extra_primitives))

    @staticmethod
    def _id(program: CausalProgram) -> str:
        raw = "|".join(program.signature).encode()
        return "GENPROGRAM::" + hashlib.sha256(raw).hexdigest()[:14]

    @staticmethod
    def _effect(sign: str) -> str:
        return "POSITIVE_EFFECT" if sign == "POS" else "NEGATIVE_EFFECT"

    @staticmethod
    def predict(program: CausalProgram, d: InterventionDescriptor) -> str:
        active = program.cause in d.targets and program.cause not in d.blocked
        for primitive in program.primitives:
            if primitive.op == "VIA":
                mediator = primitive.arg
                if mediator in d.blocked:
                    active = False
                elif mediator in d.targets:
                    active = True
            elif primitive.op == "REQUIRE":
                required = primitive.arg
                active = active and required in d.targets and required not in d.blocked
            elif primitive.op == "LAG":
                active = active and int(d.delay_steps) >= int(primitive.arg)
            elif primitive.op == "CONTEXT_GATE":
                active = active and bool(d.context_shift)
        return CompositionalCausalProgramGenesisEngine._effect(program.sign) if active else "NO_EFFECT"

    @staticmethod
    def structure(program: CausalProgram) -> Tuple[str, ...]:
        clauses: List[str] = [f"CAUSE({program.cause})"]
        clauses.extend(f"{p.op}({p.arg})" if p.arg else p.op for p in program.primitives)
        clauses.append(f"SIGN({program.sign})")
        return tuple(clauses)

    def _programs(self, variables: Sequence[str]) -> List[CausalProgram]:
        variables = tuple(sorted({str(v) for v in variables if str(v)}))
        out: Dict[Tuple[str, ...], CausalProgram] = {}
        for cause in variables:
            others = [v for v in variables if v != cause]
            primitive_options: List[CausalPrimitive] = [
                CausalPrimitive("LAG", "1"),
                CausalPrimitive("CONTEXT_GATE", ""),
            ]
            for other in others:
                primitive_options.append(CausalPrimitive("VIA", other))
                primitive_options.append(CausalPrimitive("REQUIRE", other))

            for sign in ("POS", "NEG"):
                base = CausalProgram(cause, sign, ())
                out[base.signature] = base
                for width in range(1, self.max_extra_primitives + 1):
                    for combo in itertools.combinations(primitive_options, width):
                        ops = [p.op for p in combo]
                        # Avoid internally contradictory duplicate path operators.
                        if ops.count("LAG") > 1 or ops.count("CONTEXT_GATE") > 1:
                            continue
                        via_args = {p.arg for p in combo if p.op == "VIA"}
                        req_args = {p.arg for p in combo if p.op == "REQUIRE"}
                        if via_args & req_args:
                            continue
                        program = CausalProgram(
                            cause=cause,
                            sign=sign,
                            primitives=tuple(sorted(combo, key=lambda p: (p.op, p.arg))),
                        )
                        out[program.signature] = program
        return list(out.values())

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
    def _prediction_signature(model: CausalWorldModel) -> Tuple[Tuple[str, str], ...]:
        return tuple(sorted(model.predictions))

    def generate_novel(
        self,
        variables: Sequence[str],
        descriptors: Sequence[InterventionDescriptor],
        residual_evidence: Sequence[ModelEvidence],
        existing_models: Sequence[CausalWorldModel],
    ) -> List[GeneratedCausalProgram]:
        existing_signatures = {self._prediction_signature(model) for model in existing_models}
        raw: List[Tuple[CausalProgram, CausalWorldModel]] = []
        for program in self._programs(variables):
            predictions = tuple(
                (d.intervention_id, self.predict(program, d))
                for d in descriptors
            )
            model = CausalWorldModel(
                model_id=self._id(program),
                prior=1.0,
                predictions=predictions,
                origin="GENERATED_COMPOSITIONAL",
                family="COMPOSITIONAL_PROGRAM",
                structure=self.structure(program),
                generation=2,
                parent_model_ids=tuple(sorted(model.model_id for model in existing_models if model.origin == "GENERATED")),
            )
            if self._prediction_signature(model) in existing_signatures:
                continue
            if not self._compatible(model, residual_evidence):
                continue
            raw.append((program, model))

        by_signature: Dict[Tuple[Tuple[str, str], ...], List[Tuple[CausalProgram, CausalWorldModel]]] = {}
        for item in raw:
            by_signature.setdefault(self._prediction_signature(item[1]), []).append(item)

        out: List[GeneratedCausalProgram] = []
        for signature, group in sorted(by_signature.items(), key=lambda item: item[0]):
            ordered = sorted(group, key=lambda item: (len(item[0].primitives), item[0].signature))
            program, model = ordered[0]
            equivalents = tuple("|".join(item[0].signature) for item in ordered[1:])
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
            out.append(GeneratedCausalProgram(program, model, equivalents))
            if len(out) >= self.model_budget:
                break
        return out
