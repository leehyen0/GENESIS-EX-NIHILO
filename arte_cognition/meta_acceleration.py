from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple
import hashlib
import json

from .executable_morphology import MorphologyGenome, MorphologyMutation, MorphologyMutator, MutationLevel
from .morphology_genesis import MorphologyCandidate, MorphologyEvaluation


@dataclass(frozen=True)
class MutationStrategyState:
    operation_scores: Tuple[Tuple[str, float], ...] = ()
    operation_support: Tuple[Tuple[str, int], ...] = ()
    fossilized_operations: Tuple[str, ...] = ()
    lineage_hash: str = ""

    def score_map(self) -> Dict[str, float]:
        return {key: float(value) for key, value in self.operation_scores}

    def support_map(self) -> Dict[str, int]:
        return {key: int(value) for key, value in self.operation_support}

    def score(self, operation_family: str) -> float:
        if operation_family in set(self.fossilized_operations):
            return float("-inf")
        return self.score_map().get(operation_family, 0.0)


@dataclass(frozen=True)
class MutationTemplate:
    operation: str
    level: MutationLevel
    payload: Mapping[str, object]
    rationale: Tuple[str, ...] = ()
    source_candidate_id: str = ""

    def fingerprint(self) -> str:
        raw = json.dumps(
            {
                "operation": self.operation,
                "level": int(self.level),
                "payload": dict(self.payload),
                "rationale": list(self.rationale),
                "source_candidate_id": self.source_candidate_id,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class MutationProgram:
    program_id: str
    templates: Tuple[MutationTemplate, ...]
    inherited_strategy_hash: str
    generation_uses_current_outcomes: bool = False

    @property
    def depth(self) -> int:
        return len(self.templates)


@dataclass(frozen=True)
class MutationProgramDevelopmentState:
    max_depth: int = 1
    complete_failure_receipts: Tuple[Tuple[str, str], ...] = ()
    lineage_hash: str = ""


@dataclass(frozen=True)
class GenerationMetrics:
    generation: int
    body_hash: str
    parent_body_hash: str
    external_capability_frontier: float
    transfer_score: float
    retained_competence: float
    calibration_score: float
    research_invention_score: float
    meta_improvement_ability: float
    compute_cost: float
    evidence_cost: float
    human_structural_intervention: float
    benchmark_disjoint: bool
    authority_verified: bool
    strategy_hash: str = ""

    @property
    def meta_productivity(self) -> float:
        denominator = max(
            1e-12,
            float(self.compute_cost)
            + float(self.evidence_cost)
            + float(self.human_structural_intervention),
        )
        return float(self.external_capability_frontier) / denominator


@dataclass(frozen=True)
class AccelerationAssessment:
    status: str
    generation_count: int
    frontier_trajectory: Tuple[float, ...]
    meta_productivity_trajectory: Tuple[float, ...]
    human_intervention_trajectory: Tuple[float, ...]
    strict_frontier_growth: bool
    strict_meta_productivity_growth: bool
    nonincreasing_human_intervention: bool
    retained_competence_viable: bool
    calibration_viable: bool
    meta_ability_improved: bool
    lineage_continuous: bool
    all_benchmark_disjoint: bool
    all_authority_verified: bool
    global_recursive_acceleration: bool = False


class MetaMutationLearner:
    """Learn a future mutation-search prior only from past external evidence.

    This does not authorize a BODY mutation. It only changes future search ordering.
    Current hidden outcomes are never an input to ranking.
    """

    def update(
        self,
        previous: MutationStrategyState,
        candidates: Sequence[MorphologyCandidate],
        evaluations: Sequence[MorphologyEvaluation],
    ) -> MutationStrategyState:
        by_candidate = {candidate.candidate_id: candidate for candidate in candidates}
        scores = previous.score_map()
        support = previous.support_map()
        fossils = set(previous.fossilized_operations)

        for row in evaluations:
            candidate = by_candidate.get(row.candidate_id)
            if candidate is None:
                continue
            if not (row.externally_generated and row.authority_verified and row.benchmark_disjoint):
                continue
            op = candidate.operation_family
            support[op] = support.get(op, 0) + 1
            safe = row.retained_competence_delta >= 0.0 and row.calibration_delta >= 0.0
            signal = (
                float(row.capability_delta)
                + float(row.meta_productivity_delta)
                + 0.25 * float(row.retained_competence_delta)
                + 0.25 * float(row.calibration_delta)
            )
            if safe:
                scores[op] = scores.get(op, 0.0) + signal
            else:
                scores[op] = scores.get(op, 0.0) - abs(signal) - 1.0
                if support[op] >= 2 and scores[op] < -1.0:
                    fossils.add(op)

        payload = {
            "parent": previous.lineage_hash,
            "scores": sorted((key, round(value, 12)) for key, value in scores.items()),
            "support": sorted(support.items()),
            "fossils": sorted(fossils),
        }
        lineage_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return MutationStrategyState(
            operation_scores=tuple(sorted(scores.items())),
            operation_support=tuple(sorted(support.items())),
            fossilized_operations=tuple(sorted(fossils)),
            lineage_hash=lineage_hash,
        )

    @staticmethod
    def rank(
        state: MutationStrategyState,
        candidates: Sequence[MorphologyCandidate],
    ) -> Tuple[MorphologyCandidate, ...]:
        fossils = set(state.fossilized_operations)
        return tuple(
            sorted(
                (candidate for candidate in candidates if candidate.operation_family not in fossils),
                key=lambda candidate: (
                    -state.score(candidate.operation_family),
                    candidate.operation_family,
                    candidate.candidate_id,
                ),
            )
        )


def _template(candidate: MorphologyCandidate) -> MutationTemplate:
    mutation = candidate.mutation
    return MutationTemplate(
        operation=mutation.operation,
        level=mutation.level,
        payload=dict(mutation.payload),
        rationale=tuple(mutation.rationale),
        source_candidate_id=candidate.candidate_id,
    )


def generate_mutation_programs(
    candidates: Sequence[MorphologyCandidate],
    strategy: MutationStrategyState,
    max_depth: int = 2,
    budget: int = 128,
    beam_width: int = 8,
) -> Tuple[MutationProgram, ...]:
    """Compose previously generated structural mutations before current outcomes.

    Program composition is a bootstrap meta-language; it is intentionally recorded
    as a remaining authored boundary. Programs inherit a learned prior from earlier
    externally verified generations, never from current candidate consequences.
    """
    ranked = MetaMutationLearner.rank(strategy, candidates)
    templates = tuple(_template(candidate) for candidate in ranked[: max(1, int(beam_width))])
    programs: Dict[Tuple[str, ...], MutationProgram] = {}

    def add(items: Tuple[MutationTemplate, ...]) -> None:
        key = tuple(item.fingerprint() for item in items)
        raw = "|".join(key) + "|" + strategy.lineage_hash
        program_id = "MUTATION_PROGRAM::" + hashlib.sha256(raw.encode()).hexdigest()[:20]
        programs.setdefault(
            key,
            MutationProgram(
                program_id=program_id,
                templates=items,
                inherited_strategy_hash=strategy.lineage_hash,
                generation_uses_current_outcomes=False,
            ),
        )

    depth_limit = max(1, int(max_depth))

    def extend(prefix: Tuple[MutationTemplate, ...], remaining: Tuple[MutationTemplate, ...]) -> None:
        if prefix:
            add(prefix)
        if len(prefix) >= depth_limit:
            return
        for index, item in enumerate(remaining):
            extend(prefix + (item,), remaining[:index] + remaining[index + 1 :])

    extend((), templates)

    def program_score(program: MutationProgram) -> float:
        return sum(strategy.score(item.operation) for item in program.templates) - 0.05 * max(0, program.depth - 1)

    return tuple(
        sorted(programs.values(), key=lambda p: (-program_score(p), p.depth, p.program_id))[: max(1, int(budget))]
    )


def apply_mutation_program(genome: MorphologyGenome, program: MutationProgram) -> MorphologyGenome:
    """Apply a program atomically with each step rebound to the current descendant hash."""
    current = genome
    mutator = MorphologyMutator()
    for index, template in enumerate(program.templates):
        mutation = MorphologyMutation(
            mutation_id=f"{program.program_id}::STEP::{index}",
            level=template.level,
            operation=template.operation,
            payload=dict(template.payload),
            parent_body_hash=current.fingerprint(),
            rationale=tuple(template.rationale) + (f"program::{program.program_id}",),
            reversible=True,
        )
        current = mutator.apply(current, mutation)
    return current


def observe_complete_program_failure(
    state: MutationProgramDevelopmentState,
    *,
    context_id: str,
    source_class: str,
    candidate_universe_complete: bool,
    any_success: bool,
    authority_verified: bool,
    benchmark_disjoint: bool,
    max_depth_cap: int = 8,
) -> MutationProgramDevelopmentState:
    """Open a deeper mutation-program language only after repeated complete failure.

    The depth change is a search-policy development event, not action authority. It
    requires two distinct contexts and two external evidence classes and therefore
    cannot be opened by a single lucky/failed run or by verifierless feedback.
    """
    if (
        not candidate_universe_complete
        or any_success
        or not authority_verified
        or not benchmark_disjoint
        or not source_class
        or source_class == "UNVERIFIED"
    ):
        return state

    receipts = set(state.complete_failure_receipts)
    receipts.add((str(context_id), str(source_class)))
    contexts = {context for context, _ in receipts}
    classes = {source for _, source in receipts}
    open_deeper = len(contexts) >= 2 and len(classes) >= 2 and state.max_depth < max(1, int(max_depth_cap))
    next_depth = state.max_depth + 1 if open_deeper else state.max_depth
    retained_receipts = () if open_deeper else tuple(sorted(receipts))
    payload = {
        "parent": state.lineage_hash,
        "old_depth": state.max_depth,
        "new_depth": next_depth,
        "receipts": sorted(receipts),
        "opened": open_deeper,
    }
    lineage_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return MutationProgramDevelopmentState(
        max_depth=next_depth,
        complete_failure_receipts=retained_receipts,
        lineage_hash=lineage_hash,
    )


@dataclass
class MetaAccelerationLedger:
    generations: Dict[int, GenerationMetrics] = field(default_factory=dict)

    def append(self, metrics: GenerationMetrics) -> bool:
        if metrics.generation in self.generations:
            return False
        if self.generations:
            latest_generation = max(self.generations)
            latest = self.generations[latest_generation]
            if metrics.generation != latest_generation + 1:
                return False
            if metrics.parent_body_hash != latest.body_hash:
                return False
        self.generations[metrics.generation] = metrics
        return True

    def ordered(self) -> Tuple[GenerationMetrics, ...]:
        return tuple(self.generations[key] for key in sorted(self.generations))

    def assess(
        self,
        min_transitions: int = 3,
        retained_floor: float = 0.95,
        calibration_floor: float = 0.8,
    ) -> AccelerationAssessment:
        rows = self.ordered()
        frontier = tuple(row.external_capability_frontier for row in rows)
        productivity = tuple(row.meta_productivity for row in rows)
        humans = tuple(row.human_structural_intervention for row in rows)
        enough = len(rows) >= max(2, int(min_transitions) + 1)
        strict_frontier = enough and all(b > a for a, b in zip(frontier, frontier[1:]))
        strict_productivity = enough and all(b > a for a, b in zip(productivity, productivity[1:]))
        human_nonincrease = enough and all(b <= a for a, b in zip(humans, humans[1:]))
        retained = enough and all(row.retained_competence >= retained_floor for row in rows)
        calibration = enough and all(row.calibration_score >= calibration_floor for row in rows)
        meta_ability = enough and all(
            b.meta_improvement_ability >= a.meta_improvement_ability for a, b in zip(rows, rows[1:])
        ) and any(
            b.meta_improvement_ability > a.meta_improvement_ability for a, b in zip(rows, rows[1:])
        )
        lineage = enough and all(
            child.parent_body_hash == parent.body_hash for parent, child in zip(rows, rows[1:])
        )
        disjoint = enough and all(row.benchmark_disjoint for row in rows[1:])
        authority = enough and all(row.authority_verified for row in rows[1:])
        passed = all(
            (
                enough,
                strict_frontier,
                strict_productivity,
                human_nonincrease,
                retained,
                calibration,
                meta_ability,
                lineage,
                disjoint,
                authority,
            )
        )
        status = (
            "PASS_BOUNDED_PROSPECTIVE_META_ACCELERATION_CANDIDATE"
            if passed
            else "INSUFFICIENT_PROSPECTIVE_META_ACCELERATION_EVIDENCE"
        )
        return AccelerationAssessment(
            status=status,
            generation_count=len(rows),
            frontier_trajectory=frontier,
            meta_productivity_trajectory=productivity,
            human_intervention_trajectory=humans,
            strict_frontier_growth=bool(strict_frontier),
            strict_meta_productivity_growth=bool(strict_productivity),
            nonincreasing_human_intervention=bool(human_nonincrease),
            retained_competence_viable=bool(retained),
            calibration_viable=bool(calibration),
            meta_ability_improved=bool(meta_ability),
            lineage_continuous=bool(lineage),
            all_benchmark_disjoint=bool(disjoint),
            all_authority_verified=bool(authority),
            global_recursive_acceleration=False,
        )


def choose_meta_improvement_target(
    human_dependency: float,
    candidate_search_cost: float,
    evaluator_uncertainty: float,
    transfer_failure: float,
) -> str:
    pressures = {
        "MUTATE_MUTATOR": float(human_dependency),
        "MUTATE_SEARCH_POLICY": float(candidate_search_cost),
        "MUTATE_EVALUATOR": float(evaluator_uncertainty),
        "MUTATE_REPRESENTATION_OR_TOPOLOGY": float(transfer_failure),
    }
    return sorted(pressures.items(), key=lambda item: (-item[1], item[0]))[0][0]
