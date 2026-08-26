from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple
import hashlib
import json
import re

from .executable_morphology import (
    ExperienceUnit,
    MorphologyGenome,
    MorphologyMutation,
    MorphologyMutator,
    MutationLevel,
    OrganKind,
)
from .meta_acceleration import MutationStrategyState
from .morphology_genesis import MorphologyCandidate, MorphologyGenesisEngine, MorphologyResidual


_HASH64 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _valid_hash(value: str) -> bool:
    return bool(_HASH64.fullmatch(str(value)))


@dataclass(frozen=True)
class NativeResearchProblem:
    problem_id: str
    discovered_by_body_hash: str
    pressure_kind: str
    target_surface: str
    statement: str
    falsifier: str
    evidence_refs: Tuple[str, ...]
    detector_id: str
    human_seeded: bool = False
    parent_problem_sha256: str = ""

    def validate(self) -> Tuple[str, ...]:
        errors = []
        if not self.problem_id:
            errors.append("problem_id_missing")
        if not _valid_hash(self.discovered_by_body_hash):
            errors.append("discovered_by_body_hash_invalid")
        if not self.pressure_kind:
            errors.append("pressure_kind_missing")
        if not self.target_surface:
            errors.append("target_surface_missing")
        if not self.statement:
            errors.append("statement_missing")
        if not self.falsifier:
            errors.append("falsifier_missing")
        if not self.evidence_refs:
            errors.append("evidence_refs_missing")
        if not self.detector_id:
            errors.append("detector_id_missing")
        if self.parent_problem_sha256 and not _valid_hash(self.parent_problem_sha256):
            errors.append("parent_problem_sha256_invalid")
        return tuple(sorted(set(errors)))

    def fingerprint(self) -> str:
        return _sha256(
            {
                "problem_id": self.problem_id,
                "discovered_by_body_hash": self.discovered_by_body_hash,
                "pressure_kind": self.pressure_kind,
                "target_surface": self.target_surface,
                "statement": self.statement,
                "falsifier": self.falsifier,
                "evidence_refs": list(self.evidence_refs),
                "detector_id": self.detector_id,
                "human_seeded": self.human_seeded,
                "parent_problem_sha256": self.parent_problem_sha256,
            }
        )


@dataclass(frozen=True)
class NativeResearchEvaluation:
    evaluation_id: str
    problem_sha256: str
    operation_family: str
    context_id: str
    evidence_class: str
    solved: bool
    precommitted: bool
    evaluator_reverified: bool
    removal_effect: float
    wrong_swap_effect: float
    retained_competence_delta: float
    calibration_delta: float
    problem_discovery_delta: float
    research_invention_delta: float
    meta_improvement_delta: float
    compute_cost: float
    evidence_cost: float
    human_structural_intervention: float
    outcome_receipt_sha256: str
    official_benchmark_used: bool = False

    @property
    def controls_pass(self) -> bool:
        return bool(float(self.removal_effect) > 0.0 and float(self.wrong_swap_effect) > 0.0)

    @property
    def safe(self) -> bool:
        return bool(float(self.retained_competence_delta) >= 0.0 and float(self.calibration_delta) >= 0.0)

    @property
    def admissible_native_credit(self) -> bool:
        return bool(
            self.solved
            and self.precommitted
            and self.evaluator_reverified
            and self.controls_pass
            and self.safe
            and not self.official_benchmark_used
            and _valid_hash(self.problem_sha256)
            and _valid_hash(self.outcome_receipt_sha256)
            and self.evidence_class not in {"", "UNVERIFIED"}
        )

    @property
    def research_productivity(self) -> float:
        numerator = (
            max(0.0, float(self.problem_discovery_delta))
            + max(0.0, float(self.research_invention_delta))
            + max(0.0, float(self.meta_improvement_delta))
        )
        denominator = max(
            1e-12,
            float(self.compute_cost)
            + float(self.evidence_cost)
            + float(self.human_structural_intervention),
        )
        return numerator / denominator


class NativeResearchLearner:
    """Credit self-hosted research without pretending it is external benchmark authority."""

    def update(
        self,
        previous: MutationStrategyState,
        evaluations: Sequence[NativeResearchEvaluation],
    ) -> MutationStrategyState:
        scores = previous.score_map()
        support = previous.support_map()
        fossils = set(previous.fossilized_operations)
        accepted = []

        for row in evaluations:
            if not (row.precommitted and row.evaluator_reverified):
                continue
            if not _valid_hash(row.problem_sha256) or not _valid_hash(row.outcome_receipt_sha256):
                continue
            if not row.operation_family or not row.evidence_class or row.evidence_class == "UNVERIFIED":
                continue

            op = row.operation_family
            support[op] = support.get(op, 0) + 1
            if row.admissible_native_credit:
                signal = row.research_productivity
                scores[op] = scores.get(op, 0.0) + signal
                accepted.append((row.evaluation_id, op, round(signal, 12), "ACCEPT"))
            else:
                penalty = (
                    1.0
                    + max(0.0, -float(row.retained_competence_delta))
                    + max(0.0, -float(row.calibration_delta))
                    + (0.5 if not row.controls_pass else 0.0)
                    + (0.5 if not row.solved else 0.0)
                )
                scores[op] = scores.get(op, 0.0) - penalty
                accepted.append((row.evaluation_id, op, round(-penalty, 12), "FOSSIL_EVIDENCE"))
                if support[op] >= 2 and scores[op] < -1.0:
                    fossils.add(op)

        payload = {
            "mode": "NATIVE_RECURSIVE_RESEARCH",
            "parent": previous.lineage_hash,
            "accepted": accepted,
            "scores": sorted((key, round(value, 12)) for key, value in scores.items()),
            "support": sorted(support.items()),
            "fossils": sorted(fossils),
        }
        lineage_hash = _sha256(payload)
        return MutationStrategyState(
            operation_scores=tuple(sorted(scores.items())),
            operation_support=tuple(sorted(support.items())),
            fossilized_operations=tuple(sorted(fossils)),
            lineage_hash=lineage_hash,
        )


class NativeMetaMorphologyGenesisEngine:
    """Extend the inherited morphology search with L3 shadow candidates under meta pressure.

    These candidates are not capability authority. They make generator/mutator policy
    replacement reachable and typed; a separate runtime-semantic test must prove that
    a generated implementation_ref changes executable behavior before capability credit.
    """

    def __init__(self, candidate_budget: int = 256) -> None:
        self.candidate_budget = max(1, int(candidate_budget))
        self.base = MorphologyGenesisEngine(candidate_budget=self.candidate_budget)

    @staticmethod
    def _l3_candidate(
        genome: MorphologyGenome,
        residual: MorphologyResidual,
        organ_id: str,
        kind: OrganKind,
    ) -> MorphologyCandidate:
        by_id = genome.organ_map()
        organ = by_id[organ_id]
        parent_hash = genome.fingerprint()
        suffix = _sha256(
            {
                "parent": parent_hash,
                "residual": residual.residual_id,
                "organ": organ_id,
                "kind": kind.value,
                "pressure": residual.pressure.normalized().__dict__,
            }
        )[:16]
        policy_ref = f"native-meta://{kind.value.lower()}/{residual.residual_id}/{suffix}"
        replacement = {
            "organ_id": organ.organ_id,
            "kind": organ.kind.value,
            "consumes": list(organ.consumes),
            "produces": list(organ.produces),
            "implementation_ref": policy_ref,
            "version": organ.version + 1,
            "cost_hint": organ.cost_hint,
            "provenance": list(organ.provenance) + [f"native-l3-pressure::{residual.residual_id}"],
            "enabled": organ.enabled,
        }
        operation_family = "CHANGE_GENERATOR_POLICY" if kind == OrganKind.GENERATOR else "CHANGE_MUTATOR_POLICY"
        mutation = MorphologyMutation(
            mutation_id="NATIVE_L3_MUTATION::" + suffix,
            level=MutationLevel.GENERATOR_MUTATOR,
            operation="REPLACE_ORGAN",
            payload={"organ": replacement},
            parent_body_hash=parent_hash,
            rationale=(
                f"pressure::{residual.residual_id}",
                f"human_dependency::{residual.pressure.human_dependency:.6f}",
                f"theory_blindspot::{residual.pressure.theory_blindspot:.6f}",
                "shadow_only_until_runtime_semantics_verified",
            ),
            reversible=True,
        )
        descendant = MorphologyMutator().apply(genome, mutation)
        candidate_id = "NATIVE_L3_CANDIDATE::" + _sha256(
            {
                "mutation": mutation.mutation_id,
                "descendant": descendant.fingerprint(),
                "operation_family": operation_family,
            }
        )[:20]
        return MorphologyCandidate(
            candidate_id=candidate_id,
            mutation=mutation,
            descendant_fingerprint=descendant.fingerprint(),
            origin_residual_ids=(residual.residual_id,),
            operation_family=operation_family,
            generation_uses_outcomes=False,
        )

    def generate(
        self,
        genome: MorphologyGenome,
        residuals: Sequence[MorphologyResidual],
    ) -> Tuple[MorphologyCandidate, ...]:
        candidates = list(self.base.generate(genome, residuals))
        for residual in residuals:
            pressure = residual.pressure.normalized()
            if pressure.human_dependency <= 0.0 and pressure.theory_blindspot <= 0.0:
                continue
            for organ in genome.organs:
                if organ.kind in {OrganKind.GENERATOR, OrganKind.MUTATOR}:
                    candidates.append(self._l3_candidate(genome, residual, organ.organ_id, organ.kind))

        unique: Dict[str, MorphologyCandidate] = {}
        for candidate in sorted(candidates, key=lambda row: row.candidate_id):
            unique.setdefault(candidate.descendant_fingerprint, candidate)
        return tuple(sorted(unique.values(), key=lambda row: row.candidate_id)[: self.candidate_budget])


def choose_native_meta_target(
    *,
    human_dependency: float,
    candidate_search_cost: float,
    evaluator_uncertainty: float,
    transfer_failure: float,
    strategy: Optional[MutationStrategyState] = None,
) -> str:
    pressures = {
        "MUTATE_MUTATOR": float(human_dependency),
        "MUTATE_SEARCH_POLICY": float(candidate_search_cost),
        "MUTATE_EVALUATOR": float(evaluator_uncertainty),
        "MUTATE_REPRESENTATION_OR_TOPOLOGY": float(transfer_failure),
    }
    if strategy is not None:
        for key in tuple(pressures):
            learned = strategy.score(key)
            if learned != float("-inf"):
                pressures[key] += 0.25 * max(-2.0, min(2.0, learned))
            else:
                pressures[key] = float("-inf")
    return sorted(pressures.items(), key=lambda item: (-item[1], item[0]))[0][0]


def native_research_experience(
    problem: NativeResearchProblem,
    evaluation: NativeResearchEvaluation,
    *,
    pre_body_hash: str,
    descendant_body_hash: str,
) -> ExperienceUnit:
    if evaluation.problem_sha256 != problem.fingerprint():
        raise ValueError("native research problem/evaluation mismatch")
    action_trace_hash = _sha256(
        {
            "evaluation_id": evaluation.evaluation_id,
            "problem_sha256": evaluation.problem_sha256,
            "operation_family": evaluation.operation_family,
            "context_id": evaluation.context_id,
            "outcome_receipt_sha256": evaluation.outcome_receipt_sha256,
        }
    )
    return ExperienceUnit(
        episode_id="NATIVE_RESEARCH::" + evaluation.evaluation_id,
        pre_body_hash=str(pre_body_hash),
        source_refs=tuple(problem.evidence_refs) + ("receipt://" + evaluation.outcome_receipt_sha256,),
        task_ref=problem.problem_id,
        benchmark_family="NATIVE_SELF_RESEARCH",
        precommitted_hypotheses=(problem.statement, problem.falsifier),
        selected_goal_id=problem.target_surface,
        action_trace_hash=action_trace_hash,
        outcome_summary=(
            f"solved={evaluation.solved};native_credit={evaluation.admissible_native_credit};"
            f"research_productivity={evaluation.research_productivity:.12f};official_benchmark_used={evaluation.official_benchmark_used}"
        ),
        success=evaluation.admissible_native_credit,
        uncertainty_before=1.0,
        uncertainty_after=0.0 if evaluation.solved else 1.0,
        mutation_ids=(evaluation.operation_family,),
        removal_effect=float(evaluation.removal_effect),
        wrong_swap_effect=float(evaluation.wrong_swap_effect),
        heldout_effect=None,
        delayed_replay_equal=None,
        descendant_body_hash=str(descendant_body_hash),
        notes=(
            "NATIVE_PROBLEM_DISCOVERY" if not problem.human_seeded else "HUMAN_SEEDED_PROBLEM",
            "problem_sha256=" + problem.fingerprint(),
            "detector_id=" + problem.detector_id,
            "external_claim_authority=false",
        ),
    )


def commit_native_research_to_body(body: object, problem: NativeResearchProblem, evaluation: NativeResearchEvaluation, descendant_body_hash: str) -> bool:
    """Commit research credit into existing BODY heredity surfaces, not a parallel state tree."""
    morphology = getattr(body, "morphology")
    archive = getattr(body, "experience_archive")
    strategy = getattr(body, "mutation_strategy")
    pre_body_hash = morphology.fingerprint()
    if problem.discovered_by_body_hash != pre_body_hash:
        raise ValueError("native research problem was not discovered by current BODY")
    episode = native_research_experience(
        problem,
        evaluation,
        pre_body_hash=pre_body_hash,
        descendant_body_hash=descendant_body_hash,
    )
    if not archive.append(episode):
        return False
    next_strategy = NativeResearchLearner().update(strategy, (evaluation,))
    setattr(body, "mutation_strategy", next_strategy)
    if not evaluation.admissible_native_credit:
        archive.fossilize(problem.problem_id, "native research evaluation failed causal-credit contract")
    return True


@dataclass(frozen=True)
class NativeResearchCycle:
    generation: int
    problem_sha256: str
    parent_body_hash: str
    descendant_body_hash: str
    problem_discovery_score: float
    research_invention_score: float
    meta_improvement_ability: float
    retained_competence: float
    calibration_score: float
    compute_cost: float
    evidence_cost: float
    human_structural_intervention: float
    native_problem: bool
    controls_pass: bool

    @property
    def research_productivity(self) -> float:
        numerator = (
            max(0.0, float(self.problem_discovery_score))
            + max(0.0, float(self.research_invention_score))
            + max(0.0, float(self.meta_improvement_ability))
        )
        denominator = max(
            1e-12,
            float(self.compute_cost)
            + float(self.evidence_cost)
            + float(self.human_structural_intervention),
        )
        return numerator / denominator


@dataclass(frozen=True)
class NativeRecursiveResearchAssessment:
    status: str
    cycle_count: int
    research_productivity_trajectory: Tuple[float, ...]
    human_intervention_trajectory: Tuple[float, ...]
    lineage_continuous: bool
    all_native_problem_discovery: bool
    all_controls_pass: bool
    retained_competence_viable: bool
    calibration_viable: bool
    strict_research_productivity_growth: bool
    nonincreasing_human_intervention: bool
    global_recursive_acceleration: bool = False


@dataclass
class NativeRecursiveResearchLedger:
    cycles: Dict[int, NativeResearchCycle] = field(default_factory=dict)

    def append(self, cycle: NativeResearchCycle) -> bool:
        if cycle.generation in self.cycles:
            return False
        if self.cycles:
            latest = self.cycles[max(self.cycles)]
            if cycle.generation != latest.generation + 1:
                return False
            if cycle.parent_body_hash != latest.descendant_body_hash:
                return False
        self.cycles[cycle.generation] = cycle
        return True

    def assess(
        self,
        *,
        min_transitions: int = 3,
        retained_floor: float = 0.95,
        calibration_floor: float = 0.8,
    ) -> NativeRecursiveResearchAssessment:
        rows = tuple(self.cycles[key] for key in sorted(self.cycles))
        productivity = tuple(row.research_productivity for row in rows)
        humans = tuple(row.human_structural_intervention for row in rows)
        lineage = all(
            child.parent_body_hash == parent.descendant_body_hash
            for parent, child in zip(rows, rows[1:])
        )
        native = bool(rows) and all(row.native_problem for row in rows)
        controls = bool(rows) and all(row.controls_pass for row in rows)
        retained = bool(rows) and all(row.retained_competence >= retained_floor for row in rows)
        calibration = bool(rows) and all(row.calibration_score >= calibration_floor for row in rows)
        enough = len(rows) >= max(1, int(min_transitions))
        strict_productivity = enough and all(b > a for a, b in zip(productivity, productivity[1:]))
        human_nonincrease = enough and all(b <= a for a, b in zip(humans, humans[1:]))

        recursive_candidate = all(
            (enough, lineage, native, controls, retained, calibration, strict_productivity, human_nonincrease)
        )
        one_cycle_valid = bool(rows) and all((lineage, native, controls, retained, calibration))
        if recursive_candidate:
            status = "PASS_BOUNDED_RECURSIVE_NATIVE_RESEARCH_CANDIDATE"
        elif one_cycle_valid:
            status = "PASS_BOUNDED_NATIVE_RESEARCH_CYCLE_NOT_RECURSIVE"
        else:
            status = "INSUFFICIENT_NATIVE_RESEARCH_EVIDENCE"
        return NativeRecursiveResearchAssessment(
            status=status,
            cycle_count=len(rows),
            research_productivity_trajectory=productivity,
            human_intervention_trajectory=humans,
            lineage_continuous=bool(lineage),
            all_native_problem_discovery=bool(native),
            all_controls_pass=bool(controls),
            retained_competence_viable=bool(retained),
            calibration_viable=bool(calibration),
            strict_research_productivity_growth=bool(strict_productivity),
            nonincreasing_human_intervention=bool(human_nonincrease),
            global_recursive_acceleration=False,
        )


def _problem(
    *,
    body_hash: str,
    detector_id: str,
    pressure_kind: str,
    target_surface: str,
    statement: str,
    falsifier: str,
    evidence_refs: Sequence[str],
) -> NativeResearchProblem:
    seed = {
        "body_hash": body_hash,
        "detector_id": detector_id,
        "pressure_kind": pressure_kind,
        "target_surface": target_surface,
        "statement": statement,
        "falsifier": falsifier,
        "evidence_refs": sorted(set(str(value) for value in evidence_refs)),
    }
    return NativeResearchProblem(
        problem_id="NATIVE_PROBLEM::" + _sha256(seed)[:20],
        discovered_by_body_hash=body_hash,
        pressure_kind=pressure_kind,
        target_surface=target_surface,
        statement=statement,
        falsifier=falsifier,
        evidence_refs=tuple(seed["evidence_refs"]),
        detector_id=detector_id,
        human_seeded=False,
    )


def discover_native_research_problems(
    sources: Mapping[str, str],
    *,
    body_hash: str,
) -> Tuple[NativeResearchProblem, ...]:
    """Discover source-grounded bottlenecks without requiring an official benchmark."""
    if not _valid_hash(body_hash):
        raise ValueError("body_hash must be sha256")
    combined = "\n".join(str(value) for _, value in sorted(sources.items()))
    refs = tuple(sorted(str(key) for key in sources))
    out = []

    meta = str(sources.get("arte_cognition/meta_acceleration.py", ""))
    if (
        "externally_generated" in meta
        and "benchmark_disjoint" in meta
        and "class NativeResearchLearner" not in combined
    ):
        out.append(
            _problem(
                body_hash=body_hash,
                detector_id="EXTERNAL_ONLY_META_CREDIT",
                pressure_kind="human_dependency",
                target_surface="MUTATE_SEARCH_POLICY",
                statement=(
                    "The inherited mutation-search prior can learn from externally generated benchmark-disjoint evaluations, "
                    "but there is no typed native self-research credit path for verified non-official problems."
                ),
                falsifier=(
                    "A non-official, precommitted, verifier-rechecked research outcome with REMOVE and WRONG controls can "
                    "change the inherited MutationStrategyState while granting no external claim authority."
                ),
                evidence_refs=refs,
            )
        )

    if (
        "research_invention_score" in meta
        and "research_productivity" not in combined
        and "research_meta_productivity" not in combined
    ):
        out.append(
            _problem(
                body_hash=body_hash,
                detector_id="UNCREDITED_RESEARCH_INVENTION",
                pressure_kind="theory_blindspot",
                target_surface="MUTATE_SEARCH_POLICY",
                statement=(
                    "Research invention is represented in generation metrics but has no dedicated native research-productivity "
                    "credit path, so non-official intelligence-generation gain can be structurally under-credited."
                ),
                falsifier=(
                    "A native productivity measure credits validated problem-discovery, research-invention and meta-improvement "
                    "gain per compute/evidence/human-structural cost and is consumed by future search policy."
                ),
                evidence_refs=refs,
            )
        )

    morphology = str(sources.get("arte_cognition/morphology_genesis.py", ""))
    executable = str(sources.get("arte_cognition/executable_morphology.py", ""))
    l3_reachable = bool(
        "class NativeMetaMorphologyGenesisEngine" in combined
        and "MutationLevel.GENERATOR_MUTATOR" in combined
        and ".human_dependency" in combined
        and "native-meta://" in combined
    )
    if "GENERATOR_MUTATOR" in executable and "human_dependency" in executable and not l3_reachable:
        out.append(
            _problem(
                body_hash=body_hash,
                detector_id="GENERATOR_MUTATOR_PRESSURE_UNREACHABLE",
                pressure_kind="human_dependency",
                target_surface="GENERATOR_OR_MUTATOR",
                statement=(
                    "The morphology language declares GENERATOR_MUTATOR-level change and human-dependency pressure, but the "
                    "current candidate-generation path cannot reach an L3 generator/mutator policy replacement."
                ),
                falsifier=(
                    "A pre-outcome generator/mutator candidate family becomes reachable from human-dependency pressure and "
                    "produces typed, reversible L3 descendants without using current outcomes."
                ),
                evidence_refs=refs,
            )
        )

    if l3_reachable:
        compiler_block = ""
        if "class MorphologyCompiler" in executable and "class MorphologyMutator" in executable:
            compiler_block = executable.split("class MorphologyCompiler", 1)[1].split("class MorphologyMutator", 1)[0]
        if "implementation_ref" not in compiler_block and "execute_native_meta_policy" not in combined:
            out.append(
                _problem(
                    body_hash=body_hash,
                    detector_id="L3_POLICY_SEMANTICS_UNBOUND",
                    pressure_kind="theory_blindspot",
                    target_surface="COMPILER_RUNTIME_BINDING",
                    statement=(
                        "L3 generator/mutator descendants can now change implementation_ref, but the current morphology compiler "
                        "does not bind that reference to executable policy behavior, so reachability is structural only."
                    ),
                    falsifier=(
                        "A generated native-meta policy reference is compiled into executable behavior whose precommitted probe "
                        "differs from the parent, while REMOVE restores parent behavior and WRONG policy binding fails the target probe."
                    ),
                    evidence_refs=refs,
                )
            )

    unique = {problem.fingerprint(): problem for problem in out}
    return tuple(sorted(unique.values(), key=lambda problem: problem.problem_id))
