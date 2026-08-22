from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .adaptive_cognition import AdaptiveCognitionCompiler, CognitionPlan, ModuleCredit, TaskState
from .causal_credit import OutcomeAblationCredit, OutcomeAblationCreditEngine
from .epistemic_memory import EpistemicMemory, RepresentationMutation
from .meta_router import OutcomeLearnedCognitionRouter
from .possibility_space import Fact, OperatorSpec, PossibilityCandidate, PossibilitySpaceGenerator
from .representation_genesis import MeasurementObservation, RepresentationAxis, RepresentationGenesisEngine
from .semantic_genesis import (
    ConceptCandidate,
    LawCandidate,
    ResidualObservation,
    SemanticGenesisEngine,
    SemanticQuery,
)


@dataclass
class CognitiveCycle:
    plan: CognitionPlan
    possibilities: List[PossibilityCandidate]
    representation_axes: List[RepresentationAxis]
    concepts: List[ConceptCandidate]
    laws: List[LawCandidate]
    semantic_queries: List[SemanticQuery]
    representation_mutations: List[RepresentationMutation]
    active_generated_concepts: List[str]
    shadow_generated_concepts: List[str]


class PersistentCognitiveRuntime:
    """One executable loop from task pressure to self-revising cognition.

    The runtime composes sparse routing, bounded modal generation, measurable
    representation-axis genesis, residual-driven semantic genesis, reversible
    epistemic memory, outcome-ablation credit and outcome-learned routing.
    Generated possibilities/concepts/axes never become evidence merely by existing.
    """

    def __init__(
        self,
        compiler: Optional[AdaptiveCognitionCompiler] = None,
        router: Optional[OutcomeLearnedCognitionRouter] = None,
        possibility: Optional[PossibilitySpaceGenerator] = None,
        representation: Optional[RepresentationGenesisEngine] = None,
        semantic: Optional[SemanticGenesisEngine] = None,
        memory: Optional[EpistemicMemory] = None,
        credit_engine: Optional[OutcomeAblationCreditEngine] = None,
    ) -> None:
        self.compiler = compiler or AdaptiveCognitionCompiler()
        self.router = router or OutcomeLearnedCognitionRouter(self.compiler)
        self.possibility = possibility or PossibilitySpaceGenerator()
        self.representation = representation or RepresentationGenesisEngine()
        self.semantic = semantic or SemanticGenesisEngine()
        self.memory = memory or EpistemicMemory()
        self.credit_engine = credit_engine or OutcomeAblationCreditEngine()

    def cycle(
        self,
        task: TaskState,
        facts: Sequence[Fact] = (),
        residuals: Sequence[ResidualObservation] = (),
        measurements: Sequence[MeasurementObservation] = (),
        operator_spec: Optional[OperatorSpec] = None,
        possibility_budget: int = 32,
    ) -> CognitiveCycle:
        plan = self.router.compile(task)
        possibilities = self.possibility.expand(
            facts=facts,
            modal_basis=plan.modal_basis,
            spec=operator_spec,
            budget=possibility_budget,
        )

        axes = self.representation.propose_axes(measurements) if measurements else []
        semantic_rows = (
            self.representation.augment_residuals(residuals, measurements, axes)
            if measurements and residuals
            else list(residuals)
        )

        concepts = self.semantic.propose_concepts(semantic_rows)
        semantic_queries = self.semantic.propose_queries(semantic_rows, concepts)
        laws: List[LawCandidate] = []
        mutations_before = len(self.memory.mutation_log)

        for concept in concepts:
            self.memory.remember_concept(concept)
            law = self.semantic.induce_law(concept, semantic_rows)
            laws.append(law)
            self.memory.ingest_law(law)

        return CognitiveCycle(
            plan=plan,
            possibilities=possibilities,
            representation_axes=axes,
            concepts=concepts,
            laws=laws,
            semantic_queries=semantic_queries,
            representation_mutations=list(self.memory.mutation_log[mutations_before:]),
            active_generated_concepts=self.memory.active_concepts(),
            shadow_generated_concepts=self.memory.shadow_concepts(),
        )

    def observe_world(self, observations: Iterable[ResidualObservation]) -> List[RepresentationMutation]:
        mutations: List[RepresentationMutation] = []
        for row in observations:
            mutations.extend(self.memory.observe(row))
        return mutations

    def learn_from_outcome(
        self,
        active_modules,
        baseline_decision,
        treatment_decision,
        baseline_outcome: float,
        treatment_outcome: float,
        ablation_decisions=None,
    ) -> List[ModuleCredit]:
        """Legacy behavior-sensitive credit path retained for compatibility."""
        return self.router.learn_from_outcome(
            active_modules=active_modules,
            baseline_decision=baseline_decision,
            treatment_decision=treatment_decision,
            baseline_outcome=baseline_outcome,
            treatment_outcome=treatment_outcome,
            ablation_decisions=ablation_decisions,
        )

    def learn_from_ablation_outcomes(
        self,
        active_modules,
        full_outcome: float,
        ablation_outcomes,
        matched_compute=None,
    ) -> List[OutcomeAblationCredit]:
        """Preferred learning path: credit from realized module-outcome ablations."""
        credits = self.credit_engine.assign(
            full_outcome=full_outcome,
            ablation_outcomes=ablation_outcomes,
            active_modules=active_modules,
            matched_compute=matched_compute,
        )
        self.router.policy.observe(self.credit_engine.to_router_credits(credits))
        return credits
