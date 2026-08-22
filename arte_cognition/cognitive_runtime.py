from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .adaptive_cognition import AdaptiveCognitionCompiler, CognitionPlan, ModuleCredit, TaskState
from .epistemic_memory import EpistemicMemory, RepresentationMutation
from .meta_router import OutcomeLearnedCognitionRouter
from .possibility_space import Fact, OperatorSpec, PossibilityCandidate, PossibilitySpaceGenerator
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
    concepts: List[ConceptCandidate]
    laws: List[LawCandidate]
    semantic_queries: List[SemanticQuery]
    representation_mutations: List[RepresentationMutation]
    active_generated_concepts: List[str]
    shadow_generated_concepts: List[str]


class PersistentCognitiveRuntime:
    """One executable loop from task pressure to self-revising cognition.

    The runtime composes sparse routing, bounded modal generation, residual-driven
    semantic genesis, reversible epistemic memory and outcome-learned routing.
    Generated possibilities/concepts never become evidence merely by existing.
    """

    def __init__(
        self,
        compiler: Optional[AdaptiveCognitionCompiler] = None,
        router: Optional[OutcomeLearnedCognitionRouter] = None,
        possibility: Optional[PossibilitySpaceGenerator] = None,
        semantic: Optional[SemanticGenesisEngine] = None,
        memory: Optional[EpistemicMemory] = None,
    ) -> None:
        self.compiler = compiler or AdaptiveCognitionCompiler()
        self.router = router or OutcomeLearnedCognitionRouter(self.compiler)
        self.possibility = possibility or PossibilitySpaceGenerator()
        self.semantic = semantic or SemanticGenesisEngine()
        self.memory = memory or EpistemicMemory()

    def cycle(
        self,
        task: TaskState,
        facts: Sequence[Fact] = (),
        residuals: Sequence[ResidualObservation] = (),
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

        concepts = self.semantic.propose_concepts(residuals)
        semantic_queries = self.semantic.propose_queries(residuals, concepts)
        laws: List[LawCandidate] = []
        mutations_before = len(self.memory.mutation_log)

        for concept in concepts:
            self.memory.remember_concept(concept)
            law = self.semantic.induce_law(concept, residuals)
            laws.append(law)
            self.memory.ingest_law(law)

        return CognitiveCycle(
            plan=plan,
            possibilities=possibilities,
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
        return self.router.learn_from_outcome(
            active_modules=active_modules,
            baseline_decision=baseline_decision,
            treatment_decision=treatment_decision,
            baseline_outcome=baseline_outcome,
            treatment_outcome=treatment_outcome,
            ablation_decisions=ablation_decisions,
        )
