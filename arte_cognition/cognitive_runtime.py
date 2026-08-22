from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence

from .adaptive_cognition import AdaptiveCognitionCompiler, CognitionPlan, ModuleCredit, TaskState
from .causal_credit import OutcomeAblationCredit, OutcomeAblationCreditEngine
from .causal_law import CausalLawAssessment, CausalLawEvaluator, InterventionObservation
from .epistemic_memory import EpistemicMemory, RepresentationMutation
from .experiment_genesis import ExperimentGenesisEngine, InterventionProposal
from .meta_router import OutcomeLearnedCognitionRouter
from .possibility_space import Fact, OperatorSpec, PossibilityCandidate, PossibilitySpaceGenerator
from .representation_genesis import MeasurementObservation, RepresentationAxis, RepresentationGenesisEngine
from .representation_value import RepresentationValueAssessment, RepresentationValueEvaluator
from .semantic_genesis import (
    ConceptCandidate,
    LawCandidate,
    ResidualObservation,
    SemanticGenesisEngine,
    SemanticQuery,
)
from .subgraph_credit import MinimumCausalSubgraphFinder, MinimumSufficientSubgraph, SubgraphEvaluation
from .topology_learning import CognitionTopologyLearner, MacroCognitionCandidate
from .validation_matrix import RobustPromotionGate, ValidationGateResult, ValidationObservation
from .world_coupling import AxisWorldSummary, WorldCouplingEngine, WorldExecutor, WorldOutcomePair


@dataclass
class CognitiveCycle:
    plan: CognitionPlan
    execution_order: List[str]
    macro_candidates: List[MacroCognitionCandidate]
    possibilities: List[PossibilityCandidate]
    representation_axes: List[RepresentationAxis]
    representation_value: List[RepresentationValueAssessment]
    intervention_proposals: List[InterventionProposal]
    concepts: List[ConceptCandidate]
    laws: List[LawCandidate]
    semantic_queries: List[SemanticQuery]
    representation_mutations: List[RepresentationMutation]
    active_generated_concepts: List[str]
    shadow_generated_concepts: List[str]


class PersistentCognitiveRuntime:
    """One executable loop from task pressure to self-revising cognition.

    The runtime composes sparse routing, bounded topology learning, modal
    generation, measurable representation-axis genesis, incremental-value gates,
    experiment genesis, residual-driven semantic genesis, reversible epistemic
    memory, outcome-ablation credit, causal-law staging, robust promotion and a
    narrow world-coupling memory. Externally realized intervention consequences
    can change the ordering of future experiments and that change survives BODY
    checkpoint/restore. Generated objects never become evidence merely by existing.
    """

    def __init__(
        self,
        compiler: Optional[AdaptiveCognitionCompiler] = None,
        router: Optional[OutcomeLearnedCognitionRouter] = None,
        topology: Optional[CognitionTopologyLearner] = None,
        possibility: Optional[PossibilitySpaceGenerator] = None,
        representation: Optional[RepresentationGenesisEngine] = None,
        representation_value: Optional[RepresentationValueEvaluator] = None,
        experiment: Optional[ExperimentGenesisEngine] = None,
        semantic: Optional[SemanticGenesisEngine] = None,
        memory: Optional[EpistemicMemory] = None,
        credit_engine: Optional[OutcomeAblationCreditEngine] = None,
        causal_law: Optional[CausalLawEvaluator] = None,
        subgraph_finder: Optional[MinimumCausalSubgraphFinder] = None,
        promotion_gate: Optional[RobustPromotionGate] = None,
        world_coupling: Optional[WorldCouplingEngine] = None,
    ) -> None:
        self.compiler = compiler or AdaptiveCognitionCompiler()
        self.router = router or OutcomeLearnedCognitionRouter(self.compiler)
        self.topology = topology or CognitionTopologyLearner()
        self.possibility = possibility or PossibilitySpaceGenerator()
        self.representation = representation or RepresentationGenesisEngine()
        self.representation_value = representation_value or RepresentationValueEvaluator()
        self.experiment = experiment or ExperimentGenesisEngine()
        self.semantic = semantic or SemanticGenesisEngine()
        self.memory = memory or EpistemicMemory()
        self.credit_engine = credit_engine or OutcomeAblationCreditEngine()
        self.causal_law = causal_law or CausalLawEvaluator()
        self.subgraph_finder = subgraph_finder or MinimumCausalSubgraphFinder()
        self.promotion_gate = promotion_gate or RobustPromotionGate()
        self.world_coupling = world_coupling or WorldCouplingEngine()

    def cycle(
        self,
        task: TaskState,
        facts: Sequence[Fact] = (),
        residuals: Sequence[ResidualObservation] = (),
        measurements: Sequence[MeasurementObservation] = (),
        operator_spec: Optional[OperatorSpec] = None,
        possibility_budget: int = 32,
        experiment_reference_values: Optional[Mapping[str, float]] = None,
    ) -> CognitiveCycle:
        plan = self.router.compile(task)
        execution_order = self.topology.reorder(plan.active_subgraph)
        macro_candidates = self.topology.propose_macros()
        possibilities = self.possibility.expand(
            facts=facts,
            modal_basis=plan.modal_basis,
            spec=operator_spec,
            budget=possibility_budget,
        )

        axes = self.representation.propose_axes(measurements) if measurements else []
        value_assessments = [self.representation_value.assess(axis, measurements) for axis in axes] if measurements else []
        eligible_axis_ids = {
            item.axis_id for item in value_assessments
            if item.status == "INCREMENTAL_REPRESENTATION_VALUE"
        }
        semantically_eligible_axes = [axis for axis in axes if axis.axis_id in eligible_axis_ids]

        semantic_rows = (
            self.representation.augment_residuals(residuals, measurements, semantically_eligible_axes)
            if measurements and residuals
            else list(residuals)
        )

        intervention_proposals: List[InterventionProposal] = []
        if experiment_reference_values:
            for axis in semantically_eligible_axes:
                intervention_proposals.extend(self.experiment.propose(axis, experiment_reference_values))
        intervention_proposals = self.world_coupling.rank_proposals(intervention_proposals)

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
            execution_order=execution_order,
            macro_candidates=macro_candidates,
            possibilities=possibilities,
            representation_axes=axes,
            representation_value=value_assessments,
            intervention_proposals=intervention_proposals,
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
        """Preferred module-learning path: realized outcome ablations."""
        credits = self.credit_engine.assign(
            full_outcome=full_outcome,
            ablation_outcomes=ablation_outcomes,
            active_modules=active_modules,
            matched_compute=matched_compute,
        )
        self.router.policy.observe(self.credit_engine.to_router_credits(credits))
        return credits

    def learn_topology(self, sequence, edge_synergy) -> None:
        """Update routing order only from explicit pair-synergy evidence."""
        self.topology.observe_sequence(sequence=sequence, edge_synergy=edge_synergy)

    def assess_causal_law(
        self,
        law: LawCandidate,
        observations: Sequence[InterventionObservation],
    ) -> CausalLawAssessment:
        return self.causal_law.assess(law, observations)

    def find_minimum_sufficient_subgraph(
        self,
        full_modules: Sequence[str],
        evaluations: Mapping[tuple, SubgraphEvaluation],
    ) -> MinimumSufficientSubgraph:
        return self.subgraph_finder.find(full_modules, evaluations)

    def assess_robust_promotion(
        self,
        observations: Sequence[ValidationObservation],
        protected_contexts: Iterable[str] = (),
    ) -> ValidationGateResult:
        return self.promotion_gate.assess(observations, protected_contexts)

    def execute_world_intervention(
        self,
        proposal: InterventionProposal,
        executor: WorldExecutor,
    ) -> WorldOutcomePair:
        """Enact a proposal through an external executor and consume its outcome."""
        return self.world_coupling.execute(proposal, executor)

    def world_axis_summary(self, axis_id: str) -> AxisWorldSummary:
        return self.world_coupling.summary(axis_id)

    def rank_intervention_proposals(
        self,
        proposals: Sequence[InterventionProposal],
    ) -> List[InterventionProposal]:
        return self.world_coupling.rank_proposals(proposals)
