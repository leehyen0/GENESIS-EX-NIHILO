from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
from .world_coupling import (
    AxisWorldSummary,
    WorldCouplingEngine,
    WorldExecutor,
    WorldOutcomePair,
    WorldReceiptVerifier,
    WorldTransportAssessment,
)


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
    """Executable loop from task pressure to persistent developmental cognition.

    Incrementally valuable representation phenotypes and their generated
    interventions are written into BODY memory before world execution. This lets a
    descendant recover the exact coefficients/threshold/formula and intervention
    definition from its own checkpoint rather than receiving parent-process
    proposal objects. External world receipts still require independent verifier
    authority before they can steer action.

    Projection experiment search is also allowed to become evidence-conditioned,
    but only from already persisted exact experiments whose world outcomes were
    authenticated and span the configured minimum number of verifier-derived
    independence classes. The learned search schedule is therefore reconstructed
    from BODY evidence after restart rather than trusted as a serialized scalar.
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
        adaptive_projection_search: bool = True,
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
        self.adaptive_projection_search = bool(adaptive_projection_search)

    @staticmethod
    def _proposal_probe_scale(proposal: InterventionProposal) -> Optional[float]:
        marker = "probe_scale="
        reason = str(proposal.reason)
        if marker not in reason:
            return None
        tail = reason.split(marker, 1)[1].strip().split()[0].rstrip(",;)")
        try:
            return float(tail)
        except (TypeError, ValueError):
            return None

    def projection_search_schedule(self) -> Tuple[float, ...]:
        """Derive a bounded projection-probe schedule from authenticated history.

        The default 1x/2x/4x vocabulary is never changed until every scale has at
        least one exact experiment supported by the required number of independent
        verifier classes. A dominant scale may reduce the next search to two scales
        after repeated exact-experiment evidence, and to one scale only after the
        dominant scale has reproduced across at least two world contexts. This is a
        bounded matched-family search optimization, not global transport authority.
        """
        base = tuple(float(value) for value in self.experiment.projection_margin_multipliers)
        if not self.adaptive_projection_search or len(base) <= 1:
            return base

        stats: Dict[float, Dict[str, object]] = {
            scale: {"effects": [], "contexts": set()} for scale in base
        }
        for record in self.memory.experiments.values():
            proposal = record.proposal
            scale = self._proposal_probe_scale(proposal)
            if scale is None:
                continue
            matched_scale = next((item for item in base if abs(item - scale) <= 1e-12), None)
            if matched_scale is None:
                continue

            by_class: Dict[str, WorldOutcomePair] = {}
            for pair in self.world_coupling.pairs:
                if pair.experiment_id != proposal.experiment_id:
                    continue
                if not (
                    pair.matched_budget
                    and pair.externally_generated
                    and pair.authority_verified
                    and pair.independence_class_id != "UNVERIFIED"
                ):
                    continue
                by_class.setdefault(pair.independence_class_id, pair)
            if len(by_class) < self.world_coupling.min_independent_classes:
                continue

            unique_pairs = list(by_class.values())
            mean_abs_effect = sum(abs(pair.effect) for pair in unique_pairs) / len(unique_pairs)
            effects = stats[matched_scale]["effects"]
            contexts = stats[matched_scale]["contexts"]
            assert isinstance(effects, list)
            assert isinstance(contexts, set)
            effects.append(float(mean_abs_effect))
            contexts.update(pair.context_id for pair in unique_pairs)

        # Do not narrow the vocabulary while any default scale remains untested.
        if any(not stats[scale]["effects"] for scale in base):
            return base

        scores = {
            scale: sum(stats[scale]["effects"]) / len(stats[scale]["effects"])
            for scale in base
        }
        base_index = {scale: index for index, scale in enumerate(base)}
        ordered = tuple(sorted(base, key=lambda scale: (-scores[scale], base_index[scale])))
        top = ordered[0]
        runner_up = ordered[1]

        # Material dominance is required before search-space contraction.
        if scores[top] < 0.5 or (scores[top] - scores[runner_up]) < 0.25:
            return base

        top_effects = stats[top]["effects"]
        top_contexts = stats[top]["contexts"]
        assert isinstance(top_effects, list)
        assert isinstance(top_contexts, set)

        budget = len(base)
        if len(top_effects) >= 2:
            budget = min(budget, 2)
        if len(top_effects) >= 4 and len(top_contexts) >= 2:
            budget = 1
        return ordered[:budget]

    def generate_interventions(
        self,
        axis: RepresentationAxis,
        reference_values: Mapping[str, float],
    ) -> List[InterventionProposal]:
        """Generate exact interventions using BODY-derived search scheduling."""
        engine = self.experiment
        if axis.family == "PROJECTION":
            schedule = self.projection_search_schedule()
            if schedule != tuple(self.experiment.projection_margin_multipliers):
                engine = ExperimentGenesisEngine(
                    relative_margin=self.experiment.relative_margin,
                    max_proposals=self.experiment.max_proposals,
                    projection_margin_multipliers=schedule,
                )
        generated = engine.propose(axis, reference_values)
        for proposal in generated:
            self.memory.remember_experiment(proposal)
        return generated

    def cycle(
        self,
        task: TaskState,
        facts: Sequence[Fact] = (),
        residuals: Sequence[ResidualObservation] = (),
        measurements: Sequence[MeasurementObservation] = (),
        operator_spec: Optional[OperatorSpec] = None,
        possibility_budget: int = 32,
        experiment_reference_values: Optional[Mapping[str, float]] = None,
        world_context_id: Optional[str] = None,
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

        mutations_before = len(self.memory.mutation_log)
        axes = self.representation.propose_axes(measurements) if measurements else []
        value_assessments = [self.representation_value.assess(axis, measurements) for axis in axes] if measurements else []
        value_by_axis = {item.axis_id: item for item in value_assessments}
        eligible_axis_ids = {
            item.axis_id for item in value_assessments
            if item.status == "INCREMENTAL_REPRESENTATION_VALUE"
        }
        semantically_eligible_axes = [axis for axis in axes if axis.axis_id in eligible_axis_ids]
        for axis in semantically_eligible_axes:
            self.memory.remember_representation(
                axis,
                value_status=value_by_axis[axis.axis_id].status,
            )

        semantic_rows = (
            self.representation.augment_residuals(residuals, measurements, semantically_eligible_axes)
            if measurements and residuals
            else list(residuals)
        )

        intervention_proposals: List[InterventionProposal] = []
        if experiment_reference_values:
            for axis in semantically_eligible_axes:
                intervention_proposals.extend(
                    self.generate_interventions(axis, experiment_reference_values)
                )
        intervention_proposals = self.world_coupling.rank_proposals(
            intervention_proposals,
            context_id=world_context_id,
        )

        concepts = self.semantic.propose_concepts(semantic_rows)
        semantic_queries = self.semantic.propose_queries(semantic_rows, concepts)
        laws: List[LawCandidate] = []
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

    def persisted_representation_axes(self) -> List[RepresentationAxis]:
        return self.memory.active_representation_axes()

    def persisted_intervention_proposals(self) -> List[InterventionProposal]:
        return self.memory.persisted_intervention_proposals()

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

    def learn_from_ablation_outcomes(
        self,
        active_modules,
        full_outcome: float,
        ablation_outcomes,
        matched_compute=None,
    ) -> List[OutcomeAblationCredit]:
        credits = self.credit_engine.assign(
            full_outcome=full_outcome,
            ablation_outcomes=ablation_outcomes,
            active_modules=active_modules,
            matched_compute=matched_compute,
        )
        self.router.policy.observe(self.credit_engine.to_router_credits(credits))
        return credits

    def learn_topology(self, sequence, edge_synergy) -> None:
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
        verifier: Optional[WorldReceiptVerifier] = None,
    ) -> WorldOutcomePair:
        """Consume a world intervention; unverified receipts have audit-only status."""
        return self.world_coupling.execute(proposal, executor, verifier=verifier)

    def world_axis_summary(
        self,
        axis_id: str,
        context_id: Optional[str] = None,
    ) -> AxisWorldSummary:
        return self.world_coupling.summary(axis_id, context_id=context_id)

    def assess_world_transport(
        self,
        proposals: Sequence[InterventionProposal],
    ) -> WorldTransportAssessment:
        return self.world_coupling.assess_transport(proposals)

    def rank_intervention_proposals(
        self,
        proposals: Sequence[InterventionProposal],
        context_id: Optional[str] = None,
    ) -> List[InterventionProposal]:
        return self.world_coupling.rank_proposals(proposals, context_id=context_id)
