from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .adaptive_cognition import AdaptiveCognitionCompiler, CognitionPlan, ModuleCredit, TaskState
from .causal_credit import OutcomeAblationCredit, OutcomeAblationCreditEngine
from .causal_law import CausalLawAssessment, CausalLawEvaluator, InterventionObservation
from .epistemic_memory import EpistemicMemory, RepresentationMutation
from .experiment_genesis import ExperimentGenesisEngine, InterventionProposal
from .meta_router import OutcomeLearnedCognitionRouter
from .possibility_space import Fact, OperatorSpec, PossibilityCandidate, PossibilitySpaceGenerator
from .projection_generator_metapolicy import (
    ProjectionGeneratorFrontier,
    ProjectionGeneratorPolicy,
    derive_projection_generator_frontier,
    derive_projection_generator_policy,
)
from .projection_generator_program_genesis import (
    PROGRAM_MARKER,
    ProjectionGeneratorProgramFrontier,
    ProjectionGeneratorProgramPolicy,
    derive_projection_generator_program_frontier,
    derive_projection_generator_program_policy,
    generate_projection_generator_programs,
)
from .projection_generator_transform_grammar import (
    TRANSFORM_PROGRAM_MARKER,
    ProjectionTransformFrontier,
    ProjectionTransformPolicy,
    derive_projection_transform_frontier,
    derive_projection_transform_policy,
    generate_projection_transform_programs,
)
from .projection_scale_genesis import (
    ProjectionScaleFrontier,
    derive_projection_scale_frontier,
    validated_generated_projection_scales,
)
from .projection_search_metapolicy import ProjectionSearchMetaPolicy, derive_projection_search_metapolicy
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

    Projection experiment search can become evidence-conditioned. The BODY searches
    a bounded policy space for the smallest cross-context sufficient probe subset.
    When the authored numeric probe vocabulary itself leaves a strong-effect world
    residual, the BODY can generate off-grid scale atoms. Repeated authenticated
    success of generated atoms can induce reusable interpolation parameters, bounded
    generator programs, and compositional transform ASTs built from primitive unary
    operators. Search subsets, generated atoms, and generator policies are always
    reconstructed from reverified BODY evidence after restart rather than trusted as
    serialized authority scalars.
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

    def _persisted_proposals(self) -> List[InterventionProposal]:
        return [record.proposal for record in self.memory.experiments.values()]

    def _authored_projection_scales(self) -> Tuple[float, ...]:
        return tuple(float(value) for value in self.experiment.projection_margin_multipliers)

    def projection_probe_vocabulary(self) -> Tuple[float, ...]:
        """Return authored scales plus externally validated BODY-generated scales."""
        authored = self._authored_projection_scales()
        if not self.adaptive_projection_search:
            return authored
        generated = validated_generated_projection_scales(
            authored_scales=authored,
            proposals=self._persisted_proposals(),
            world_pairs=self.world_coupling.pairs,
            min_independent_classes=self.world_coupling.min_independent_classes,
            probe_scale=self._proposal_probe_scale,
            strong_effect_threshold=0.9,
        )
        return tuple(sorted(set(authored + generated)))

    def projection_search_metapolicy(self) -> ProjectionSearchMetaPolicy:
        """Derive the smallest authenticated cross-context sufficient probe policy."""
        authored = self._authored_projection_scales()
        if not self.adaptive_projection_search:
            return ProjectionSearchMetaPolicy(
                schedule=authored,
                observed_contexts=(),
                covered_contexts=(),
                candidate_count=(2 ** len(authored)) - 1 if authored else 0,
                material_effect_threshold=0.5,
                reason="adaptive metapolicy application disabled",
            )
        vocabulary = self.projection_probe_vocabulary()
        return derive_projection_search_metapolicy(
            base_scales=vocabulary,
            proposals=self._persisted_proposals(),
            world_pairs=self.world_coupling.pairs,
            min_independent_classes=self.world_coupling.min_independent_classes,
            probe_scale=self._proposal_probe_scale,
            material_effect_threshold=0.5,
        )

    def projection_search_schedule(self) -> Tuple[float, ...]:
        return self.projection_search_metapolicy().schedule

    def projection_generator_policy(self) -> ProjectionGeneratorPolicy:
        """Reconstruct a reusable scalar refinement generator from authenticated history."""
        if not self.adaptive_projection_search:
            return ProjectionGeneratorPolicy(
                status="GENERATOR_POLICY_DISABLED",
                alpha=None,
                supporting_contexts=(),
                candidate_alpha_count=0,
                strong_effect_threshold=0.9,
                reason="adaptive projection search is disabled",
            )
        return derive_projection_generator_policy(
            authored_scales=self._authored_projection_scales(),
            proposals=self._persisted_proposals(),
            world_pairs=self.world_coupling.pairs,
            min_independent_classes=self.world_coupling.min_independent_classes,
            probe_scale=self._proposal_probe_scale,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )

    def projection_generator_frontier(
        self,
        context_id: Optional[str],
        max_candidates: int = 16,
    ) -> ProjectionGeneratorFrontier:
        """Generate a shadow or learned scalar-generator refinement frontier."""
        policy = self.projection_generator_policy()
        return derive_projection_generator_frontier(
            authored_scales=self._authored_projection_scales(),
            proposals=self._persisted_proposals(),
            world_pairs=self.world_coupling.pairs,
            min_independent_classes=self.world_coupling.min_independent_classes,
            probe_scale=self._proposal_probe_scale,
            context_id=context_id,
            learned_policy=policy,
            strong_effect_threshold=0.9,
            max_candidates=max_candidates,
        )

    def generate_projection_generator_interventions(
        self,
        axis: RepresentationAxis,
        reference_values: Mapping[str, float],
        context_id: Optional[str],
        max_candidates: int = 16,
    ) -> List[InterventionProposal]:
        """Instantiate proposal-only atoms from the BODY's scalar-generator frontier."""
        if axis.family != "PROJECTION":
            return []
        frontier = self.projection_generator_frontier(
            context_id=context_id,
            max_candidates=max_candidates,
        )
        if not frontier.candidate_scales:
            return []
        engine = ExperimentGenesisEngine(
            relative_margin=self.experiment.relative_margin,
            max_proposals=max(self.experiment.max_proposals, len(frontier.candidate_scales) * len(axis.coefficients)),
            projection_margin_multipliers=frontier.candidate_scales,
        )
        generated = engine.propose(axis, reference_values)
        for proposal in generated:
            self.memory.remember_experiment(proposal)
        return generated

    def projection_generator_program_policy(self) -> ProjectionGeneratorProgramPolicy:
        """Reconstruct the minimum-complexity repeated-success generator program."""
        if not self.adaptive_projection_search:
            return ProjectionGeneratorProgramPolicy(
                status="GENERATOR_PROGRAM_POLICY_DISABLED",
                program_id=None,
                family=None,
                alpha=None,
                supporting_contexts=(),
                candidate_program_count=0,
                reason="adaptive projection search is disabled",
            )
        programs = generate_projection_generator_programs()
        return derive_projection_generator_program_policy(
            proposals=self._persisted_proposals(),
            world_pairs=self.world_coupling.pairs,
            min_independent_classes=self.world_coupling.min_independent_classes,
            programs=programs,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )

    def projection_generator_program_frontier(
        self,
        context_id: str,
        left: float,
        right: float,
        max_candidates: int = 32,
        apply_learned_program: bool = True,
    ) -> ProjectionGeneratorProgramFrontier:
        """Open a bounded generator-program frontier for one authenticated weak bracket.

        `apply_learned_program=False` is an explicit REMOVE-ablation surface: it
        preserves the same BODY memory and world evidence but suppresses only the
        application of the causally learned generator program.
        """
        policy = self.projection_generator_program_policy() if apply_learned_program else None
        return derive_projection_generator_program_frontier(
            proposals=self._persisted_proposals(),
            world_pairs=self.world_coupling.pairs,
            min_independent_classes=self.world_coupling.min_independent_classes,
            probe_scale=self._proposal_probe_scale,
            context_id=context_id,
            left=left,
            right=right,
            policy=policy,
            programs=generate_projection_generator_programs(),
            strong_effect_threshold=0.9,
            max_candidates=max_candidates,
        )

    def generate_projection_generator_program_interventions(
        self,
        axis: RepresentationAxis,
        reference_values: Mapping[str, float],
        context_id: str,
        left: float,
        right: float,
        max_candidates: int = 32,
        apply_learned_program: bool = True,
    ) -> List[InterventionProposal]:
        """Generate proposal-only atoms and bind exact generator-program provenance."""
        if axis.family != "PROJECTION":
            return []
        frontier = self.projection_generator_program_frontier(
            context_id=context_id,
            left=left,
            right=right,
            max_candidates=max_candidates,
            apply_learned_program=apply_learned_program,
        )
        if not frontier.candidates:
            return []
        generated: List[InterventionProposal] = []
        for candidate in frontier.candidates:
            engine = ExperimentGenesisEngine(
                relative_margin=self.experiment.relative_margin,
                max_proposals=max(self.experiment.max_proposals, len(axis.coefficients)),
                projection_margin_multipliers=(candidate.scale,),
            )
            for proposal in engine.propose(axis, reference_values):
                reason = (
                    f"{proposal.reason} {PROGRAM_MARKER}"
                    f"{'|'.join(candidate.program_ids)}"
                )
                bound = replace(proposal, reason=reason)
                self.memory.remember_experiment(bound)
                generated.append(bound)
        return generated

    def projection_transform_program_policy(self) -> ProjectionTransformPolicy:
        """Reconstruct a repeated-success transform AST from authenticated history."""
        if not self.adaptive_projection_search:
            return ProjectionTransformPolicy(
                status="TRANSFORM_PROGRAM_POLICY_DISABLED",
                program_id=None,
                operations=(),
                alpha=None,
                supporting_contexts=(),
                candidate_program_count=0,
                reason="adaptive projection search is disabled",
            )
        programs = generate_projection_transform_programs()
        return derive_projection_transform_policy(
            proposals=self._persisted_proposals(),
            world_pairs=self.world_coupling.pairs,
            min_independent_classes=self.world_coupling.min_independent_classes,
            programs=programs,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )

    def projection_transform_frontier(
        self,
        context_id: str,
        left: float,
        right: float,
        max_candidates: int = 64,
        apply_learned_program: bool = True,
    ) -> ProjectionTransformFrontier:
        """Open a bounded compositional transform frontier for a weak bracket.

        The REMOVE surface preserves identical evidence and only suppresses transfer
        of the learned transform AST into the fresh bracket.
        """
        policy = self.projection_transform_program_policy() if apply_learned_program else None
        return derive_projection_transform_frontier(
            proposals=self._persisted_proposals(),
            world_pairs=self.world_coupling.pairs,
            min_independent_classes=self.world_coupling.min_independent_classes,
            probe_scale=self._proposal_probe_scale,
            context_id=context_id,
            left=left,
            right=right,
            policy=policy,
            programs=generate_projection_transform_programs(),
            strong_effect_threshold=0.9,
            max_candidates=max_candidates,
        )

    def generate_projection_transform_interventions(
        self,
        axis: RepresentationAxis,
        reference_values: Mapping[str, float],
        context_id: str,
        left: float,
        right: float,
        max_candidates: int = 64,
        apply_learned_program: bool = True,
    ) -> List[InterventionProposal]:
        """Generate proposal-only atoms with exact compositional transform provenance."""
        if axis.family != "PROJECTION":
            return []
        frontier = self.projection_transform_frontier(
            context_id=context_id,
            left=left,
            right=right,
            max_candidates=max_candidates,
            apply_learned_program=apply_learned_program,
        )
        if not frontier.candidates:
            return []
        generated: List[InterventionProposal] = []
        for candidate in frontier.candidates:
            engine = ExperimentGenesisEngine(
                relative_margin=self.experiment.relative_margin,
                max_proposals=max(self.experiment.max_proposals, len(axis.coefficients)),
                projection_margin_multipliers=(candidate.scale,),
            )
            for proposal in engine.propose(axis, reference_values):
                reason = (
                    f"{proposal.reason} {TRANSFORM_PROGRAM_MARKER}"
                    f"{'|'.join(candidate.program_ids)}"
                )
                bound = replace(proposal, reason=reason)
                self.memory.remember_experiment(bound)
                generated.append(bound)
        return generated

    def projection_scale_frontier(
        self,
        context_id: Optional[str] = None,
    ) -> ProjectionScaleFrontier:
        """Legacy bounded midpoint atom frontier retained for causal comparison."""
        authored = self._authored_projection_scales()
        return derive_projection_scale_frontier(
            authored_scales=authored,
            proposals=self._persisted_proposals(),
            world_pairs=self.world_coupling.pairs,
            min_independent_classes=self.world_coupling.min_independent_classes,
            probe_scale=self._proposal_probe_scale,
            context_id=context_id,
            strong_effect_threshold=0.9,
            max_candidates=8,
        )

    def generate_projection_scale_frontier_interventions(
        self,
        axis: RepresentationAxis,
        reference_values: Mapping[str, float],
        context_id: Optional[str] = None,
    ) -> List[InterventionProposal]:
        """Instantiate proposal-only experiments for legacy midpoint scale atoms."""
        if axis.family != "PROJECTION":
            return []
        frontier = self.projection_scale_frontier(context_id=context_id)
        if not frontier.candidate_scales:
            return []
        engine = ExperimentGenesisEngine(
            relative_margin=self.experiment.relative_margin,
            max_proposals=self.experiment.max_proposals,
            projection_margin_multipliers=frontier.candidate_scales,
        )
        generated = engine.propose(axis, reference_values)
        for proposal in generated:
            self.memory.remember_experiment(proposal)
        return generated

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
