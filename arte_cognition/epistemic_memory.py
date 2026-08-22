from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .experiment_genesis import InterventionProposal
from .representation_genesis import RepresentationAxis
from .semantic_genesis import ConceptCandidate, LawCandidate, ResidualObservation


@dataclass
class ConceptRecord:
    concept: ConceptCandidate
    status: str = "SHADOW_PROPOSAL"
    revisions: int = 0
    last_law_id: Optional[str] = None


@dataclass
class LawRecord:
    law: LawCandidate
    status: str
    refutations: List[str] = field(default_factory=list)


@dataclass
class RepresentationRecord:
    axis: RepresentationAxis
    status: str = "ACTIVE_VALIDATED"
    value_status: str = "INCREMENTAL_REPRESENTATION_VALUE"
    revisions: int = 0
    history: List[RepresentationAxis] = field(default_factory=list)


@dataclass
class ExperimentRecord:
    proposal: InterventionProposal
    status: str = "PROPOSAL_ONLY"
    revisions: int = 0
    history: List[InterventionProposal] = field(default_factory=list)


@dataclass(frozen=True)
class RepresentationMutation:
    mutation_id: str
    action: str
    target: str
    reason: str
    reversible: bool = True


class EpistemicMemory:
    """Persistent, reversible BODY memory for generated cognition.

    Concepts and predictive laws are preserved together with the exact generated
    representation phenotype that created them: coefficients, bias, threshold,
    formula and partition. Generated intervention definitions are also retained.
    This prevents descendant checkpoints from depending on parent-process Python
    objects to reconstruct what the BODY had learned.
    """

    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptRecord] = {}
        self.laws: Dict[str, LawRecord] = {}
        self.representations: Dict[str, RepresentationRecord] = {}
        self.experiments: Dict[str, ExperimentRecord] = {}
        self.mutation_log: List[RepresentationMutation] = []

    def remember_representation(
        self,
        axis: RepresentationAxis,
        value_status: str = "INCREMENTAL_REPRESENTATION_VALUE",
    ) -> RepresentationRecord:
        if value_status != "INCREMENTAL_REPRESENTATION_VALUE":
            raise ValueError("only incrementally validated representation axes enter persistent BODY memory")

        record = self.representations.get(axis.axis_id)
        if record is None:
            record = RepresentationRecord(axis=axis, value_status=value_status)
            self.representations[axis.axis_id] = record
            self._append_mutation(RepresentationMutation(
                mutation_id="PERSIST_AXIS::" + axis.axis_id,
                action="EXTEND",
                target=axis.axis_id,
                reason="incremental representation value plus held-out reproduction entered BODY phenotype memory",
            ))
            return record

        if record.axis != axis:
            record.history.append(record.axis)
            record.axis = axis
            record.revisions += 1
            record.status = "ACTIVE_VALIDATED"
            record.value_status = value_status
            self._append_mutation(RepresentationMutation(
                mutation_id=f"REVISE_AXIS::{axis.axis_id}::{record.revisions}",
                action="REVISE",
                target=axis.axis_id,
                reason="fresh evidence changed the exact generated representation phenotype",
            ))
        return record

    def remember_experiment(self, proposal: InterventionProposal) -> ExperimentRecord:
        record = self.experiments.get(proposal.experiment_id)
        if record is None:
            record = ExperimentRecord(proposal=proposal)
            self.experiments[proposal.experiment_id] = record
            self._append_mutation(RepresentationMutation(
                mutation_id="PERSIST_EXPERIMENT::" + proposal.experiment_id,
                action="EXTEND",
                target=proposal.experiment_id,
                reason="generated threshold-crossing intervention entered BODY phenotype memory as proposal-only",
            ))
            return record

        if record.proposal != proposal:
            record.history.append(record.proposal)
            record.proposal = proposal
            record.revisions += 1
            record.status = "PROPOSAL_ONLY"
            self._append_mutation(RepresentationMutation(
                mutation_id=f"REVISE_EXPERIMENT::{proposal.experiment_id}::{record.revisions}",
                action="REVISE",
                target=proposal.experiment_id,
                reason="representation revision changed the generated intervention definition",
            ))
        return record

    def active_representation_axes(self) -> List[RepresentationAxis]:
        return [
            self.representations[axis_id].axis
            for axis_id in sorted(self.representations)
            if self.representations[axis_id].status == "ACTIVE_VALIDATED"
        ]

    def persisted_intervention_proposals(self) -> List[InterventionProposal]:
        active_axis_ids = {
            axis.axis_id for axis in self.active_representation_axes()
        }
        return [
            self.experiments[experiment_id].proposal
            for experiment_id in sorted(self.experiments)
            if self.experiments[experiment_id].status == "PROPOSAL_ONLY"
            and self.experiments[experiment_id].proposal.axis_id in active_axis_ids
        ]

    def remember_concept(self, concept: ConceptCandidate) -> ConceptRecord:
        record = self.concepts.get(concept.concept_id)
        if record is None:
            record = ConceptRecord(concept=concept)
            self.concepts[concept.concept_id] = record
        return record

    def ingest_law(self, law: LawCandidate) -> LawRecord:
        concept = self.concepts.get(law.concept_id)
        if concept is None:
            raise KeyError(f"unknown concept: {law.concept_id}")

        active = law.status == "BOUNDED_LAW"
        status = "ACTIVE_BOUNDED" if active else "SHADOW_UNVERIFIED"
        record = LawRecord(law=law, status=status)
        self.laws[law.law_id] = record
        concept.last_law_id = law.law_id

        if active:
            concept.status = "ACTIVE_BOUNDED"
            mutation = RepresentationMutation(
                mutation_id="ACTIVATE::" + law.concept_id,
                action="EXTEND",
                target=law.concept_id,
                reason="held-out reproduced generated distinction",
            )
            self._append_mutation(mutation)
        return record

    def observe(self, row: ResidualObservation) -> List[RepresentationMutation]:
        """Apply a new realized observation against active generated laws."""
        mutations: List[RepresentationMutation] = []
        row_features = set(row.features)

        for law_id, record in list(self.laws.items()):
            if record.status != "ACTIVE_BOUNDED":
                continue
            concept_record = self.concepts.get(record.law.concept_id)
            if concept_record is None:
                continue
            if not set(concept_record.concept.defining_features).issubset(row_features):
                continue
            if row.outcome == record.law.predicted_outcome:
                continue

            record.status = "DEMOTED_REFUTED"
            if row.residual_id not in record.refutations:
                record.refutations.append(row.residual_id)
            concept_record.status = "SHADOW_REFUTED"
            concept_record.revisions += 1
            mutation = RepresentationMutation(
                mutation_id=f"DEMOTE::{law_id}::{row.residual_id}",
                action="DEMOTE",
                target=record.law.concept_id,
                reason="new realized counterexample violated active bounded law",
            )
            self._append_mutation(mutation)
            mutations.append(mutation)

        return mutations

    def active_concepts(self) -> List[str]:
        return sorted(
            concept_id for concept_id, record in self.concepts.items()
            if record.status == "ACTIVE_BOUNDED"
        )

    def shadow_concepts(self) -> List[str]:
        return sorted(
            concept_id for concept_id, record in self.concepts.items()
            if record.status != "ACTIVE_BOUNDED"
        )

    def _append_mutation(self, mutation: RepresentationMutation) -> None:
        if mutation.mutation_id not in {item.mutation_id for item in self.mutation_log}:
            self.mutation_log.append(mutation)
