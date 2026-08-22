from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional

from .experiment_genesis import InterventionProposal
from .representation_genesis import RepresentationAxis
from .semantic_genesis import ConceptCandidate, LawCandidate, ResidualObservation


_TRANSFORM_PROGRAM_MARKER = "generator_transform_programs="


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


def _transform_program_ids(reason: str) -> tuple[str, ...]:
    text = str(reason)
    if _TRANSFORM_PROGRAM_MARKER not in text:
        return ()
    tail = text.split(_TRANSFORM_PROGRAM_MARKER, 1)[1].strip().split()[0].rstrip(",;)")
    return tuple(sorted(set(item for item in tail.split("|") if item)))


def _same_exact_intervention_semantics(
    left: InterventionProposal,
    right: InterventionProposal,
) -> bool:
    """Compare every exact-action field while deliberately excluding provenance text."""
    return bool(
        left.experiment_id == right.experiment_id
        and left.axis_id == right.axis_id
        and left.manipulated_variable == right.manipulated_variable
        and left.held_fixed == right.held_fixed
        and float(left.low_value) == float(right.low_value)
        and float(left.high_value) == float(right.high_value)
        and left.predicted_low_side == right.predicted_low_side
        and left.predicted_high_side == right.predicted_high_side
        and left.status == right.status
    )


def _merge_transform_provenance(
    existing: InterventionProposal,
    incoming: InterventionProposal,
) -> Optional[InterventionProposal]:
    """Union generator ancestry when two programs collapse to the same exact intervention.

    Floating-point realization can make distinct nominal generator scales produce the
    same LOW/HIGH intervention phenotype. Evidence must remain bound to that one exact
    ExperimentID, but BODY lineage must not lose any generator program that produced
    the action. This merge is intentionally unavailable when any action-semantic field
    differs.
    """
    if not _same_exact_intervention_semantics(existing, incoming):
        return None
    old_ids = _transform_program_ids(existing.reason)
    new_ids = _transform_program_ids(incoming.reason)
    if not old_ids or not new_ids:
        return None
    merged_ids = tuple(sorted(set(old_ids + new_ids)))
    if merged_ids == old_ids:
        return existing
    prefix = str(existing.reason).split(_TRANSFORM_PROGRAM_MARKER, 1)[0].rstrip()
    merged_reason = f"{prefix} {_TRANSFORM_PROGRAM_MARKER}{'|'.join(merged_ids)}".strip()
    return replace(existing, reason=merged_reason)


class EpistemicMemory:
    """Persistent, reversible BODY memory for generated cognition.

    Concepts and predictive laws are preserved together with the exact generated
    representation phenotype that created them: coefficients, bias, threshold,
    formula and partition. Generated intervention definitions are also retained.
    This prevents descendant checkpoints from depending on parent-process Python
    objects to reconstruct what the BODY had learned.

    World-caused revision is lineage preserving. Strong authenticated
    counterevidence can demote an active representation phenotype, its generated
    experiments, and directly dependent concept/law state without deleting any of
    those objects. A later fresh cycle can only reactivate that axis identity when
    it generates a materially revised phenotype; reproducing the same refuted
    object does not silently self-authorize it again.
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

        merged = _merge_transform_provenance(record.proposal, proposal)
        if merged is not None:
            if merged != record.proposal:
                record.history.append(record.proposal)
                record.proposal = merged
                record.revisions += 1
                self._append_mutation(RepresentationMutation(
                    mutation_id=f"MERGE_EXPERIMENT_PROVENANCE::{proposal.experiment_id}::{record.revisions}",
                    action="EXTEND",
                    target=proposal.experiment_id,
                    reason=(
                        "distinct generated transform programs realized the same exact intervention phenotype; "
                        "their ancestry was unioned without duplicating world-action authority"
                    ),
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

    def demote_world_refuted_axis(
        self,
        axis_id: str,
        experiment_ids: Iterable[str],
        evidence_tag: str,
    ) -> List[RepresentationMutation]:
        """Demote a generated cognition phenotype after robust world refutation.

        The caller is responsible for the external-evidence gate. Once that gate
        closes, this method performs the same-BODY state transition while keeping
        every refuted object addressable for delayed replay and wrong-swap tests.
        """
        axis_record = self.representations.get(axis_id)
        if axis_record is None or axis_record.status != "ACTIVE_VALIDATED":
            return []

        contradicted = tuple(sorted(set(str(item) for item in experiment_ids)))
        if not contradicted:
            return []

        mutations: List[RepresentationMutation] = []
        axis_record.status = "SHADOW_WORLD_REFUTED"
        axis_record.revisions += 1
        axis_mutation = RepresentationMutation(
            mutation_id=f"WORLD_DEMOTE_AXIS::{axis_id}::{axis_record.revisions}",
            action="DEMOTE",
            target=axis_id,
            reason=(
                "authenticated independent world counterevidence refuted multiple exact experiments "
                f"for this representation phenotype; evidence={evidence_tag}"
            ),
        )
        self._append_mutation(axis_mutation)
        mutations.append(axis_mutation)

        for experiment_id, record in sorted(self.experiments.items()):
            if record.proposal.axis_id != axis_id:
                continue
            if record.status != "PROPOSAL_ONLY":
                continue
            record.status = "SHADOW_WORLD_REFUTED"
            record.revisions += 1
            mutation = RepresentationMutation(
                mutation_id=f"WORLD_DEMOTE_EXPERIMENT::{experiment_id}::{record.revisions}",
                action="DEMOTE",
                target=experiment_id,
                reason=(
                    "parent representation phenotype was refuted by authenticated world counterevidence; "
                    f"contradicted_exact_experiments={','.join(contradicted)}"
                ),
            )
            self._append_mutation(mutation)
            mutations.append(mutation)

        for concept_id, concept_record in sorted(self.concepts.items()):
            if axis_id not in set(concept_record.concept.defining_features):
                continue
            if concept_record.status != "ACTIVE_BOUNDED":
                continue
            concept_record.status = "SHADOW_WORLD_REFUTED"
            concept_record.revisions += 1
            mutation = RepresentationMutation(
                mutation_id=f"WORLD_DEMOTE_CONCEPT::{concept_id}::{concept_record.revisions}",
                action="DEMOTE",
                target=concept_id,
                reason="directly dependent generated concept lost support when its representation phenotype was world-refuted",
            )
            self._append_mutation(mutation)
            mutations.append(mutation)

            if concept_record.last_law_id:
                law_record = self.laws.get(concept_record.last_law_id)
                if law_record is not None and law_record.status == "ACTIVE_BOUNDED":
                    law_record.status = "DEMOTED_WORLD_REFUTED"
                    if evidence_tag not in law_record.refutations:
                        law_record.refutations.append(evidence_tag)
                    law_mutation = RepresentationMutation(
                        mutation_id=f"WORLD_DEMOTE_LAW::{law_record.law.law_id}::{evidence_tag}",
                        action="DEMOTE",
                        target=law_record.law.law_id,
                        reason="authenticated world counterevidence invalidated the active representation supporting this bounded law",
                    )
                    self._append_mutation(law_mutation)
                    mutations.append(law_mutation)

        return mutations

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
