from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

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


@dataclass(frozen=True)
class RepresentationMutation:
    mutation_id: str
    action: str
    target: str
    reason: str
    reversible: bool = True


class EpistemicMemory:
    """Persistent but reversible memory for generated representations.

    Concept generation is cheap and remains in shadow. A concept becomes active
    only when a corresponding law survives the configured held-out gate. Later
    world counterexamples demote the law and concept without deleting lineage.
    """

    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptRecord] = {}
        self.laws: Dict[str, LawRecord] = {}
        self.mutation_log: List[RepresentationMutation] = []

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
