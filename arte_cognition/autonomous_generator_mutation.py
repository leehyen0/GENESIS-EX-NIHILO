from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import hashlib
import json

from .executable_morphology import (
    ExperienceUnit,
    MorphologyGenome,
    MorphologyMutation,
    MorphologyMutator,
    MutationLevel,
    PressureVector,
)
from .meta_acceleration import MutationStrategyState
from .morphology_genesis import MorphologyResidual
from .native_recursive_research import NativeMetaMorphologyGenesisEngine
from .native_representation_generator_language import derive_generator_language_mutation
from .self_evolving_body_checkpoint import SelfEvolvingResearchBody


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class MetaFailureCertificate:
    certificate_id: str
    parent_body_hash: str
    failure_layer: str
    more_compute_exhausted: bool
    independent_contexts: Tuple[str, ...]
    prior_generator_language_gain: float
    source_receipt_hashes: Tuple[str, ...]
    current_hidden_task_information_present: bool = False

    def fingerprint(self) -> str:
        return _sha(
            {
                "certificate_id": self.certificate_id,
                "parent_body_hash": self.parent_body_hash,
                "failure_layer": self.failure_layer,
                "more_compute_exhausted": self.more_compute_exhausted,
                "independent_contexts": self.independent_contexts,
                "prior_generator_language_gain": self.prior_generator_language_gain,
                "source_receipt_hashes": self.source_receipt_hashes,
                "current_hidden_task_information_present": self.current_hidden_task_information_present,
            }
        )


@dataclass(frozen=True)
class AutonomousMutationProposal:
    proposal_id: str
    family: str
    mutation: MorphologyMutation
    score: float
    rationale: Tuple[str, ...]


@dataclass(frozen=True)
class AutonomousMutationSelection:
    certificate_fingerprint: str
    proposals: Tuple[AutonomousMutationProposal, ...]
    selected_proposal_id: str
    generation_uses_current_outcomes: bool = False

    @property
    def selected(self) -> AutonomousMutationProposal:
        rows = [row for row in self.proposals if row.proposal_id == self.selected_proposal_id]
        if len(rows) != 1:
            raise ValueError("AUTONOMOUS_SELECTION_ID_NOT_UNIQUE")
        return rows[0]

    def trace_hash(self) -> str:
        return _sha(
            {
                "certificate": self.certificate_fingerprint,
                "selected": self.selected_proposal_id,
                "proposals": [
                    [row.proposal_id, row.family, row.mutation.mutation_id, row.score, list(row.rationale)]
                    for row in self.proposals
                ],
                "generation_uses_current_outcomes": self.generation_uses_current_outcomes,
            }
        )


class AutonomousMetaMutationCompiler:
    """Select a structural proposal from inherited evidence, never current outcomes.

    The compiler is intentionally bounded. It chooses among inherited morphology
    mutation families; it is not source-code self rewriting and confers no external
    authority. Its purpose is to move target-operation selection from the evaluator
    into the persistent research BODY's own typed failure-to-mutation path.
    """

    def _generator_proposal(self, genome: MorphologyGenome, certificate: MetaFailureCertificate) -> AutonomousMutationProposal:
        mutation = derive_generator_language_mutation(
            genome,
            origin_residual_id=certificate.certificate_id,
            failure_fossil=certificate.fingerprint(),
        )
        score = 0.0
        if certificate.failure_layer == "REPRESENTATION_GENERATOR_LANGUAGE":
            score += 2.0
        if certificate.more_compute_exhausted:
            score += 1.0
        score += max(0.0, float(certificate.prior_generator_language_gain))
        return AutonomousMutationProposal(
            proposal_id="AUTO::GENERATOR_LANGUAGE::" + mutation.mutation_id[-12:],
            family="AUTONOMOUS_GENERATOR_LANGUAGE",
            mutation=mutation,
            score=score,
            rationale=(
                f"certificate-layer::{certificate.failure_layer}",
                f"more-compute-exhausted::{certificate.more_compute_exhausted}",
                f"prior-generator-gain::{certificate.prior_generator_language_gain:.6f}",
            ),
        )

    def _mutator_proposal(self, genome: MorphologyGenome, certificate: MetaFailureCertificate) -> AutonomousMutationProposal:
        residual = MorphologyResidual(
            residual_id=certificate.certificate_id,
            pressure=PressureVector(human_dependency=1.0, theory_blindspot=0.5),
            source_refs=tuple(certificate.source_receipt_hashes),
        )
        candidates = NativeMetaMorphologyGenesisEngine(candidate_budget=16).generate(genome, (residual,))
        rows = [row for row in candidates if row.operation_family == "CHANGE_MUTATOR_POLICY"]
        if len(rows) != 1:
            raise ValueError("AUTONOMOUS_MUTATOR_PROPOSAL_NOT_UNIQUE")
        mutation = rows[0].mutation
        score = 2.5 if certificate.failure_layer == "MUTATOR_SEARCH_POLICY" else 0.25
        return AutonomousMutationProposal(
            proposal_id="AUTO::MUTATOR_POLICY::" + mutation.mutation_id[-12:],
            family="AUTONOMOUS_MUTATOR_POLICY",
            mutation=mutation,
            score=score,
            rationale=(f"certificate-layer::{certificate.failure_layer}", "bounded-alternative-proposal"),
        )

    def _abstain_proposal(self, genome: MorphologyGenome, certificate: MetaFailureCertificate) -> AutonomousMutationProposal:
        mutation_id = "AUTO_NOOP::" + _sha((genome.fingerprint(), certificate.fingerprint()))[:20]
        mutation = MorphologyMutation(
            mutation_id=mutation_id,
            level=MutationLevel.STRATEGY,
            operation="SET_EVENT_ORDER",
            payload={"event_order": list(genome.event_order)},
            parent_body_hash=genome.fingerprint(),
            rationale=("fail-closed-abstention-candidate",),
            reversible=True,
        )
        score = 1.0
        if certificate.more_compute_exhausted and len(set(certificate.independent_contexts)) >= 2:
            score = -0.5
        if certificate.failure_layer not in {"REPRESENTATION_GENERATOR_LANGUAGE", "MUTATOR_SEARCH_POLICY"}:
            score = 3.0
        return AutonomousMutationProposal(
            proposal_id="AUTO::ABSTAIN::" + mutation_id[-12:],
            family="AUTONOMOUS_ABSTAIN",
            mutation=mutation,
            score=score,
            rationale=("uncertainty-preserving-null-option",),
        )

    def propose(self, genome: MorphologyGenome, certificate: MetaFailureCertificate) -> AutonomousMutationSelection:
        if certificate.parent_body_hash != genome.fingerprint():
            raise ValueError("AUTONOMOUS_CERTIFICATE_PARENT_MISMATCH")
        if certificate.current_hidden_task_information_present:
            raise ValueError("AUTONOMOUS_CERTIFICATE_CONTAMINATED_BY_CURRENT_HIDDEN_TASK")
        if len(set(certificate.independent_contexts)) < 2:
            raise ValueError("AUTONOMOUS_CERTIFICATE_INSUFFICIENT_CONTEXTS")
        proposals = (
            self._generator_proposal(genome, certificate),
            self._mutator_proposal(genome, certificate),
            self._abstain_proposal(genome, certificate),
        )
        ordered = tuple(sorted(proposals, key=lambda row: (-row.score, row.proposal_id)))
        return AutonomousMutationSelection(
            certificate_fingerprint=certificate.fingerprint(),
            proposals=ordered,
            selected_proposal_id=ordered[0].proposal_id,
            generation_uses_current_outcomes=False,
        )


def apply_autonomous_selection(genome: MorphologyGenome, selection: AutonomousMutationSelection) -> MorphologyGenome:
    if selection.generation_uses_current_outcomes:
        raise ValueError("AUTONOMOUS_SELECTION_OUTCOME_CONTAMINATED")
    return MorphologyMutator().apply(genome, selection.selected.mutation)


def credit_autonomous_selection(
    body: SelfEvolvingResearchBody,
    selection: AutonomousMutationSelection,
    *,
    full_useful_rate: float,
    remove_useful_rate: float,
    wrong_useful_rate: float,
    task_ref: str,
) -> float:
    full = float(full_useful_rate)
    remove = float(remove_useful_rate)
    wrong = float(wrong_useful_rate)
    positive = bool(
        not selection.generation_uses_current_outcomes
        and selection.selected.family != "AUTONOMOUS_ABSTAIN"
        and full > remove
        and full > wrong
    )
    effect = max(0.0, (full - remove) + (full - wrong)) if positive else 0.0

    scores = body.mutation_strategy.score_map()
    support = body.mutation_strategy.support_map()
    family = selection.selected.family
    if effect > 0.0:
        scores[family] = scores.get(family, 0.0) + effect
        support[family] = support.get(family, 0) + 1
    lineage_hash = _sha(
        {
            "previous": body.mutation_strategy.lineage_hash,
            "selection": selection.trace_hash(),
            "effect": effect,
            "full": full,
            "remove": remove,
            "wrong": wrong,
        }
    )
    body.mutation_strategy = MutationStrategyState(
        operation_scores=tuple(sorted((key, float(value)) for key, value in scores.items())),
        operation_support=tuple(sorted((key, int(value)) for key, value in support.items())),
        fossilized_operations=body.mutation_strategy.fossilized_operations,
        lineage_hash=lineage_hash,
    )

    episode_id = "NATIVE_AUTONOMOUS_META_MUTATION::" + selection.trace_hash()[:20]
    episode = ExperienceUnit(
        episode_id=episode_id,
        pre_body_hash=selection.selected.mutation.parent_body_hash,
        source_refs=("native://cycle6-receipt", "native://cycle7-receipt"),
        task_ref=str(task_ref),
        benchmark_family="NATIVE_FRESH_RESEARCH",
        precommitted_hypotheses=("AUTONOMOUS_GENERATOR_MUTATION_CAN_TRANSFER",),
        selected_goal_id=selection.selected.family,
        action_trace_hash=selection.trace_hash(),
        outcome_summary=(
            f"full={full:.6f};remove={remove:.6f};wrong={wrong:.6f};effect={effect:.6f}"
        ),
        success=effect > 0.0,
        uncertainty_before=1.0,
        uncertainty_after=max(0.0, 1.0 - effect / 2.0),
        mutation_ids=(selection.selected.mutation.mutation_id,),
        removal_effect=full - remove,
        wrong_swap_effect=full - wrong,
        heldout_effect=full,
        delayed_replay_equal=None,
        descendant_body_hash=body.morphology.fingerprint(),
        notes=(
            "proposal-selected-before-current-hidden-task-seed",
            "credit-assigned-after-hidden-outcome",
            "source-code-autonomous-self-modification-not-claimed",
        ),
    )
    if not body.experience_archive.append(episode):
        raise ValueError("AUTONOMOUS_CREDIT_DUPLICATE_EXPERIENCE")
    return effect
