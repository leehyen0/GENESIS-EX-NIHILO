from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .software_failure_extractor_program_genesis import ExtractorPatchCandidate
from .world_coupling import WorldOutcomePair


SEMANTIC_DISCRIMINATOR_MARKER = "repair_semantic_discriminator="
SEMANTIC_SELECTION_MARKER = "repair_semantic_selection="

_SELECT_DROP_OPTIONAL = "SELECT_DROP_OPTIONAL_BOUND_POSITIONAL"
_SELECT_DROP_FIRST = "SELECT_DROP_FIRST_POSITIONAL"


@dataclass(frozen=True)
class RepairSemanticDiscriminator:
    selection_rule: str

    @property
    def discriminator_id(self) -> str:
        payload = {"selection_rule": self.selection_rule}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        return f"REPAIR_SEMANTIC_DISCRIMINATOR::{digest}"


@dataclass(frozen=True)
class RepairSemanticDiscriminatorProposal:
    discriminator: RepairSemanticDiscriminator
    proposal: InterventionProposal


@dataclass(frozen=True)
class RepairSemanticDiscriminatorPolicy:
    status: str
    discriminator_id: Optional[str]
    supporting_contexts: Tuple[str, ...]
    candidate_discriminator_count: int
    reason: str


@dataclass(frozen=True)
class CallBindingObservation:
    candidate_id: str
    positional_fingerprints: Tuple[str, ...]
    keyword_fingerprints: Tuple[Tuple[str, str], ...]
    dropped_positional_index: Optional[int]


def generate_repair_semantic_discriminators() -> Tuple[RepairSemanticDiscriminator, ...]:
    """Generate a tiny outcome-independent semantic-selection shadow language.

    Neither rule contains API names, exception names, argument values, test outputs,
    later human fixes, or concrete argument ordinals. The useful rule is structural:
    prefer removing a positional binding whose callable parameter has a default.
    The control rule prefers the first positional edit. External semantic outcomes,
    not this generator, decide which rule can enter BODY authority.
    """
    return (
        RepairSemanticDiscriminator(selection_rule=_SELECT_DROP_OPTIONAL),
        RepairSemanticDiscriminator(selection_rule=_SELECT_DROP_FIRST),
    )


def propose_repair_semantic_discriminator(
    discriminator: RepairSemanticDiscriminator,
) -> RepairSemanticDiscriminatorProposal:
    digest = hashlib.sha256(discriminator.discriminator_id.encode("utf-8")).hexdigest()[:20]
    proposal = InterventionProposal(
        experiment_id=f"SOFTWARE_REPAIR_SEMANTIC_DISCRIMINATOR::{digest}",
        axis_id=f"AXIS::SOFTWARE_REPAIR_SEMANTICS::{digest}",
        manipulated_variable=discriminator.discriminator_id,
        held_fixed=(),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="EXECUTABLE_PATCH_AMBIGUITY_REMAINS",
        predicted_high_side="SEMANTICALLY_DISCRIMINATED_REPAIR",
        reason=(
            "outcome-independent repair semantic discriminator proposal; "
            f"{SEMANTIC_DISCRIMINATOR_MARKER}{discriminator.discriminator_id} "
            f"{SEMANTIC_SELECTION_MARKER}{discriminator.selection_rule}"
        ),
        status="PROPOSAL_ONLY",
    )
    return RepairSemanticDiscriminatorProposal(discriminator=discriminator, proposal=proposal)


def _edit_index(candidate: ExtractorPatchCandidate) -> Optional[int]:
    parts = str(candidate.edit_id).split("::")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def select_patch_candidate(
    discriminator: RepairSemanticDiscriminator,
    candidates: Sequence[ExtractorPatchCandidate],
    parameter_optional_by_position: Sequence[bool],
) -> Optional[ExtractorPatchCandidate]:
    indexed = []
    for candidate in candidates:
        index = _edit_index(candidate)
        if index is not None:
            indexed.append((index, candidate))
    indexed.sort(key=lambda item: (item[0], item[1].edit_id))
    if discriminator.selection_rule == _SELECT_DROP_FIRST:
        return indexed[0][1] if indexed else None
    if discriminator.selection_rule != _SELECT_DROP_OPTIONAL:
        return None
    eligible = [
        candidate
        for index, candidate in indexed
        if 0 <= index < len(parameter_optional_by_position)
        and bool(parameter_optional_by_position[index])
    ]
    return eligible[0] if len(eligible) == 1 else None


def required_binding_preserved(
    baseline_bindings: Mapping[str, str],
    candidate_bindings: Mapping[str, str],
    required_parameter_names: Sequence[str],
) -> bool:
    required = tuple(str(name) for name in required_parameter_names)
    return bool(required) and all(
        name in baseline_bindings
        and name in candidate_bindings
        and str(candidate_bindings[name]) == str(baseline_bindings[name])
        for name in required
    )


def _parse_discriminator_id(proposal: InterventionProposal) -> Optional[str]:
    reason = str(proposal.reason)
    if SEMANTIC_DISCRIMINATOR_MARKER not in reason:
        return None
    return reason.split(SEMANTIC_DISCRIMINATOR_MARKER, 1)[1].strip().split()[0].rstrip(",;)") or None


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def derive_repair_semantic_discriminator_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> RepairSemanticDiscriminatorPolicy:
    discriminator_by_experiment: Dict[str, str] = {}
    all_ids = set()
    for proposal in proposals:
        discriminator_id = _parse_discriminator_id(proposal)
        if discriminator_id:
            discriminator_by_experiment[proposal.experiment_id] = discriminator_id
            all_ids.add(discriminator_id)
    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if pair.experiment_id not in discriminator_by_experiment or not _authoritative(pair):
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(
            pair.independence_class_id, pair
        )
    required_classes = max(1, int(min_independent_classes))
    support: Dict[str, Dict[str, float]] = {}
    for (experiment_id, context_id), by_class in grouped.items():
        if len(by_class) < required_classes:
            continue
        score = sum(max(0.0, float(pair.effect)) for pair in by_class.values()) / len(by_class)
        if score < float(strong_effect_threshold):
            continue
        discriminator_id = discriminator_by_experiment[experiment_id]
        support.setdefault(discriminator_id, {})[context_id] = score
    eligible = []
    for discriminator_id, contexts in support.items():
        if len(contexts) < max(1, int(min_contexts)):
            continue
        mean_score = sum(contexts.values()) / len(contexts)
        eligible.append((-len(contexts), -mean_score, discriminator_id, tuple(sorted(contexts))))
    eligible.sort()
    if not eligible:
        return RepairSemanticDiscriminatorPolicy(
            status="NO_REPRODUCED_REPAIR_SEMANTIC_DISCRIMINATOR",
            discriminator_id=None,
            supporting_contexts=(),
            candidate_discriminator_count=len(all_ids),
            reason="no semantic repair discriminator has repeated verifier-derived required-binding-preserving success",
        )
    chosen = eligible[0]
    return RepairSemanticDiscriminatorPolicy(
        status="REPRODUCED_REPAIR_SEMANTIC_DISCRIMINATOR",
        discriminator_id=chosen[2],
        supporting_contexts=chosen[3],
        candidate_discriminator_count=len(all_ids),
        reason="repair semantic discriminator retained by repeated external required-binding-preserving executable outcomes",
    )


def select_authorized_repair_semantic_discriminator(
    discriminators: Sequence[RepairSemanticDiscriminator],
    policy: RepairSemanticDiscriminatorPolicy,
) -> Optional[RepairSemanticDiscriminator]:
    if policy.status != "REPRODUCED_REPAIR_SEMANTIC_DISCRIMINATOR" or not policy.discriminator_id:
        return None
    return next(
        (item for item in discriminators if item.discriminator_id == policy.discriminator_id),
        None,
    )


class RepairSemanticDiscriminatorOrgan:
    def __init__(self, body) -> None:
        self.body = body

    def remember(self, proposals: Sequence[RepairSemanticDiscriminatorProposal]) -> None:
        for item in proposals:
            self.body.memory.remember_experiment(item.proposal)

    def policy(self) -> RepairSemanticDiscriminatorPolicy:
        return derive_repair_semantic_discriminator_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )
