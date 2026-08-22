from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .software_repair_grammar_expansion import SoftwareRepairAlphabetAssessment
from .software_task_acquisition import SoftwarePatchCandidate
from .world_coupling import WorldOutcomePair


STATE_CONFLICT_REPAIR_MARKER = "software_state_conflict_strategy="
STATE_CONFLICT_STRATEGIES: Tuple[str, ...] = (
    "LAST_WRITE",
    "KEEP_FIRST",
    "ID_ONLY_PROVENANCE_UNION",
    "EXACT_ACTION_PROVENANCE_UNION",
)


@dataclass(frozen=True)
class StateConflictRepairPolicy:
    status: str
    strategy_id: Optional[str]
    supporting_contexts: Tuple[str, ...]
    candidate_strategy_count: int
    reason: str


@dataclass(frozen=True)
class StateConflictRepairSelection:
    status: str
    candidates: Tuple[SoftwarePatchCandidate, ...]
    policy_strategy_id: Optional[str]
    total_candidate_count: int
    reason: str


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def _generated_override(strategy_id: str) -> str:
    strategy = str(strategy_id)
    if strategy not in STATE_CONFLICT_STRATEGIES:
        raise ValueError(f"unknown state-conflict strategy: {strategy}")
    return f'''\n\n# ARTE generated bounded state-conflict repair candidate.\nfrom dataclasses import replace as _arte_state_replace\n\n_ARTE_STATE_STRATEGY = {strategy!r}\n_ARTE_TRANSFORM_MARKER = "generator_transform_programs="\n\ndef _arte_transform_ids(reason):\n    text = str(reason)\n    if _ARTE_TRANSFORM_MARKER not in text:\n        return ()\n    tail = text.split(_ARTE_TRANSFORM_MARKER, 1)[1].strip().split()[0].rstrip(",;)")\n    return tuple(sorted(set(item for item in tail.split("|") if item)))\n\ndef _arte_same_exact_action(left, right):\n    return bool(\n        left.experiment_id == right.experiment_id\n        and left.axis_id == right.axis_id\n        and left.manipulated_variable == right.manipulated_variable\n        and left.held_fixed == right.held_fixed\n        and float(left.low_value) == float(right.low_value)\n        and float(left.high_value) == float(right.high_value)\n        and left.predicted_low_side == right.predicted_low_side\n        and left.predicted_high_side == right.predicted_high_side\n        and left.status == right.status\n    )\n\ndef _arte_union_provenance(existing, incoming):\n    old_ids = _arte_transform_ids(existing.reason)\n    new_ids = _arte_transform_ids(incoming.reason)\n    if not old_ids or not new_ids:\n        return None\n    if _ARTE_STATE_STRATEGY == "EXACT_ACTION_PROVENANCE_UNION" and not _arte_same_exact_action(existing, incoming):\n        return None\n    if _ARTE_STATE_STRATEGY not in ("ID_ONLY_PROVENANCE_UNION", "EXACT_ACTION_PROVENANCE_UNION"):\n        return None\n    merged_ids = tuple(sorted(set(old_ids + new_ids)))\n    prefix = str(existing.reason).split(_ARTE_TRANSFORM_MARKER, 1)[0].rstrip()\n    reason = f"{prefix} {_ARTE_TRANSFORM_MARKER}{'|'.join(merged_ids)}".strip()\n    return _arte_state_replace(existing, reason=reason)\n\ndef _arte_state_conflict_remember_experiment(self, proposal):\n    record = self.experiments.get(proposal.experiment_id)\n    if record is None:\n        record = ExperimentRecord(proposal=proposal)\n        self.experiments[proposal.experiment_id] = record\n        self._append_mutation(RepresentationMutation(\n            mutation_id="PERSIST_EXPERIMENT::" + proposal.experiment_id,\n            action="EXTEND",\n            target=proposal.experiment_id,\n            reason="generated threshold-crossing intervention entered BODY phenotype memory as proposal-only",\n        ))\n        return record\n\n    if record.proposal == proposal:\n        return record\n\n    if _ARTE_STATE_STRATEGY == "KEEP_FIRST":\n        return record\n\n    merged = _arte_union_provenance(record.proposal, proposal)\n    if merged is not None:\n        if merged != record.proposal:\n            record.history.append(record.proposal)\n            record.proposal = merged\n            record.revisions += 1\n            record.status = "PROPOSAL_ONLY"\n            self._append_mutation(RepresentationMutation(\n                mutation_id=f"MERGE_EXPERIMENT_PROVENANCE::{proposal.experiment_id}::{record.revisions}",\n                action="EXTEND",\n                target=proposal.experiment_id,\n                reason="generated state-conflict candidate unioned transform ancestry under its bounded conflict semantics",\n            ))\n        return record\n\n    record.history.append(record.proposal)\n    record.proposal = proposal\n    record.revisions += 1\n    record.status = "PROPOSAL_ONLY"\n    self._append_mutation(RepresentationMutation(\n        mutation_id=f"REVISE_EXPERIMENT::{proposal.experiment_id}::{record.revisions}",\n        action="REVISE",\n        target=proposal.experiment_id,\n        reason="representation revision changed the generated intervention definition",\n    ))\n    return record\n\nEpistemicMemory.remember_experiment = _arte_state_conflict_remember_experiment\n'''


def apply_state_conflict_strategy(historical_source: str, strategy_id: str) -> str:
    source = str(historical_source)
    if "class EpistemicMemory" not in source or "def remember_experiment" not in source:
        raise AssertionError("historical source lacks EpistemicMemory.remember_experiment")
    return source.rstrip() + _generated_override(strategy_id) + "\n"


class PythonStateConflictRepairGenerator:
    """Bounded outcome-independent repair grammar for experiment-memory conflict semantics."""

    def generate(
        self,
        task_id: str,
        historical_source: str,
        assessment: SoftwareRepairAlphabetAssessment,
    ) -> Tuple[SoftwarePatchCandidate, ...]:
        if assessment.status != "SOFTWARE_REPAIR_ALPHABET_FALSIFIED_OPEN_NEXT":
            return ()
        source = str(historical_source)
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        candidates = []
        for strategy_index, strategy_id in enumerate(STATE_CONFLICT_STRATEGIES):
            patched_source = apply_state_conflict_strategy(source, strategy_id)
            operator_id = f"STATE_CONFLICT::{strategy_id}"
            payload = {
                "task_id": str(task_id),
                "source_hash": source_hash,
                "strategy_id": strategy_id,
                "patched_source_hash": hashlib.sha256(patched_source.encode("utf-8")).hexdigest(),
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:20]
            proposal = InterventionProposal(
                experiment_id=f"SOFTWARE_STATE_CONFLICT_PATCH::{source_hash[:12]}::{strategy_id}::{digest}",
                axis_id=f"AXIS::SOFTWARE_STATE_CONFLICT_REPAIR::{source_hash[:16]}",
                manipulated_variable=operator_id,
                held_fixed=(),
                low_value=0.0,
                high_value=1.0,
                predicted_low_side="HISTORICAL_MEMORY_CONFLICT_BUG",
                predicted_high_side="STATE_CONFLICT_REPAIR_CANDIDATE",
                reason=(
                    "execute world-gated state-conflict repair candidate; "
                    f"{STATE_CONFLICT_REPAIR_MARKER}{strategy_id} "
                    f"historical_source_hash={source_hash} strategy_index={strategy_index}"
                ),
                status="PROPOSAL_ONLY",
            )
            candidates.append(SoftwarePatchCandidate(
                task_id=str(task_id),
                source_hash=source_hash,
                site_index=strategy_index,
                operator_id=operator_id,
                patched_source=patched_source,
                proposal=proposal,
            ))
        return tuple(candidates)


def parse_state_conflict_strategy(proposal: InterventionProposal) -> Optional[str]:
    reason = str(proposal.reason)
    if STATE_CONFLICT_REPAIR_MARKER not in reason:
        return None
    tail = reason.split(STATE_CONFLICT_REPAIR_MARKER, 1)[1].strip().split()[0].rstrip(",;)")
    return tail or None


def derive_state_conflict_repair_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> StateConflictRepairPolicy:
    proposal_list = tuple(proposals)
    strategy_by_experiment = {
        proposal.experiment_id: parse_state_conflict_strategy(proposal)
        for proposal in proposal_list
        if parse_state_conflict_strategy(proposal) is not None
    }
    strategy_space = tuple(sorted(set(strategy_by_experiment.values())))
    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair) or pair.experiment_id not in strategy_by_experiment:
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(
            pair.independence_class_id, pair
        )
    minimum_classes = max(1, int(min_independent_classes))
    support: Dict[str, Dict[str, float]] = {}
    for (experiment_id, context_id), by_class in grouped.items():
        if len(by_class) < minimum_classes:
            continue
        score = sum(abs(pair.effect) for pair in by_class.values()) / len(by_class)
        if score < float(strong_effect_threshold):
            continue
        strategy_id = strategy_by_experiment[experiment_id]
        support.setdefault(strategy_id, {})[context_id] = float(score)
    required = max(1, int(min_contexts))
    eligible = []
    for strategy_id, contexts in support.items():
        if len(contexts) < required:
            continue
        mean_score = sum(contexts.values()) / len(contexts)
        eligible.append((-len(contexts), -mean_score, strategy_id, tuple(sorted(contexts))))
    eligible.sort()
    if not eligible:
        return StateConflictRepairPolicy(
            status="NO_REPRODUCED_STATE_CONFLICT_REPAIR_STRATEGY",
            strategy_id=None,
            supporting_contexts=(),
            candidate_strategy_count=len(strategy_space),
            reason="no state-conflict semantics has repeated authenticated cross-context success",
        )
    chosen = eligible[0]
    return StateConflictRepairPolicy(
        status="REPRODUCED_STATE_CONFLICT_REPAIR_STRATEGY",
        strategy_id=chosen[2],
        supporting_contexts=chosen[3],
        candidate_strategy_count=len(strategy_space),
        reason="world-selected state-conflict semantics reproduced across historical regression contexts",
    )


def select_state_conflict_repairs(
    candidates: Sequence[SoftwarePatchCandidate],
    policy: Optional[StateConflictRepairPolicy],
    max_candidates: Optional[int] = None,
) -> StateConflictRepairSelection:
    ordered = tuple(candidates)
    policy_strategy_id = None
    status = "STATE_CONFLICT_SHADOW_SEARCH"
    reason = "no learned state-conflict policy applied"
    if policy is not None and policy.status == "REPRODUCED_STATE_CONFLICT_REPAIR_STRATEGY" and policy.strategy_id:
        policy_strategy_id = policy.strategy_id
        expected_operator = f"STATE_CONFLICT::{policy.strategy_id}"
        matching = tuple(item for item in ordered if item.operator_id == expected_operator)
        nonmatching = tuple(item for item in ordered if item.operator_id != expected_operator)
        ordered = matching + nonmatching
        status = "LEARNED_STATE_CONFLICT_STRATEGY_PRIORITIZED"
        reason = "reproduced state-conflict semantics prioritized on fresh historical regression"
    if max_candidates is not None:
        ordered = ordered[: max(0, int(max_candidates))]
    return StateConflictRepairSelection(
        status=status,
        candidates=ordered,
        policy_strategy_id=policy_strategy_id,
        total_candidate_count=len(candidates),
        reason=reason,
    )


class StateConflictRepairOrgan:
    """Stateless organ deriving state-conflict repair policy from canonical BODY evidence."""

    def __init__(self, body, generator: Optional[PythonStateConflictRepairGenerator] = None) -> None:
        self.body = body
        self.generator = generator or PythonStateConflictRepairGenerator()

    def propose(self, task_id: str, historical_source: str, assessment: SoftwareRepairAlphabetAssessment):
        candidates = self.generator.generate(task_id, historical_source, assessment)
        for candidate in candidates:
            self.body.memory.remember_experiment(candidate.proposal)
        return candidates

    def policy(self):
        return derive_state_conflict_repair_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )

    def select(self, candidates, max_candidates=None, apply_learned_policy=True):
        policy = self.policy() if apply_learned_policy else None
        return select_state_conflict_repairs(candidates, policy, max_candidates=max_candidates)
