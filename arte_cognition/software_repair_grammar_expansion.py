from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .software_task_acquisition import SoftwarePatchCandidate
from .world_coupling import WorldOutcomePair


GENERATED_REPAIR_MARKER = "software_generated_repair_operator="


_BINOP_MUTATIONS = {
    ast.Add: (ast.Sub, "BINOP::Add->Sub"),
    ast.Sub: (ast.Add, "BINOP::Sub->Add"),
    ast.Mult: (ast.Add, "BINOP::Mult->Add"),
}


@dataclass(frozen=True)
class SoftwareRepairAlphabetAssessment:
    status: str
    complete_contexts: Tuple[str, ...]
    falsified_contexts: Tuple[str, ...]
    supported_contexts: Tuple[str, ...]
    missing_experiment_ids: Tuple[str, ...]
    evaluated_candidate_count: int
    reason: str


@dataclass(frozen=True)
class GeneratedSoftwareRepairPolicy:
    status: str
    operator_id: Optional[str]
    supporting_contexts: Tuple[str, ...]
    candidate_operator_count: int
    reason: str


@dataclass(frozen=True)
class GeneratedSoftwareRepairSelection:
    status: str
    candidates: Tuple[SoftwarePatchCandidate, ...]
    policy_operator_id: Optional[str]
    total_candidate_count: int
    reason: str


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def assess_software_repair_alphabet_failure(
    old_candidates_by_context: Mapping[str, Sequence[SoftwarePatchCandidate]],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> SoftwareRepairAlphabetAssessment:
    """Require complete external failure of the current source-derived repair alphabet.

    Absence never counts as refutation. Every old candidate in each counted context
    must have enough verifier-derived independent world pairs. Any strong current-
    alphabet patch blocks grammar expansion.
    """
    minimum_classes = max(1, int(min_independent_classes))
    pairs_by_key: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair):
            continue
        pairs_by_key.setdefault((pair.context_id, pair.experiment_id), {}).setdefault(
            pair.independence_class_id, pair
        )

    complete_contexts: List[str] = []
    falsified_contexts: List[str] = []
    supported_contexts: List[str] = []
    missing: List[str] = []
    evaluated = 0
    for context_id, candidates in old_candidates_by_context.items():
        if not candidates:
            continue
        context_complete = True
        context_supported = False
        for candidate in candidates:
            by_class = pairs_by_key.get((str(context_id), candidate.proposal.experiment_id), {})
            if len(by_class) < minimum_classes:
                context_complete = False
                missing.append(candidate.proposal.experiment_id)
                continue
            evaluated += 1
            score = sum(abs(pair.effect) for pair in by_class.values()) / len(by_class)
            if score >= float(strong_effect_threshold):
                context_supported = True
        if context_complete:
            complete_contexts.append(str(context_id))
            if context_supported:
                supported_contexts.append(str(context_id))
            else:
                falsified_contexts.append(str(context_id))

    if supported_contexts:
        return SoftwareRepairAlphabetAssessment(
            status="CURRENT_SOFTWARE_REPAIR_ALPHABET_RETAINS_SUPPORTED_PATCH",
            complete_contexts=tuple(sorted(complete_contexts)),
            falsified_contexts=tuple(sorted(falsified_contexts)),
            supported_contexts=tuple(sorted(supported_contexts)),
            missing_experiment_ids=tuple(sorted(set(missing))),
            evaluated_candidate_count=evaluated,
            reason="at least one completely evaluated current-alphabet context contains a strong repair",
        )
    if len(falsified_contexts) < max(1, int(min_contexts)):
        return SoftwareRepairAlphabetAssessment(
            status="INSUFFICIENT_COMPLETE_SOFTWARE_REPAIR_ALPHABET_FAILURE",
            complete_contexts=tuple(sorted(complete_contexts)),
            falsified_contexts=tuple(sorted(falsified_contexts)),
            supported_contexts=(),
            missing_experiment_ids=tuple(sorted(set(missing))),
            evaluated_candidate_count=evaluated,
            reason="grammar expansion requires repeated complete current-alphabet failure; missing candidates are not refutations",
        )
    return SoftwareRepairAlphabetAssessment(
        status="SOFTWARE_REPAIR_ALPHABET_FALSIFIED_OPEN_NEXT",
        complete_contexts=tuple(sorted(complete_contexts)),
        falsified_contexts=tuple(sorted(falsified_contexts)),
        supported_contexts=(),
        missing_experiment_ids=tuple(sorted(set(missing))),
        evaluated_candidate_count=evaluated,
        reason="all current-alphabet candidates were independently executed and weak in repeated source-disjoint contexts",
    )


class _ArithmeticMutationTransformer(ast.NodeTransformer):
    def __init__(self, target_index: int) -> None:
        self.target_index = int(target_index)
        self.current_index = -1
        self.applied_operator_id: Optional[str] = None

    def visit_BinOp(self, node: ast.BinOp):
        mutation = _BINOP_MUTATIONS.get(type(node.op))
        if mutation is not None:
            self.current_index += 1
            if self.current_index == self.target_index:
                replacement, operator_id = mutation
                node.op = replacement()
                self.applied_operator_id = operator_id
        self.generic_visit(node)
        return node


class PythonArithmeticRepairGenerator:
    """Outcome-independent next repair grammar over arithmetic AST operators."""

    @staticmethod
    def _operator_ids(source: str) -> Tuple[str, ...]:
        ids: List[str] = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.BinOp):
                mutation = _BINOP_MUTATIONS.get(type(node.op))
                if mutation is not None:
                    ids.append(mutation[1])
        return tuple(ids)

    def generate(
        self,
        task_id: str,
        source: str,
        assessment: SoftwareRepairAlphabetAssessment,
    ) -> Tuple[SoftwarePatchCandidate, ...]:
        if assessment.status != "SOFTWARE_REPAIR_ALPHABET_FALSIFIED_OPEN_NEXT":
            return ()
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        candidates: List[SoftwarePatchCandidate] = []
        for site_index, expected_operator_id in enumerate(self._operator_ids(source)):
            tree = ast.parse(source)
            transformer = _ArithmeticMutationTransformer(site_index)
            mutated = transformer.visit(tree)
            ast.fix_missing_locations(mutated)
            if transformer.applied_operator_id != expected_operator_id:
                raise AssertionError("arithmetic repair site identity drifted")
            patched_source = ast.unparse(mutated) + "\n"
            payload = {
                "task_id": str(task_id),
                "source_hash": source_hash,
                "site_index": int(site_index),
                "operator_id": expected_operator_id,
                "patched_source": patched_source,
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:20]
            proposal = InterventionProposal(
                experiment_id=f"SOFTWARE_GENERATED_PATCH::{source_hash[:12]}::{site_index}::{digest}",
                axis_id=f"AXIS::SOFTWARE_REPAIR_EXPANDED::{source_hash[:16]}",
                manipulated_variable=expected_operator_id,
                held_fixed=(),
                low_value=0.0,
                high_value=1.0,
                predicted_low_side="BUGGY_SOURCE",
                predicted_high_side="PATCHED_SOURCE",
                reason=(
                    "execute world-gated expanded Python AST repair candidate; "
                    f"{GENERATED_REPAIR_MARKER}{expected_operator_id} "
                    f"source_hash={source_hash} site_index={site_index}"
                ),
                status="PROPOSAL_ONLY",
            )
            candidates.append(SoftwarePatchCandidate(
                task_id=str(task_id),
                source_hash=source_hash,
                site_index=site_index,
                operator_id=expected_operator_id,
                patched_source=patched_source,
                proposal=proposal,
            ))
        return tuple(candidates)


def parse_generated_repair_operator(proposal: InterventionProposal) -> Optional[str]:
    reason = str(proposal.reason)
    if GENERATED_REPAIR_MARKER not in reason:
        return None
    tail = reason.split(GENERATED_REPAIR_MARKER, 1)[1].strip().split()[0].rstrip(",;)")
    return tail or None


def derive_generated_software_repair_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> GeneratedSoftwareRepairPolicy:
    proposal_list = tuple(proposals)
    operator_by_experiment = {
        proposal.experiment_id: parse_generated_repair_operator(proposal)
        for proposal in proposal_list
        if parse_generated_repair_operator(proposal) is not None
    }
    operator_space = tuple(sorted(set(operator_by_experiment.values())))
    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair) or pair.experiment_id not in operator_by_experiment:
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
        operator_id = operator_by_experiment[experiment_id]
        support.setdefault(operator_id, {})[context_id] = float(score)
    eligible = []
    required = max(1, int(min_contexts))
    for operator_id, contexts in support.items():
        if len(contexts) < required:
            continue
        mean_score = sum(contexts.values()) / len(contexts)
        eligible.append((-len(contexts), -mean_score, operator_id, tuple(sorted(contexts))))
    eligible.sort()
    if not eligible:
        return GeneratedSoftwareRepairPolicy(
            status="NO_REPRODUCED_EXPANDED_SOFTWARE_REPAIR_OPERATOR",
            operator_id=None,
            supporting_contexts=(),
            candidate_operator_count=len(operator_space),
            reason="no expanded repair operator has repeated authenticated cross-task success",
        )
    chosen = eligible[0]
    return GeneratedSoftwareRepairPolicy(
        status="REPRODUCED_EXPANDED_SOFTWARE_REPAIR_OPERATOR",
        operator_id=chosen[2],
        supporting_contexts=chosen[3],
        candidate_operator_count=len(operator_space),
        reason="world-selected arithmetic AST repair operator reproduced across source-disjoint tasks",
    )


def select_generated_software_repairs(
    candidates: Sequence[SoftwarePatchCandidate],
    policy: Optional[GeneratedSoftwareRepairPolicy],
    max_candidates: Optional[int] = None,
) -> GeneratedSoftwareRepairSelection:
    ordered = tuple(candidates)
    policy_operator_id = None
    status = "EXPANDED_REPAIR_SHADOW_SEARCH"
    reason = "no learned expanded repair policy applied"
    if policy is not None and policy.status == "REPRODUCED_EXPANDED_SOFTWARE_REPAIR_OPERATOR" and policy.operator_id:
        policy_operator_id = policy.operator_id
        matching = tuple(item for item in ordered if item.operator_id == policy.operator_id)
        nonmatching = tuple(item for item in ordered if item.operator_id != policy.operator_id)
        ordered = matching + nonmatching
        status = "LEARNED_EXPANDED_REPAIR_OPERATOR_PRIORITIZED"
        reason = "reproduced expanded repair operator prioritized on a fresh source"
    if max_candidates is not None:
        ordered = ordered[: max(0, int(max_candidates))]
    return GeneratedSoftwareRepairSelection(
        status=status,
        candidates=ordered,
        policy_operator_id=policy_operator_id,
        total_candidate_count=len(candidates),
        reason=reason,
    )


class SoftwareRepairGrammarExpansionOrgan:
    """Stateless organ deriving repair-grammar expansion from canonical BODY evidence."""

    def __init__(self, body, generator: Optional[PythonArithmeticRepairGenerator] = None) -> None:
        self.body = body
        self.generator = generator or PythonArithmeticRepairGenerator()

    def assess_old_alphabet(self, old_candidates_by_context):
        return assess_software_repair_alphabet_failure(
            old_candidates_by_context=old_candidates_by_context,
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )

    def propose(self, task_id: str, source: str, old_candidates_by_context):
        assessment = self.assess_old_alphabet(old_candidates_by_context)
        candidates = self.generator.generate(task_id, source, assessment)
        for candidate in candidates:
            self.body.memory.remember_experiment(candidate.proposal)
        return candidates

    def policy(self):
        return derive_generated_software_repair_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )

    def select(self, candidates, max_candidates=None, apply_learned_policy=True):
        policy = self.policy() if apply_learned_policy else None
        return select_generated_software_repairs(candidates, policy, max_candidates=max_candidates)
