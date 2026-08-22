from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import subprocess
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .world_coupling import WorldOutcomePair, WorldOutcomeReceipt


SOFTWARE_REPAIR_MARKER = "software_repair_operator="


_COMPARE_MUTATIONS = {
    ast.Eq: (ast.NotEq, "COMPARE::Eq->NotEq"),
    ast.NotEq: (ast.Eq, "COMPARE::NotEq->Eq"),
    ast.Gt: (ast.GtE, "COMPARE::Gt->GtE"),
    ast.GtE: (ast.Gt, "COMPARE::GtE->Gt"),
    ast.Lt: (ast.LtE, "COMPARE::Lt->LtE"),
    ast.LtE: (ast.Lt, "COMPARE::LtE->Lt"),
}
_BOOL_MUTATIONS = {
    ast.And: (ast.Or, "BOOL::And->Or"),
    ast.Or: (ast.And, "BOOL::Or->And"),
}


@dataclass(frozen=True)
class SoftwarePatchCandidate:
    task_id: str
    source_hash: str
    site_index: int
    operator_id: str
    patched_source: str
    proposal: InterventionProposal


@dataclass(frozen=True)
class SoftwareRepairPolicy:
    status: str
    operator_id: Optional[str]
    supporting_contexts: Tuple[str, ...]
    candidate_operator_count: int
    reason: str


@dataclass(frozen=True)
class SoftwareRepairSelection:
    status: str
    candidates: Tuple[SoftwarePatchCandidate, ...]
    policy_operator_id: Optional[str]
    total_candidate_count: int
    reason: str


class _SingleMutationTransformer(ast.NodeTransformer):
    def __init__(self, target_index: int) -> None:
        self.target_index = int(target_index)
        self.current_index = -1
        self.applied_operator_id: Optional[str] = None

    def _advance(self, operator_id: str) -> bool:
        self.current_index += 1
        if self.current_index == self.target_index:
            self.applied_operator_id = operator_id
            return True
        return False

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        if len(node.ops) != 1:
            return node
        mutation = _COMPARE_MUTATIONS.get(type(node.ops[0]))
        if mutation is None:
            return node
        replacement, operator_id = mutation
        if self._advance(operator_id):
            node.ops[0] = replacement()
        return node

    def visit_BoolOp(self, node: ast.BoolOp):
        self.generic_visit(node)
        mutation = _BOOL_MUTATIONS.get(type(node.op))
        if mutation is None:
            return node
        replacement, operator_id = mutation
        if self._advance(operator_id):
            node.op = replacement()
        return node


class PythonASTRepairGenerator:
    """Generate one-node Python AST repairs without consuming test outcomes.

    The generator intentionally knows only a bounded mutation alphabet. It does not
    inspect hidden tests, expected outputs, or world receipts. Candidates are source-
    dependent exact patches and carry a source-disjoint abstract operator id that can
    later earn cross-task authority from external execution outcomes.
    """

    @staticmethod
    def _site_operator_ids(source: str) -> Tuple[str, ...]:
        tree = ast.parse(source)
        operator_ids: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                mutation = _COMPARE_MUTATIONS.get(type(node.ops[0]))
                if mutation is not None:
                    operator_ids.append(mutation[1])
            elif isinstance(node, ast.BoolOp):
                mutation = _BOOL_MUTATIONS.get(type(node.op))
                if mutation is not None:
                    operator_ids.append(mutation[1])
        return tuple(operator_ids)

    @staticmethod
    def _source_hash(source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def generate(self, task_id: str, source: str) -> Tuple[SoftwarePatchCandidate, ...]:
        source_hash = self._source_hash(source)
        candidates: List[SoftwarePatchCandidate] = []
        for site_index, expected_operator_id in enumerate(self._site_operator_ids(source)):
            tree = ast.parse(source)
            transformer = _SingleMutationTransformer(site_index)
            mutated = transformer.visit(tree)
            ast.fix_missing_locations(mutated)
            if transformer.applied_operator_id != expected_operator_id:
                raise AssertionError("AST repair site identity drifted during regeneration")
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
            experiment_id = f"SOFTWARE_PATCH::{source_hash[:12]}::{site_index}::{digest}"
            proposal = InterventionProposal(
                experiment_id=experiment_id,
                axis_id=f"AXIS::SOFTWARE_REPAIR::{source_hash[:16]}",
                manipulated_variable=expected_operator_id,
                held_fixed=(),
                low_value=0.0,
                high_value=1.0,
                predicted_low_side="BUGGY_SOURCE",
                predicted_high_side="PATCHED_SOURCE",
                reason=(
                    "execute source-disjoint Python AST repair candidate; "
                    f"{SOFTWARE_REPAIR_MARKER}{expected_operator_id} "
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


def parse_software_repair_operator(proposal: InterventionProposal) -> Optional[str]:
    reason = str(proposal.reason)
    if SOFTWARE_REPAIR_MARKER not in reason:
        return None
    tail = reason.split(SOFTWARE_REPAIR_MARKER, 1)[1].strip().split()[0].rstrip(",;)")
    return tail or None


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def derive_software_repair_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> SoftwareRepairPolicy:
    proposal_list = tuple(proposals)
    operator_by_experiment = {
        proposal.experiment_id: parse_software_repair_operator(proposal)
        for proposal in proposal_list
        if parse_software_repair_operator(proposal) is not None
    }
    candidate_operators = tuple(sorted(set(operator_by_experiment.values())))
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
        contexts = support.setdefault(operator_id, {})
        contexts[context_id] = max(contexts.get(context_id, 0.0), float(score))

    eligible = []
    required_contexts = max(1, int(min_contexts))
    for operator_id, contexts in support.items():
        if len(contexts) < required_contexts:
            continue
        mean_score = sum(contexts.values()) / len(contexts)
        eligible.append((-len(contexts), -mean_score, operator_id, tuple(sorted(contexts))))
    eligible.sort()
    if not eligible:
        return SoftwareRepairPolicy(
            status="NO_REPRODUCED_SOFTWARE_REPAIR_OPERATOR",
            operator_id=None,
            supporting_contexts=(),
            candidate_operator_count=len(candidate_operators),
            reason="no abstract AST repair operator has repeated authenticated cross-task success",
        )
    chosen = eligible[0]
    return SoftwareRepairPolicy(
        status="REPRODUCED_SOFTWARE_REPAIR_OPERATOR",
        operator_id=chosen[2],
        supporting_contexts=chosen[3],
        candidate_operator_count=len(candidate_operators),
        reason="source-disjoint AST mutation operator retained by repeated authenticated hidden-test success",
    )


def select_software_patch_candidates(
    candidates: Sequence[SoftwarePatchCandidate],
    policy: Optional[SoftwareRepairPolicy] = None,
    max_candidates: Optional[int] = None,
) -> SoftwareRepairSelection:
    ordered = tuple(candidates)
    policy_operator_id = None
    status = "FULL_SOURCE_DERIVED_REPAIR_SEARCH"
    reason = "no learned repair operator applied; preserve source-derived mutation order"
    if policy is not None and policy.status == "REPRODUCED_SOFTWARE_REPAIR_OPERATOR" and policy.operator_id:
        policy_operator_id = policy.operator_id
        matching = tuple(candidate for candidate in ordered if candidate.operator_id == policy.operator_id)
        nonmatching = tuple(candidate for candidate in ordered if candidate.operator_id != policy.operator_id)
        ordered = matching + nonmatching
        status = "LEARNED_REPAIR_OPERATOR_PRIORITIZED"
        reason = "reproduced source-disjoint repair operator prioritized on fresh source"
    if max_candidates is not None:
        ordered = ordered[: max(0, int(max_candidates))]
    return SoftwareRepairSelection(
        status=status,
        candidates=ordered,
        policy_operator_id=policy_operator_id,
        total_candidate_count=len(candidates),
        reason=reason,
    )


class SubprocessPythonRepairExecutor:
    """Execute baseline/patched Python in an isolated child process under hidden tests."""

    _HARNESS = r'''
import json, sys
payload = json.loads(sys.stdin.read())
namespace = {}
try:
    exec(payload["source"], namespace, namespace)
    function = namespace[payload["function_name"]]
    ok = True
    for case in payload["cases"]:
        try:
            result = function(*case["args"])
        except Exception:
            ok = False
            break
        if result != case["expected"]:
            ok = False
            break
    print(json.dumps({"ok": bool(ok)}))
except Exception:
    print(json.dumps({"ok": False}))
'''

    def __init__(
        self,
        baseline_source: str,
        patched_source: str,
        function_name: str,
        hidden_cases: Sequence[Mapping[str, object]],
        signer,
        source_id: str,
        context_id: str,
        challenge_id: str,
        epoch: int,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.baseline_source = str(baseline_source)
        self.patched_source = str(patched_source)
        self.function_name = str(function_name)
        self.hidden_cases = tuple(dict(case) for case in hidden_cases)
        self.signer = signer
        self.source_id = str(source_id)
        self.context_id = str(context_id)
        self.challenge_id = str(challenge_id)
        self.epoch = int(epoch)
        self.timeout_seconds = max(0.5, float(timeout_seconds))

    def _run(self, source: str) -> float:
        payload = {
            "source": source,
            "function_name": self.function_name,
            "cases": self.hidden_cases,
        }
        try:
            completed = subprocess.run(
                [sys.executable, "-c", self._HARNESS],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                return 0.0
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            return 1.0 if bool(result.get("ok")) else 0.0
        except Exception:
            return 0.0

    def execute(self, proposal: InterventionProposal, arm: str, value: float) -> WorldOutcomeReceipt:
        source = self.baseline_source if str(arm).upper() == "LOW" else self.patched_source
        outcome = self._run(source)
        return self.signer.sign(WorldOutcomeReceipt(
            receipt_id=f"{self.challenge_id}::{proposal.experiment_id}::{arm}",
            experiment_id=proposal.experiment_id,
            axis_id=proposal.axis_id,
            arm=str(arm).upper(),
            intervention_value=float(value),
            outcome=float(outcome),
            source_id=self.source_id,
            context_id=self.context_id,
            challenge_id=self.challenge_id,
            epoch=self.epoch,
            budget_token=f"software-hidden-tests::{self.challenge_id}",
            externally_generated=True,
        ))


class SoftwareTaskAcquisitionOrgan:
    """Stateless source-repair organ backed only by canonical BODY evidence."""

    def __init__(self, body, generator: Optional[PythonASTRepairGenerator] = None) -> None:
        self.body = body
        self.generator = generator or PythonASTRepairGenerator()

    def propose(self, task_id: str, source: str) -> Tuple[SoftwarePatchCandidate, ...]:
        candidates = self.generator.generate(task_id, source)
        for candidate in candidates:
            self.body.memory.remember_experiment(candidate.proposal)
        return candidates

    def policy(self) -> SoftwareRepairPolicy:
        return derive_software_repair_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )

    def select(
        self,
        candidates: Sequence[SoftwarePatchCandidate],
        max_candidates: Optional[int] = None,
        apply_learned_policy: bool = True,
    ) -> SoftwareRepairSelection:
        policy = self.policy() if apply_learned_policy else None
        return select_software_patch_candidates(candidates, policy=policy, max_candidates=max_candidates)
