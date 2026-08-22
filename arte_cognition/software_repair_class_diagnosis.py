from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .software_repair_grammar_expansion import PythonArithmeticRepairGenerator, SoftwareRepairAlphabetAssessment
from .software_task_acquisition import PythonASTRepairGenerator
from .world_coupling import WorldOutcomePair


REPAIR_CLASSES: Tuple[str, ...] = ("CONTENT", "TRAVERSAL", "STATE_CONFLICT")
REPAIR_CLASS_MARKER = "software_repair_class="
DIAGNOSIS_FINGERPRINT_MARKER = "diagnosis_fingerprint="


@dataclass(frozen=True)
class RepairClassApplicability:
    class_id: str
    status: str
    candidate_count: int
    reason: str


@dataclass(frozen=True)
class RepairClassCandidate:
    class_id: str
    diagnosis_fingerprint: str
    proposal: InterventionProposal


@dataclass(frozen=True)
class RepairClassDiagnosisPolicy:
    status: str
    diagnosis_fingerprint: str
    class_id: Optional[str]
    supporting_contexts: Tuple[str, ...]
    candidate_class_count: int
    reason: str


@dataclass(frozen=True)
class RepairClassSelection:
    status: str
    class_ids: Tuple[str, ...]
    learned_class_id: Optional[str]
    diagnosis_fingerprint: str
    reason: str


def _permissive_content_assessment() -> SoftwareRepairAlphabetAssessment:
    return SoftwareRepairAlphabetAssessment(
        status="SOFTWARE_REPAIR_ALPHABET_FALSIFIED_OPEN_NEXT",
        complete_contexts=(),
        falsified_contexts=(),
        supported_contexts=(),
        missing_experiment_ids=(),
        evaluated_candidate_count=0,
        reason="enumerate existing bounded arithmetic repair candidates for source applicability only",
    )


def _has_named_method(tree: ast.AST, class_name: str, method_name: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        if any(isinstance(item, ast.FunctionDef) and item.name == method_name for item in node.body):
            return True
    return False


def assess_repair_class_applicability(source: str) -> Mapping[str, RepairClassApplicability]:
    text = str(source)
    tree = ast.parse(text)
    base = PythonASTRepairGenerator().generate("repair-class-applicability", text)
    arithmetic = PythonArithmeticRepairGenerator().generate(
        "repair-class-applicability-arithmetic", text, _permissive_content_assessment()
    )
    content_count = len({item.proposal.experiment_id for item in (*base, *arithmetic)})
    traversal = _has_named_method(tree, "PythonASTRepairGenerator", "_site_operator_ids")
    state_conflict = _has_named_method(tree, "EpistemicMemory", "remember_experiment")
    return {
        "CONTENT": RepairClassApplicability(
            class_id="CONTENT",
            status="APPLICABLE" if content_count else "INAPPLICABLE",
            candidate_count=content_count,
            reason="existing comparator/boolean/arithmetic source-derived candidates exist" if content_count else "no existing content mutation sites",
        ),
        "TRAVERSAL": RepairClassApplicability(
            class_id="TRAVERSAL",
            status="APPLICABLE" if traversal else "INAPPLICABLE",
            candidate_count=4 if traversal else 0,
            reason="bounded traversal-strategy target exists" if traversal else "required traversal target absent",
        ),
        "STATE_CONFLICT": RepairClassApplicability(
            class_id="STATE_CONFLICT",
            status="APPLICABLE" if state_conflict else "INAPPLICABLE",
            candidate_count=4 if state_conflict else 0,
            reason="bounded experiment-memory conflict target exists" if state_conflict else "required state-conflict target absent",
        ),
    }


def normalized_structural_fingerprint(source: str) -> str:
    """Identifier/literal/path-free bounded source-family fingerprint.

    This intentionally does not claim source-disjoint diagnosis. It retains AST node
    type counts, parent->child type-edge counts, top-level statement types, and the
    source-derived applicability vector while erasing names, literal values, source
    locations, filenames, and raw source hashes.
    """
    tree = ast.parse(str(source))
    node_counts = Counter(type(node).__name__ for node in ast.walk(tree))
    edge_counts: Counter[str] = Counter()
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            edge_counts[f"{type(parent).__name__}>{type(child).__name__}"] += 1
    top_level = tuple(type(node).__name__ for node in getattr(tree, "body", ()))
    applicability = assess_repair_class_applicability(source)
    payload = {
        "node_counts": tuple(sorted(node_counts.items())),
        "edge_counts": tuple(sorted(edge_counts.items())),
        "top_level": top_level,
        "applicability": tuple(
            (class_id, applicability[class_id].status, applicability[class_id].candidate_count)
            for class_id in REPAIR_CLASSES
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


def generate_repair_class_candidates(source: str) -> Tuple[RepairClassCandidate, ...]:
    fingerprint = normalized_structural_fingerprint(source)
    applicability = assess_repair_class_applicability(source)
    candidates = []
    for class_id in REPAIR_CLASSES:
        state = applicability[class_id]
        if state.status != "APPLICABLE":
            continue
        digest = hashlib.sha256(f"{fingerprint}|{class_id}".encode("utf-8")).hexdigest()[:20]
        proposal = InterventionProposal(
            experiment_id=f"SOFTWARE_REPAIR_CLASS::{fingerprint}::{class_id}::{digest}",
            axis_id=f"AXIS::SOFTWARE_REPAIR_CLASS_DIAGNOSIS::{fingerprint}",
            manipulated_variable=f"REPAIR_CLASS::{class_id}",
            held_fixed=(),
            low_value=0.0,
            high_value=1.0,
            predicted_low_side="BASELINE_FAILURE",
            predicted_high_side="REPAIR_CLASS_SEARCH",
            reason=(
                "execute one bounded repair-class search; "
                f"{REPAIR_CLASS_MARKER}{class_id} "
                f"{DIAGNOSIS_FINGERPRINT_MARKER}{fingerprint}"
            ),
            status="PROPOSAL_ONLY",
        )
        candidates.append(RepairClassCandidate(class_id=class_id, diagnosis_fingerprint=fingerprint, proposal=proposal))
    return tuple(candidates)


def parse_repair_class_candidate(proposal: InterventionProposal) -> Tuple[Optional[str], Optional[str]]:
    reason = str(proposal.reason)
    if REPAIR_CLASS_MARKER not in reason or DIAGNOSIS_FINGERPRINT_MARKER not in reason:
        return None, None
    class_id = reason.split(REPAIR_CLASS_MARKER, 1)[1].strip().split()[0].rstrip(",;)")
    fingerprint = reason.split(DIAGNOSIS_FINGERPRINT_MARKER, 1)[1].strip().split()[0].rstrip(",;)")
    if class_id not in REPAIR_CLASSES:
        return None, None
    return class_id, fingerprint or None


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def derive_repair_class_diagnosis_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    diagnosis_fingerprint: str,
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> RepairClassDiagnosisPolicy:
    target_fp = str(diagnosis_fingerprint)
    class_by_experiment: Dict[str, str] = {}
    for proposal in proposals:
        class_id, fingerprint = parse_repair_class_candidate(proposal)
        if class_id is not None and fingerprint == target_fp:
            class_by_experiment[proposal.experiment_id] = class_id
    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair) or pair.experiment_id not in class_by_experiment:
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(pair.independence_class_id, pair)
    required_classes = max(1, int(min_independent_classes))
    support: Dict[str, Dict[str, float]] = {}
    for (experiment_id, context_id), by_class in grouped.items():
        if len(by_class) < required_classes:
            continue
        score = sum(abs(pair.effect) for pair in by_class.values()) / len(by_class)
        if score < float(strong_effect_threshold):
            continue
        class_id = class_by_experiment[experiment_id]
        support.setdefault(class_id, {})[context_id] = float(score)
    required_contexts = max(1, int(min_contexts))
    eligible = []
    for class_id, contexts in support.items():
        if len(contexts) < required_contexts:
            continue
        eligible.append((-len(contexts), -sum(contexts.values()) / len(contexts), class_id, tuple(sorted(contexts))))
    eligible.sort()
    if not eligible:
        return RepairClassDiagnosisPolicy(
            status="NO_REPRODUCED_REPAIR_CLASS_DIAGNOSIS",
            diagnosis_fingerprint=target_fp,
            class_id=None,
            supporting_contexts=(),
            candidate_class_count=len(set(class_by_experiment.values())),
            reason="no repair class has repeated authenticated capability for this structural fingerprint",
        )
    chosen = eligible[0]
    return RepairClassDiagnosisPolicy(
        status="REPRODUCED_REPAIR_CLASS_DIAGNOSIS",
        diagnosis_fingerprint=target_fp,
        class_id=chosen[2],
        supporting_contexts=chosen[3],
        candidate_class_count=len(set(class_by_experiment.values())),
        reason="world outcomes reproduced one repair class for this bounded structural failure family",
    )


def select_repair_classes(
    candidates: Sequence[RepairClassCandidate],
    policy: Optional[RepairClassDiagnosisPolicy],
    max_classes: Optional[int] = None,
) -> RepairClassSelection:
    ordered = tuple(candidates)
    fingerprint = ordered[0].diagnosis_fingerprint if ordered else ""
    learned = None
    status = "REPAIR_CLASS_SHADOW_SEARCH"
    reason = "no learned repair-class diagnosis applied"
    if policy is not None and policy.status == "REPRODUCED_REPAIR_CLASS_DIAGNOSIS" and policy.class_id:
        if policy.diagnosis_fingerprint == fingerprint:
            learned = policy.class_id
            matching = tuple(item for item in ordered if item.class_id == learned)
            nonmatching = tuple(item for item in ordered if item.class_id != learned)
            ordered = matching + nonmatching
            status = "LEARNED_REPAIR_CLASS_PRIORITIZED"
            reason = "reverified world evidence prioritized one repair class for the matching structural fingerprint"
    if max_classes is not None:
        ordered = ordered[: max(0, int(max_classes))]
    return RepairClassSelection(
        status=status,
        class_ids=tuple(item.class_id for item in ordered),
        learned_class_id=learned,
        diagnosis_fingerprint=fingerprint,
        reason=reason,
    )


class RepairClassDiagnosisOrgan:
    """Stateless BODY view over repair-class proposals and externally reverified outcomes."""

    def __init__(self, body) -> None:
        self.body = body

    def propose(self, source: str) -> Tuple[RepairClassCandidate, ...]:
        candidates = generate_repair_class_candidates(source)
        for candidate in candidates:
            self.body.memory.remember_experiment(candidate.proposal)
        return candidates

    def policy(self, source: str) -> RepairClassDiagnosisPolicy:
        fingerprint = normalized_structural_fingerprint(source)
        return derive_repair_class_diagnosis_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            diagnosis_fingerprint=fingerprint,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )

    def select(self, source: str, max_classes: Optional[int] = None, apply_learned_policy: bool = True) -> RepairClassSelection:
        candidates = generate_repair_class_candidates(source)
        policy = self.policy(source) if apply_learned_policy else None
        return select_repair_classes(candidates, policy, max_classes=max_classes)
