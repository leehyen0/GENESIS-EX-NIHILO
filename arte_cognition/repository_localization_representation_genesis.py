from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import hashlib
import json
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .repository_task_acquisition import (
    RepositoryPatchCandidate,
    PythonRepositoryRepairGenerator,
    REPOSITORY_FILE_ROLE_MARKER,
    REPOSITORY_REPAIR_OPERATOR_MARKER,
)
from .world_coupling import WorldOutcomePair


GRAPH_FINGERPRINT_MARKER = "repository_graph_fingerprint="
GRAPH_FINGERPRINT_DEPTH_MARKER = "repository_graph_fingerprint_depth="


@dataclass(frozen=True)
class LocalizationLanguageAssessment:
    status: str
    ambiguous_contexts: Tuple[str, ...]
    complete_contexts: Tuple[str, ...]
    missing_experiment_ids: Tuple[str, ...]
    evaluated_candidate_count: int
    reason: str


@dataclass(frozen=True)
class RepositoryGraphLocalizationPolicy:
    status: str
    fingerprint: Optional[str]
    operator_id: Optional[str]
    fingerprint_depth: Optional[int]
    supporting_contexts: Tuple[str, ...]
    candidate_signature_count: int
    reason: str


@dataclass(frozen=True)
class RepositoryGraphLocalizationSelection:
    status: str
    candidates: Tuple[RepositoryPatchCandidate, ...]
    policy_fingerprint: Optional[str]
    policy_operator_id: Optional[str]
    policy_fingerprint_depth: Optional[int]
    total_candidate_count: int
    reason: str


def _module_name(path: str) -> str:
    clean = str(path).replace("\\", "/")
    if clean.endswith(".py"):
        clean = clean[:-3]
    return clean.replace("/", ".")


def _import_graph(files: Mapping[str, str]) -> Tuple[Dict[str, set[str]], Dict[str, set[str]]]:
    python_files = {str(path): str(source) for path, source in files.items() if str(path).endswith(".py")}
    module_to_path = {_module_name(path): path for path in python_files}
    outgoing: Dict[str, set[str]] = {path: set() for path in python_files}
    incoming: Dict[str, set[str]] = {path: set() for path in python_files}

    for path, source in python_files.items():
        tree = ast.parse(source)
        targets: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                targets.add(str(node.module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    targets.add(str(alias.name))
        for target in targets:
            resolved = module_to_path.get(target)
            if resolved is None and "." in target:
                resolved = module_to_path.get(target.split(".", 1)[0])
            if resolved is None or resolved == path:
                continue
            outgoing[path].add(resolved)
            incoming[resolved].add(path)
    return outgoing, incoming


def _digest_structural_label(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]


def derive_import_graph_fingerprints(files: Mapping[str, str], depth: int) -> Dict[str, str]:
    """Generate filename-independent Weisfeiler-Lehman-style import-graph fingerprints.

    Initial labels contain only in/out degree. Each refinement round replaces a
    node label with a digest of its current label and the multisets of incoming and
    outgoing neighbor labels. Concrete module names and hidden outcomes never enter
    the representation.
    """
    outgoing, incoming = _import_graph(files)
    labels = {
        path: _digest_structural_label(["DEGREE", len(incoming[path]), len(outgoing[path])])
        for path in sorted(outgoing)
    }
    for _ in range(max(0, int(depth))):
        previous = dict(labels)
        labels = {
            path: _digest_structural_label([
                "WL",
                previous[path],
                sorted(previous[item] for item in incoming[path]),
                sorted(previous[item] for item in outgoing[path]),
            ])
            for path in sorted(outgoing)
        }
    return labels


def minimal_role_collision_escape_depth(
    files: Mapping[str, str],
    candidates: Sequence[RepositoryPatchCandidate],
    max_depth: int = 4,
) -> Optional[int]:
    """Find the shallowest graph refinement that separates duplicate old signatures.

    This search sees source structure and the old `(file_role, operator)` vocabulary
    only. It does not inspect hidden test outcomes.
    """
    groups: Dict[Tuple[str, str], list[RepositoryPatchCandidate]] = {}
    for candidate in candidates:
        groups.setdefault((candidate.file_role, candidate.operator_id), []).append(candidate)
    collisions = [tuple(items) for items in groups.values() if len(items) > 1]
    if not collisions:
        return 0
    for depth in range(max(0, int(max_depth)) + 1):
        fingerprints = derive_import_graph_fingerprints(files, depth)
        if all(
            len({fingerprints[item.file_path] for item in group}) == len(group)
            for group in collisions
        ):
            return depth
    return None


def enrich_repository_candidates_with_graph_fingerprint(
    files: Mapping[str, str],
    candidates: Sequence[RepositoryPatchCandidate],
    max_depth: int = 4,
) -> Tuple[Tuple[RepositoryPatchCandidate, ...], Optional[int]]:
    depth = minimal_role_collision_escape_depth(files, candidates, max_depth=max_depth)
    if depth is None:
        return tuple(candidates), None
    fingerprints = derive_import_graph_fingerprints(files, depth)
    enriched = []
    for candidate in candidates:
        fingerprint = fingerprints[candidate.file_path]
        reason = (
            f"{candidate.proposal.reason} "
            f"{GRAPH_FINGERPRINT_MARKER}{fingerprint} "
            f"{GRAPH_FINGERPRINT_DEPTH_MARKER}{depth}"
        )
        enriched.append(replace(
            candidate,
            proposal=replace(candidate.proposal, reason=reason),
        ))
    return tuple(enriched), depth


def _parse_marker(reason: str, marker: str) -> Optional[str]:
    if marker not in str(reason):
        return None
    return str(reason).split(marker, 1)[1].strip().split()[0].rstrip(",;)") or None


def parse_graph_localization_signature(
    proposal: InterventionProposal,
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    reason = str(proposal.reason)
    fingerprint = _parse_marker(reason, GRAPH_FINGERPRINT_MARKER)
    operator_id = _parse_marker(reason, REPOSITORY_REPAIR_OPERATOR_MARKER)
    depth_text = _parse_marker(reason, GRAPH_FINGERPRINT_DEPTH_MARKER)
    if fingerprint is None or operator_id is None or depth_text is None:
        return None, None, None
    try:
        depth = int(depth_text)
    except ValueError:
        return None, None, None
    return fingerprint, operator_id, depth


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def assess_named_role_localization_language(
    candidates_by_context: Mapping[str, Sequence[RepositoryPatchCandidate]],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> LocalizationLanguageAssessment:
    """Open representation escape only when the old named-role language is non-identifying.

    A context is ambiguous only if every externally successful exact patch shares
    its old `(file_role, operator)` signature with at least one completely evaluated
    weak patch. Missing evidence never counts as ambiguity or refutation.
    """
    minimum_classes = max(1, int(min_independent_classes))
    by_key: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair):
            continue
        by_key.setdefault((pair.context_id, pair.experiment_id), {}).setdefault(
            pair.independence_class_id, pair
        )

    complete_contexts = []
    ambiguous_contexts = []
    missing = []
    evaluated = 0
    for context_id, candidates in candidates_by_context.items():
        if not candidates:
            continue
        scores: Dict[str, float] = {}
        context_complete = True
        for candidate in candidates:
            classes = by_key.get((str(context_id), candidate.proposal.experiment_id), {})
            if len(classes) < minimum_classes:
                context_complete = False
                missing.append(candidate.proposal.experiment_id)
                continue
            evaluated += 1
            scores[candidate.proposal.experiment_id] = (
                sum(abs(pair.effect) for pair in classes.values()) / len(classes)
            )
        if not context_complete:
            continue
        complete_contexts.append(str(context_id))
        strong = [
            candidate for candidate in candidates
            if scores.get(candidate.proposal.experiment_id, 0.0) >= float(strong_effect_threshold)
        ]
        if not strong:
            continue
        groups: Dict[Tuple[str, str], list[RepositoryPatchCandidate]] = {}
        for candidate in candidates:
            groups.setdefault((candidate.file_role, candidate.operator_id), []).append(candidate)
        every_strong_is_ambiguous = True
        for candidate in strong:
            peers = groups[(candidate.file_role, candidate.operator_id)]
            weak_peer_exists = any(
                peer.proposal.experiment_id != candidate.proposal.experiment_id
                and scores.get(peer.proposal.experiment_id, 0.0) < float(strong_effect_threshold)
                for peer in peers
            )
            if not weak_peer_exists:
                every_strong_is_ambiguous = False
                break
        if every_strong_is_ambiguous:
            ambiguous_contexts.append(str(context_id))

    if len(ambiguous_contexts) >= max(1, int(min_contexts)):
        status = "NAMED_ROLE_LOCALIZATION_NON_IDENTIFYING_OPEN_GRAPH_REPRESENTATION"
        reason = (
            "repeated complete executable evidence shows that successful patches are not identifiable "
            "within the authored file-role plus repair-operator vocabulary"
        )
    else:
        status = "NO_AUTHORIZED_LOCALIZATION_REPRESENTATION_ESCAPE"
        reason = (
            "graph representation escape requires repeated complete old-language ambiguity; "
            "absence or an ordinary repair failure is insufficient"
        )
    return LocalizationLanguageAssessment(
        status=status,
        ambiguous_contexts=tuple(sorted(ambiguous_contexts)),
        complete_contexts=tuple(sorted(complete_contexts)),
        missing_experiment_ids=tuple(sorted(set(missing))),
        evaluated_candidate_count=evaluated,
        reason=reason,
    )


def derive_graph_localization_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    language_assessment: LocalizationLanguageAssessment,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> RepositoryGraphLocalizationPolicy:
    if language_assessment.status != "NAMED_ROLE_LOCALIZATION_NON_IDENTIFYING_OPEN_GRAPH_REPRESENTATION":
        return RepositoryGraphLocalizationPolicy(
            status="GRAPH_REPRESENTATION_NOT_AUTHORIZED",
            fingerprint=None,
            operator_id=None,
            fingerprint_depth=None,
            supporting_contexts=(),
            candidate_signature_count=0,
            reason="old localization representation has not been causally shown non-identifying",
        )

    identity_by_experiment = {}
    for proposal in proposals:
        identity = parse_graph_localization_signature(proposal)
        if all(item is not None for item in identity):
            identity_by_experiment[proposal.experiment_id] = identity

    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair) or pair.experiment_id not in identity_by_experiment:
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(
            pair.independence_class_id, pair
        )

    minimum_classes = max(1, int(min_independent_classes))
    support: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for (experiment_id, context_id), classes in grouped.items():
        if len(classes) < minimum_classes:
            continue
        score = sum(abs(pair.effect) for pair in classes.values()) / len(classes)
        if score < float(strong_effect_threshold):
            continue
        fingerprint, operator_id, depth = identity_by_experiment[experiment_id]
        support.setdefault((str(fingerprint), str(operator_id), int(depth)), {})[context_id] = float(score)

    required = max(1, int(min_contexts))
    eligible = []
    for (fingerprint, operator_id, depth), contexts in support.items():
        if len(contexts) < required:
            continue
        mean_score = sum(contexts.values()) / len(contexts)
        eligible.append((depth, -len(contexts), -mean_score, fingerprint, operator_id, tuple(sorted(contexts))))
    eligible.sort()
    signature_space = set(identity_by_experiment.values())
    if not eligible:
        return RepositoryGraphLocalizationPolicy(
            status="NO_REPRODUCED_GRAPH_LOCALIZATION",
            fingerprint=None,
            operator_id=None,
            fingerprint_depth=None,
            supporting_contexts=(),
            candidate_signature_count=len(signature_space),
            reason="no generated structural fingerprint has repeated authenticated repair success",
        )
    chosen = eligible[0]
    return RepositoryGraphLocalizationPolicy(
        status="REPRODUCED_GENERATED_GRAPH_LOCALIZATION",
        fingerprint=str(chosen[3]),
        operator_id=str(chosen[4]),
        fingerprint_depth=int(chosen[0]),
        supporting_contexts=chosen[5],
        candidate_signature_count=len(signature_space),
        reason="minimum-depth filename-independent import-graph fingerprint reproduced across repositories",
    )


def select_graph_localized_candidates(
    candidates: Sequence[RepositoryPatchCandidate],
    policy: Optional[RepositoryGraphLocalizationPolicy],
    max_candidates: Optional[int] = None,
) -> RepositoryGraphLocalizationSelection:
    ordered = tuple(candidates)
    fingerprint = None
    operator_id = None
    depth = None
    status = "FULL_GRAPH_LOCALIZATION_SEARCH"
    reason = "no learned graph localization applied"
    if (
        policy is not None
        and policy.status == "REPRODUCED_GENERATED_GRAPH_LOCALIZATION"
        and policy.fingerprint
        and policy.operator_id
        and policy.fingerprint_depth is not None
    ):
        fingerprint = policy.fingerprint
        operator_id = policy.operator_id
        depth = int(policy.fingerprint_depth)
        matching = tuple(
            candidate for candidate in ordered
            if parse_graph_localization_signature(candidate.proposal) == (fingerprint, operator_id, depth)
        )
        nonmatching = tuple(candidate for candidate in ordered if candidate not in matching)
        ordered = matching + nonmatching
        status = "LEARNED_GENERATED_GRAPH_LOCALIZATION_PRIORITIZED"
        reason = "reproduced generated graph fingerprint prioritized on fresh repository"
    if max_candidates is not None:
        ordered = ordered[: max(0, int(max_candidates))]
    return RepositoryGraphLocalizationSelection(
        status=status,
        candidates=ordered,
        policy_fingerprint=fingerprint,
        policy_operator_id=operator_id,
        policy_fingerprint_depth=depth,
        total_candidate_count=len(candidates),
        reason=reason,
    )


class RepositoryLocalizationRepresentationOrgan:
    """Stateless representation-escape organ over the canonical BODY evidence ledger."""

    def __init__(
        self,
        body,
        generator: Optional[PythonRepositoryRepairGenerator] = None,
        max_fingerprint_depth: int = 4,
    ) -> None:
        self.body = body
        self.generator = generator or PythonRepositoryRepairGenerator()
        self.max_fingerprint_depth = max(0, int(max_fingerprint_depth))

    def propose(
        self,
        task_id: str,
        files: Mapping[str, str],
    ) -> Tuple[Tuple[RepositoryPatchCandidate, ...], Optional[int]]:
        base = self.generator.generate(task_id, files)
        candidates, depth = enrich_repository_candidates_with_graph_fingerprint(
            files, base, max_depth=self.max_fingerprint_depth
        )
        for candidate in candidates:
            self.body.memory.remember_experiment(candidate.proposal)
        return candidates, depth

    def assess_old_language(
        self,
        candidates_by_context: Mapping[str, Sequence[RepositoryPatchCandidate]],
    ) -> LocalizationLanguageAssessment:
        return assess_named_role_localization_language(
            candidates_by_context=candidates_by_context,
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )

    def policy(
        self,
        language_assessment: LocalizationLanguageAssessment,
    ) -> RepositoryGraphLocalizationPolicy:
        return derive_graph_localization_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            language_assessment=language_assessment,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )

    def select(
        self,
        candidates: Sequence[RepositoryPatchCandidate],
        policy: Optional[RepositoryGraphLocalizationPolicy],
        max_candidates: Optional[int] = None,
    ) -> RepositoryGraphLocalizationSelection:
        return select_graph_localized_candidates(candidates, policy, max_candidates=max_candidates)
