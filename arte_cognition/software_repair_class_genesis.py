from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .software_task_acquisition import SoftwarePatchCandidate
from .world_coupling import WorldOutcomePair


GENERATED_CLASS_MARKER = "generated_repair_class="
GENERATED_CLASS_PHASE_MARKER = "generated_repair_phase="
GENERATED_CLASS_RESOURCE_MARKER = "generated_repair_resource="
GENERATED_CLASS_GOAL_MARKER = "generated_repair_goal="
GENERATED_MECHANISM_MARKER = "generated_repair_mechanism="


@dataclass(frozen=True)
class FixedRepairClassContextResult:
    context_id: str
    applicable_class_ids: Tuple[str, ...]
    evaluated_candidate_count: int
    missing_candidate_count: int
    capability: float


@dataclass(frozen=True)
class FixedRepairClassFrontierAssessment:
    status: str
    complete_contexts: Tuple[str, ...]
    applicable_class_ids: Tuple[str, ...]
    evaluated_candidate_count: int
    missing_candidate_count: int
    reason: str


@dataclass(frozen=True)
class GeneratedRepairClassPhenotype:
    failure_phase: str
    resource_relation: str
    repair_goal: str

    @property
    def class_id(self) -> str:
        payload = {
            "failure_phase": self.failure_phase,
            "resource_relation": self.resource_relation,
            "repair_goal": self.repair_goal,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        return f"GEN_REPAIR_CLASS::{digest}"


@dataclass(frozen=True)
class GeneratedRepairClassCandidate:
    phenotype: GeneratedRepairClassPhenotype
    proposal: InterventionProposal


@dataclass(frozen=True)
class GeneratedRepairClassPolicy:
    status: str
    class_id: Optional[str]
    supporting_contexts: Tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class GeneratedRepairMechanism:
    class_id: str
    mechanism_id: str
    patched_source: str
    candidate: SoftwarePatchCandidate


def assess_fixed_repair_class_frontier(
    results: Sequence[FixedRepairClassContextResult],
    min_contexts: int = 2,
) -> FixedRepairClassFrontierAssessment:
    rows = tuple(results)
    required = max(1, int(min_contexts))
    complete = tuple(
        row for row in rows
        if row.applicable_class_ids
        and row.evaluated_candidate_count > 0
        and row.missing_candidate_count == 0
    )
    supported = tuple(row for row in complete if float(row.capability) > 0.0)
    classes = tuple(sorted({class_id for row in complete for class_id in row.applicable_class_ids}))
    evaluated = sum(max(0, int(row.evaluated_candidate_count)) for row in complete)
    missing = sum(max(0, int(row.missing_candidate_count)) for row in rows)
    if supported:
        return FixedRepairClassFrontierAssessment(
            status="FIXED_REPAIR_CLASS_FRONTIER_STILL_CAPABLE",
            complete_contexts=tuple(row.context_id for row in complete),
            applicable_class_ids=classes,
            evaluated_candidate_count=evaluated,
            missing_candidate_count=missing,
            reason="at least one previously available repair class retains world capability",
        )
    if len(complete) < required or missing:
        return FixedRepairClassFrontierAssessment(
            status="FIXED_REPAIR_CLASS_FAILURE_INCOMPLETE",
            complete_contexts=tuple(row.context_id for row in complete),
            applicable_class_ids=classes,
            evaluated_candidate_count=evaluated,
            missing_candidate_count=missing,
            reason="new repair-class synthesis is blocked until applicable fixed classes are completely evaluated",
        )
    return FixedRepairClassFrontierAssessment(
        status="FIXED_REPAIR_CLASSES_FALSIFIED_OPEN_CLASS_GENESIS",
        complete_contexts=tuple(row.context_id for row in complete),
        applicable_class_ids=classes,
        evaluated_candidate_count=evaluated,
        missing_candidate_count=0,
        reason="all applicable fixed repair classes failed across the required complete contexts",
    )


def _missing_module(stderr: str) -> Optional[str]:
    match = re.search(r"ModuleNotFoundError:\s*No module named ['\"]([^'\"]+)['\"]", str(stderr))
    return match.group(1) if match else None


def _is_local_missing_module(missing: str, target_path: str, repository_paths: Sequence[str]) -> bool:
    normalized = {str(path).replace("\\", "/").lstrip("./") for path in repository_paths}
    token = str(missing).split(".", 1)[0]
    if any(path == token or path.startswith(token + "/") for path in normalized):
        return True
    target = str(target_path).replace("\\", "/").lstrip("./")
    parent = target.rsplit("/", 1)[0] if "/" in target else ""
    sibling = f"{parent}/{token}.py" if parent else f"{token}.py"
    return sibling in normalized


def generate_repair_class_from_failure(
    stderr: str,
    target_path: str,
    repository_paths: Sequence[str],
    frontier: FixedRepairClassFrontierAssessment,
) -> Tuple[GeneratedRepairClassCandidate, ...]:
    if frontier.status != "FIXED_REPAIR_CLASSES_FALSIFIED_OPEN_CLASS_GENESIS":
        return ()
    missing = _missing_module(stderr)
    if not missing or not _is_local_missing_module(missing, target_path, repository_paths):
        return ()
    phenotype = GeneratedRepairClassPhenotype(
        failure_phase="MODULE_IMPORT",
        resource_relation="LOCAL_MODULE_UNRESOLVED",
        repair_goal="RESTORE_MODULE_REACHABILITY",
    )
    class_id = phenotype.class_id
    proposal = InterventionProposal(
        experiment_id=f"SOFTWARE_REPAIR_CLASS_GENESIS::{class_id.split('::')[-1]}",
        axis_id=f"AXIS::SOFTWARE_REPAIR_CLASS_GENESIS::{class_id.split('::')[-1]}",
        manipulated_variable=class_id,
        held_fixed=(),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="FIXED_REPAIR_CLASS_FAILURE",
        predicted_high_side="GENERATED_REPAIR_CLASS_SEARCH",
        reason=(
            "world-gated compositional software repair-class proposal; "
            f"{GENERATED_CLASS_MARKER}{class_id} "
            f"{GENERATED_CLASS_PHASE_MARKER}{phenotype.failure_phase} "
            f"{GENERATED_CLASS_RESOURCE_MARKER}{phenotype.resource_relation} "
            f"{GENERATED_CLASS_GOAL_MARKER}{phenotype.repair_goal}"
        ),
        status="PROPOSAL_ONLY",
    )
    return (GeneratedRepairClassCandidate(phenotype=phenotype, proposal=proposal),)


def parse_generated_repair_class(proposal: InterventionProposal) -> Optional[str]:
    reason = str(proposal.reason)
    if GENERATED_CLASS_MARKER not in reason:
        return None
    value = reason.split(GENERATED_CLASS_MARKER, 1)[1].strip().split()[0].rstrip(",;)")
    return value if value.startswith("GEN_REPAIR_CLASS::") else None


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def derive_generated_repair_class_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_contexts: int = 2,
) -> GeneratedRepairClassPolicy:
    class_by_experiment: Dict[str, str] = {}
    for proposal in proposals:
        class_id = parse_generated_repair_class(proposal)
        if class_id:
            class_by_experiment[proposal.experiment_id] = class_id
    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if not _authoritative(pair) or pair.experiment_id not in class_by_experiment:
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(
            pair.independence_class_id, pair
        )
    support: Dict[str, Dict[str, float]] = {}
    required_classes = max(1, int(min_independent_classes))
    for (experiment_id, context_id), by_class in grouped.items():
        if len(by_class) < required_classes:
            continue
        score = sum(abs(pair.effect) for pair in by_class.values()) / len(by_class)
        if score < float(strong_effect_threshold):
            continue
        class_id = class_by_experiment[experiment_id]
        support.setdefault(class_id, {})[context_id] = float(score)
    required_contexts = max(1, int(min_contexts))
    eligible = [
        (-len(contexts), -sum(contexts.values()) / len(contexts), class_id, tuple(sorted(contexts)))
        for class_id, contexts in support.items()
        if len(contexts) >= required_contexts
    ]
    eligible.sort()
    if not eligible:
        return GeneratedRepairClassPolicy(
            status="NO_REPRODUCED_GENERATED_REPAIR_CLASS",
            class_id=None,
            supporting_contexts=(),
            reason="no generated repair class has repeated externally reverified capability",
        )
    chosen = eligible[0]
    return GeneratedRepairClassPolicy(
        status="REPRODUCED_GENERATED_REPAIR_CLASS",
        class_id=chosen[2],
        supporting_contexts=chosen[3],
        reason="generated repair class reproduced across complete fixed-class-failure contexts",
    )


def _insert_repo_root_bootstrap(source: str, depth: int) -> str:
    text = str(source)
    lines = text.splitlines()
    insertion = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("from arte_cognition") or stripped.startswith("import arte_cognition"):
            insertion = index
            break
    if insertion is None:
        raise ValueError("source exposes no package-qualified local import site for search-context repair")
    bootstrap = []
    if not any(line.strip() == "import sys" for line in lines):
        bootstrap.append("import sys")
    if not any(line.strip() == "from pathlib import Path" for line in lines):
        bootstrap.append("from pathlib import Path")
    bootstrap.extend([
        f"_arte_generated_root = Path(__file__).resolve().parents[{int(depth)}]",
        "if str(_arte_generated_root) not in sys.path:",
        "    sys.path.insert(0, str(_arte_generated_root))",
        "",
    ])
    return "\n".join(lines[:insertion] + bootstrap + lines[insertion:]) + "\n"


class _QualifySiblingImports(ast.NodeTransformer):
    def __init__(self, package_name: str, sibling_modules: Sequence[str]) -> None:
        self.package_name = str(package_name)
        self.sibling_modules = set(str(item) for item in sibling_modules)
        self.changed = 0

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.level == 0 and node.module in self.sibling_modules:
            node.module = f"{self.package_name}.{node.module}"
            self.changed += 1
        return node


def _qualify_sibling_imports(
    source: str,
    target_path: str,
    repository_paths: Sequence[str],
) -> Optional[str]:
    target = str(target_path).replace("\\", "/").lstrip("./")
    if "/" not in target:
        return None
    package_name = target.split("/", 1)[0]
    prefix = package_name + "/"
    sibling_modules = {
        path[len(prefix):-3]
        for path in (str(item).replace("\\", "/").lstrip("./") for item in repository_paths)
        if path.startswith(prefix) and path.endswith(".py") and "/" not in path[len(prefix):]
    }
    if not sibling_modules:
        return None
    tree = ast.parse(str(source))
    transformer = _QualifySiblingImports(package_name, tuple(sorted(sibling_modules)))
    changed = transformer.visit(tree)
    ast.fix_missing_locations(changed)
    if transformer.changed <= 0:
        return None
    return ast.unparse(changed) + "\n"


def generate_repair_mechanisms(
    class_id: str,
    source: str,
    target_path: str,
    stderr: str,
    repository_paths: Sequence[str],
) -> Tuple[GeneratedRepairMechanism, ...]:
    phenotype = GeneratedRepairClassPhenotype(
        failure_phase="MODULE_IMPORT",
        resource_relation="LOCAL_MODULE_UNRESOLVED",
        repair_goal="RESTORE_MODULE_REACHABILITY",
    )
    if str(class_id) != phenotype.class_id:
        return ()
    missing = _missing_module(stderr)
    if not missing or not _is_local_missing_module(missing, target_path, repository_paths):
        return ()
    target = str(target_path).replace("\\", "/").lstrip("./")
    normalized_paths = {str(item).replace("\\", "/").lstrip("./") for item in repository_paths}
    mechanisms = []

    # Search-context candidates are generated only when the missing name is a
    # top-level repository package already referenced through package-qualified imports.
    top_token = missing.split(".", 1)[0]
    qualified_reference = any(
        isinstance(node, ast.ImportFrom)
        and node.module is not None
        and (node.module == top_token or node.module.startswith(top_token + "."))
        for node in ast.walk(ast.parse(str(source)))
    )
    top_level_package_exists = any(
        path == top_token or path.startswith(top_token + "/") for path in normalized_paths
    )
    if qualified_reference and top_level_package_exists:
        for depth in (0, 1, 2):
            patched = _insert_repo_root_bootstrap(source, depth)
            mechanism_id = f"SEARCH_CONTEXT::FILE_PARENT_DEPTH::{depth}::PREPEND"
            mechanisms.append((mechanism_id, patched))

    # Package-qualification is generated only for an unresolved bare sibling module.
    parent = target.rsplit("/", 1)[0] if "/" in target else ""
    sibling = f"{parent}/{top_token}.py" if parent else f"{top_token}.py"
    if "." not in missing and sibling in normalized_paths:
        qualified = _qualify_sibling_imports(source, target, repository_paths)
        if qualified is not None:
            mechanisms.append(("IMPORT_REFERENCE::QUALIFY_LOCAL_PACKAGE", qualified))

    source_hash = hashlib.sha256(str(source).encode("utf-8")).hexdigest()
    out = []
    for index, (mechanism_id, patched_source) in enumerate(mechanisms):
        digest = hashlib.sha256(
            f"{class_id}|{source_hash}|{target}|{mechanism_id}".encode("utf-8")
        ).hexdigest()[:20]
        proposal = InterventionProposal(
            experiment_id=f"SOFTWARE_GENERATED_REPAIR_MECHANISM::{digest}",
            axis_id=f"AXIS::{class_id}",
            manipulated_variable=mechanism_id,
            held_fixed=(),
            low_value=0.0,
            high_value=1.0,
            predicted_low_side="UNRESOLVED_LOCAL_MODULE",
            predicted_high_side="GENERATED_MODULE_RESOLUTION_REPAIR",
            reason=(
                "outcome-independent mechanism generated inside inherited repair class; "
                f"{GENERATED_CLASS_MARKER}{class_id} "
                f"{GENERATED_MECHANISM_MARKER}{mechanism_id}"
            ),
            status="PROPOSAL_ONLY",
        )
        candidate = SoftwarePatchCandidate(
            task_id=f"generated-class::{class_id}",
            source_hash=source_hash,
            site_index=index,
            operator_id=mechanism_id,
            patched_source=patched_source,
            proposal=proposal,
        )
        out.append(GeneratedRepairMechanism(
            class_id=class_id,
            mechanism_id=mechanism_id,
            patched_source=patched_source,
            candidate=candidate,
        ))
    return tuple(out)


class GeneratedRepairClassOrgan:
    """Stateless view over generated repair-class proposal/evidence lineage."""

    def __init__(self, body) -> None:
        self.body = body

    def remember_class_candidates(self, candidates: Sequence[GeneratedRepairClassCandidate]) -> None:
        for candidate in candidates:
            self.body.memory.remember_experiment(candidate.proposal)

    def policy(self) -> GeneratedRepairClassPolicy:
        return derive_generated_repair_class_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )
