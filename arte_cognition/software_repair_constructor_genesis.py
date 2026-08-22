from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .software_repair_class_genesis import GeneratedRepairClassPhenotype, GeneratedRepairMechanism
from .software_task_acquisition import SoftwarePatchCandidate
from .world_coupling import WorldOutcomePair


CONSTRUCTOR_FAMILY_MARKER = "repair_constructor_family="
CONSTRUCTOR_PRIMITIVE_MARKER = "repair_constructor_primitive="
CONSTRUCTOR_EXCEPTION_MARKER = "repair_constructor_exception="
CONSTRUCTOR_LOCUS_MARKER = "repair_constructor_locus="
CONSTRUCTOR_RELATION_MARKER = "repair_constructor_relation="
CONSTRUCTOR_GOAL_MARKER = "repair_constructor_goal="
CONSTRUCTOR_FAMILY = "RELATIONAL_BINDING"


@dataclass(frozen=True)
class ConstructorInexpressivityContext:
    context_id: str
    baseline_capability: float
    old_constructor_candidate_count: int
    failure_signature: str


@dataclass(frozen=True)
class ConstructorInexpressivityAssessment:
    status: str
    complete_contexts: Tuple[str, ...]
    old_constructor_candidate_count: int
    reason: str


@dataclass(frozen=True)
class RelationalConstructorPrimitive:
    exception_family: str
    locus_kind: str
    binding_relation: str
    repair_goal: str

    @property
    def primitive_id(self) -> str:
        payload = {
            "exception_family": self.exception_family,
            "locus_kind": self.locus_kind,
            "binding_relation": self.binding_relation,
            "repair_goal": self.repair_goal,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        return f"REPAIR_CONSTRUCTOR_PRIMITIVE::{digest}"

    @property
    def class_phenotype(self) -> GeneratedRepairClassPhenotype:
        return GeneratedRepairClassPhenotype(
            failure_phase=f"EXCEPTION::{self.exception_family}",
            resource_relation=self.binding_relation,
            repair_goal=self.repair_goal,
        )


@dataclass(frozen=True)
class RelationalRepairClassCandidate:
    primitive: RelationalConstructorPrimitive
    proposal: InterventionProposal

    @property
    def class_id(self) -> str:
        return self.primitive.class_phenotype.class_id


@dataclass(frozen=True)
class RelationalRepairConstructorPolicy:
    status: str
    constructor_family: Optional[str]
    supporting_primitive_ids: Tuple[str, ...]
    supporting_exception_families: Tuple[str, ...]
    supporting_contexts: Tuple[str, ...]
    reason: str


_IMPORT_ERROR_RE = re.compile(
    r"ImportError:\s*cannot import name ['\"]([^'\"]+)['\"] from ['\"]([^'\"]+)['\"]"
)
_TYPE_ERROR_KEYWORD_RE = re.compile(
    r"TypeError:\s*[^\n]*got an unexpected keyword argument ['\"]([^'\"]+)['\"]"
)
_NAME_ERROR_RE = re.compile(r"NameError:\s*name ['\"]([^'\"]+)['\"] is not defined")
_EXCEPTION_RE = re.compile(r"(?m)^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):\s*(.+)$")


def assess_constructor_inexpressivity(
    contexts: Sequence[ConstructorInexpressivityContext],
    min_contexts: int = 1,
) -> ConstructorInexpressivityAssessment:
    rows = tuple(contexts)
    complete = tuple(
        row for row in rows
        if float(row.baseline_capability) == 0.0
        and int(row.old_constructor_candidate_count) == 0
        and bool(str(row.failure_signature).strip())
    )
    count = sum(max(0, int(row.old_constructor_candidate_count)) for row in rows)
    required = max(1, int(min_contexts))
    if len(complete) < required:
        return ConstructorInexpressivityAssessment(
            status="OLD_REPAIR_CLASS_CONSTRUCTOR_INEXPRESSIVITY_INCOMPLETE",
            complete_contexts=tuple(row.context_id for row in complete),
            old_constructor_candidate_count=count,
            reason="meta-constructor remains closed until a reproducible failure has zero old-constructor proposals",
        )
    return ConstructorInexpressivityAssessment(
        status="OLD_REPAIR_CLASS_CONSTRUCTOR_INEXPRESSIVE_OPEN_RELATIONAL_CONSTRUCTOR",
        complete_contexts=tuple(row.context_id for row in complete),
        old_constructor_candidate_count=count,
        reason="reproducible failure is outside the currently authored repair-class constructor phenotype",
    )


def _module_path(module_name: str, repository_sources: Mapping[str, str]) -> Optional[str]:
    token = str(module_name).strip().strip(".")
    if not token:
        return None
    candidate = token.replace(".", "/") + ".py"
    normalized = {str(path).replace("\\", "/").lstrip("./"): path for path in repository_sources}
    if candidate in normalized:
        return str(normalized[candidate])
    tail = token.split(".")[-1] + ".py"
    matches = [path for path in normalized if path.endswith("/" + tail) or path == tail]
    return matches[0] if len(matches) == 1 else None


def _top_level_exports(source: str) -> Tuple[str, ...]:
    try:
        tree = ast.parse(str(source))
    except SyntaxError:
        return ()
    exports = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exports.append(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if isinstance(target, ast.Name):
                    exports.append(target.id)
    return tuple(sorted(set(exports)))


def _exception_signature(stderr: str) -> Optional[Tuple[str, str]]:
    matches = tuple(_EXCEPTION_RE.finditer(str(stderr)))
    if not matches:
        return None
    match = matches[-1]
    return match.group(1), match.group(2).strip()


def infer_relational_constructor_primitive(
    stderr: str,
    source: str,
    target_path: str,
    repository_sources: Mapping[str, str],
    assessment: ConstructorInexpressivityAssessment,
) -> Optional[RelationalConstructorPrimitive]:
    if assessment.status != "OLD_REPAIR_CLASS_CONSTRUCTOR_INEXPRESSIVE_OPEN_RELATIONAL_CONSTRUCTOR":
        return None
    text = str(stderr)

    import_match = _IMPORT_ERROR_RE.search(text)
    if import_match:
        symbol, module_name = import_match.groups()
        local_path = _module_path(module_name, repository_sources)
        if local_path is None:
            return None
        if symbol in _top_level_exports(repository_sources[local_path]):
            return None
        try:
            tree = ast.parse(str(source))
        except SyntaxError:
            return None
        has_site = any(
            isinstance(node, ast.ImportFrom)
            and any(alias.name == symbol for alias in node.names)
            for node in ast.walk(tree)
        )
        if not has_site:
            return None
        return RelationalConstructorPrimitive(
            exception_family="IMPORT_ERROR",
            locus_kind="IMPORT_FROM",
            binding_relation="LOCAL_EXPORT_ABSENT",
            repair_goal="RESTORE_BINDING_COMPATIBILITY",
        )

    keyword_match = _TYPE_ERROR_KEYWORD_RE.search(text)
    if keyword_match:
        keyword = keyword_match.group(1)
        try:
            tree = ast.parse(str(source))
        except SyntaxError:
            return None
        if not any(
            isinstance(node, ast.Call) and any(item.arg == keyword for item in node.keywords)
            for node in ast.walk(tree)
        ):
            return None
        return RelationalConstructorPrimitive(
            exception_family="TYPE_ERROR",
            locus_kind="CALL_KEYWORD",
            binding_relation="UNACCEPTED_KEYWORD_ARGUMENT",
            repair_goal="RESTORE_BINDING_COMPATIBILITY",
        )

    name_match = _NAME_ERROR_RE.search(text)
    if name_match:
        symbol = name_match.group(1)
        try:
            tree = ast.parse(str(source))
        except SyntaxError:
            return None
        used = any(isinstance(node, ast.Name) and node.id == symbol and isinstance(node.ctx, ast.Load) for node in ast.walk(tree))
        if not used:
            return None
        providers = tuple(
            path for path, candidate_source in repository_sources.items()
            if str(path).replace("\\", "/").lstrip("./") != str(target_path).replace("\\", "/").lstrip("./")
            and symbol in _top_level_exports(candidate_source)
        )
        if not providers:
            return None
        return RelationalConstructorPrimitive(
            exception_family="NAME_ERROR",
            locus_kind="NAME_LOAD",
            binding_relation="LOCAL_EXPORT_UNBOUND",
            repair_goal="RESTORE_BINDING_COMPATIBILITY",
        )
    return None


def propose_relational_repair_class(
    primitive: RelationalConstructorPrimitive,
) -> RelationalRepairClassCandidate:
    phenotype = primitive.class_phenotype
    class_id = phenotype.class_id
    digest = hashlib.sha256(
        f"{CONSTRUCTOR_FAMILY}|{primitive.primitive_id}|{class_id}".encode("utf-8")
    ).hexdigest()[:20]
    proposal = InterventionProposal(
        experiment_id=f"SOFTWARE_REPAIR_CONSTRUCTOR_GENESIS::{digest}",
        axis_id=f"AXIS::SOFTWARE_REPAIR_CONSTRUCTOR::{digest}",
        manipulated_variable=primitive.primitive_id,
        held_fixed=(),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="OLD_REPAIR_CLASS_CONSTRUCTOR_INEXPRESSIVE",
        predicted_high_side="RELATIONAL_CONSTRUCTOR_GENERATED_CLASS_SEARCH",
        reason=(
            "world-gated relational repair-constructor primitive proposal; "
            f"{CONSTRUCTOR_FAMILY_MARKER}{CONSTRUCTOR_FAMILY} "
            f"{CONSTRUCTOR_PRIMITIVE_MARKER}{primitive.primitive_id} "
            f"{CONSTRUCTOR_EXCEPTION_MARKER}{primitive.exception_family} "
            f"{CONSTRUCTOR_LOCUS_MARKER}{primitive.locus_kind} "
            f"{CONSTRUCTOR_RELATION_MARKER}{primitive.binding_relation} "
            f"{CONSTRUCTOR_GOAL_MARKER}{primitive.repair_goal} "
            f"generated_class={class_id}"
        ),
        status="PROPOSAL_ONLY",
    )
    return RelationalRepairClassCandidate(primitive=primitive, proposal=proposal)


def _marker(reason: str, marker: str) -> Optional[str]:
    if marker not in reason:
        return None
    return reason.split(marker, 1)[1].strip().split()[0].rstrip(",;)") or None


def _authoritative(pair: WorldOutcomePair) -> bool:
    return bool(
        pair.matched_budget
        and pair.externally_generated
        and pair.authority_verified
        and pair.independence_class_id != "UNVERIFIED"
    )


def derive_relational_constructor_policy(
    proposals: Iterable[InterventionProposal],
    world_pairs: Sequence[WorldOutcomePair],
    min_independent_classes: int,
    strong_effect_threshold: float = 0.9,
    min_distinct_primitives: int = 2,
) -> RelationalRepairConstructorPolicy:
    metadata: Dict[str, Tuple[str, str, str]] = {}
    for proposal in proposals:
        reason = str(proposal.reason)
        family = _marker(reason, CONSTRUCTOR_FAMILY_MARKER)
        primitive = _marker(reason, CONSTRUCTOR_PRIMITIVE_MARKER)
        exception = _marker(reason, CONSTRUCTOR_EXCEPTION_MARKER)
        if family == CONSTRUCTOR_FAMILY and primitive and exception:
            metadata[proposal.experiment_id] = (family, primitive, exception)
    grouped: Dict[Tuple[str, str], Dict[str, WorldOutcomePair]] = {}
    for pair in world_pairs:
        if pair.experiment_id not in metadata or not _authoritative(pair):
            continue
        grouped.setdefault((pair.experiment_id, pair.context_id), {}).setdefault(
            pair.independence_class_id, pair
        )
    required_classes = max(1, int(min_independent_classes))
    supported = []
    for (experiment_id, context_id), by_class in grouped.items():
        if len(by_class) < required_classes:
            continue
        score = sum(abs(pair.effect) for pair in by_class.values()) / len(by_class)
        if score < float(strong_effect_threshold):
            continue
        family, primitive, exception = metadata[experiment_id]
        supported.append((context_id, family, primitive, exception))
    primitive_ids = tuple(sorted({item[2] for item in supported}))
    exceptions = tuple(sorted({item[3] for item in supported}))
    contexts = tuple(sorted({item[0] for item in supported}))
    required_primitives = max(1, int(min_distinct_primitives))
    if len(primitive_ids) < required_primitives or len(exceptions) < required_primitives:
        return RelationalRepairConstructorPolicy(
            status="NO_REPRODUCED_RELATIONAL_REPAIR_CONSTRUCTOR",
            constructor_family=None,
            supporting_primitive_ids=primitive_ids,
            supporting_exception_families=exceptions,
            supporting_contexts=contexts,
            reason="relational constructor lacks strong externally reverified support across distinct generated primitives",
        )
    return RelationalRepairConstructorPolicy(
        status="REPRODUCED_RELATIONAL_REPAIR_CONSTRUCTOR",
        constructor_family=CONSTRUCTOR_FAMILY,
        supporting_primitive_ids=primitive_ids,
        supporting_exception_families=exceptions,
        supporting_contexts=contexts,
        reason="distinct failure relations were converted into causally successful generated repair classes",
    )


class _DropMissingImportAndAnnotation(ast.NodeTransformer):
    def __init__(self, symbol: str) -> None:
        self.symbol = str(symbol)
        self.changed = 0

    def visit_ImportFrom(self, node: ast.ImportFrom):
        names = [alias for alias in node.names if alias.name != self.symbol]
        if len(names) != len(node.names):
            self.changed += 1
            if not names:
                return None
            node.names = names
        return node

    def visit_arg(self, node: ast.arg):
        if isinstance(node.annotation, ast.Name) and node.annotation.id == self.symbol:
            node.annotation = None
            self.changed += 1
        return node


def _drop_unaccepted_keyword(source: str, keyword: str) -> Optional[str]:
    class Transformer(ast.NodeTransformer):
        def __init__(self) -> None:
            self.changed = 0

        def visit_Call(self, node: ast.Call):
            self.generic_visit(node)
            before = len(node.keywords)
            node.keywords = [item for item in node.keywords if item.arg != keyword]
            self.changed += before - len(node.keywords)
            return node

    tree = ast.parse(str(source))
    transformer = Transformer()
    changed = transformer.visit(tree)
    ast.fix_missing_locations(changed)
    return ast.unparse(changed) + "\n" if transformer.changed else None


def _drop_missing_import(source: str, symbol: str) -> Optional[str]:
    tree = ast.parse(str(source))
    transformer = _DropMissingImportAndAnnotation(symbol)
    changed = transformer.visit(tree)
    ast.fix_missing_locations(changed)
    return ast.unparse(changed) + "\n" if transformer.changed else None


def _insert_import(source: str, module_name: str, symbol: str) -> str:
    tree = ast.parse(str(source))
    body = list(tree.body)
    insertion = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        insertion = 1
    while insertion < len(body):
        node = body[insertion]
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insertion += 1
            continue
        break
    body.insert(insertion, ast.ImportFrom(module=str(module_name), names=[ast.alias(name=str(symbol))], level=0))
    tree.body = body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _path_to_module(path: str) -> Optional[str]:
    normalized = str(path).replace("\\", "/").lstrip("./")
    if not normalized.endswith(".py") or normalized.endswith("/__init__.py"):
        return None
    return normalized[:-3].replace("/", ".")


def generate_relational_repair_mechanisms(
    candidate: RelationalRepairClassCandidate,
    stderr: str,
    source: str,
    target_path: str,
    repository_sources: Mapping[str, str],
) -> Tuple[GeneratedRepairMechanism, ...]:
    primitive = candidate.primitive
    class_id = candidate.class_id
    mechanisms = []

    if primitive.binding_relation == "LOCAL_EXPORT_ABSENT":
        match = _IMPORT_ERROR_RE.search(str(stderr))
        if match:
            symbol = match.group(1)
            patched = _drop_missing_import(source, symbol)
            if patched is not None:
                mechanisms.append(("IMPORT_BINDING::DROP_STALE_LOCAL_SYMBOL", patched))

    elif primitive.binding_relation == "UNACCEPTED_KEYWORD_ARGUMENT":
        match = _TYPE_ERROR_KEYWORD_RE.search(str(stderr))
        if match:
            patched = _drop_unaccepted_keyword(source, match.group(1))
            if patched is not None:
                mechanisms.append(("CALL_BINDING::DROP_UNACCEPTED_KEYWORD", patched))

    elif primitive.binding_relation == "LOCAL_EXPORT_UNBOUND":
        match = _NAME_ERROR_RE.search(str(stderr))
        if match:
            symbol = match.group(1)
            for path, candidate_source in sorted(repository_sources.items()):
                normalized = str(path).replace("\\", "/").lstrip("./")
                if normalized == str(target_path).replace("\\", "/").lstrip("./"):
                    continue
                if symbol not in _top_level_exports(candidate_source):
                    continue
                module_name = _path_to_module(normalized)
                if module_name is None:
                    continue
                patched = _insert_import(source, module_name, symbol)
                module_token = hashlib.sha256(module_name.encode("utf-8")).hexdigest()[:10]
                mechanisms.append((f"NAME_BINDING::IMPORT_LOCAL_EXPORT::{module_token}", patched))

    source_hash = hashlib.sha256(str(source).encode("utf-8")).hexdigest()
    out = []
    for index, (mechanism_id, patched_source) in enumerate(mechanisms):
        digest = hashlib.sha256(
            f"{candidate.primitive.primitive_id}|{class_id}|{source_hash}|{target_path}|{mechanism_id}".encode("utf-8")
        ).hexdigest()[:20]
        proposal = InterventionProposal(
            experiment_id=f"SOFTWARE_RELATIONAL_REPAIR_MECHANISM::{digest}",
            axis_id=f"AXIS::{class_id}",
            manipulated_variable=mechanism_id,
            held_fixed=(),
            low_value=0.0,
            high_value=1.0,
            predicted_low_side="UNRESOLVED_BINDING_FAILURE",
            predicted_high_side="RELATIONAL_REPAIR_MECHANISM",
            reason=(
                "outcome-independent mechanism generated from a world-authorized relational constructor; "
                f"{CONSTRUCTOR_FAMILY_MARKER}{CONSTRUCTOR_FAMILY} "
                f"{CONSTRUCTOR_PRIMITIVE_MARKER}{primitive.primitive_id} "
                f"generated_class={class_id} generated_repair_mechanism={mechanism_id}"
            ),
            status="PROPOSAL_ONLY",
        )
        patch = SoftwarePatchCandidate(
            task_id=f"relational-constructor::{primitive.primitive_id}",
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
            candidate=patch,
        ))
    return tuple(out)


class RelationalRepairConstructorOrgan:
    """Stateless authority view over relational constructor proposals and world receipts."""

    def __init__(self, body) -> None:
        self.body = body

    def remember(self, candidates: Sequence[RelationalRepairClassCandidate]) -> None:
        for candidate in candidates:
            self.body.memory.remember_experiment(candidate.proposal)

    def policy(self) -> RelationalRepairConstructorPolicy:
        return derive_relational_constructor_policy(
            proposals=(record.proposal for record in self.body.memory.experiments.values()),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            strong_effect_threshold=0.9,
            min_distinct_primitives=2,
        )
