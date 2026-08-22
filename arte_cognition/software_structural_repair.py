from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from typing import List, Tuple

from .experiment_genesis import InterventionProposal
from .software_repair_grammar_expansion import (
    GENERATED_REPAIR_MARKER,
    SoftwareRepairAlphabetAssessment,
)
from .software_task_acquisition import SoftwarePatchCandidate


TRAVERSAL_STRATEGIES: Tuple[str, ...] = (
    "BFS",
    "DFS_PRE",
    "DFS_POST",
    "REVERSED_BFS",
)


def _method_body_for_strategy(strategy_id: str) -> List[ast.stmt]:
    if strategy_id == "BFS":
        body = '''
tree = ast.parse(source)
operator_ids = []
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
'''
    elif strategy_id in {"DFS_PRE", "DFS_POST"}:
        pre = strategy_id == "DFS_PRE"
        before = '''
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        mutation = _COMPARE_MUTATIONS.get(type(node.ops[0]))
        if mutation is not None:
            operator_ids.append(mutation[1])
    elif isinstance(node, ast.BoolOp):
        mutation = _BOOL_MUTATIONS.get(type(node.op))
        if mutation is not None:
            operator_ids.append(mutation[1])
''' if pre else ""
        after = before if not pre else ""
        body = f'''
tree = ast.parse(source)
operator_ids = []
def collect(node):
{before if before else ''}    for child in ast.iter_child_nodes(node):
        collect(child)
{after if after else ''}collect(tree)
return tuple(operator_ids)
'''
    elif strategy_id == "REVERSED_BFS":
        body = '''
tree = ast.parse(source)
operator_ids = []
for node in reversed(tuple(ast.walk(tree))):
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        mutation = _COMPARE_MUTATIONS.get(type(node.ops[0]))
        if mutation is not None:
            operator_ids.append(mutation[1])
    elif isinstance(node, ast.BoolOp):
        mutation = _BOOL_MUTATIONS.get(type(node.op))
        if mutation is not None:
            operator_ids.append(mutation[1])
return tuple(operator_ids)
'''
    else:
        raise ValueError(f"unknown traversal strategy: {strategy_id}")

    wrapper = "def _generated(source):\n" + "\n".join(
        "    " + line if line else "" for line in body.strip("\n").splitlines()
    ) + "\n"
    parsed = ast.parse(wrapper)
    function = parsed.body[0]
    if not isinstance(function, ast.FunctionDef):
        raise AssertionError("generated traversal strategy did not parse as a function")
    return list(function.body)


class _TraversalStrategyTransformer(ast.NodeTransformer):
    def __init__(self, strategy_id: str) -> None:
        self.strategy_id = str(strategy_id)
        self.matches = 0
        self.inside_generator = False

    def visit_ClassDef(self, node: ast.ClassDef):
        old_inside = self.inside_generator
        if node.name == "PythonASTRepairGenerator":
            self.inside_generator = True
        self.generic_visit(node)
        self.inside_generator = old_inside
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.inside_generator and node.name == "_site_operator_ids":
            self.matches += 1
            node.body = _method_body_for_strategy(self.strategy_id)
            return node
        return self.generic_visit(node)


def apply_traversal_strategy(source: str, strategy_id: str) -> str:
    tree = ast.parse(source)
    transformer = _TraversalStrategyTransformer(strategy_id)
    mutated = transformer.visit(tree)
    ast.fix_missing_locations(mutated)
    if transformer.matches != 1:
        raise AssertionError(
            f"expected exactly one PythonASTRepairGenerator._site_operator_ids method, got {transformer.matches}"
        )
    return ast.unparse(mutated) + "\n"


class PythonTraversalStrategyRepairGenerator:
    """Outcome-independent structural repair grammar for AST-site traversal identity.

    This generator is intentionally bounded. World evidence may open this structural
    repair class after the previously available content-mutation alphabet is fully
    falsified, but hidden outcomes never choose a strategy or synthesize candidate
    source. The current four-strategy metalanguage is human-authored and therefore
    does not establish unrestricted software-operator invention.
    """

    def generate(
        self,
        task_id: str,
        historical_source: str,
        assessment: SoftwareRepairAlphabetAssessment,
    ) -> Tuple[SoftwarePatchCandidate, ...]:
        if assessment.status != "SOFTWARE_REPAIR_ALPHABET_FALSIFIED_OPEN_NEXT":
            return ()
        source_hash = hashlib.sha256(historical_source.encode("utf-8")).hexdigest()
        candidates = []
        for strategy_index, strategy_id in enumerate(TRAVERSAL_STRATEGIES):
            patched_source = apply_traversal_strategy(historical_source, strategy_id)
            operator_id = f"TRAVERSAL::{strategy_id}"
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
                experiment_id=f"SOFTWARE_STRUCTURAL_PATCH::{source_hash[:12]}::{strategy_id}::{digest}",
                axis_id=f"AXIS::SOFTWARE_STRUCTURAL_REPAIR::{source_hash[:16]}",
                manipulated_variable=operator_id,
                held_fixed=(),
                low_value=0.0,
                high_value=1.0,
                predicted_low_side="HISTORICAL_BUGGY_GENERATOR",
                predicted_high_side="STRUCTURALLY_REPAIRED_GENERATOR",
                reason=(
                    "execute world-gated structural Python repair candidate; "
                    f"{GENERATED_REPAIR_MARKER}{operator_id} "
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
