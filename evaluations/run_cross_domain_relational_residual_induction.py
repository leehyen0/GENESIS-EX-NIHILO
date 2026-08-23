from __future__ import annotations

import hashlib
import json
import secrets
import subprocess
import sys
from dataclasses import asdict
from typing import Dict, Iterable, Optional, Sequence, Tuple

from arte_cognition.relational_residual_induction import (
    GeneratedRelationalPathSchema,
    RelationalContext,
    RelationalEdge,
    RelationalResidualInducer,
    derive_relational_path_policy,
    make_context,
    select_authorized_relational_path_schema,
)
from arte_cognition.world_coupling import WorldOutcomePair


STATUS = "PASS_BOUNDED_CROSS_DOMAIN_RELATIONAL_RESIDUAL_INDUCTION_AND_PREOUTCOME_TRANSFER"


def _suffix() -> str:
    return f"{secrets.randbelow(10**9):09d}"


def _software_context(context_id: str, suffix: str) -> RelationalContext:
    return make_context(
        context_id,
        (
            RelationalEdge(f"builder_{suffix}", "PRODUCES", f"mapping_{suffix}", "FUNCTION", "MAPPING"),
            RelationalEdge(f"mapping_{suffix}", "BOUND_AS", f"binding_{suffix}", "MAPPING", "BINDING"),
            RelationalEdge(f"binding_{suffix}", "UNPACKED_INTO", f"call_{suffix}", "BINDING", "CALL"),
            RelationalEdge(f"noise_{suffix}", "MENTIONS", f"call_{suffix}", "NAME", "CALL"),
            RelationalEdge(f"builder_{suffix}", "DECORATED_BY", f"noise_{suffix}", "FUNCTION", "NAME"),
        ),
        f"builder_{suffix}",
        f"call_{suffix}",
        "SOFTWARE",
    )


def _causal_context(context_id: str, suffix: str) -> RelationalContext:
    return make_context(
        context_id,
        (
            RelationalEdge(f"do_{suffix}", "EMITS", f"signal_{suffix}", "INTERVENTION", "SIGNAL"),
            RelationalEdge(f"signal_{suffix}", "MEDIATED_BY", f"state_{suffix}", "SIGNAL", "STATE"),
            RelationalEdge(f"state_{suffix}", "REACHES", f"outcome_{suffix}", "STATE", "OUTCOME"),
            RelationalEdge(f"context_{suffix}", "COEXISTS", f"outcome_{suffix}", "CONTEXT", "OUTCOME"),
            RelationalEdge(f"do_{suffix}", "TAGGED_BY", f"context_{suffix}", "INTERVENTION", "CONTEXT"),
        ),
        f"do_{suffix}",
        f"outcome_{suffix}",
        "CAUSAL_WORLD",
    )


def _context_payload(context: RelationalContext) -> Dict[str, object]:
    return {
        "context_id": context.context_id,
        "domain": context.domain,
        "source_anchor": context.source_anchor,
        "target_anchor": context.target_anchor,
        "edges": [asdict(edge) for edge in context.edges],
    }


def _hash_context(context: RelationalContext) -> str:
    raw = json.dumps(_context_payload(context), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _external_execute(
    context: RelationalContext,
    schema: Optional[GeneratedRelationalPathSchema],
) -> float:
    """Evaluate in a fresh Python subprocess that does not import the ARTE inducer."""
    payload = {
        "context": _context_payload(context),
        "steps": list(schema.steps) if schema is not None else [],
    }
    child = r'''
import json, sys
p = json.loads(sys.stdin.read())
c = p["context"]
steps = tuple(p["steps"])
if not steps:
    print("0")
    raise SystemExit(0)
adj = {}
for e in c["edges"]:
    adj.setdefault(e["source"], []).append(e)
for v in adj.values():
    v.sort(key=lambda e: (e["relation"], e["source_kind"], e["target_kind"], e["target"]))
found = set()
def walk(node, visited, path):
    if len(path) >= 8:
        return
    for e in adj.get(node, ()):
        if e["target"] in visited:
            continue
        step = f'{e["source_kind"]}-[{e["relation"]}]->{e["target_kind"]}'
        nxt = path + (step,)
        if e["target"] == c["target_anchor"]:
            found.add(nxt)
        walk(e["target"], visited + (e["target"],), nxt)
walk(c["source_anchor"], (c["source_anchor"],), ())
print("1" if steps in found else "0")
'''
    result = subprocess.run(
        [sys.executable, "-c", child],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    return float(result.stdout.strip())


def _pair(
    schema: GeneratedRelationalPathSchema,
    context: RelationalContext,
    independence_class: str,
    source_id: str,
    effect: float,
    *,
    verified: bool = True,
) -> WorldOutcomePair:
    return WorldOutcomePair(
        pair_id=f"PAIR::{schema.schema_id}::{context.context_id}::{independence_class}::{source_id}",
        experiment_id=schema.schema_id,
        axis_id=f"RELATIONAL::{context.domain}",
        source_id=source_id,
        context_id=context.context_id,
        challenge_id=f"challenge::{context.context_id}",
        epoch=1,
        low_outcome=0.0,
        high_outcome=float(effect),
        low_value=0.0,
        high_value=1.0,
        matched_budget=True,
        externally_generated=True,
        issuer_id=f"issuer::{independence_class}",
        independence_class_id=independence_class if verified else "UNVERIFIED",
        authority_verified=bool(verified),
    )


def _freeze_candidates(
    inducer: RelationalResidualInducer,
    contexts: Sequence[RelationalContext],
) -> Tuple[GeneratedRelationalPathSchema, ...]:
    assessment = inducer.assess_repeated_residual(contexts, (0,) * len(contexts), min_contexts=2)
    if assessment.status != "RELATIONAL_RESIDUAL_OPEN_INDUCTION":
        raise AssertionError("repeated residual gate did not open")
    schemas = inducer.generate_candidates(assessment, contexts)
    if not schemas:
        raise AssertionError("no relational schema generated")
    return schemas


def _authority_pairs(
    schemas: Sequence[GeneratedRelationalPathSchema],
    contexts: Sequence[RelationalContext],
) -> Tuple[WorldOutcomePair, ...]:
    pairs = []
    for schema in schemas:
        for context in contexts:
            effect = _external_execute(context, schema)
            for cls in ("AUTH_A", "AUTH_B"):
                pairs.append(_pair(schema, context, cls, f"{context.domain.lower()}::{cls}", effect))
    return tuple(pairs)


def _assert_domain(
    inducer: RelationalResidualInducer,
    train: Sequence[RelationalContext],
    heldout: RelationalContext,
) -> Tuple[GeneratedRelationalPathSchema, int, int, Tuple[WorldOutcomePair, ...]]:
    schemas = _freeze_candidates(inducer, train)
    frozen_ids = tuple(schema.schema_id for schema in schemas)

    # Candidate freeze occurs before any training outcome is produced.
    pairs = _authority_pairs(schemas, train)
    if tuple(schema.schema_id for schema in schemas) != frozen_ids:
        raise AssertionError("candidate set changed after world outcomes")

    one_context_pairs = tuple(pair for pair in pairs if pair.context_id == train[0].context_id)
    one_context_policy = derive_relational_path_policy(schemas, one_context_pairs, 2, 2)
    if select_authorized_relational_path_schema(schemas, one_context_policy) is not None:
        raise AssertionError("one context incorrectly authorized a relation path")

    verifierless_pairs = tuple(
        _pair(
            next(schema for schema in schemas if schema.schema_id == pair.experiment_id),
            next(context for context in train if context.context_id == pair.context_id),
            "UNVERIFIED",
            pair.source_id,
            pair.effect,
            verified=False,
        )
        for pair in pairs
    )
    verifierless_policy = derive_relational_path_policy(schemas, verifierless_pairs, 2, 2)
    if select_authorized_relational_path_schema(schemas, verifierless_policy) is not None:
        raise AssertionError("verifierless evidence incorrectly authorized a relation path")

    policy = derive_relational_path_policy(schemas, pairs, 2, 2)
    selected = select_authorized_relational_path_schema(schemas, policy)
    if selected is None:
        raise AssertionError("authoritative repeated evidence did not select a relation path")

    # Freeze the heldout action before exposing heldout execution outcome.
    selected_before_heldout_outcome = int(inducer.matches(selected, heldout))
    if selected_before_heldout_outcome != 1:
        raise AssertionError("learned relation path did not transfer structurally to heldout")
    return selected, len(schemas), selected_before_heldout_outcome, pairs


def main() -> Dict[str, object]:
    inducer = RelationalResidualInducer(max_depth=6, candidate_budget=64)

    software_train = (
        _software_context("software-train-a", _suffix()),
        _software_context("software-train-b", _suffix()),
    )
    causal_train = (
        _causal_context("causal-train-a", _suffix()),
        _causal_context("causal-train-b", _suffix()),
    )

    software_schemas_pre = _freeze_candidates(inducer, software_train)
    causal_schemas_pre = _freeze_candidates(inducer, causal_train)
    pre_outcome_freeze = {
        "software": tuple(schema.schema_id for schema in software_schemas_pre),
        "causal": tuple(schema.schema_id for schema in causal_schemas_pre),
    }

    software_heldout = _software_context("software-heldout", _suffix())
    causal_heldout = _causal_context("causal-heldout", _suffix())

    selected_software, software_count, software_preheldout, software_pairs = _assert_domain(
        inducer, software_train, software_heldout
    )
    selected_causal, causal_count, causal_preheldout, causal_pairs = _assert_domain(
        inducer, causal_train, causal_heldout
    )

    if pre_outcome_freeze["software"] != tuple(schema.schema_id for schema in software_schemas_pre):
        raise AssertionError("software candidate freeze changed")
    if pre_outcome_freeze["causal"] != tuple(schema.schema_id for schema in causal_schemas_pre):
        raise AssertionError("causal candidate freeze changed")

    treatment = (
        _external_execute(software_heldout, selected_software),
        _external_execute(causal_heldout, selected_causal),
    )
    remove = (
        _external_execute(software_heldout, None),
        _external_execute(causal_heldout, None),
    )
    # Cross-domain wrong-swap is a genuine learned schema with matched execution budget.
    wrong = (
        _external_execute(software_heldout, selected_causal),
        _external_execute(causal_heldout, selected_software),
    )

    treatment_capability = sum(treatment) / len(treatment)
    remove_capability = sum(remove) / len(remove)
    wrong_capability = sum(wrong) / len(wrong)
    if treatment_capability != 1.0:
        raise AssertionError(f"treatment capability mismatch: {treatment}")
    if remove_capability != 0.0:
        raise AssertionError(f"REMOVE retained capability: {remove}")
    if wrong_capability != 0.0:
        raise AssertionError(f"WRONG retained capability: {wrong}")

    software_hashes = tuple(_hash_context(context) for context in (*software_train, software_heldout))
    causal_hashes = tuple(_hash_context(context) for context in (*causal_train, causal_heldout))
    if len(set(software_hashes)) != 3 or len(set(causal_hashes)) != 3:
        raise AssertionError("train/heldout contexts are not source-disjoint")

    report: Dict[str, object] = {
        "status": STATUS,
        "repository": "leehyen0/GENESIS-EX-NIHILO",
        "same_domain_agnostic_inducer_class": True,
        "domains": ["SOFTWARE", "CAUSAL_WORLD"],
        "candidate_generation_uses_world_outcomes": False,
        "candidate_freeze_before_world_outcomes": True,
        "software_generated_schema_count": software_count,
        "causal_generated_schema_count": causal_count,
        "software_learned_schema_id": selected_software.schema_id,
        "causal_learned_schema_id": selected_causal.schema_id,
        "software_learned_steps": list(selected_software.steps),
        "causal_learned_steps": list(selected_causal.steps),
        "software_selected_before_heldout_outcome": bool(software_preheldout),
        "causal_selected_before_heldout_outcome": bool(causal_preheldout),
        "software_training_authority_pair_count": len(software_pairs),
        "causal_training_authority_pair_count": len(causal_pairs),
        "one_context_insufficient_for_authority": True,
        "verifierless_policy_authority": False,
        "heldout_source_disjoint": True,
        "external_execution": "fresh_python_subprocess_independent_path_evaluator",
        "treatment_external_execution_count": 2,
        "treatment_capability": treatment_capability,
        "remove_external_execution_count": 2,
        "remove_same_checkpoint_capability": remove_capability,
        "wrong_external_execution_count": 2,
        "wrong_cross_domain_schema_swap_capability": wrong_capability,
        "edge_relation_vocabulary_human_authored": True,
        "node_kind_vocabulary_human_authored": True,
        "domain_graph_adapters_human_authored": True,
        "path_enumerator_human_authored": True,
        "max_depth_human_authored": True,
        "natural_historical_cross_domain_failure": False,
        "unrestricted_meta_language_genesis": False,
        "unrestricted_operator_genesis": False,
        "global_recursive_acceleration": False,
        "independent_organizational_custody": False,
        "physical_world": False,
        "foundation_weight_change": False,
        "AGI": False,
        "ASI": False,
    }
    print(json.dumps(report, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
