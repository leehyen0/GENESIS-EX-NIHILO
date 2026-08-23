from __future__ import annotations

import hashlib
import json
import secrets
import subprocess
import sys
from dataclasses import asdict
from typing import Dict, Optional

from arte_cognition.latent_relation_ontology_genesis import (
    GeneratedLatentPathSchema,
    OpaqueInterventionalWorld,
    WorldDerivedLatentRelationInducer,
    contrast,
    derive_latent_path_policy,
    select_authorized_latent_path,
)
from arte_cognition.world_coupling import WorldOutcomePair


STATUS = "PASS_BOUNDED_WORLD_DERIVED_LATENT_RELATION_ONTOLOGY_AND_CROSS_DOMAIN_PREOUTCOME_TRANSFER"


def _suffix() -> str:
    return f"{secrets.randbelow(10**9):09d}"


def _local_trace_world(context_id: str, prefix: str, domain: str, magnitude: float) -> OpaqueInterventionalWorld:
    """Expose local one-step intervention contrasts, never a root->target observation."""
    nodes = [f"{prefix}_{i}" for i in range(4)]
    decoy = f"{prefix}_decoy"
    all_nodes = nodes + [decoy]
    zero = {node: 0.0 for node in all_nodes}
    rows = []
    for source_index in range(3):
        for repeat in range(2):
            low = [dict(zero), dict(zero)]
            high = [dict(zero), dict(zero)]
            high[0][nodes[source_index]] = float(magnitude)
            high[1][nodes[source_index + 1]] = float(magnitude)
            low[1][decoy] = float(repeat)
            high[1][decoy] = float(repeat)
            rows.append(contrast(
                f"{context_id}:{source_index}:{repeat}",
                nodes[source_index],
                low,
                high,
            ))
    return OpaqueInterventionalWorld(
        context_id=context_id,
        domain=domain,
        source_anchor=nodes[0],
        target_anchor=nodes[-1],
        contrasts=tuple(rows),
    )


def _world_payload(world: OpaqueInterventionalWorld) -> Dict[str, object]:
    return {
        "context_id": world.context_id,
        "domain": world.domain,
        "source_anchor": world.source_anchor,
        "target_anchor": world.target_anchor,
        "contrasts": [asdict(row) for row in world.contrasts],
    }


def _world_hash(world: OpaqueInterventionalWorld) -> str:
    return hashlib.sha256(
        json.dumps(_world_payload(world), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _expected_token() -> str:
    return "LATENT_REL::" + hashlib.sha256(b"lag=1|sign=POS").hexdigest()[:16]


def _external_hidden_chain_query(
    world: OpaqueInterventionalWorld,
    schema: Optional[GeneratedLatentPathSchema],
) -> float:
    """Fresh subprocess independently composes the actual world's local traces.

    The inducer never receives an explicit root->target outcome. The external
    executor receives the concrete heldout world plus the already-frozen action,
    reconstructs repeated positive one-step effects from raw low/high snapshots,
    and asks whether that action exactly explains a path from this world's source
    anchor to this world's target anchor. Thus success cannot be obtained from a
    token string alone or from a heldout world that is never consumed.
    """
    payload = {
        "world": _world_payload(world),
        "tokens": list(schema.relation_tokens) if schema is not None else [],
    }
    child = r'''
import hashlib, json, sys
p = json.loads(sys.stdin.read())
w = p["world"]
tokens = tuple(p["tokens"])
if not tokens:
    print("0")
    raise SystemExit(0)

# Reconstruct repeated lag-1 positive intervention effects directly from this
# concrete world's raw contrasts. This subprocess imports no ARTE inducer code.
support = {}
for row in w["contrasts"]:
    source = row["source_node"]
    low = dict(row["low_timeline"][1])
    high = dict(row["high_timeline"][1])
    for target in sorted(set(low) | set(high)):
        if target == source:
            continue
        effect = float(high.get(target, 0.0)) - float(low.get(target, 0.0))
        if effect > 0.1:
            support[(source, target)] = support.get((source, target), 0) + 1

adj = {}
for (source, target), count in support.items():
    if count >= 2:
        adj.setdefault(source, []).append(target)
for values in adj.values():
    values.sort()

expected = "LATENT_REL::" + hashlib.sha256(b"lag=1|sign=POS").hexdigest()[:16]
found_lengths = set()
def walk(node, visited, depth):
    if depth > 8:
        return
    for target in adj.get(node, ()):
        if target in visited:
            continue
        next_depth = depth + 1
        if target == w["target_anchor"]:
            found_lengths.add(next_depth)
        walk(target, visited + (target,), next_depth)
walk(w["source_anchor"], (w["source_anchor"],), 0)

ok = (
    len(tokens) in found_lengths
    and all(token == expected for token in tokens)
)
print("1" if ok else "0")
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
    schema: GeneratedLatentPathSchema,
    context_id: str,
    cls: str,
    effect: float,
    verified: bool = True,
) -> WorldOutcomePair:
    return WorldOutcomePair(
        pair_id=f"PAIR::{schema.schema_id}::{context_id}::{cls}",
        experiment_id=schema.schema_id,
        axis_id="LATENT_RELATION_ONTOLOGY",
        source_id=f"external::{context_id}::{cls}",
        context_id=context_id,
        challenge_id=f"challenge::{context_id}",
        epoch=1,
        low_outcome=0.0,
        high_outcome=float(effect),
        low_value=0.0,
        high_value=1.0,
        matched_budget=True,
        externally_generated=True,
        issuer_id=f"issuer::{cls}",
        independence_class_id=cls if verified else "UNVERIFIED",
        authority_verified=verified,
    )


def main() -> Dict[str, object]:
    inducer = WorldDerivedLatentRelationInducer(
        lag=1,
        min_effect=0.1,
        min_repeats=2,
        max_path_depth=6,
        candidate_budget=64,
    )

    software_train = (
        _local_trace_world("software-a", f"softa_{_suffix()}", "SOFTWARE", 1.0),
        _local_trace_world("software-b", f"softb_{_suffix()}", "SOFTWARE", 7.0),
    )
    assessment = inducer.assess_residual(software_train, (0, 0), min_contexts=2)
    schemas = inducer.generate_candidates(assessment, software_train)
    if len(schemas) != 1:
        raise AssertionError(f"expected exactly one software latent schema, got {len(schemas)}")
    schema = schemas[0]
    frozen_schema_id = schema.schema_id
    if len(schema.relation_tokens) != 3 or len(set(schema.relation_tokens)) != 1:
        raise AssertionError("software world did not generate a three-step domain-opaque relation schema")
    if schema.relation_tokens[0] != _expected_token():
        raise AssertionError("relation fingerprint differs from independent evaluator derivation")

    # Candidate ontology and schema are frozen before any external composed outcome.
    training_pairs = []
    for world in software_train:
        effect = _external_hidden_chain_query(world, schema)
        for cls in ("AUTH_A", "AUTH_B"):
            training_pairs.append(_pair(schema, world.context_id, cls, effect))

    one_context = tuple(pair for pair in training_pairs if pair.context_id == software_train[0].context_id)
    if select_authorized_latent_path(
        (schema,), derive_latent_path_policy((schema,), one_context, 2, 2)
    ) is not None:
        raise AssertionError("one software context incorrectly authorized latent ontology")

    verifierless = tuple(
        _pair(schema, pair.context_id, "UNVERIFIED", pair.effect, verified=False)
        for pair in training_pairs
    )
    if select_authorized_latent_path(
        (schema,), derive_latent_path_policy((schema,), verifierless, 2, 2)
    ) is not None:
        raise AssertionError("verifierless latent ontology incorrectly authorized")

    policy = derive_latent_path_policy((schema,), tuple(training_pairs), 2, 2)
    authorized = select_authorized_latent_path((schema,), policy)
    if authorized is None or authorized.schema_id != frozen_schema_id:
        raise AssertionError("repeated authoritative software outcomes did not preserve frozen schema")

    # Generate a distinct causal heldout only after software-world authority exists.
    causal_heldout = _local_trace_world(
        "causal-heldout", f"causal_{_suffix()}", "CAUSAL_WORLD", 3.5
    )
    if causal_heldout.domain == software_train[0].domain:
        raise AssertionError("heldout domain is not distinct")
    if not inducer.matches(authorized, causal_heldout):
        raise AssertionError("software-derived latent schema did not structurally transfer")

    # Freeze actions before the external executor composes the causal heldout world.
    wrong_short = GeneratedLatentPathSchema(authorized.relation_tokens[:-1])
    wrong_token = GeneratedLatentPathSchema((
        "LATENT_REL::ffffffffffffffff",
        "LATENT_REL::ffffffffffffffff",
        "LATENT_REL::ffffffffffffffff",
    ))
    treatment = _external_hidden_chain_query(causal_heldout, authorized)
    remove = _external_hidden_chain_query(causal_heldout, None)
    wrong_short_effect = _external_hidden_chain_query(causal_heldout, wrong_short)
    wrong_token_effect = _external_hidden_chain_query(causal_heldout, wrong_token)
    if treatment != 1.0:
        raise AssertionError("cross-domain treatment failed heldout-world composed outcome")
    if remove != 0.0 or wrong_short_effect != 0.0 or wrong_token_effect != 0.0:
        raise AssertionError("REMOVE/WRONG retained heldout-world capability")

    hashes = tuple(_world_hash(world) for world in (*software_train, causal_heldout))
    if len(set(hashes)) != 3:
        raise AssertionError("training and causal-heldout worlds are not source-disjoint")

    report: Dict[str, object] = {
        "status": STATUS,
        "repository": "leehyen0/GENESIS-EX-NIHILO",
        "predecessor_semantic_relation_labels_consumed": False,
        "predecessor_node_kind_labels_consumed": False,
        "relation_vocabulary_generated_from_world_intervention_traces": True,
        "relation_token_contains_domain_name": False,
        "relation_token_contains_node_identifier": False,
        "software_training_world_count": 2,
        "software_generated_schema_count": len(schemas),
        "learned_schema_id": authorized.schema_id,
        "learned_relation_tokens": list(authorized.relation_tokens),
        "same_exact_schema_transferred_software_to_causal_world": True,
        "software_authority_pair_count": len(training_pairs),
        "one_context_insufficient_for_authority": True,
        "verifierless_policy_authority": False,
        "candidate_schema_frozen_before_external_composed_outcomes": True,
        "causal_heldout_action_frozen_before_target_outcome": True,
        "heldout_root_to_target_outcome_present_in_induction_traces": False,
        "external_outcome_consumes_concrete_heldout_world": True,
        "external_outcome_reconstructs_edges_from_raw_contrasts": True,
        "heldout_source_disjoint": True,
        "external_execution": "fresh_python_subprocess_composes_concrete_world_raw_contrasts",
        "treatment_capability": treatment,
        "remove_same_checkpoint_capability": remove,
        "wrong_short_path_capability": wrong_short_effect,
        "wrong_relation_token_capability": wrong_token_effect,
        "matched_intervention_trace_representation_human_authored": True,
        "lag_window_human_authored": True,
        "effect_threshold_and_sign_fingerprint_human_authored": True,
        "path_enumerator_human_authored": True,
        "evaluator_composition_semantics_human_authored": True,
        "unrestricted_relation_ontology_genesis": False,
        "unrestricted_meta_language_genesis": False,
        "unrestricted_operator_genesis": False,
        "natural_historical_cross_domain_failure": False,
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
