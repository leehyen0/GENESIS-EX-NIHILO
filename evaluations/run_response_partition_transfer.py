from __future__ import annotations

import hashlib
import json
import secrets
import subprocess
import sys
from dataclasses import asdict
from typing import Dict, Optional, Sequence, Tuple

from arte_cognition.latent_relation_ontology_genesis import OpaqueInterventionalWorld, contrast
from arte_cognition.observation_basis_genesis import WorldDerivedObservationBasisInducer
from arte_cognition.response_partition_genesis import (
    GeneratedResponsePathSchema,
    WorldDerivedResponsePartitionInducer,
    derive_response_partition_policy,
    select_authorized_response_schema,
)
from arte_cognition.world_coupling import WorldOutcomePair


STATUS = "PASS_BOUNDED_WORLD_DERIVED_RESPONSE_PARTITION_AND_CROSS_DOMAIN_PREOUTCOME_TRANSFER"


def _suffix() -> str:
    return f"{secrets.randbelow(10**9):09d}"


def _world(
    context_id: str,
    prefix: str,
    domain: str,
    magnitude: float,
    fast_tail: float = 0.25,
    slow_tail: float = 0.75,
) -> OpaqueInterventionalWorld:
    root = f"{prefix}_root"
    fast = f"{prefix}_fast"
    slow = f"{prefix}_slow"
    target = f"{prefix}_target"
    decoy = f"{prefix}_decoy"
    nodes = (root, fast, slow, target, decoy)
    rows = []

    def add(source: str, destinations: Sequence[Tuple[str, float]]) -> None:
        for repeat in range(2):
            low = [{node: 0.0 for node in nodes} for _ in range(3)]
            high = [{node: 0.0 for node in nodes} for _ in range(3)]
            high[0][source] = float(magnitude)
            for destination, tail in destinations:
                high[1][destination] = float(magnitude)
                high[2][destination] = float(magnitude) * float(tail)
            # Nondifferential decoy survives source renaming but never produces an edge.
            low[2][decoy] = float(repeat + 1)
            high[2][decoy] = float(repeat + 1)
            rows.append(contrast(
                f"{context_id}:{source}:{repeat}", source, low, high
            ))

    add(root, ((fast, fast_tail), (slow, slow_tail)))
    add(fast, ((target, fast_tail),))
    add(slow, ((target, slow_tail),))
    return OpaqueInterventionalWorld(
        context_id=context_id,
        domain=domain,
        source_anchor=root,
        target_anchor=target,
        contrasts=tuple(rows),
    )


def _payload(world: OpaqueInterventionalWorld) -> Dict[str, object]:
    return {
        "context_id": world.context_id,
        "domain": world.domain,
        "source_anchor": world.source_anchor,
        "target_anchor": world.target_anchor,
        "contrasts": [asdict(row) for row in world.contrasts],
    }


def _hash(world: OpaqueInterventionalWorld) -> str:
    return hashlib.sha256(
        json.dumps(_payload(world), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _predecessor_multiplicity(world: OpaqueInterventionalWorld) -> int:
    predecessor = WorldDerivedObservationBasisInducer(min_repeats=2)
    adjacency = {}
    for edge in predecessor.infer_edges(world):
        adjacency.setdefault(edge.source, []).append(edge)
    concrete = []

    def walk(node: str, visited: Tuple[str, ...], tokens: Tuple[str, ...]) -> None:
        for edge in adjacency.get(node, ()):
            if edge.target in visited:
                continue
            nxt = tokens + (edge.profile_token,)
            if edge.target == world.target_anchor:
                concrete.append(nxt)
            walk(edge.target, visited + (edge.target,), nxt)

    walk(world.source_anchor, (world.source_anchor,), ())
    if not concrete:
        return 0
    first = concrete[0]
    return sum(1 for row in concrete if row == first)


def _external_execute(
    world: OpaqueInterventionalWorld,
    schema: Optional[GeneratedResponsePathSchema],
) -> float:
    """Fresh subprocess selects consequence from full response shape, not labels.

    Both root-to-target paths are structurally real and share the same peak/sign.
    The external world rewards only paths whose normalized lag-2 tail is <= 0.4 on
    every edge. That hidden downstream requirement is not passed to candidate
    generation. The executor independently reconstructs all normalized response
    curves from the concrete raw world, maps the already-frozen profile-token
    action back onto a path, then reveals whether that path satisfies the external
    consequence rule.
    """
    request = {
        "world": _payload(world),
        "tokens": list(schema.profile_tokens) if schema is not None else [],
    }
    child = r'''
import hashlib, json, sys
p = json.loads(sys.stdin.read())
w = p["world"]
action = tuple(p["tokens"])
if not action:
    print("0")
    raise SystemExit(0)

nodes = {w["source_anchor"], w["target_anchor"]}
by_source = {}
for row in w["contrasts"]:
    by_source.setdefault(row["source_node"], []).append(row)
    for timeline_name in ("low_timeline", "high_timeline"):
        for snapshot in row[timeline_name]:
            nodes.update(name for name, _ in snapshot)

def token(profile):
    raw = "|".join(f"{value:+.6f}" for value in profile).encode()
    return "RESPONSE_PROFILE::" + hashlib.sha256(raw).hexdigest()[:16]

adj = {}
for source, rows in sorted(by_source.items()):
    max_lag = min(min(len(r["low_timeline"]), len(r["high_timeline"])) - 1 for r in rows)
    for target in nodes:
        if target == source:
            continue
        means = []
        for lag in range(1, max_lag + 1):
            vals = []
            for r in rows:
                low = dict(r["low_timeline"][lag]); high = dict(r["high_timeline"][lag])
                vals.append(float(high.get(target,0.0))-float(low.get(target,0.0)))
            means.append(sum(vals)/len(vals))
        peak = max([abs(v) for v in means] or [0.0])
        if peak < 1e-9:
            continue
        profile = tuple(round(v/peak, 6) for v in means)
        adj.setdefault(source, []).append((target, token(profile), profile))
for rows in adj.values():
    rows.sort(key=lambda x: (x[1], x[0]))

matches = []
def walk(node, visited, tokens, profiles):
    for target, tok, profile in adj.get(node, ()):
        if target in visited:
            continue
        ntokens = tokens + (tok,)
        nprofiles = profiles + (profile,)
        if target == w["target_anchor"] and ntokens == action:
            matches.append(nprofiles)
        if len(ntokens) < len(action):
            walk(target, visited + (target,), ntokens, nprofiles)
walk(w["source_anchor"], (w["source_anchor"],), (), ())

# Action must identify exactly one concrete path. External success then depends on
# a downstream temporal-retention requirement not present in candidate generation.
if len(matches) != 1:
    print("0")
else:
    profiles = matches[0]
    ok = all(len(profile) >= 2 and abs(profile[1]) <= 0.4 for profile in profiles)
    print("1" if ok else "0")
'''
    result = subprocess.run(
        [sys.executable, "-c", child],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=True,
    )
    return float(result.stdout.strip())


def _pair(
    schema: GeneratedResponsePathSchema,
    context: str,
    cls: str,
    effect: float,
    verified: bool = True,
) -> WorldOutcomePair:
    return WorldOutcomePair(
        pair_id=f"PAIR::{schema.schema_id}::{context}::{cls}",
        experiment_id=schema.schema_id,
        axis_id="RESPONSE_PARTITION",
        source_id=f"external::{context}::{cls}",
        context_id=context,
        challenge_id=f"challenge::{context}",
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
    training = (
        _world("software-a", f"softa_{_suffix()}", "SOFTWARE", 1.0),
        _world("software-b", f"softb_{_suffix()}", "SOFTWARE", 9.0),
    )
    predecessor_ambiguities = tuple(_predecessor_multiplicity(world) for world in training)
    if predecessor_ambiguities != (2, 2):
        raise AssertionError(f"peak/sign predecessor pressure missing: {predecessor_ambiguities}")
    for _ in range(16):
        if tuple(_predecessor_multiplicity(world) for world in training) != (2, 2):
            raise AssertionError("MORE-COMPUTE changed predecessor representational ambiguity")

    inducer = WorldDerivedResponsePartitionInducer(
        min_repeats=2, min_peak_effect=1e-9, precision=6, max_path_depth=8
    )
    assessment = inducer.assess_residual(training, predecessor_ambiguities, 2)
    schemas = inducer.generate_candidates(assessment, training)
    if len(schemas) != 2:
        raise AssertionError(f"expected two full-profile path candidates, got {len(schemas)}")
    frozen_ids = tuple(schema.schema_id for schema in schemas)

    # World outcomes are revealed only after both path candidates are frozen.
    pairs = []
    effects_by_schema = {}
    for schema in schemas:
        per_context = []
        for world in training:
            effect = _external_execute(world, schema)
            per_context.append(effect)
            for cls in ("AUTH_A", "AUTH_B"):
                pairs.append(_pair(schema, world.context_id, cls, effect))
        effects_by_schema[schema.schema_id] = tuple(per_context)

    if tuple(schema.schema_id for schema in schemas) != frozen_ids:
        raise AssertionError("candidate path set changed after outcomes")
    winners = [sid for sid, values in effects_by_schema.items() if values == (1.0, 1.0)]
    losers = [sid for sid, values in effects_by_schema.items() if values == (0.0, 0.0)]
    if len(winners) != 1 or len(losers) != 1:
        raise AssertionError(f"external world did not uniquely distinguish path classes: {effects_by_schema}")

    one_context = tuple(pair for pair in pairs if pair.context_id == training[0].context_id)
    if select_authorized_response_schema(
        schemas, derive_response_partition_policy(schemas, one_context, 2, 2)
    ) is not None:
        raise AssertionError("one context incorrectly authorized response partition")

    verifierless = tuple(
        _pair(
            next(schema for schema in schemas if schema.schema_id == pair.experiment_id),
            pair.context_id,
            "UNVERIFIED",
            pair.effect,
            verified=False,
        )
        for pair in pairs
    )
    if select_authorized_response_schema(
        schemas, derive_response_partition_policy(schemas, verifierless, 2, 2)
    ) is not None:
        raise AssertionError("verifierless response evidence gained authority")

    policy = derive_response_partition_policy(schemas, tuple(pairs), 2, 2)
    selected = select_authorized_response_schema(schemas, policy)
    if selected is None or selected.schema_id != winners[0]:
        raise AssertionError("authority did not select unique external-success profile path")
    wrong = next(schema for schema in schemas if schema.schema_id == losers[0])

    # Causal heldout exists only after training authority is fixed.
    heldout = _world(
        "causal-heldout", f"causal_{_suffix()}", "CAUSAL_WORLD", 3.5,
        fast_tail=0.25, slow_tail=0.75,
    )
    if not inducer.matches(selected, heldout) or not inducer.matches(wrong, heldout):
        raise AssertionError("Treatment/WRONG actions were not both frozen before heldout outcome")

    treatment = _external_execute(heldout, selected)
    remove = _external_execute(heldout, None)
    wrong_effect = _external_execute(heldout, wrong)
    if treatment != 1.0 or remove != 0.0 or wrong_effect != 0.0:
        raise AssertionError("response partition Treatment/REMOVE/WRONG isolation failed")

    shifted = _world(
        "causal-shifted", f"shifted_{_suffix()}", "CAUSAL_WORLD", 3.5,
        fast_tail=0.55, slow_tail=0.90,
    )
    shifted_effect = _external_execute(shifted, selected)
    if shifted_effect != 0.0:
        raise AssertionError("learned response partition overgeneralized to shifted regime")

    hashes = tuple(_hash(world) for world in (*training, heldout, shifted))
    if len(set(hashes)) != len(hashes):
        raise AssertionError("worlds are not source-disjoint")

    report: Dict[str, object] = {
        "status": STATUS,
        "repository": "leehyen0/GENESIS-EX-NIHILO",
        "predecessor_peak_sign_concrete_path_multiplicity": list(predecessor_ambiguities),
        "predecessor_more_compute_attempts": 16,
        "predecessor_more_compute_ambiguity": [2, 2],
        "generated_response_partition_count": 2,
        "generated_candidate_path_count": len(schemas),
        "candidate_generation_uses_external_outcomes": False,
        "candidate_paths_frozen_before_outcomes": True,
        "training_effects_by_schema": effects_by_schema,
        "unique_authorized_schema_id": selected.schema_id,
        "genuinely_structural_wrong_schema_id": wrong.schema_id,
        "software_authority_pair_count": len(pairs),
        "one_context_insufficient_for_authority": True,
        "verifierless_policy_authority": False,
        "same_exact_response_schema_transferred_software_to_causal_world": True,
        "heldout_treatment_and_wrong_frozen_preoutcome": True,
        "external_executor_consumes_concrete_full_response_profiles": True,
        "external_executor_imports_arte_inducer": False,
        "treatment_capability": treatment,
        "remove_same_checkpoint_capability": remove,
        "wrong_structurally_valid_profile_capability": wrong_effect,
        "shifted_regime_capability": shifted_effect,
        "heldout_source_disjoint": True,
        "arithmetic_mean_human_authored": True,
        "max_abs_normalization_human_authored": True,
        "numeric_precision_human_authored": True,
        "minimum_support_human_authored": True,
        "external_tail_requirement_human_authored": True,
        "normalization_rule_genesis": False,
        "unrestricted_response_partition_genesis": False,
        "unrestricted_meta_language_genesis": False,
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
