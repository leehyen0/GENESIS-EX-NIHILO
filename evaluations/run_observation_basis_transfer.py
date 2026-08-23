from __future__ import annotations

import hashlib
import json
import secrets
import subprocess
import sys
from dataclasses import asdict
from typing import Dict, Optional, Sequence, Tuple

from arte_cognition.latent_relation_ontology_genesis import (
    OpaqueInterventionalWorld,
    WorldDerivedLatentRelationInducer,
    contrast,
)
from arte_cognition.observation_basis_genesis import (
    GeneratedObservationPathSchema,
    WorldDerivedObservationBasisInducer,
    derive_observation_basis_policy,
    select_authorized_observation_schema,
)
from arte_cognition.world_coupling import WorldOutcomePair


STATUS = "PASS_BOUNDED_WORLD_DERIVED_OBSERVATION_BASIS_AND_CROSS_DOMAIN_PREOUTCOME_TRANSFER"


def _suffix() -> str:
    return f"{secrets.randbelow(10**9):09d}"


def _world(
    context_id: str,
    prefix: str,
    domain: str,
    magnitude: float,
    lags: Tuple[int, int, int] = (1, 2, 1),
    signs: Tuple[int, int, int] = (1, 1, 1),
) -> OpaqueInterventionalWorld:
    nodes = [f"{prefix}_{i}" for i in range(4)]
    decoy = f"{prefix}_decoy"
    all_nodes = nodes + [decoy]
    rows = []
    for source_index in range(3):
        lag = int(lags[source_index])
        sign = int(signs[source_index])
        for repeat in range(2):
            low = [{node: 0.0 for node in all_nodes} for _ in range(3)]
            high = [{node: 0.0 for node in all_nodes} for _ in range(3)]
            high[0][nodes[source_index]] = float(magnitude)
            high[lag][nodes[source_index + 1]] = float(sign) * float(magnitude)
            # Nondifferential temporal clutter cannot reveal the hidden path.
            clutter_lag = 1 if repeat == 0 else 2
            low[clutter_lag][decoy] = float(repeat + 1)
            high[clutter_lag][decoy] = float(repeat + 1)
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


def _payload(world: OpaqueInterventionalWorld) -> Dict[str, object]:
    return {
        "context_id": world.context_id,
        "domain": world.domain,
        "source_anchor": world.source_anchor,
        "target_anchor": world.target_anchor,
        "contrasts": [asdict(row) for row in world.contrasts],
    }


def _world_hash(world: OpaqueInterventionalWorld) -> str:
    raw = json.dumps(_payload(world), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _external_execute(
    world: OpaqueInterventionalWorld,
    schema: Optional[GeneratedObservationPathSchema],
) -> float:
    """Fresh-process evaluator reconstructs the concrete world's multi-lag path.

    It imports no ARTE cognition module. The frozen action consists only of the
    profile-token path. The executor independently finds each repeatedly supported
    source-target peak across all observed lags, derives the same opaque profile
    token rule, and asks whether the frozen action exactly follows a path from this
    world's source anchor to target anchor.
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
for row in w["contrasts"]:
    nodes.add(row["source_node"])
    for timeline_name in ("low_timeline", "high_timeline"):
        for snapshot in row[timeline_name]:
            nodes.update(name for name, _ in snapshot)

values = {}
for row in w["contrasts"]:
    source = row["source_node"]
    max_lag = min(len(row["low_timeline"]), len(row["high_timeline"])) - 1
    for lag in range(1, max_lag + 1):
        low = dict(row["low_timeline"][lag])
        high = dict(row["high_timeline"][lag])
        for target in nodes:
            if target == source:
                continue
            effect = float(high.get(target, 0.0)) - float(low.get(target, 0.0))
            values.setdefault((source, target, lag), []).append(effect)

by_pair = {}
for (source, target, lag), effects in values.items():
    nonzero = [x for x in effects if abs(x) >= 1e-9]
    if len(nonzero) < 2:
        continue
    mean = sum(nonzero) / len(nonzero)
    if abs(mean) < 1e-9:
        continue
    by_pair.setdefault((source, target), []).append((abs(mean), -lag, mean))

def token(lag, effect):
    sign = "POS" if effect > 0 else "NEG"
    raw = f"peak_lag={lag}|sign={sign}".encode()
    return "OBS_REL::" + hashlib.sha256(raw).hexdigest()[:16]

adj = {}
for (source, target), candidates in sorted(by_pair.items()):
    candidates.sort(reverse=True)
    _, neg_lag, effect = candidates[0]
    lag = -neg_lag
    adj.setdefault(source, []).append((target, token(lag, effect)))
for edges in adj.values():
    edges.sort()

paths = set()
def walk(node, visited, path):
    if len(path) >= 8:
        return
    for target, tok in adj.get(node, ()):
        if target in visited:
            continue
        nxt = path + (tok,)
        if target == w["target_anchor"]:
            paths.add(nxt)
        walk(target, visited + (target,), nxt)
walk(w["source_anchor"], (w["source_anchor"],), ())
print("1" if action in paths else "0")
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
    schema: GeneratedObservationPathSchema,
    context_id: str,
    cls: str,
    effect: float,
    verified: bool = True,
) -> WorldOutcomePair:
    return WorldOutcomePair(
        pair_id=f"PAIR::{schema.schema_id}::{context_id}::{cls}",
        experiment_id=schema.schema_id,
        axis_id="OBSERVATION_BASIS",
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
    training = (
        _world("software-a", f"softa_{_suffix()}", "SOFTWARE", 1.0),
        _world("software-b", f"softb_{_suffix()}", "SOFTWARE", 8.0),
    )

    # Complete predecessor failure is frozen before new-basis generation.
    predecessor = WorldDerivedLatentRelationInducer(
        lag=1, min_effect=0.1, min_repeats=2, max_path_depth=8
    )
    predecessor_assessment = predecessor.assess_residual(training, (0, 0), 2)
    predecessor_candidates = predecessor.generate_candidates(predecessor_assessment, training)
    if predecessor_candidates:
        raise AssertionError("fixed-lag predecessor unexpectedly solved multi-lag residual")
    for _ in range(16):
        if predecessor.generate_candidates(predecessor_assessment, training):
            raise AssertionError("MORE-COMPUTE changed fixed-lag predecessor expressivity")

    inducer = WorldDerivedObservationBasisInducer(
        min_repeats=2, min_peak_effect=1e-9, max_path_depth=8, candidate_budget=64
    )
    assessment = inducer.assess_residual(training, (0, 0), min_contexts=2)
    schemas = inducer.generate_candidates(assessment, training)
    if len(schemas) != 1:
        raise AssertionError(f"expected one multi-lag schema, got {len(schemas)}")
    schema = schemas[0]
    frozen_schema_id = schema.schema_id
    bases = tuple(inducer.derive_basis(world) for world in training)
    if any(basis is None or basis.lag_offsets != (1, 2) for basis in bases):
        raise AssertionError("raw timelines did not generate expected observation coordinates")

    # Only after candidate freeze do external consequences become available.
    training_pairs = []
    for world in training:
        effect = _external_execute(world, schema)
        for cls in ("AUTH_A", "AUTH_B"):
            training_pairs.append(_pair(schema, world.context_id, cls, effect))

    one_context = tuple(pair for pair in training_pairs if pair.context_id == training[0].context_id)
    one_policy = derive_observation_basis_policy((schema,), one_context, 2, 2)
    if select_authorized_observation_schema((schema,), one_policy) is not None:
        raise AssertionError("one context incorrectly authorized observation basis")

    verifierless = tuple(
        _pair(schema, pair.context_id, "UNVERIFIED", pair.effect, verified=False)
        for pair in training_pairs
    )
    verifierless_policy = derive_observation_basis_policy((schema,), verifierless, 2, 2)
    if select_authorized_observation_schema((schema,), verifierless_policy) is not None:
        raise AssertionError("verifierless evidence incorrectly authorized observation basis")

    policy = derive_observation_basis_policy((schema,), tuple(training_pairs), 2, 2)
    authorized = select_authorized_observation_schema((schema,), policy)
    if authorized is None or authorized.schema_id != frozen_schema_id:
        raise AssertionError("external training consequences did not authorize frozen basis")

    # Generate causal heldout only after software authority exists.
    heldout = _world(
        "causal-heldout", f"causal_{_suffix()}", "CAUSAL_WORLD", 3.5,
        lags=(1, 2, 1), signs=(1, 1, 1),
    )
    if not inducer.matches(authorized, heldout):
        raise AssertionError("authorized multi-lag basis did not transfer before heldout outcome")

    # Freeze matched-resource counterfactual actions before heldout execution.
    lag1_basis = inducer.derive_basis(_world(
        "wrong-lag-shadow", f"wl_{_suffix()}", "SHADOW", 2.0,
        lags=(1, 1, 1), signs=(1, 1, 1),
    ))
    wrong_lag_world = _world(
        "wrong-lag", f"wronglag_{_suffix()}", "CAUSAL_WORLD", 3.5,
        lags=(1, 1, 1), signs=(1, 1, 1),
    )
    wrong_sign_world = _world(
        "wrong-sign", f"wrongsign_{_suffix()}", "CAUSAL_WORLD", 3.5,
        lags=(1, 2, 1), signs=(1, -1, 1),
    )

    treatment = _external_execute(heldout, authorized)
    remove = _external_execute(heldout, None)
    wrong_lag_same_action = _external_execute(wrong_lag_world, authorized)
    wrong_sign_same_action = _external_execute(wrong_sign_world, authorized)
    if treatment != 1.0:
        raise AssertionError("Treatment failed concrete causal heldout world")
    if remove != 0.0 or wrong_lag_same_action != 0.0 or wrong_sign_same_action != 0.0:
        raise AssertionError("REMOVE/WRONG retained heldout capability")

    hashes = tuple(_world_hash(world) for world in (*training, heldout, wrong_lag_world, wrong_sign_world))
    if len(set(hashes)) != len(hashes):
        raise AssertionError("training/heldout/counterexample worlds are not source-disjoint")

    report: Dict[str, object] = {
        "status": STATUS,
        "repository": "leehyen0/GENESIS-EX-NIHILO",
        "predecessor_fixed_lag": 1,
        "predecessor_selected_count": len(predecessor_candidates),
        "predecessor_more_compute_attempts": 16,
        "predecessor_more_compute_selected_count": 0,
        "generated_observation_lags": [1, 2],
        "generated_schema_count": len(schemas),
        "learned_schema_id": authorized.schema_id,
        "learned_profile_tokens": list(authorized.profile_tokens),
        "candidate_generation_uses_external_outcomes": False,
        "candidate_frozen_before_external_outcomes": True,
        "software_authority_pair_count": len(training_pairs),
        "one_context_insufficient_for_authority": True,
        "verifierless_policy_authority": False,
        "same_exact_schema_transferred_software_to_causal_world": True,
        "heldout_action_frozen_before_outcome": True,
        "external_executor_consumes_concrete_multilag_raw_timelines": True,
        "external_executor_imports_arte_inducer": False,
        "heldout_source_disjoint": True,
        "treatment_capability": treatment,
        "remove_same_checkpoint_capability": remove,
        "wrong_temporal_basis_capability": wrong_lag_same_action,
        "wrong_relation_sign_capability": wrong_sign_same_action,
        "matched_trace_representation_human_authored": True,
        "arithmetic_differencing_human_authored": True,
        "peak_selection_rule_human_authored": True,
        "minimum_repeat_rule_human_authored": True,
        "path_enumerator_human_authored": True,
        "response_partition_genesis": False,
        "unrestricted_observation_basis_genesis": False,
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
