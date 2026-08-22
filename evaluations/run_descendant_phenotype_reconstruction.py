from __future__ import annotations

import json
import random
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.adaptive_cognition import TaskState
from arte_cognition.body_checkpoint import checkpoint_json, restore_json
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.world_action_policy import EvidenceBoundWorldActionPolicy
from arte_cognition.world_coupling import HMACWorldReceiptSigner, HMACWorldReceiptVerifier
from evaluations.run_end_to_end_world_genesis import HiddenAffineWorld, build_observations


def main(seed_path: str) -> None:
    seed = int(Path(seed_path).read_text().strip())
    rng = random.Random(seed)

    scale = rng.choice((-1.0, 1.0)) * rng.uniform(0.6, 3.0)
    swap = bool(rng.getrandbits(1))
    feature_names = (
        f"phenotype_sensor_{rng.randrange(10_000, 99_999)}",
        f"phenotype_sensor_{rng.randrange(100_000, 999_999)}",
    )
    label_flip = bool(rng.getrandbits(1))
    issuer_ids = (
        f"phenotype-evaluator-a-{rng.randrange(1_000_000, 9_999_999)}",
        f"phenotype-evaluator-b-{rng.randrange(1_000_000, 9_999_999)}",
    )
    secrets_by_issuer = {
        issuer_ids[0]: secrets.token_bytes(32),
        issuer_ids[1]: secrets.token_bytes(32),
    }
    signers = {
        issuer: HMACWorldReceiptSigner(issuer, secret)
        for issuer, secret in secrets_by_issuer.items()
    }
    verifier = HMACWorldReceiptVerifier(
        secrets_by_issuer,
        independence_classes={
            issuer_ids[0]: "phenotype-independent-a",
            issuer_ids[1]: "phenotype-independent-b",
        },
    )

    discovery_world = HiddenAffineWorld(
        scale=scale,
        swap=swap,
        feature_names=feature_names,
        source_id="phenotype-discovery-source",
        challenge_id="phenotype-discovery-challenge",
        epoch=0,
        signer=signers[issuer_ids[0]],
    )
    measurements, residuals = build_observations(discovery_world, label_flip)
    reference_values = {name: 0.0 for name in feature_names}

    runtime = PersistentCognitiveRuntime()
    cycle = runtime.cycle(
        TaskState(
            goal="discover and persist a latent representation phenotype",
            novelty=0.95,
            residuals=[row.residual_id for row in residuals if not row.heldout],
            external_world=True,
            action_required=True,
        ),
        residuals=residuals,
        measurements=measurements,
        experiment_reference_values=reference_values,
        world_context_id=discovery_world.context_id,
    )

    assessment_by_axis = {item.axis_id: item for item in cycle.representation_value}
    eligible = [
        axis for axis in cycle.representation_axes
        if axis.family == "PROJECTION"
        and assessment_by_axis.get(axis.axis_id) is not None
        and assessment_by_axis[axis.axis_id].status == "INCREMENTAL_REPRESENTATION_VALUE"
    ]
    if not eligible:
        raise AssertionError("no validated projection phenotype was generated")
    selected_axis = max(eligible, key=lambda axis: assessment_by_axis[axis.axis_id].incremental_gain)

    parent_axes = runtime.persisted_representation_axes()
    parent_proposals = runtime.persisted_intervention_proposals()
    if selected_axis not in parent_axes:
        raise AssertionError("validated generated axis was not written into BODY phenotype memory")
    selected_proposals = [p for p in parent_proposals if p.axis_id == selected_axis.axis_id]
    if not selected_proposals:
        raise AssertionError("generated intervention definition was not written into BODY phenotype memory")
    selected_proposal = selected_proposals[0]

    policy = EvidenceBoundWorldActionPolicy()
    before = policy.select(parent_proposals, runtime.world_coupling, context_id=discovery_world.context_id)
    if before.status != "EXPLORE_ONLY_NO_WORLD_SUPPORTED_ACTION":
        raise AssertionError("persisted phenotype self-promoted before world evidence")

    for index, issuer_id in enumerate(issuer_ids, start=1):
        world = HiddenAffineWorld(
            scale=scale,
            swap=swap,
            feature_names=feature_names,
            source_id=f"phenotype-source-{index}",
            challenge_id=f"phenotype-challenge-{index}",
            epoch=index,
            signer=signers[issuer_id],
        )
        pair = runtime.execute_world_intervention(selected_proposal, world, verifier=verifier)
        if not pair.authority_verified:
            raise AssertionError("phenotype world outcome failed authority verification")

    after = policy.select(parent_proposals, runtime.world_coupling, context_id=discovery_world.context_id)
    if after.status != "WORLD_SUPPORTED_ACTION" or after.proposal is None:
        raise AssertionError("persisted phenotype did not become a world-supported action")

    selected_axis_id = selected_axis.axis_id
    selected_coefficients = tuple(selected_axis.coefficients)
    selected_threshold = float(selected_axis.threshold)
    selected_experiment_id = selected_proposal.experiment_id
    selected_low = float(selected_proposal.low_value)
    selected_high = float(selected_proposal.high_value)
    encoded = checkpoint_json(runtime)
    payload = json.loads(encoded)
    if payload.get("schema") != "arte.cognition_body_checkpoint/v3":
        raise AssertionError("generated phenotype changed the authenticated BODY authority envelope")
    if payload.get("phenotype_schema") != "arte.cognition_generated_phenotype/v1":
        raise AssertionError("generated phenotype sub-schema was not declared")
    if selected_axis_id not in payload.get("memory", {}).get("representations", {}):
        raise AssertionError("checkpoint omitted exact generated representation phenotype")
    if selected_experiment_id not in payload.get("memory", {}).get("experiments", {}):
        raise AssertionError("checkpoint omitted generated experiment definition")
    if any(secret.hex() in encoded for secret in secrets_by_issuer.values()):
        raise AssertionError("external verifier secret leaked into phenotype checkpoint")

    # From this point the parent cycle/proposal collections are deliberately not
    # used. The descendant must recover its own phenotype from BODY state.
    del cycle, parent_axes, parent_proposals, selected_axis, selected_proposal

    unverified_descendant = restore_json(encoded)
    unverified_axes = unverified_descendant.persisted_representation_axes()
    unverified_proposals = unverified_descendant.persisted_intervention_proposals()
    reconstructed_axis = next((axis for axis in unverified_axes if axis.axis_id == selected_axis_id), None)
    reconstructed_proposal = next((p for p in unverified_proposals if p.experiment_id == selected_experiment_id), None)
    if reconstructed_axis is None or reconstructed_proposal is None:
        raise AssertionError("descendant could not reconstruct generated phenotype from checkpoint")
    if tuple(reconstructed_axis.coefficients) != selected_coefficients or float(reconstructed_axis.threshold) != selected_threshold:
        raise AssertionError("descendant reconstructed different latent coefficients or threshold")
    if float(reconstructed_proposal.low_value) != selected_low or float(reconstructed_proposal.high_value) != selected_high:
        raise AssertionError("descendant reconstructed different intervention values")
    unverified_action = policy.select(
        unverified_proposals,
        unverified_descendant.world_coupling,
        context_id=discovery_world.context_id,
    )
    if unverified_action.status != "EXPLORE_ONLY_NO_WORLD_SUPPORTED_ACTION":
        raise AssertionError("phenotype reconstruction incorrectly restored external authority")

    descendant = restore_json(encoded, world_verifier=verifier)
    descendant_axes = descendant.persisted_representation_axes()
    descendant_proposals = descendant.persisted_intervention_proposals()
    inherited_axis = next((axis for axis in descendant_axes if axis.axis_id == selected_axis_id), None)
    inherited_proposal = next((p for p in descendant_proposals if p.experiment_id == selected_experiment_id), None)
    if inherited_axis is None or inherited_proposal is None:
        raise AssertionError("reverified descendant lost its generated phenotype")
    descendant_action = policy.select(
        descendant_proposals,
        descendant.world_coupling,
        context_id=discovery_world.context_id,
    )
    if descendant_action.status != "WORLD_SUPPORTED_ACTION" or descendant_action.proposal is None:
        raise AssertionError("descendant did not recover world-supported action from its own phenotype")
    if descendant_action.proposal.experiment_id != selected_experiment_id:
        raise AssertionError("descendant selected an action other than its inherited generated experiment")

    print(json.dumps({
        "status": "PASS_BOUNDED_DESCENDANT_RECONSTRUCTS_GENERATED_PHENOTYPE_FROM_BODY",
        "checkpoint_schema": payload["schema"],
        "phenotype_schema": payload["phenotype_schema"],
        "axis_id": selected_axis_id,
        "axis_coefficients": list(selected_coefficients),
        "axis_threshold": selected_threshold,
        "experiment_id": selected_experiment_id,
        "experiment_low": selected_low,
        "experiment_high": selected_high,
        "parent_objects_reused_after_restore": False,
        "unverified_descendant_action_status": unverified_action.status,
        "reverified_descendant_action_status": descendant_action.status,
        "reverified_descendant_experiment_id": descendant_action.proposal.experiment_id,
        "independent_world_evidence_classes": descendant.world_axis_summary(
            selected_axis_id,
            context_id=discovery_world.context_id,
        ).independent_evidence_classes,
        "external_verifier_secret_persisted": False,
        "independent_organizational_custody": False,
        "physical_world": False,
        "recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_descendant_phenotype_reconstruction.py <evaluator-owned-seed-file>")
    main(sys.argv[1])
