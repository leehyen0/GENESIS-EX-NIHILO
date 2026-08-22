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
from arte_cognition.representation_genesis import MeasurementObservation
from arte_cognition.semantic_genesis import ResidualObservation
from arte_cognition.world_action_policy import EvidenceBoundWorldActionPolicy
from arte_cognition.world_coupling import HMACWorldReceiptSigner, HMACWorldReceiptVerifier, WorldOutcomeReceipt
from arte_cognition.world_revision import AuthenticatedWorldCognitionReviser
from evaluations.run_world_caused_cognition_rewrite import (
    HiddenRegimeWorld,
    NEW_HELDOUT,
    NEW_TRAIN,
    OLD_HELDOUT,
    OLD_TRAIN,
    best_incremental,
    build_observations,
    execute_many,
    remember_reference_experiments,
)


ABS_TRAIN = (
    ("ap1", 1.00, 0.00),
    ("ap2", -1.00, 0.00),
    ("ap3", 0.00, 1.00),
    ("ap4", 0.00, -1.00),
    ("an1", 1.00, 1.00),
    ("an2", -1.00, -1.00),
    ("an3", 0.50, 0.50),
    ("an4", -0.50, -0.50),
)
ABS_HELDOUT = (
    ("ahp1", 0.75, 0.00),
    ("ahp2", 0.00, -0.75),
    ("ahn1", 0.75, 0.75),
    ("ahn2", -0.75, -0.75),
)


class ThreeRegimeWorld(HiddenRegimeWorld):
    """Same hidden sensor surface with SUM, DIFF, then ABS_DIFF mechanisms."""

    def positive(self, base_x: float, base_y: float) -> bool:
        if self.rule == "ABS_DIFF":
            return abs(float(base_x) - float(base_y)) > 0.50
        return super().positive(base_x, base_y)


def build_three_regime_observations(world, train, heldout, label_flip):
    measurements = []
    residuals = []
    for is_heldout, rows in ((False, train), (True, heldout)):
        for observation_id, base_x, base_y in rows:
            label = "B" if world.positive(base_x, base_y) else "A"
            if label_flip:
                label = "A" if label == "B" else "B"
            measurements.append(MeasurementObservation(
                observation_id=observation_id,
                values=world.encode(base_x, base_y),
                outcome=label,
                heldout=is_heldout,
                context_id=world.context_id,
            ))
            residuals.append(ResidualObservation(
                residual_id=observation_id,
                features=("raw-multi-episode-residual",),
                outcome=label,
                source_class=f"hidden-{world.context_id}",
                heldout=is_heldout,
            ))
    return measurements, residuals


def execute_abs_many(
    runtime,
    proposals,
    scale,
    swap,
    feature_names,
    context_id,
    issuer_ids,
    signers,
    verifier,
    epoch_base,
):
    effects = {}
    for proposal_index, proposal in enumerate(proposals):
        effects[proposal.experiment_id] = []
        for issuer_index, issuer_id in enumerate(issuer_ids):
            world = ThreeRegimeWorld(
                scale=scale,
                swap=swap,
                feature_names=feature_names,
                rule="ABS_DIFF",
                context_id=context_id,
                source_id=f"{context_id}-source-{proposal_index}-{issuer_index}",
                challenge_id=f"{context_id}-challenge-{proposal_index}-{issuer_index}",
                epoch=epoch_base + proposal_index * 10 + issuer_index,
                signer=signers[issuer_id],
            )
            pair = runtime.execute_world_intervention(proposal, world, verifier=verifier)
            if not pair.authority_verified:
                raise AssertionError("ABS_DIFF world receipt failed external authentication")
            effects[proposal.experiment_id].append(pair.effect)
    return effects


def cycle_for(runtime, goal, context_id, measurements, residuals, revision_residual=None):
    all_residuals = ([revision_residual] if revision_residual is not None else []) + list(residuals)
    return runtime.cycle(
        TaskState(
            goal=goal,
            novelty=0.99,
            residuals=[row.residual_id for row in all_residuals if not row.heldout],
            external_world=True,
            action_required=True,
        ),
        residuals=all_residuals,
        measurements=measurements,
        world_context_id=context_id,
    )


def material_proposals(proposals, effects, minimum=0.5):
    return [
        proposal for proposal in proposals
        if effects.get(proposal.experiment_id)
        and min(abs(value) for value in effects[proposal.experiment_id]) >= minimum
    ]


def require_semantic_reentry(cycle, axis_id):
    concepts = [concept for concept in cycle.concepts if axis_id in concept.defining_features]
    concept_ids = {concept.concept_id for concept in concepts}
    laws = [law for law in cycle.laws if law.concept_id in concept_ids]
    if not concepts or not any(law.status == "BOUNDED_LAW" for law in laws):
        raise AssertionError(f"representation {axis_id} failed bounded semantic re-entry")
    return len(concepts), sum(law.status == "BOUNDED_LAW" for law in laws)


def main(seed_path: str) -> None:
    seed = int(Path(seed_path).read_text().strip())
    rng = random.Random(seed)

    scale = rng.choice((-1.0, 1.0)) * rng.uniform(0.65, 2.75)
    swap = bool(rng.getrandbits(1))
    feature_names = (
        f"episode_sensor_{rng.randrange(10_000, 99_999)}",
        f"episode_sensor_{rng.randrange(100_000, 999_999)}",
    )
    label_flip = bool(rng.getrandbits(1))
    issuer_ids = (
        f"episode-evaluator-a-{rng.randrange(1_000_000, 9_999_999)}",
        f"episode-evaluator-b-{rng.randrange(1_000_000, 9_999_999)}",
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
            issuer_ids[0]: "episode-independent-a",
            issuer_ids[1]: "episode-independent-b",
        },
    )
    policy = EvidenceBoundWorldActionPolicy()
    reviser = AuthenticatedWorldCognitionReviser()
    runtime = PersistentCognitiveRuntime()

    sum_context = "episode-1-sum"
    diff_context = "episode-2-diff"
    abs_context = "episode-3-abs-diff"

    # Episode 1: the hidden SUM mechanism requires a generated latent projection.
    sum_world = ThreeRegimeWorld(
        scale, swap, feature_names, "SUM", sum_context,
        "sum-observation-source", "sum-observation-challenge", 0,
        signers[issuer_ids[0]],
    )
    sum_measurements, sum_residuals = build_three_regime_observations(
        sum_world, OLD_TRAIN, OLD_HELDOUT, label_flip
    )
    sum_cycle = cycle_for(
        runtime,
        "discover first hidden world mechanism",
        sum_context,
        sum_measurements,
        sum_residuals,
    )
    projection_axis, projection_assessments = best_incremental(sum_cycle, "PROJECTION")
    if projection_assessments[projection_axis.axis_id].heldout_accuracy != 1.0:
        raise AssertionError("episode-1 projection failed held-out reproduction")
    require_semantic_reentry(sum_cycle, projection_axis.axis_id)
    projection_proposals = remember_reference_experiments(runtime, projection_axis, sum_measurements)
    projection_effects = execute_many(
        runtime,
        projection_proposals,
        scale,
        swap,
        feature_names,
        "SUM",
        sum_context,
        issuer_ids,
        signers,
        verifier,
        1000,
    )
    projection_supported = material_proposals(projection_proposals, projection_effects)
    if len(projection_supported) < 2:
        raise AssertionError("episode-1 produced fewer than two material exact experiments")
    sum_action = policy.select(projection_proposals, runtime.world_coupling, context_id=sum_context)
    if sum_action.status != "WORLD_SUPPORTED_ACTION" or sum_action.proposal is None:
        raise AssertionError("episode-1 cognition failed to earn action authority")

    # Shift 1: SUM -> DIFF. The old projection is contradicted by multiple exact
    # experiments before any new observations are provided.
    execute_many(
        runtime,
        projection_supported,
        scale,
        swap,
        feature_names,
        "DIFF",
        diff_context,
        issuer_ids,
        signers,
        verifier,
        5000,
    )
    revision_1 = reviser.assess_and_apply(
        runtime.memory,
        runtime.world_coupling,
        projection_axis.axis_id,
        sum_context,
        diff_context,
    )
    if revision_1.status != "PASS_BOUNDED_WORLD_CAUSED_COGNITION_DEMOTION":
        raise AssertionError(f"first world rewrite failed: {revision_1.status}")
    if len(revision_1.counterevidence) < 2 or revision_1.residual is None:
        raise AssertionError("first rewrite lacked robust exact counterevidence or residual")
    projection_stale_ids = {item.experiment_id for item in revision_1.counterevidence}

    # Episode 2: fresh source-disjoint DIFF observations contract cognition to the
    # simpler minimum-sufficient DIFFERENCE family.
    diff_world = ThreeRegimeWorld(
        scale, swap, feature_names, "DIFF", diff_context,
        "diff-observation-source", "diff-observation-challenge", 9000,
        signers[issuer_ids[1]],
    )
    diff_measurements, diff_residuals = build_three_regime_observations(
        diff_world, NEW_TRAIN, NEW_HELDOUT, label_flip
    )
    diff_cycle = cycle_for(
        runtime,
        "rebuild cognition after first authenticated world shift",
        diff_context,
        diff_measurements,
        diff_residuals,
        revision_residual=revision_1.residual,
    )
    difference_axis, difference_assessments = best_incremental(diff_cycle, "DIFFERENCE")
    if difference_assessments[difference_axis.axis_id].heldout_accuracy != 1.0:
        raise AssertionError("episode-2 difference representation failed held-out reproduction")
    if difference_axis.axis_id == projection_axis.axis_id:
        raise AssertionError("first rewrite did not change representation identity")
    require_semantic_reentry(diff_cycle, difference_axis.axis_id)
    difference_proposals = remember_reference_experiments(runtime, difference_axis, diff_measurements)
    difference_effects = execute_many(
        runtime,
        difference_proposals,
        scale,
        swap,
        feature_names,
        "DIFF",
        diff_context,
        issuer_ids,
        signers,
        verifier,
        12000,
    )
    difference_supported = material_proposals(difference_proposals, difference_effects)
    if len(difference_supported) < 2:
        raise AssertionError("episode-2 produced fewer than two material exact experiments")
    diff_action = policy.select(difference_proposals, runtime.world_coupling, context_id=diff_context)
    if diff_action.status != "WORLD_SUPPORTED_ACTION" or diff_action.proposal is None:
        raise AssertionError("episode-2 cognition failed to earn action authority")

    # Shift 2: DIFF -> ABS_DIFF. The local signed-difference probes now collapse:
    # both intervention arms lie inside the hidden |x-y| <= .5 region.
    execute_abs_many(
        runtime,
        difference_supported,
        scale,
        swap,
        feature_names,
        abs_context,
        issuer_ids,
        signers,
        verifier,
        18000,
    )
    revision_2 = reviser.assess_and_apply(
        runtime.memory,
        runtime.world_coupling,
        difference_axis.axis_id,
        diff_context,
        abs_context,
    )
    if revision_2.status != "PASS_BOUNDED_WORLD_CAUSED_COGNITION_DEMOTION":
        raise AssertionError(f"second world rewrite failed: {revision_2.status}")
    if len(revision_2.counterevidence) < 2 or revision_2.residual is None:
        raise AssertionError("second rewrite lacked robust exact counterevidence or residual")
    difference_stale_ids = {item.experiment_id for item in revision_2.counterevidence}

    # Episode 3: ABS_DIFF data are symmetric. No raw parent or linear projection
    # can represent the two-sided separation; ABS_DIFFERENCE must add value.
    abs_world = ThreeRegimeWorld(
        scale, swap, feature_names, "ABS_DIFF", abs_context,
        "abs-observation-source", "abs-observation-challenge", 24000,
        signers[issuer_ids[0]],
    )
    abs_measurements, abs_residuals = build_three_regime_observations(
        abs_world, ABS_TRAIN, ABS_HELDOUT, label_flip
    )
    abs_cycle = cycle_for(
        runtime,
        "rebuild cognition after second authenticated world shift",
        abs_context,
        abs_measurements,
        abs_residuals,
        revision_residual=revision_2.residual,
    )
    abs_axis, abs_assessments = best_incremental(abs_cycle, "ABS_DIFFERENCE")
    if abs_assessments[abs_axis.axis_id].heldout_accuracy != 1.0:
        raise AssertionError("episode-3 absolute-difference representation failed held-out reproduction")
    if abs_axis.axis_id in {projection_axis.axis_id, difference_axis.axis_id}:
        raise AssertionError("second rewrite did not produce a third representation identity")
    require_semantic_reentry(abs_cycle, abs_axis.axis_id)
    abs_proposals = remember_reference_experiments(runtime, abs_axis, abs_measurements)
    abs_effects = execute_abs_many(
        runtime,
        abs_proposals,
        scale,
        swap,
        feature_names,
        abs_context,
        issuer_ids,
        signers,
        verifier,
        30000,
    )
    abs_supported = material_proposals(abs_proposals, abs_effects)
    if not abs_supported:
        raise AssertionError("episode-3 generated no material exact experiment")
    abs_action = policy.select(abs_proposals, runtime.world_coupling, context_id=abs_context)
    if abs_action.status != "WORLD_SUPPORTED_ACTION" or abs_action.proposal is None:
        raise AssertionError("episode-3 cognition failed to earn action authority")
    if abs_action.proposal.experiment_id not in {p.experiment_id for p in abs_supported}:
        raise AssertionError("episode-3 selected a non-material experiment")

    # Stale exact actions from both earlier mechanisms must remain non-active even
    # if a representation identity is later observationally revised. Action-level
    # revival requires a new exact experiment ID plus new world evidence.
    stale_ids = projection_stale_ids | difference_stale_ids
    stale_active = {
        experiment_id
        for experiment_id in stale_ids
        if experiment_id in runtime.memory.experiments
        and runtime.memory.experiments[experiment_id].status == "PROPOSAL_ONLY"
    }
    if stale_active:
        raise AssertionError(f"previously contradicted exact actions revived: {sorted(stale_active)}")

    demotion_events = [
        mutation for mutation in runtime.memory.mutation_log
        if mutation.mutation_id.startswith("WORLD_DEMOTE_AXIS::")
    ]
    demoted_targets = {mutation.target for mutation in demotion_events}
    if projection_axis.axis_id not in demoted_targets or difference_axis.axis_id not in demoted_targets:
        raise AssertionError("multi-episode BODY lost one of the two world-caused demotion events")

    encoded = checkpoint_json(runtime)
    if any(secret.hex() in encoded for secret in secrets_by_issuer.values()):
        raise AssertionError("external verifier secret leaked into multi-episode BODY checkpoint")

    final_experiment_id = abs_action.proposal.experiment_id
    projection_axis_id = projection_axis.axis_id
    difference_axis_id = difference_axis.axis_id
    abs_axis_id = abs_axis.axis_id
    del sum_cycle, diff_cycle, abs_cycle, projection_axis, difference_axis, abs_axis

    unverified_descendant = restore_json(encoded)
    unverified_action = policy.select(
        unverified_descendant.persisted_intervention_proposals(),
        unverified_descendant.world_coupling,
        context_id=abs_context,
    )
    if unverified_action.status != "EXPLORE_ONLY_NO_WORLD_SUPPORTED_ACTION":
        raise AssertionError("multi-episode descendant self-restored external action authority")

    descendant = restore_json(encoded, world_verifier=verifier)
    descendant_demotion_targets = {
        mutation.target for mutation in descendant.memory.mutation_log
        if mutation.mutation_id.startswith("WORLD_DEMOTE_AXIS::")
    }
    if {projection_axis_id, difference_axis_id} - descendant_demotion_targets:
        raise AssertionError("descendant lost one or more historical world-refutation events")
    if abs_axis_id not in descendant.memory.representations:
        raise AssertionError("descendant lost final replacement representation")
    if descendant.memory.representations[abs_axis_id].status != "ACTIVE_VALIDATED":
        raise AssertionError("descendant final replacement representation is not active")
    if descendant.memory.representations[abs_axis_id].axis.family != "ABS_DIFFERENCE":
        raise AssertionError("descendant reconstructed wrong final representation family")
    descendant_stale_active = {
        experiment_id
        for experiment_id in stale_ids
        if experiment_id in descendant.memory.experiments
        and descendant.memory.experiments[experiment_id].status == "PROPOSAL_ONLY"
    }
    if descendant_stale_active:
        raise AssertionError("descendant revived an exact action refuted in an earlier episode")
    descendant_action = policy.select(
        descendant.persisted_intervention_proposals(),
        descendant.world_coupling,
        context_id=abs_context,
    )
    if descendant_action.status != "WORLD_SUPPORTED_ACTION" or descendant_action.proposal is None:
        raise AssertionError("reverified multi-episode descendant lost final world-supported action")
    if descendant_action.proposal.experiment_id != final_experiment_id:
        raise AssertionError("descendant selected a different final exact experiment")

    episode_stats = [
        {
            "episode": 1,
            "context": sum_context,
            "family": "PROJECTION",
            "heldout_accuracy": projection_assessments[projection_axis_id].heldout_accuracy,
            "material_exact_experiments": len(projection_supported),
            "next_shift_contradicted_exact_experiments": len(revision_1.counterevidence),
            "fresh_observations": len(sum_measurements),
        },
        {
            "episode": 2,
            "context": diff_context,
            "family": "DIFFERENCE",
            "heldout_accuracy": difference_assessments[difference_axis_id].heldout_accuracy,
            "material_exact_experiments": len(difference_supported),
            "next_shift_contradicted_exact_experiments": len(revision_2.counterevidence),
            "fresh_observations": len(diff_measurements),
        },
        {
            "episode": 3,
            "context": abs_context,
            "family": "ABS_DIFFERENCE",
            "heldout_accuracy": abs_assessments[abs_axis_id].heldout_accuracy,
            "material_exact_experiments": len(abs_supported),
            "next_shift_contradicted_exact_experiments": None,
            "fresh_observations": len(abs_measurements),
        },
    ]

    print(json.dumps({
        "status": "PASS_BOUNDED_MULTI_EPISODE_WORLD_CAUSED_COGNITION_REWRITE_AND_DESCENDANT",
        "same_body_across_episodes": True,
        "world_regime_sequence": ["SUM", "DIFF", "ABS_DIFF"],
        "representation_family_sequence": ["PROJECTION", "DIFFERENCE", "ABS_DIFFERENCE"],
        "world_caused_rewrite_count": 2,
        "demotion_event_count": len(demotion_events),
        "first_shift_counterevidence_modes": sorted({item.contradiction for item in revision_1.counterevidence}),
        "second_shift_counterevidence_modes": sorted({item.contradiction for item in revision_2.counterevidence}),
        "first_shift_contradicted_exact_experiments": len(revision_1.counterevidence),
        "second_shift_contradicted_exact_experiments": len(revision_2.counterevidence),
        "stale_exact_action_revival_count": 0,
        "episode_stats": episode_stats,
        "final_action_status": abs_action.status,
        "final_action_experiment_id": final_experiment_id,
        "unverified_descendant_action_status": unverified_action.status,
        "reverified_descendant_action_status": descendant_action.status,
        "reverified_descendant_experiment_id": descendant_action.proposal.experiment_id,
        "two_refutation_events_inherited": True,
        "recursive_acceleration_measured": False,
        "recursive_acceleration": False,
        "independent_organizational_custody": False,
        "physical_world": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_multi_episode_world_rewrite.py <evaluator-owned-seed-file>")
    main(sys.argv[1])
