from __future__ import annotations

import json
import random
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.canonical_body_checkpoint import checkpoint_json, restore_json
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.experiment_genesis import ExperimentGenesisEngine
from arte_cognition.representation_genesis import RepresentationAxis
from arte_cognition.world_action_policy import EvidenceBoundWorldActionPolicy
from arte_cognition.world_coupling import (
    HMACWorldReceiptSigner,
    HMACWorldReceiptVerifier,
    WorldOutcomeReceipt,
)


AUTHORED_SCALES = (1.0, 2.0, 4.0)
STRONG_EFFECT = 0.9


class HiddenSmoothScaleWorld:
    """Evaluator-owned response landscape with optimum outside authored vocabulary."""

    def __init__(self, feature_names, target_scale, context_id, source_id, challenge_id, epoch, signer):
        self.feature_names = tuple(feature_names)
        self.target_scale = float(target_scale)
        self.context_id = str(context_id)
        self.source_id = str(source_id)
        self.challenge_id = str(challenge_id)
        self.epoch = int(epoch)
        self.signer = signer

    def execute(self, proposal, arm, value):
        state = {name: 0.0 for name in self.feature_names}
        state.update({name: float(v) for name, v in proposal.held_fixed})
        state[proposal.manipulated_variable] = float(value)
        latent_score = sum(float(state[name]) for name in self.feature_names)
        target_amplitude = 0.15 * self.target_scale
        response = max(0.0, 1.0 - abs(latent_score - target_amplitude) / 0.30)
        receipt = WorldOutcomeReceipt(
            receipt_id=(
                f"scale-genesis::{self.context_id}::{self.source_id}::"
                f"{self.challenge_id}::{proposal.experiment_id}::{arm}"
            ),
            experiment_id=proposal.experiment_id,
            axis_id=proposal.axis_id,
            arm=arm,
            intervention_value=float(value),
            outcome=float(response),
            source_id=self.source_id,
            context_id=self.context_id,
            challenge_id=self.challenge_id,
            epoch=self.epoch,
            budget_token=f"scale-genesis::{self.context_id}::{self.source_id}::{self.challenge_id}",
            externally_generated=True,
        )
        return self.signer.sign(receipt)


def make_axis(feature_names, label):
    a, b = feature_names
    return RepresentationAxis(
        axis_id=f"AXIS::PROJECTION::{a}|{b}",
        family="PROJECTION",
        inputs=(a, b),
        threshold=0.0,
        direction="GT",
        information_gain=1.0,
        train_support=8,
        positive_partition=(f"{label}-positive",),
        formula=f"(1)*{a} + (1)*{b}",
        coefficients=((a, 1.0), (b, 1.0)),
        bias=0.0,
        status="PROPOSAL_ONLY",
    )


def references(feature_names):
    a, b = feature_names
    return ({a: 0.0, b: 0.0}, {a: 0.05, b: 0.05})


def dedup(items):
    out = {}
    for item in items:
        out.setdefault(item.experiment_id, item)
    return list(out.values())


def scale_of(runtime, proposal):
    value = runtime._proposal_probe_scale(proposal)
    if value is None:
        raise AssertionError("projection proposal missing scale phenotype")
    return float(value)


def execute_proposals(runtime, proposals, feature_names, target_scale, context_id, issuer_ids, signers, verifier, run_label, epoch_base):
    for proposal_index, proposal in enumerate(proposals):
        for issuer_index, issuer_id in enumerate(issuer_ids):
            world = HiddenSmoothScaleWorld(
                feature_names,
                target_scale,
                context_id,
                source_id=f"{run_label}-source-{proposal_index}-{issuer_index}",
                challenge_id=f"{run_label}-challenge-{proposal_index}-{issuer_index}",
                epoch=epoch_base + proposal_index * 10 + issuer_index,
                signer=signers[issuer_id],
            )
            pair = runtime.execute_world_intervention(proposal, world, verifier=verifier)
            if not pair.authority_verified:
                raise AssertionError("generated-scale world evidence failed authority")


def run_current_schedule(runtime, label, feature_names, target_scale, issuer_ids, signers, verifier, run_label, epoch_base):
    context_id = f"scale-genesis-{label}"
    axis = make_axis(feature_names, label)
    runtime.memory.remember_representation(axis)
    schedule_before = tuple(runtime.projection_search_schedule())
    vocabulary_before = tuple(runtime.projection_probe_vocabulary())
    proposals = []
    for ref in references(feature_names):
        proposals.extend(runtime.generate_interventions(axis, ref))
    proposals = dedup(proposals)
    execute_proposals(
        runtime, proposals, feature_names, target_scale, context_id,
        issuer_ids, signers, verifier, run_label, epoch_base,
    )
    decision = EvidenceBoundWorldActionPolicy().select(
        proposals, runtime.world_coupling, context_id=context_id
    )
    strong_capability = float(
        decision.status == "WORLD_SUPPORTED_ACTION"
        and decision.routing_score >= STRONG_EFFECT
    )
    return {
        "context": context_id,
        "schedule_before": list(schedule_before),
        "vocabulary_before": list(vocabulary_before),
        "proposal_count": len(proposals),
        "external_pair_count": len(proposals) * len(issuer_ids),
        "selected_score": decision.routing_score,
        "strong_capability": strong_capability,
        "selected_experiment_id": decision.proposal.experiment_id if decision.proposal else None,
        "scales_present": sorted({scale_of(runtime, p) for p in proposals}),
    }


def make_feature_sets(rng, count):
    result = []
    used = set()
    while len(result) < count:
        pair = (
            f"scale_sensor_{len(result)+1}_{rng.randrange(10_000, 99_999)}",
            f"scale_sensor_{len(result)+1}_{rng.randrange(100_000, 999_999)}",
        )
        if set(pair) & used:
            continue
        used.update(pair)
        result.append(pair)
    return result


def main(seed_path):
    rng = random.Random(int(Path(seed_path).read_text().strip()))
    hidden_target = rng.choice((1.5, 3.0))
    if hidden_target in AUTHORED_SCALES:
        raise AssertionError("hidden target must be absent from authored vocabulary")
    features = make_feature_sets(rng, 3)

    issuer_ids = (
        f"scale-evaluator-a-{rng.randrange(1_000_000, 9_999_999)}",
        f"scale-evaluator-b-{rng.randrange(1_000_000, 9_999_999)}",
    )
    secrets_by_issuer = {
        issuer_ids[0]: secrets.token_bytes(32),
        issuer_ids[1]: secrets.token_bytes(32),
    }
    signers = {issuer: HMACWorldReceiptSigner(issuer, secret) for issuer, secret in secrets_by_issuer.items()}
    verifier = HMACWorldReceiptVerifier(
        secrets_by_issuer,
        independence_classes={
            issuer_ids[0]: "scale-independent-a",
            issuer_ids[1]: "scale-independent-b",
        },
    )

    # First episode: the authored {1,2,4} vocabulary cannot reach the strong-effect
    # criterion. BODY must derive midpoint candidates from the authenticated residual.
    runtime = PersistentCognitiveRuntime()
    initial = run_current_schedule(
        runtime, "train", features[0], hidden_target,
        issuer_ids, signers, verifier, "initial", 100_000,
    )
    if initial["strong_capability"] != 0.0:
        raise AssertionError("authored vocabulary unexpectedly solved hidden off-grid optimum")
    frontier = runtime.projection_scale_frontier(context_id=initial["context"])
    if frontier.status != "GENERATED_NUMERIC_REFINEMENT":
        raise AssertionError(f"BODY did not open numeric scale refinement: {frontier}")
    if hidden_target not in frontier.candidate_scales:
        raise AssertionError(
            f"bounded midpoint frontier failed to contain evaluator-frozen target {hidden_target}: {frontier.candidate_scales}"
        )

    axis = make_axis(features[0], "train")
    generated = []
    for ref in references(features[0]):
        generated.extend(
            runtime.generate_projection_scale_frontier_interventions(
                axis, ref, context_id=initial["context"]
            )
        )
    generated = dedup(generated)
    if not generated:
        raise AssertionError("scale frontier produced no executable proposal phenotypes")
    execute_proposals(
        runtime, generated, features[0], hidden_target, initial["context"],
        issuer_ids, signers, verifier, "generated", 300_000,
    )
    generated_decision = EvidenceBoundWorldActionPolicy().select(
        generated, runtime.world_coupling, context_id=initial["context"]
    )
    if generated_decision.status != "WORLD_SUPPORTED_ACTION" or generated_decision.routing_score < STRONG_EFFECT:
        raise AssertionError("generated scale failed external strong-effect validation")

    vocabulary_after = tuple(runtime.projection_probe_vocabulary())
    if hidden_target not in vocabulary_after:
        raise AssertionError("externally validated generated scale did not enter BODY vocabulary")
    if not any(
        abs(scale_of(runtime, p) - hidden_target) <= 1e-12
        for p in generated
    ):
        raise AssertionError("hidden generated target absent from executed BODY proposals")

    checkpoint1 = checkpoint_json(runtime)
    if any(secret.hex() in checkpoint1 for secret in secrets_by_issuer.values()):
        raise AssertionError("external verifier secret leaked into canonical checkpoint")
    verifierless = restore_json(checkpoint1)
    if hidden_target in verifierless.projection_probe_vocabulary():
        raise AssertionError("verifierless descendant self-authorized generated scale")

    # Fresh sensor surface, same response family: descendant reconstructs generated
    # scale from reverified evidence and spends only a two-scale search.
    descendant2 = restore_json(checkpoint1, world_verifier=verifier)
    gen2 = run_current_schedule(
        descendant2, "fresh-2", features[1], hidden_target,
        issuer_ids, signers, verifier, "descendant2", 500_000,
    )
    if gen2["strong_capability"] != 1.0:
        raise AssertionError("first descendant failed fresh off-grid target")
    if hidden_target not in gen2["scales_present"] or gen2["proposal_count"] != 8:
        raise AssertionError(f"first descendant did not use generated scale in contracted policy: {gen2}")

    checkpoint2 = checkpoint_json(descendant2)
    descendant3 = restore_json(checkpoint2, world_verifier=verifier)
    schedule3 = tuple(descendant3.projection_search_schedule())
    if schedule3 != (hidden_target,):
        raise AssertionError(f"two-context generated scale did not become singleton metapolicy: {schedule3}")
    gen3 = run_current_schedule(
        descendant3, "fresh-3", features[2], hidden_target,
        issuer_ids, signers, verifier, "descendant3", 700_000,
    )
    if gen3["strong_capability"] != 1.0 or gen3["proposal_count"] != 4 or gen3["external_pair_count"] != 8:
        raise AssertionError("second descendant failed accelerated singleton generated-scale policy")

    # REMOVE preserves the learned generated atom but removes policy contraction.
    remove = restore_json(checkpoint2, world_verifier=verifier)
    full_learned_vocabulary = remove.projection_probe_vocabulary()
    if hidden_target not in full_learned_vocabulary or len(full_learned_vocabulary) != 4:
        raise AssertionError(f"unexpected learned full vocabulary: {full_learned_vocabulary}")
    remove.experiment = ExperimentGenesisEngine(
        projection_margin_multipliers=full_learned_vocabulary
    )
    remove.adaptive_projection_search = False
    remove_stat = run_current_schedule(
        remove, "fresh-3", features[2], hidden_target,
        issuer_ids, signers, verifier, "remove", 900_000,
    )
    if remove_stat["strong_capability"] != 1.0:
        raise AssertionError("REMOVE full learned vocabulary should preserve strong capability")
    if remove_stat["proposal_count"] != 16 or remove_stat["external_pair_count"] != 32:
        raise AssertionError("REMOVE did not expose full learned-vocabulary resource cost")

    # WRONG-SWAP spends the same one-scale budget as descendant3 but substitutes an
    # authored near-miss scale. It must lose the strong-effect capability.
    wrong_scale = 2.0 if hidden_target == 1.5 else 4.0
    wrong = restore_json(checkpoint2, world_verifier=verifier)
    wrong.experiment = ExperimentGenesisEngine(projection_margin_multipliers=(wrong_scale,))
    wrong.adaptive_projection_search = False
    wrong_stat = run_current_schedule(
        wrong, "fresh-3", features[2], hidden_target,
        issuer_ids, signers, verifier, "wrong", 1_100_000,
    )
    if wrong_stat["proposal_count"] != gen3["proposal_count"] or wrong_stat["external_pair_count"] != gen3["external_pair_count"]:
        raise AssertionError("WRONG-SWAP resource budget not matched")
    if wrong_stat["strong_capability"] != 0.0:
        raise AssertionError("near-miss one-scale WRONG-SWAP retained strong capability")

    reset = PersistentCognitiveRuntime()
    reset_stat = run_current_schedule(
        reset, "fresh-3", features[2], hidden_target,
        issuer_ids, signers, verifier, "reset", 1_300_000,
    )
    if reset_stat["strong_capability"] != 0.0:
        raise AssertionError("RESET authored vocabulary unexpectedly inherited off-grid capability")

    candidate_reduction_vs_remove = 1.0 - gen3["proposal_count"] / remove_stat["proposal_count"]
    evidence_reduction_vs_remove = 1.0 - gen3["external_pair_count"] / remove_stat["external_pair_count"]
    if abs(candidate_reduction_vs_remove - 0.75) > 1e-12:
        raise AssertionError(f"unexpected generated-scale candidate reduction: {candidate_reduction_vs_remove}")
    if abs(evidence_reduction_vs_remove - 0.75) > 1e-12:
        raise AssertionError(f"unexpected generated-scale evidence reduction: {evidence_reduction_vs_remove}")

    print(json.dumps({
        "status": "PASS_BOUNDED_WORLD_RESIDUAL_GENERATED_PROBE_SCALE_AND_DESCENDANT_ACCELERATION",
        "authored_probe_scales": list(AUTHORED_SCALES),
        "hidden_target_scale": hidden_target,
        "hidden_target_absent_from_authored_vocabulary": True,
        "hidden_target_exposed_to_body_before_world_evidence": False,
        "initial": initial,
        "frontier_status": frontier.status,
        "generated_candidate_scales": list(frontier.candidate_scales),
        "generated_proposal_count": len(generated),
        "generated_best_external_score": generated_decision.routing_score,
        "learned_vocabulary_after_external_validation": list(vocabulary_after),
        "verifierless_generated_scale_authority": hidden_target in verifierless.projection_probe_vocabulary(),
        "descendant_generation_2": gen2,
        "descendant_generation_3": gen3,
        "descendant_schedule_trajectory": [initial["schedule_before"], gen2["schedule_before"], gen3["schedule_before"]],
        "external_pair_trajectory_after_scale_acquisition": [gen2["external_pair_count"], gen3["external_pair_count"]],
        "strong_capability_trajectory_after_scale_acquisition": [gen2["strong_capability"], gen3["strong_capability"]],
        "remove": remove_stat,
        "wrong_swap": wrong_stat,
        "reset": reset_stat,
        "candidate_reduction_vs_remove_g3": candidate_reduction_vs_remove,
        "external_evidence_reduction_vs_remove_g3": evidence_reduction_vs_remove,
        "wrong_swap_matched_resource_capability_loss": 1.0,
        "generated_scale_reconstructed_from_reverified_evidence": True,
        "generated_scale_serialized_as_authoritative_scalar": False,
        "post_hidden_human_structural_repairs": 0,
        "numeric_midpoint_refinement_operator_human_authored": True,
        "unrestricted_numeric_operator_genesis": False,
        "recursive_acceleration_candidate": True,
        "global_recursive_acceleration": False,
        "physical_world": False,
        "independent_organizational_custody": False,
        "foundation_weight_change": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_generated_probe_scale_genesis.py <evaluator-owned-seed-file>")
    main(sys.argv[1])
