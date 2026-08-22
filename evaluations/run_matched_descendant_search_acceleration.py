from __future__ import annotations

import json
import random
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.body_checkpoint import checkpoint_json, restore_json
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.experiment_genesis import ExperimentGenesisEngine
from arte_cognition.representation_genesis import RepresentationAxis
from arte_cognition.world_action_policy import EvidenceBoundWorldActionPolicy
from arte_cognition.world_coupling import (
    HMACWorldReceiptSigner,
    HMACWorldReceiptVerifier,
    WorldOutcomeReceipt,
)


BASE_SCALES = (1.0, 2.0, 4.0)


class HiddenBoundaryWorld:
    """Evaluator-owned matched software world.

    BODY sees only signed LOW/HIGH outcomes. The hidden causal boundary is frozen
    after checkout and is shared across generations; sensor names, sources,
    challenges, and exact experiment identities remain generation-specific.
    """

    def __init__(
        self,
        feature_names,
        boundary,
        context_id,
        source_id,
        challenge_id,
        epoch,
        signer,
    ):
        self.feature_names = tuple(feature_names)
        self.boundary = float(boundary)
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
        receipt = WorldOutcomeReceipt(
            receipt_id=(
                f"matched-search::{self.context_id}::{self.source_id}::"
                f"{self.challenge_id}::{proposal.experiment_id}::{arm}"
            ),
            experiment_id=proposal.experiment_id,
            axis_id=proposal.axis_id,
            arm=arm,
            intervention_value=float(value),
            outcome=1.0 if latent_score > self.boundary else 0.0,
            source_id=self.source_id,
            context_id=self.context_id,
            challenge_id=self.challenge_id,
            epoch=self.epoch,
            budget_token=f"matched::{self.context_id}::{self.source_id}::{self.challenge_id}",
            externally_generated=True,
        )
        return self.signer.sign(receipt)


def make_axis(feature_names, generation):
    a, b = feature_names
    return RepresentationAxis(
        axis_id=f"AXIS::PROJECTION::{a}|{b}",
        family="PROJECTION",
        inputs=(a, b),
        threshold=0.0,
        direction="GT",
        information_gain=1.0,
        train_support=8,
        positive_partition=(f"g{generation}-positive",),
        formula=f"(1)*{a} + (1)*{b}",
        coefficients=((a, 1.0), (b, 1.0)),
        bias=0.0,
        status="PROPOSAL_ONLY",
    )


def references(feature_names):
    a, b = feature_names
    return (
        {a: 0.0, b: 0.0},
        {a: 0.05, b: 0.05},
    )


def dedup(proposals):
    out = {}
    for proposal in proposals:
        out.setdefault(proposal.experiment_id, proposal)
    return list(out.values())


def proposal_scale(runtime, proposal):
    scale = runtime._proposal_probe_scale(proposal)
    if scale is None:
        raise AssertionError("projection proposal missing probe_scale phenotype")
    return float(scale)


def execute_generation(
    runtime,
    generation,
    feature_names,
    boundary,
    issuer_ids,
    signers,
    verifier,
    run_label,
    epoch_base,
):
    context_id = f"matched-generation-{generation}"
    axis = make_axis(feature_names, generation)
    runtime.memory.remember_representation(axis)
    schedule_before = tuple(runtime.projection_search_schedule())

    proposals = []
    for reference in references(feature_names):
        proposals.extend(runtime.generate_interventions(axis, reference))
    proposals = dedup(proposals)
    if not proposals:
        raise AssertionError("no exact experiments generated")

    scales_present = tuple(sorted({proposal_scale(runtime, proposal) for proposal in proposals}))
    material = []
    pair_count = 0
    for proposal_index, proposal in enumerate(proposals):
        effects = []
        for issuer_index, issuer_id in enumerate(issuer_ids):
            world = HiddenBoundaryWorld(
                feature_names=feature_names,
                boundary=boundary,
                context_id=context_id,
                source_id=f"{run_label}-source-{generation}-{proposal_index}-{issuer_index}",
                challenge_id=f"{run_label}-challenge-{generation}-{proposal_index}-{issuer_index}",
                epoch=epoch_base + proposal_index * 10 + issuer_index,
                signer=signers[issuer_id],
            )
            pair = runtime.execute_world_intervention(proposal, world, verifier=verifier)
            pair_count += 1
            if not pair.authority_verified:
                raise AssertionError("matched world receipt failed verifier authority")
            effects.append(pair.effect)
        if len(effects) != len(issuer_ids):
            raise AssertionError("missing independent world effects")
        if min(abs(value) for value in effects) >= 0.5:
            material.append(proposal.experiment_id)

    decision = EvidenceBoundWorldActionPolicy().select(
        proposals,
        runtime.world_coupling,
        context_id=context_id,
    )
    capability = float(
        decision.status == "WORLD_SUPPORTED_ACTION"
        and decision.proposal is not None
        and decision.proposal.experiment_id in set(material)
    )
    return {
        "generation": generation,
        "context": context_id,
        "schedule_before": list(schedule_before),
        "scales_present": list(scales_present),
        "proposal_count": len(proposals),
        "external_pair_count": pair_count,
        "material_exact_experiments": len(material),
        "capability": capability,
        "action_status": decision.status,
        "selected_experiment_id": (
            decision.proposal.experiment_id if decision.proposal is not None else None
        ),
        "capability_per_external_pair": capability / pair_count if pair_count else 0.0,
    }


def main(seed_path):
    rng = random.Random(int(Path(seed_path).read_text().strip()))
    boundary = rng.uniform(0.36, 0.54)
    generation_features = []
    used = set()
    for generation in (1, 2, 3):
        while True:
            names = (
                f"accel_sensor_{generation}_{rng.randrange(10_000, 99_999)}",
                f"accel_sensor_{generation}_{rng.randrange(100_000, 999_999)}",
            )
            if not (set(names) & used):
                used.update(names)
                generation_features.append(names)
                break

    issuer_ids = (
        f"accel-evaluator-a-{rng.randrange(1_000_000, 9_999_999)}",
        f"accel-evaluator-b-{rng.randrange(1_000_000, 9_999_999)}",
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
            issuer_ids[0]: "accel-independent-a",
            issuer_ids[1]: "accel-independent-b",
        },
    )

    # TREATMENT: each fresh generation starts from the previous descendant BODY.
    treatment = PersistentCognitiveRuntime()
    treatment_stats = []
    checkpoints = {}
    for generation, feature_names in enumerate(generation_features, start=1):
        stat = execute_generation(
            treatment,
            generation,
            feature_names,
            boundary,
            issuer_ids,
            signers,
            verifier,
            run_label="treatment",
            epoch_base=generation * 100_000,
        )
        treatment_stats.append(stat)
        encoded = checkpoint_json(treatment)
        if any(secret.hex() in encoded for secret in secrets_by_issuer.values()):
            raise AssertionError("external verifier secret leaked into BODY checkpoint")
        checkpoints[generation] = encoded
        if generation < 3:
            treatment = restore_json(encoded, world_verifier=verifier)

    expected_schedules = [list(BASE_SCALES), [4.0, 1.0], [4.0]]
    actual_schedules = [item["schedule_before"] for item in treatment_stats]
    if actual_schedules != expected_schedules:
        raise AssertionError(
            f"descendant search schedule did not contract 3->2->1: {actual_schedules}"
        )
    proposal_counts = [item["proposal_count"] for item in treatment_stats]
    pair_counts = [item["external_pair_count"] for item in treatment_stats]
    efficiencies = [item["capability_per_external_pair"] for item in treatment_stats]
    if proposal_counts != [12, 8, 4]:
        raise AssertionError(f"unexpected candidate-space trajectory: {proposal_counts}")
    if pair_counts != [24, 16, 8]:
        raise AssertionError(f"unexpected external-evidence trajectory: {pair_counts}")
    if [item["capability"] for item in treatment_stats] != [1.0, 1.0, 1.0]:
        raise AssertionError("treatment lost fresh matched-task capability")
    if not (efficiencies[0] < efficiencies[1] < efficiencies[2]):
        raise AssertionError(f"validated capability per external pair did not accelerate: {efficiencies}")

    # A verifierless descendant must not reconstruct learned search authority from
    # serialized booleans or receipts alone.
    unverified_after_g2 = restore_json(checkpoints[2])
    if tuple(unverified_after_g2.projection_search_schedule()) != BASE_SCALES:
        raise AssertionError("verifierless descendant self-restored search contraction")

    # REMOVE: preserve the exact G2 BODY/evidence, remove only adaptive application.
    remove = restore_json(checkpoints[2], world_verifier=verifier)
    remove.adaptive_projection_search = False
    remove_stat = execute_generation(
        remove,
        3,
        generation_features[2],
        boundary,
        issuer_ids,
        signers,
        verifier,
        run_label="remove",
        epoch_base=900_000,
    )
    if remove_stat["capability"] != 1.0:
        raise AssertionError("REMOVE should preserve capability by spending more search evidence")
    if remove_stat["proposal_count"] != 12 or remove_stat["external_pair_count"] != 24:
        raise AssertionError("REMOVE did not restore the fixed full-search resource cost")

    # WRONG-SWAP: same one-scale candidate/evidence budget as treatment G3, but a
    # causally wrong 1x probe vocabulary. This must lose fresh capability.
    wrong = restore_json(checkpoints[2], world_verifier=verifier)
    wrong.adaptive_projection_search = False
    wrong.experiment = ExperimentGenesisEngine(projection_margin_multipliers=(1.0,))
    wrong_stat = execute_generation(
        wrong,
        3,
        generation_features[2],
        boundary,
        issuer_ids,
        signers,
        verifier,
        run_label="wrong",
        epoch_base=1_100_000,
    )
    if wrong_stat["proposal_count"] != treatment_stats[2]["proposal_count"]:
        raise AssertionError("WRONG-SWAP candidate budget does not match treatment G3")
    if wrong_stat["external_pair_count"] != treatment_stats[2]["external_pair_count"]:
        raise AssertionError("WRONG-SWAP external evidence budget does not match treatment G3")
    if wrong_stat["capability"] != 0.0:
        raise AssertionError("wrong one-scale swap unexpectedly retained fresh capability")

    # RESET: no inherited BODY evidence. All generations remain at fixed full search.
    reset_stats = []
    for generation, feature_names in enumerate(generation_features, start=1):
        reset = PersistentCognitiveRuntime()
        reset_stats.append(execute_generation(
            reset,
            generation,
            feature_names,
            boundary,
            issuer_ids,
            signers,
            verifier,
            run_label=f"reset-{generation}",
            epoch_base=1_300_000 + generation * 100_000,
        ))
    if [item["proposal_count"] for item in reset_stats] != [12, 12, 12]:
        raise AssertionError("RESET unexpectedly inherited candidate-space contraction")
    if [item["external_pair_count"] for item in reset_stats] != [24, 24, 24]:
        raise AssertionError("RESET unexpectedly reduced external evidence cost")
    if [item["capability"] for item in reset_stats] != [1.0, 1.0, 1.0]:
        raise AssertionError("RESET baseline failed matched tasks")

    treatment_g3 = treatment_stats[2]
    candidate_reduction_vs_remove = 1.0 - treatment_g3["proposal_count"] / remove_stat["proposal_count"]
    evidence_reduction_vs_remove = 1.0 - treatment_g3["external_pair_count"] / remove_stat["external_pair_count"]
    if candidate_reduction_vs_remove <= 0.0 or evidence_reduction_vs_remove <= 0.0:
        raise AssertionError("adaptive descendant produced no causal resource reduction")

    print(json.dumps({
        "status": "PASS_BOUNDED_MATCHED_DESCENDANT_SEARCH_ACCELERATION",
        "world_boundary_hidden_from_body": True,
        "matched_hidden_boundary": boundary,
        "source_disjoint_sensor_surfaces": True,
        "same_task_family_difficulty": True,
        "treatment": treatment_stats,
        "reset": reset_stats,
        "remove_g3": remove_stat,
        "wrong_swap_g3": wrong_stat,
        "proposal_count_trajectory": proposal_counts,
        "external_pair_count_trajectory": pair_counts,
        "validated_capability_trajectory": [item["capability"] for item in treatment_stats],
        "capability_per_external_pair_trajectory": efficiencies,
        "strict_efficiency_increase": True,
        "candidate_reduction_vs_remove_g3": candidate_reduction_vs_remove,
        "external_evidence_reduction_vs_remove_g3": evidence_reduction_vs_remove,
        "wrong_swap_matched_resource_capability_loss": 1.0,
        "verifierless_descendant_schedule": list(unverified_after_g2.projection_search_schedule()),
        "verified_descendant_schedule_g3": treatment_stats[2]["schedule_before"],
        "post_hidden_human_structural_repairs": 0,
        "independent_evidence_classes": 2,
        "independent_organizational_custody": False,
        "physical_world": False,
        "global_recursive_acceleration": False,
        "foundation_weight_change": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_matched_descendant_search_acceleration.py <evaluator-owned-seed-file>")
    main(sys.argv[1])
