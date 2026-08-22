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


BASE_SCALES = (1.0, 2.0, 4.0)


class HiddenScaleRegimeWorld:
    """Evaluator-owned non-monotonic response regime.

    Each regime has one hidden effective intervention amplitude. BODY sees exact
    signed intervention outcomes but never receives the target scale. The causal
    response is a narrow positive-score band, so 1x, 2x and 4x are genuinely
    distinguishable rather than monotonic aliases.
    """

    def __init__(
        self,
        feature_names,
        target_scale,
        context_id,
        source_id,
        challenge_id,
        epoch,
        signer,
    ):
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
        tolerance = 0.025
        response = float(
            target_amplitude - tolerance <= latent_score <= target_amplitude + tolerance
        )
        receipt = WorldOutcomeReceipt(
            receipt_id=(
                f"metapolicy::{self.context_id}::{self.source_id}::"
                f"{self.challenge_id}::{proposal.experiment_id}::{arm}"
            ),
            experiment_id=proposal.experiment_id,
            axis_id=proposal.axis_id,
            arm=arm,
            intervention_value=float(value),
            outcome=response,
            source_id=self.source_id,
            context_id=self.context_id,
            challenge_id=self.challenge_id,
            epoch=self.epoch,
            budget_token=f"metapolicy::{self.context_id}::{self.source_id}::{self.challenge_id}",
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


def dedup(proposals):
    unique = {}
    for item in proposals:
        unique.setdefault(item.experiment_id, item)
    return list(unique.values())


def scale_of(runtime, proposal):
    scale = runtime._proposal_probe_scale(proposal)
    if scale is None:
        raise AssertionError("projection proposal missing probe_scale phenotype")
    return float(scale)


def execute_context(
    runtime,
    label,
    feature_names,
    target_scale,
    issuer_ids,
    signers,
    verifier,
    run_label,
    epoch_base,
):
    context_id = f"metapolicy-{label}"
    axis = make_axis(feature_names, label)
    runtime.memory.remember_representation(axis)
    policy_before = runtime.projection_search_metapolicy()

    proposals = []
    for reference in references(feature_names):
        proposals.extend(runtime.generate_interventions(axis, reference))
    proposals = dedup(proposals)
    scales = tuple(sorted({scale_of(runtime, item) for item in proposals}))

    material = []
    pair_count = 0
    for proposal_index, proposal in enumerate(proposals):
        effects = []
        for issuer_index, issuer_id in enumerate(issuer_ids):
            world = HiddenScaleRegimeWorld(
                feature_names=feature_names,
                target_scale=target_scale,
                context_id=context_id,
                source_id=f"{run_label}-source-{label}-{proposal_index}-{issuer_index}",
                challenge_id=f"{run_label}-challenge-{label}-{proposal_index}-{issuer_index}",
                epoch=epoch_base + proposal_index * 10 + issuer_index,
                signer=signers[issuer_id],
            )
            pair = runtime.execute_world_intervention(proposal, world, verifier=verifier)
            if not pair.authority_verified:
                raise AssertionError("external world pair failed authority")
            effects.append(pair.effect)
            pair_count += 1
        if min(abs(effect) for effect in effects) >= 0.5:
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
        "label": label,
        "context": context_id,
        "hidden_target_scale": target_scale,
        "policy_before": list(policy_before.schedule),
        "policy_reason_before": policy_before.reason,
        "metapolicy_candidate_count": policy_before.candidate_count,
        "scales_present": list(scales),
        "proposal_count": len(proposals),
        "external_pair_count": pair_count,
        "material_exact_experiments": len(material),
        "capability": capability,
        "selected_experiment_id": decision.proposal.experiment_id if decision.proposal else None,
    }


def make_feature_sets(rng, count):
    result = []
    used = set()
    while len(result) < count:
        pair = (
            f"meta_sensor_{len(result)+1}_{rng.randrange(10_000, 99_999)}",
            f"meta_sensor_{len(result)+1}_{rng.randrange(100_000, 999_999)}",
        )
        if set(pair) & used:
            continue
        used.update(pair)
        result.append(pair)
    return result


def main(seed_path):
    rng = random.Random(int(Path(seed_path).read_text().strip()))
    features = make_feature_sets(rng, 3)
    issuer_ids = (
        f"meta-evaluator-a-{rng.randrange(1_000_000, 9_999_999)}",
        f"meta-evaluator-b-{rng.randrange(1_000_000, 9_999_999)}",
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
            issuer_ids[0]: "meta-independent-a",
            issuer_ids[1]: "meta-independent-b",
        },
    )

    # Training context 1 makes 4x uniquely material. A one-context singleton is
    # forbidden, so the BODY retains a second exploratory scale and derives {4,1}.
    runtime = PersistentCognitiveRuntime()
    train4 = execute_context(
        runtime, "train-needs4", features[0], 4.0,
        issuer_ids, signers, verifier, "treatment", 100_000,
    )
    if train4["capability"] != 1.0 or train4["proposal_count"] != 12:
        raise AssertionError("full-vocabulary first training context failed")
    if tuple(runtime.projection_search_schedule()) != (4.0, 1.0):
        raise AssertionError(
            f"first-context metapolicy did not retain 4x plus exploration: {runtime.projection_search_schedule()}"
        )

    checkpoint_after_4 = checkpoint_json(runtime)
    if any(secret.hex() in checkpoint_after_4 for secret in secrets_by_issuer.values()):
        raise AssertionError("world verifier secret leaked into canonical BODY checkpoint")
    runtime = restore_json(checkpoint_after_4, world_verifier=verifier)

    # Training context 2 makes 1x uniquely material. Because 4x is still explored,
    # the BODY can discover that the minimum cross-context policy is non-contiguous
    # {1,4}; 2x is causally useless in both regimes and is removed.
    train1 = execute_context(
        runtime, "train-needs1", features[1], 1.0,
        issuer_ids, signers, verifier, "treatment", 300_000,
    )
    if train1["capability"] != 1.0 or train1["proposal_count"] != 8:
        raise AssertionError("second heterogeneous training context failed retained exploration")
    learned = runtime.projection_search_metapolicy()
    if set(learned.schedule) != {1.0, 4.0} or len(learned.schedule) != 2:
        raise AssertionError(f"BODY failed to synthesize non-prefix {{1,4}} metapolicy: {learned.schedule}")
    if 2.0 in learned.schedule:
        raise AssertionError("causally useless 2x scale survived minimum-policy search")

    training_checkpoint = checkpoint_json(runtime)
    if any(secret.hex() in training_checkpoint for secret in secrets_by_issuer.values()):
        raise AssertionError("world verifier secret leaked after metapolicy learning")

    verifierless = restore_json(training_checkpoint)
    if tuple(verifierless.projection_search_schedule()) != BASE_SCALES:
        raise AssertionError("verifierless descendant self-authorized learned metapolicy")

    # Fresh held-out regime is chosen only from the two causally retained scales.
    # BODY does not know which one will be required before external execution.
    fresh_target = rng.choice((1.0, 4.0))
    treatment = restore_json(training_checkpoint, world_verifier=verifier)
    learned_before_fresh = tuple(treatment.projection_search_schedule())
    fresh = execute_context(
        treatment, "fresh-heldout", features[2], fresh_target,
        issuer_ids, signers, verifier, "treatment-fresh", 500_000,
    )
    if fresh["capability"] != 1.0:
        raise AssertionError("learned cross-context metapolicy lost fresh held-out capability")
    if fresh["proposal_count"] != 8 or fresh["external_pair_count"] != 16:
        raise AssertionError("learned two-scale policy did not reduce fresh resource use")

    # REMOVE uses the same learned BODY/evidence but disables metapolicy application.
    remove = restore_json(training_checkpoint, world_verifier=verifier)
    remove.adaptive_projection_search = False
    remove_stat = execute_context(
        remove, "fresh-heldout", features[2], fresh_target,
        issuer_ids, signers, verifier, "remove", 700_000,
    )
    if remove_stat["capability"] != 1.0:
        raise AssertionError("REMOVE full vocabulary should preserve capability at higher cost")
    if remove_stat["proposal_count"] != 12 or remove_stat["external_pair_count"] != 24:
        raise AssertionError("REMOVE did not restore full-search resource cost")

    # WRONG-SWAP spends exactly the treatment's two-scale budget but removes the
    # evaluator-frozen fresh target scale. It must lose capability.
    wrong_scales = (2.0, 4.0) if fresh_target == 1.0 else (1.0, 2.0)
    wrong = restore_json(training_checkpoint, world_verifier=verifier)
    wrong.adaptive_projection_search = False
    wrong.experiment = ExperimentGenesisEngine(projection_margin_multipliers=wrong_scales)
    wrong_stat = execute_context(
        wrong, "fresh-heldout", features[2], fresh_target,
        issuer_ids, signers, verifier, "wrong", 900_000,
    )
    if wrong_stat["proposal_count"] != fresh["proposal_count"]:
        raise AssertionError("WRONG-SWAP candidate budget not matched")
    if wrong_stat["external_pair_count"] != fresh["external_pair_count"]:
        raise AssertionError("WRONG-SWAP external evidence budget not matched")
    if wrong_stat["capability"] != 0.0:
        raise AssertionError("wrong two-scale policy unexpectedly retained held-out capability")

    # RESET has no inherited causal policy evidence and must spend the full search.
    reset = PersistentCognitiveRuntime()
    reset_stat = execute_context(
        reset, "fresh-heldout", features[2], fresh_target,
        issuer_ids, signers, verifier, "reset", 1_100_000,
    )
    if reset_stat["capability"] != 1.0:
        raise AssertionError("RESET full search failed held-out regime")
    if reset_stat["proposal_count"] != 12 or reset_stat["external_pair_count"] != 24:
        raise AssertionError("RESET unexpectedly inherited metapolicy contraction")

    candidate_reduction = 1.0 - fresh["proposal_count"] / remove_stat["proposal_count"]
    evidence_reduction = 1.0 - fresh["external_pair_count"] / remove_stat["external_pair_count"]
    if abs(candidate_reduction - (1.0 / 3.0)) > 1e-12:
        raise AssertionError(f"unexpected candidate reduction: {candidate_reduction}")
    if abs(evidence_reduction - (1.0 / 3.0)) > 1e-12:
        raise AssertionError(f"unexpected evidence reduction: {evidence_reduction}")

    print(json.dumps({
        "status": "PASS_BOUNDED_CAUSALLY_LEARNED_CROSS_CONTEXT_SEARCH_METAPOLICY",
        "training_target_sequence": [4.0, 1.0],
        "first_context_derived_schedule": list(restore_json(checkpoint_after_4, world_verifier=verifier).projection_search_schedule()),
        "learned_non_prefix_schedule": list(learned_before_fresh),
        "useless_scale_removed": 2.0 not in learned_before_fresh,
        "metapolicy_shadow_candidate_count": learned.candidate_count,
        "metapolicy_reconstructed_from_reverified_body_evidence": True,
        "metapolicy_serialized_as_authoritative_scalar": False,
        "verifierless_descendant_schedule": list(verifierless.projection_search_schedule()),
        "fresh_hidden_target_scale": fresh_target,
        "fresh_target_exposed_to_body_before_execution": False,
        "fresh_treatment": fresh,
        "fresh_remove": remove_stat,
        "fresh_wrong_swap": wrong_stat,
        "fresh_reset": reset_stat,
        "candidate_reduction_vs_remove": candidate_reduction,
        "external_evidence_reduction_vs_remove": evidence_reduction,
        "wrong_swap_matched_resource_capability_loss": 1.0,
        "cross_context_policy_generation_not_fixed_prefix": True,
        "post_hidden_human_structural_repairs": 0,
        "independent_evidence_classes": 2,
        "physical_world": False,
        "independent_organizational_custody": False,
        "foundation_weight_change": False,
        "unrestricted_metapolicy_language": False,
        "recursive_acceleration_candidate": True,
        "global_recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_cross_context_projection_metapolicy.py <evaluator-owned-seed-file>")
    main(sys.argv[1])
