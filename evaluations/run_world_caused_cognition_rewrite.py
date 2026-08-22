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


OLD_TRAIN = (
    ("oa1", -1.00, 0.50), ("oa2", 0.50, -1.00),
    ("oa3", -0.04, 0.02), ("oa4", 0.02, -0.04),
    ("ob1", 1.00, -0.50), ("ob2", -0.50, 1.00),
    ("ob3", 0.04, -0.02), ("ob4", -0.02, 0.04),
)
OLD_HELDOUT = (("oh1", -0.20, 0.10), ("oh2", 0.20, -0.10))

# Same marginal x distribution as the old regime, but y is reflected. Neither
# raw variable alone cleanly separates the labels; x-y does. This ensures the
# changed-world task genuinely requires a relational representation rather than
# allowing a raw parent feature to solve it by itself.
NEW_TRAIN = (
    ("na1", -1.00, -0.50), ("na2", 0.50, 1.00),
    ("na3", -0.04, -0.02), ("na4", 0.02, 0.04),
    ("nb1", 1.00, 0.50), ("nb2", -0.50, -1.00),
    ("nb3", 0.04, 0.02), ("nb4", -0.02, -0.04),
)
NEW_HELDOUT = (("nh1", -0.20, -0.10), ("nh2", 0.20, 0.10))


class HiddenRegimeWorld:
    def __init__(self, scale, swap, feature_names, rule, context_id, source_id, challenge_id, epoch, signer):
        self.scale = float(scale)
        self.swap = bool(swap)
        self.feature_names = feature_names
        self.rule = rule
        self.context_id = context_id
        self.source_id = source_id
        self.challenge_id = challenge_id
        self.epoch = int(epoch)
        self.signer = signer

    def encode(self, base_x, base_y):
        a, b = (base_y, base_x) if self.swap else (base_x, base_y)
        return {self.feature_names[0]: self.scale * float(a), self.feature_names[1]: self.scale * float(b)}

    def decode(self, observed):
        a = float(observed[self.feature_names[0]]) / self.scale
        b = float(observed[self.feature_names[1]]) / self.scale
        return (b, a) if self.swap else (a, b)

    def positive(self, base_x, base_y):
        if self.rule == "SUM":
            return float(base_x) + float(base_y) > 0.0
        if self.rule == "DIFF":
            return float(base_x) - float(base_y) > 0.0
        raise ValueError("unknown hidden world rule")

    def execute(self, proposal, arm, value):
        state = {name: 0.0 for name in self.feature_names}
        state.update({name: float(v) for name, v in proposal.held_fixed})
        state[proposal.manipulated_variable] = float(value)
        base_x, base_y = self.decode(state)
        receipt = WorldOutcomeReceipt(
            receipt_id=f"rewrite::{self.context_id}::{self.source_id}::{self.challenge_id}::{proposal.experiment_id}::{arm}",
            experiment_id=proposal.experiment_id,
            axis_id=proposal.axis_id,
            arm=arm,
            intervention_value=float(value),
            outcome=1.0 if self.positive(base_x, base_y) else 0.0,
            source_id=self.source_id,
            context_id=self.context_id,
            challenge_id=self.challenge_id,
            epoch=self.epoch,
            budget_token=f"matched::{self.context_id}::{self.source_id}::{self.challenge_id}",
            externally_generated=True,
        )
        return self.signer.sign(receipt)


def build_observations(world, train, heldout, label_flip):
    measurements, residuals = [], []
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
                features=("raw-regime-residual",),
                outcome=label,
                source_class=f"hidden-{world.context_id}",
                heldout=is_heldout,
            ))
    return measurements, residuals


def best_incremental(cycle, family):
    assessments = {item.axis_id: item for item in cycle.representation_value}
    eligible = [
        axis for axis in cycle.representation_axes
        if axis.family == family
        and axis.axis_id in assessments
        and assessments[axis.axis_id].status == "INCREMENTAL_REPRESENTATION_VALUE"
    ]
    if not eligible:
        observed = [(axis.family, assessments[axis.axis_id].status) for axis in cycle.representation_axes if axis.axis_id in assessments]
        raise AssertionError(f"no incrementally valuable {family} representation; observed={observed}")
    axis = max(eligible, key=lambda item: assessments[item.axis_id].incremental_gain)
    return axis, assessments


def remember_reference_experiments(runtime, axis, measurements):
    for row in (row for row in measurements if not row.heldout):
        for proposal in runtime.experiment.propose(axis, row.values):
            runtime.memory.remember_experiment(proposal)
    return [proposal for proposal in runtime.persisted_intervention_proposals() if proposal.axis_id == axis.axis_id]


def execute_many(runtime, proposals, scale, swap, feature_names, rule, context_id, issuer_ids, signers, verifier, epoch_base):
    effects = {}
    for proposal_index, proposal in enumerate(proposals):
        effects[proposal.experiment_id] = []
        for issuer_index, issuer_id in enumerate(issuer_ids):
            world = HiddenRegimeWorld(
                scale, swap, feature_names, rule, context_id,
                f"{context_id}-source-{proposal_index}-{issuer_index}",
                f"{context_id}-challenge-{proposal_index}-{issuer_index}",
                epoch_base + proposal_index * 10 + issuer_index,
                signers[issuer_id],
            )
            pair = runtime.execute_world_intervention(proposal, world, verifier=verifier)
            if not pair.authority_verified:
                raise AssertionError("hidden world receipt failed external authentication")
            effects[proposal.experiment_id].append(pair.effect)
    return effects


def main(seed_path):
    rng = random.Random(int(Path(seed_path).read_text().strip()))
    scale = rng.choice((-1.0, 1.0)) * rng.uniform(0.65, 2.75)
    swap = bool(rng.getrandbits(1))
    feature_names = (f"rewrite_sensor_{rng.randrange(10_000, 99_999)}", f"rewrite_sensor_{rng.randrange(100_000, 999_999)}")
    label_flip = bool(rng.getrandbits(1))
    issuer_ids = (
        f"rewrite-evaluator-a-{rng.randrange(1_000_000, 9_999_999)}",
        f"rewrite-evaluator-b-{rng.randrange(1_000_000, 9_999_999)}",
    )
    secrets_by_issuer = {issuer_ids[0]: secrets.token_bytes(32), issuer_ids[1]: secrets.token_bytes(32)}
    signers = {issuer: HMACWorldReceiptSigner(issuer, secret) for issuer, secret in secrets_by_issuer.items()}
    verifier = HMACWorldReceiptVerifier(
        secrets_by_issuer,
        independence_classes={issuer_ids[0]: "rewrite-independent-a", issuer_ids[1]: "rewrite-independent-b"},
    )

    old_context, new_context = "hidden-regime-sum", "hidden-regime-diff"
    old_world = HiddenRegimeWorld(scale, swap, feature_names, "SUM", old_context, "old-observation-source", "old-observation-challenge", 0, signers[issuer_ids[0]])
    old_measurements, old_residuals = build_observations(old_world, OLD_TRAIN, OLD_HELDOUT, label_flip)

    runtime = PersistentCognitiveRuntime()
    old_cycle = runtime.cycle(
        TaskState(goal="discover current hidden world representation", novelty=0.95, residuals=[r.residual_id for r in old_residuals if not r.heldout], external_world=True, action_required=True),
        residuals=old_residuals,
        measurements=old_measurements,
        world_context_id=old_context,
    )
    old_axis, old_assessments = best_incremental(old_cycle, "PROJECTION")
    old_assessment = old_assessments[old_axis.axis_id]
    old_coefficients = dict(old_axis.coefficients)
    old_weight_product = old_coefficients[feature_names[0]] * old_coefficients[feature_names[1]]
    if old_assessment.heldout_accuracy != 1.0 or old_weight_product <= 0.0:
        raise AssertionError("old SUM regime failed latent projection gate")
    old_concepts = [c for c in old_cycle.concepts if old_axis.axis_id in c.defining_features]
    old_laws = [l for l in old_cycle.laws if l.concept_id in {c.concept_id for c in old_concepts}]
    if not old_concepts or not any(l.status == "BOUNDED_LAW" for l in old_laws):
        raise AssertionError("old representation did not enter bounded semantic cognition")

    old_proposals = remember_reference_experiments(runtime, old_axis, old_measurements)
    old_effects = execute_many(runtime, old_proposals, scale, swap, feature_names, "SUM", old_context, issuer_ids, signers, verifier, 1000)
    old_supported = [p for p in old_proposals if min(abs(v) for v in old_effects[p.experiment_id]) >= 0.5]
    if len(old_supported) < 2:
        raise AssertionError("fewer than two old exact experiments had material effects")
    policy = EvidenceBoundWorldActionPolicy()
    old_action = policy.select(old_proposals, runtime.world_coupling, context_id=old_context)
    if old_action.status != "WORLD_SUPPORTED_ACTION":
        raise AssertionError("old world did not create a supported action")

    execute_many(runtime, old_supported, scale, swap, feature_names, "DIFF", new_context, issuer_ids, signers, verifier, 5000)
    revision = AuthenticatedWorldCognitionReviser().assess_and_apply(runtime.memory, runtime.world_coupling, old_axis.axis_id, old_context, new_context)
    if revision.status != "PASS_BOUNDED_WORLD_CAUSED_COGNITION_DEMOTION" or len(revision.counterevidence) < 2 or revision.residual is None:
        raise AssertionError(f"changed world failed robust cognition demotion: {revision.status}")
    if runtime.memory.representations[old_axis.axis_id].status != "SHADOW_WORLD_REFUTED":
        raise AssertionError("old representation remained active")
    if any(r.status == "PROPOSAL_ONLY" and r.proposal.axis_id == old_axis.axis_id for r in runtime.memory.experiments.values()):
        raise AssertionError("old experiments remained actionable")

    new_world = HiddenRegimeWorld(scale, swap, feature_names, "DIFF", new_context, "new-observation-source", "new-observation-challenge", 9000, signers[issuer_ids[1]])
    new_measurements, new_residuals = build_observations(new_world, NEW_TRAIN, NEW_HELDOUT, label_flip)
    regeneration_residuals = [revision.residual] + new_residuals
    new_cycle = runtime.cycle(
        TaskState(goal="regenerate minimum-sufficient cognition after authenticated world mismatch", novelty=0.99, residuals=[r.residual_id for r in regeneration_residuals if not r.heldout], external_world=True, action_required=True),
        residuals=regeneration_residuals,
        measurements=new_measurements,
        world_context_id=new_context,
    )
    new_axis, new_assessments = best_incremental(new_cycle, "DIFFERENCE")
    new_assessment = new_assessments[new_axis.axis_id]
    if new_assessment.heldout_accuracy != 1.0:
        raise AssertionError("new difference representation failed held-out reproduction")
    if new_axis.axis_id == old_axis.axis_id or new_axis.family == old_axis.family:
        raise AssertionError("world change did not produce a different representation family")
    if runtime.memory.representations[old_axis.axis_id].status != "SHADOW_WORLD_REFUTED":
        raise AssertionError("fresh cognition silently reactivated the refuted parent")
    if runtime.memory.representations[new_axis.axis_id].status != "ACTIVE_VALIDATED":
        raise AssertionError("new minimum-sufficient representation did not activate")

    new_concepts = [c for c in new_cycle.concepts if new_axis.axis_id in c.defining_features]
    new_laws = [l for l in new_cycle.laws if l.concept_id in {c.concept_id for c in new_concepts}]
    if not new_concepts or not any(l.status == "BOUNDED_LAW" for l in new_laws):
        raise AssertionError("new representation did not re-enter bounded semantic cognition")

    new_proposals = remember_reference_experiments(runtime, new_axis, new_measurements)
    new_effects = execute_many(runtime, new_proposals, scale, swap, feature_names, "DIFF", new_context, issuer_ids, signers, verifier, 12000)
    new_action = policy.select(new_proposals, runtime.world_coupling, context_id=new_context)
    if new_action.status != "WORLD_SUPPORTED_ACTION" or new_action.proposal is None:
        raise AssertionError("new cognition failed to earn world action authority")
    if min(abs(v) for v in new_effects[new_action.proposal.experiment_id]) < 0.5:
        raise AssertionError("new selected experiment lacked material world effect")

    encoded = checkpoint_json(runtime)
    if any(secret.hex() in encoded for secret in secrets_by_issuer.values()):
        raise AssertionError("verifier secret leaked into BODY checkpoint")
    old_axis_id, new_axis_id = old_axis.axis_id, new_axis.axis_id
    selected_new_experiment_id = new_action.proposal.experiment_id
    del old_cycle, new_cycle, old_proposals, new_proposals, old_axis, new_axis

    unverified_descendant = restore_json(encoded)
    unverified_action = policy.select(unverified_descendant.persisted_intervention_proposals(), unverified_descendant.world_coupling, context_id=new_context)
    if unverified_action.status != "EXPLORE_ONLY_NO_WORLD_SUPPORTED_ACTION":
        raise AssertionError("descendant self-restored external authority")

    descendant = restore_json(encoded, world_verifier=verifier)
    if descendant.memory.representations[old_axis_id].status != "SHADOW_WORLD_REFUTED":
        raise AssertionError("descendant forgot refuted parent representation")
    if descendant.memory.representations[new_axis_id].status != "ACTIVE_VALIDATED":
        raise AssertionError("descendant lost new active representation")
    inherited_new_axis = descendant.memory.representations[new_axis_id].axis
    if inherited_new_axis.family != "DIFFERENCE":
        raise AssertionError("descendant reconstructed wrong new representation family")
    descendant_action = policy.select(descendant.persisted_intervention_proposals(), descendant.world_coupling, context_id=new_context)
    if descendant_action.status != "WORLD_SUPPORTED_ACTION" or descendant_action.proposal is None:
        raise AssertionError("reverified descendant lost post-rewrite action")
    if descendant_action.proposal.experiment_id != selected_new_experiment_id:
        raise AssertionError("descendant chose a different post-rewrite exact experiment")

    print(json.dumps({
        "status": "PASS_BOUNDED_AUTHENTICATED_WORLD_CAUSED_REPRESENTATION_FAMILY_REWRITE_AND_DESCENDANT",
        "old_context": old_context,
        "new_context": new_context,
        "old_representation_family": "PROJECTION",
        "new_representation_family": inherited_new_axis.family,
        "old_axis_id": old_axis_id,
        "new_axis_id": new_axis_id,
        "representation_family_changed": True,
        "minimum_sufficient_contraction": True,
        "old_projection_weight_product": old_weight_product,
        "old_heldout_accuracy": old_assessment.heldout_accuracy,
        "new_heldout_accuracy": new_assessment.heldout_accuracy,
        "old_supported_exact_experiments": len(old_supported),
        "contradicted_exact_experiments": len(revision.counterevidence),
        "counterevidence_modes": sorted({item.contradiction for item in revision.counterevidence}),
        "world_revision_status": revision.status,
        "world_revision_residual_id": revision.residual.residual_id,
        "world_revision_mutation_count": len(revision.mutations),
        "old_action_status": old_action.status,
        "new_action_status": new_action.status,
        "new_action_experiment_id": selected_new_experiment_id,
        "unverified_descendant_action_status": unverified_action.status,
        "reverified_descendant_action_status": descendant_action.status,
        "reverified_descendant_experiment_id": descendant_action.proposal.experiment_id,
        "refuted_parent_representation_preserved": True,
        "external_verifier_secret_persisted": False,
        "independent_organizational_custody": False,
        "physical_world": False,
        "recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_world_caused_cognition_rewrite.py <evaluator-owned-seed-file>")
    main(sys.argv[1])
