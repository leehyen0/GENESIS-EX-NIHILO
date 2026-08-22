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
from arte_cognition.representation_genesis import MeasurementObservation, RepresentationGenesisEngine
from arte_cognition.semantic_genesis import ResidualObservation
from arte_cognition.world_action_policy import EvidenceBoundWorldActionPolicy
from arte_cognition.world_coupling import (
    HMACWorldReceiptSigner,
    HMACWorldReceiptVerifier,
    WorldOutcomeReceipt,
)


BASE_TRAIN = (
    ("a1", -3.0,  2.0, "A"),
    ("a2", -2.0,  1.0, "A"),
    ("a3", -1.0,  0.0, "A"),
    ("a4",  0.0, -1.0, "A"),
    ("b1", -2.0,  3.0, "B"),
    ("b2", -1.0,  2.0, "B"),
    ("b3",  0.0,  1.0, "B"),
    ("b4",  1.0,  0.0, "B"),
)
BASE_HELDOUT = (
    ("ha", -4.0,  3.0, "A"),
    ("hb",  3.0, -2.0, "B"),
)


class HiddenAffineWorld:
    """Post-freeze software world used only by the evaluator.

    The BODY sees transformed measurements and signed consequence receipts, never
    the inverse transform or the hidden rule `base_x + base_y > 0`.
    """

    def __init__(
        self,
        scale: float,
        swap: bool,
        feature_names: tuple[str, str],
        source_id: str,
        challenge_id: str,
        epoch: int,
        signer: HMACWorldReceiptSigner,
    ) -> None:
        self.scale = float(scale)
        self.swap = bool(swap)
        self.feature_names = feature_names
        self.source_id = source_id
        self.challenge_id = challenge_id
        self.epoch = int(epoch)
        self.signer = signer
        self.context_id = "e2e-hidden-affine-world"

    def encode(self, base_x: float, base_y: float) -> dict[str, float]:
        a, b = (base_y, base_x) if self.swap else (base_x, base_y)
        return {
            self.feature_names[0]: self.scale * float(a),
            self.feature_names[1]: self.scale * float(b),
        }

    def decode(self, observed: dict[str, float]) -> tuple[float, float]:
        a = float(observed[self.feature_names[0]]) / self.scale
        b = float(observed[self.feature_names[1]]) / self.scale
        return (b, a) if self.swap else (a, b)

    @staticmethod
    def hidden_outcome(base_x: float, base_y: float) -> float:
        return 1.0 if float(base_x) + float(base_y) > 0.0 else 0.0

    def execute(self, proposal, arm, value):
        state = {name: 0.0 for name in self.feature_names}
        state.update({name: float(v) for name, v in proposal.held_fixed})
        state[proposal.manipulated_variable] = float(value)
        base_x, base_y = self.decode(state)
        outcome = self.hidden_outcome(base_x, base_y)
        receipt = WorldOutcomeReceipt(
            receipt_id=(
                f"e2e::{self.source_id}::{self.challenge_id}::"
                f"{proposal.experiment_id}::{arm}"
            ),
            experiment_id=proposal.experiment_id,
            axis_id=proposal.axis_id,
            arm=arm,
            intervention_value=float(value),
            outcome=outcome,
            source_id=self.source_id,
            context_id=self.context_id,
            challenge_id=self.challenge_id,
            epoch=self.epoch,
            budget_token=f"matched::{self.source_id}::{self.challenge_id}",
            externally_generated=True,
        )
        return self.signer.sign(receipt)


def build_observations(world: HiddenAffineWorld, label_flip: bool):
    measurements = []
    residuals = []
    for heldout, rows in ((False, BASE_TRAIN), (True, BASE_HELDOUT)):
        for observation_id, base_x, base_y, base_label in rows:
            label = ("B" if base_label == "A" else "A") if label_flip else base_label
            values = world.encode(base_x, base_y)
            measurements.append(MeasurementObservation(
                observation_id=observation_id,
                values=values,
                outcome=label,
                heldout=heldout,
                context_id=world.context_id,
            ))
            residuals.append(ResidualObservation(
                residual_id=observation_id,
                features=("raw-unexplained-residual",),
                outcome=label,
                source_class="e2e-hidden-measurement",
                heldout=heldout,
            ))
    return measurements, residuals


def main(seed_path: str) -> None:
    seed = int(Path(seed_path).read_text().strip())
    rng = random.Random(seed)

    scale = rng.choice((-1.0, 1.0)) * rng.uniform(0.55, 3.25)
    swap = bool(rng.getrandbits(1))
    feature_names = (
        f"sensor_{rng.randrange(10_000, 99_999)}",
        f"sensor_{rng.randrange(100_000, 999_999)}",
    )
    label_flip = bool(rng.getrandbits(1))
    issuer_ids = (
        f"e2e-hidden-evaluator-a-{rng.randrange(1_000_000, 9_999_999)}",
        f"e2e-hidden-evaluator-b-{rng.randrange(1_000_000, 9_999_999)}",
    )
    receipt_secrets = (secrets.token_bytes(32), secrets.token_bytes(32))
    signers = tuple(
        HMACWorldReceiptSigner(issuer_id, secret)
        for issuer_id, secret in zip(issuer_ids, receipt_secrets)
    )
    verifier = HMACWorldReceiptVerifier(
        dict(zip(issuer_ids, receipt_secrets)),
        independence_classes={
            issuer_ids[0]: "e2e-independent-world-class-a",
            issuer_ids[1]: "e2e-independent-world-class-b",
        },
    )

    discovery_world = HiddenAffineWorld(
        scale=scale,
        swap=swap,
        feature_names=feature_names,
        source_id="discovery-source",
        challenge_id="discovery-challenge",
        epoch=0,
        signer=signers[0],
    )
    measurements, residuals = build_observations(discovery_world, label_flip)

    runtime = PersistentCognitiveRuntime()
    zero_reference = {name: 0.0 for name in feature_names}
    cycle = runtime.cycle(
        TaskState(
            goal="discover a representation that explains the unseen residual structure",
            novelty=0.95,
            residuals=[row.residual_id for row in residuals if not row.heldout],
            external_world=True,
            action_required=True,
        ),
        residuals=residuals,
        measurements=measurements,
        experiment_reference_values=zero_reference,
        world_context_id=discovery_world.context_id,
    )

    assessment_by_axis = {item.axis_id: item for item in cycle.representation_value}
    eligible_projection_axes = [
        axis for axis in cycle.representation_axes
        if axis.family == "PROJECTION"
        and assessment_by_axis.get(axis.axis_id) is not None
        and assessment_by_axis[axis.axis_id].status == "INCREMENTAL_REPRESENTATION_VALUE"
    ]
    if not eligible_projection_axes:
        raise AssertionError("raw unseen measurements did not produce an incrementally valuable latent projection")
    selected_axis = max(
        eligible_projection_axes,
        key=lambda axis: assessment_by_axis[axis.axis_id].incremental_gain,
    )
    selected_assessment = assessment_by_axis[selected_axis.axis_id]
    if selected_assessment.heldout_accuracy != 1.0:
        raise AssertionError("generated latent axis did not reproduce on held-out observations")

    generated_concepts = [
        concept for concept in cycle.concepts
        if selected_axis.axis_id in concept.defining_features
    ]
    if not generated_concepts:
        raise AssertionError("generated latent axis did not enter semantic concept formation")
    concept_ids = {concept.concept_id for concept in generated_concepts}
    bounded_laws = [
        law for law in cycle.laws
        if law.concept_id in concept_ids and law.status == "BOUNDED_LAW"
    ]
    if not bounded_laws:
        raise AssertionError("latent-axis concept did not survive predictive held-out law gate")

    # Expand the BODY's own experiment generator over several already observed
    # reference states. Exact intervention fingerprints make each numerical
    # experiment a distinct evidence target rather than aliasing them by axis.
    train_measurements = [row for row in measurements if not row.heldout]
    for row in train_measurements[:6]:
        for generated in runtime.experiment.propose(selected_axis, row.values):
            runtime.memory.remember_experiment(generated)

    selected_proposals = [
        proposal for proposal in runtime.persisted_intervention_proposals()
        if proposal.axis_id == selected_axis.axis_id
    ]
    if not selected_proposals:
        raise AssertionError("generated latent representation did not create any persistent experiment candidates")
    if len({p.experiment_id for p in selected_proposals}) != len(selected_proposals):
        raise AssertionError("numerically distinct generated experiments aliased to one experiment identity")

    remove_runtime = PersistentCognitiveRuntime(
        representation=RepresentationGenesisEngine(enable_projection=False)
    )
    remove_cycle = remove_runtime.cycle(
        TaskState(
            goal="discover a representation that explains the unseen residual structure",
            novelty=0.95,
            residuals=[row.residual_id for row in residuals if not row.heldout],
            external_world=True,
            action_required=True,
        ),
        residuals=residuals,
        measurements=measurements,
        experiment_reference_values=zero_reference,
        world_context_id=discovery_world.context_id,
    )
    if any(axis.family == "PROJECTION" for axis in remove_cycle.representation_axes):
        raise AssertionError("REMOVE-PROJECTION control regenerated the removed latent family")
    if any(proposal.axis_id == selected_axis.axis_id for proposal in remove_cycle.intervention_proposals):
        raise AssertionError("REMOVE-PROJECTION control retained the selected generated representation")

    action_policy = EvidenceBoundWorldActionPolicy()
    before_action = action_policy.select(
        selected_proposals,
        runtime.world_coupling,
        context_id=discovery_world.context_id,
    )
    if before_action.status != "EXPLORE_ONLY_NO_WORLD_SUPPORTED_ACTION" or before_action.proposal is not None:
        raise AssertionError("generated experiments self-promoted to action before world evidence")

    effects_by_experiment = {}
    for proposal_index, proposal in enumerate(selected_proposals, start=1):
        effects = []
        suffix = proposal.experiment_id.rsplit("::", 1)[-1]
        for issuer_index, signer in enumerate(signers, start=1):
            world = HiddenAffineWorld(
                scale=scale,
                swap=swap,
                feature_names=feature_names,
                source_id=f"e2e-source-{proposal_index}-{issuer_index}-{suffix}",
                challenge_id=f"e2e-challenge-{proposal_index}-{issuer_index}-{suffix}",
                epoch=proposal_index * 10 + issuer_index,
                signer=signer,
            )
            pair = runtime.execute_world_intervention(proposal, world, verifier=verifier)
            if not pair.authority_verified:
                raise AssertionError("generated experiment outcome failed external receipt authentication")
            effects.append(abs(pair.effect))
        effects_by_experiment[proposal.experiment_id] = effects

    after_action = action_policy.select(
        selected_proposals,
        runtime.world_coupling,
        context_id=discovery_world.context_id,
    )
    if after_action.status != "WORLD_SUPPORTED_ACTION" or after_action.proposal is None:
        raise AssertionError("none of the BODY-generated exact experiments produced reproducible world consequence")
    chosen_proposal = after_action.proposal
    chosen_effects = effects_by_experiment[chosen_proposal.experiment_id]
    if min(chosen_effects) < 0.5:
        raise AssertionError("selected exact experiment did not reproduce its consequence in both verifier classes")
    if after_action.independent_evidence_classes != 2 or after_action.routing_score <= 0.0:
        raise AssertionError("selected exact experiment lacked two verifier-bound evidence classes")

    encoded = checkpoint_json(runtime)
    if any(secret.hex() in encoded for secret in receipt_secrets) or "trusted_keys" in encoded:
        raise AssertionError("external verifier secret leaked into persistent BODY checkpoint")

    chosen_experiment_id = chosen_proposal.experiment_id
    unverified_descendant = restore_json(encoded)
    unverified_proposals = unverified_descendant.persisted_intervention_proposals()
    if chosen_experiment_id not in {p.experiment_id for p in unverified_proposals}:
        raise AssertionError("descendant BODY lost the exact world-tested generated experiment")
    unverified_action = action_policy.select(
        unverified_proposals,
        unverified_descendant.world_coupling,
        context_id=discovery_world.context_id,
    )
    if unverified_action.status != "EXPLORE_ONLY_NO_WORLD_SUPPORTED_ACTION" or unverified_action.proposal is not None:
        raise AssertionError("descendant used external evidence without re-verification")

    descendant = restore_json(encoded, world_verifier=verifier)
    descendant_proposals = descendant.persisted_intervention_proposals()
    descendant_action = action_policy.select(
        descendant_proposals,
        descendant.world_coupling,
        context_id=discovery_world.context_id,
    )
    if descendant_action.status != "WORLD_SUPPORTED_ACTION" or descendant_action.proposal is None:
        raise AssertionError("reverified descendant did not recover exact world-supported experiment")
    if descendant_action.proposal.experiment_id != chosen_experiment_id:
        raise AssertionError("descendant selected a different experiment than the exact world-supported parent phenotype")

    print(json.dumps({
        "status": "PASS_BOUNDED_EXACT_EXPERIMENT_RAW_RESIDUAL_TO_REVERIFIED_DESCENDANT_ACTION",
        "generated_axis_family": selected_axis.family,
        "generated_axis_id": selected_axis.axis_id,
        "generated_axis_coefficients": list(selected_axis.coefficients),
        "incremental_gain": selected_assessment.incremental_gain,
        "best_parent_information_gain": selected_assessment.best_parent_information_gain,
        "heldout_accuracy": selected_assessment.heldout_accuracy,
        "generated_concept_count": len(generated_concepts),
        "bounded_predictive_law_count": len(bounded_laws),
        "generated_experiment_count": len(selected_proposals),
        "world_effects_by_experiment": effects_by_experiment,
        "selected_experiment_id": chosen_experiment_id,
        "selected_manipulated_variable": chosen_proposal.manipulated_variable,
        "selected_world_effects": chosen_effects,
        "remove_projection_intervention_count": len(remove_cycle.intervention_proposals),
        "before_world_action_status": before_action.status,
        "after_world_action_status": after_action.status,
        "after_world_action_experiment_id": chosen_experiment_id,
        "exact_experiment_evidence_binding": True,
        "cryptographic_world_issuers": 2,
        "verifier_independence_classes": 2,
        "independent_world_evidence_classes": after_action.independent_evidence_classes,
        "unverified_descendant_action_status": unverified_action.status,
        "reverified_descendant_action_status": descendant_action.status,
        "reverified_descendant_experiment_id": descendant_action.proposal.experiment_id,
        "descendant_parent_proposal_reuse": False,
        "authority_reverification_required_after_restart": True,
        "hidden_rule_exposed_to_body": False,
        "post_checkout_random_transform": True,
        "independent_organizational_custody": False,
        "physical_world": False,
        "recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_end_to_end_world_genesis.py <evaluator-owned-seed-file>")
    main(sys.argv[1])
