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
            label = (
                ("B" if base_label == "A" else "A")
                if label_flip
                else base_label
            )
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

    # The transform, feature names, label orientation and receipt key are chosen
    # after checkout from evaluator-owned randomness.
    scale = rng.choice((-1.0, 1.0)) * rng.uniform(0.55, 3.25)
    swap = bool(rng.getrandbits(1))
    feature_names = (
        f"sensor_{rng.randrange(10_000, 99_999)}",
        f"sensor_{rng.randrange(100_000, 999_999)}",
    )
    label_flip = bool(rng.getrandbits(1))
    issuer_id = f"e2e-hidden-evaluator-{rng.randrange(1_000_000, 9_999_999)}"
    receipt_secret = secrets.token_bytes(32)
    signer = HMACWorldReceiptSigner(issuer_id, receipt_secret)
    verifier = HMACWorldReceiptVerifier({issuer_id: receipt_secret})

    discovery_world = HiddenAffineWorld(
        scale=scale,
        swap=swap,
        feature_names=feature_names,
        source_id="discovery-source",
        challenge_id="discovery-challenge",
        epoch=0,
        signer=signer,
    )
    measurements, residuals = build_observations(discovery_world, label_flip)

    runtime = PersistentCognitiveRuntime()
    reference_values = {name: 0.0 for name in feature_names}
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
        experiment_reference_values=reference_values,
        world_context_id=discovery_world.context_id,
    )

    assessment_by_axis = {item.axis_id: item for item in cycle.representation_value}
    axis_by_id = {axis.axis_id: axis for axis in cycle.representation_axes}
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

    selected_proposals = [
        proposal for proposal in cycle.intervention_proposals
        if proposal.axis_id == selected_axis.axis_id
    ]
    if not selected_proposals:
        raise AssertionError("generated latent representation did not create an actionable experiment")
    # At the zero reference state the fixed ratio family has no admissible
    # threshold-crossing intervention; require that the executable candidate is
    # genuinely generated from the learned projection rather than a raw parent.
    executable_families = {
        axis_by_id[proposal.axis_id].family
        for proposal in cycle.intervention_proposals
        if proposal.axis_id in axis_by_id
    }
    if "PROJECTION" not in executable_families:
        raise AssertionError("learned projection was not present in executable experiment candidates")

    selected_proposal = selected_proposals[0]
    remove_projection_proposals = [
        proposal for proposal in cycle.intervention_proposals
        if proposal.axis_id != selected_axis.axis_id
    ]
    if any(proposal.axis_id == selected_axis.axis_id for proposal in remove_projection_proposals):
        raise AssertionError("REMOVE control retained the selected generated representation")

    # Two separately identified challenge receipts provide bounded independent
    # consequence evidence for the BODY's generated experiment.
    observed_effects = []
    for index in (1, 2):
        world = HiddenAffineWorld(
            scale=scale,
            swap=swap,
            feature_names=feature_names,
            source_id=f"e2e-source-{index}",
            challenge_id=f"e2e-challenge-{index}",
            epoch=index,
            signer=signer,
        )
        pair = runtime.execute_world_intervention(
            selected_proposal,
            world,
            verifier=verifier,
        )
        if not pair.authority_verified:
            raise AssertionError("generated experiment outcome failed external receipt authentication")
        observed_effects.append(abs(pair.effect))

    summary = runtime.world_axis_summary(
        selected_axis.axis_id,
        context_id=discovery_world.context_id,
    )
    if summary.independent_evidence_classes != 2:
        raise AssertionError("generated representation did not receive two independent world evidence classes")
    if min(observed_effects) < 0.5 or summary.routing_score <= 0.0:
        raise AssertionError("generated experiment did not causally distinguish the hidden world")

    # The world consequence must change a future choice, not merely be logged.
    ranking_input = list(reversed(cycle.intervention_proposals))
    if not ranking_input:
        raise AssertionError("no generated intervention candidates available for future routing")
    before_world_top = ranking_input[0].axis_id
    after_world_ranked = runtime.rank_intervention_proposals(
        ranking_input,
        context_id=discovery_world.context_id,
    )
    if after_world_ranked[0].axis_id != selected_axis.axis_id:
        raise AssertionError("world consequence did not make the generated representation control future intervention choice")

    encoded = checkpoint_json(runtime)
    if receipt_secret.hex() in encoded or "trusted_keys" in encoded:
        raise AssertionError("external verifier secret leaked into persistent BODY checkpoint")

    # A descendant without the external authority surface cannot self-authorize.
    unverified_descendant = restore_json(encoded)
    if unverified_descendant.world_axis_summary(
        selected_axis.axis_id,
        context_id=discovery_world.context_id,
    ).routing_score != 0.0:
        raise AssertionError("descendant used external evidence without re-verification")

    descendant = restore_json(encoded, world_verifier=verifier)
    descendant_ranked = descendant.rank_intervention_proposals(
        ranking_input,
        context_id=discovery_world.context_id,
    )
    if descendant_ranked[0].axis_id != selected_axis.axis_id:
        raise AssertionError("reverified descendant did not inherit world-caused intervention preference")
    descendant_summary = descendant.world_axis_summary(
        selected_axis.axis_id,
        context_id=discovery_world.context_id,
    )
    if descendant_summary != summary:
        raise AssertionError("world-caused representation evidence changed across descendant reconstruction")

    print(json.dumps({
        "status": "PASS_BOUNDED_RAW_RESIDUAL_TO_REVERIFIED_DESCENDANT_WORLD_GENESIS",
        "generated_axis_family": selected_axis.family,
        "generated_axis_id": selected_axis.axis_id,
        "generated_axis_coefficients": list(selected_axis.coefficients),
        "incremental_gain": selected_assessment.incremental_gain,
        "best_parent_information_gain": selected_assessment.best_parent_information_gain,
        "heldout_accuracy": selected_assessment.heldout_accuracy,
        "generated_concept_count": len(generated_concepts),
        "bounded_predictive_law_count": len(bounded_laws),
        "generated_experiment_id": selected_proposal.experiment_id,
        "generated_experiment_manipulated_variable": selected_proposal.manipulated_variable,
        "remove_projection_experiment_count": len(remove_projection_proposals),
        "world_effects": observed_effects,
        "independent_world_evidence_classes": summary.independent_evidence_classes,
        "before_world_top_axis": before_world_top,
        "after_world_top_axis": after_world_ranked[0].axis_id,
        "descendant_top_axis": descendant_ranked[0].axis_id,
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
