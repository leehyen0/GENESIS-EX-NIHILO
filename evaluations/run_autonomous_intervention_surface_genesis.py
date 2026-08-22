from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.epistemic_depth_runtime import (
    EpistemicallyDeepPersistentCognitiveRuntime,
    epistemic_checkpoint_dict,
    restore_epistemic_runtime,
)
from arte_cognition.experiment_genesis import InterventionProposal
from arte_cognition.intervention_surface_genesis import InterventionSurfaceGenesisEngine
from arte_cognition.world_coupling import HMACWorldReceiptSigner, HMACWorldReceiptVerifier, WorldOutcomeReceipt
from arte_cognition.world_model_ecology import CausalWorldModel


def proposal_from_descriptor(d):
    return InterventionProposal(
        experiment_id=d.intervention_id,
        axis_id="AUTONOMOUS_INTERVENTION_SURFACE_AXIS",
        manipulated_variable=d.targets[0] if d.targets else "__context__",
        held_fixed=tuple((f"blocked::{name}", 1.0) for name in d.blocked),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="LOW",
        predicted_high_side="HIGH",
        reason=(
            f"BODY-generated intervention targets={d.targets} blocked={d.blocked} "
            f"delay={d.delay_steps} context={d.context_shift}"
        ),
    )


class HiddenModelWorld:
    def __init__(self, hidden_model, signer, source_id, context_id, challenge_id):
        self.hidden_model = hidden_model
        self.signer = signer
        self.source_id = source_id
        self.context_id = context_id
        self.challenge_id = challenge_id

    def execute(self, proposal, arm: str, value: float):
        label = self.hidden_model.prediction_for(proposal.experiment_id) or "NO_EFFECT"
        effect = {
            "POSITIVE_EFFECT": 1.0,
            "NEGATIVE_EFFECT": -1.0,
            "NO_EFFECT": 0.0,
        }[label]
        receipt = WorldOutcomeReceipt(
            receipt_id=f"{self.challenge_id}::{proposal.experiment_id}::{arm}",
            experiment_id=proposal.experiment_id,
            axis_id=proposal.axis_id,
            arm=arm,
            intervention_value=float(value),
            outcome=0.0 if arm.upper() == "LOW" else effect,
            source_id=self.source_id,
            context_id=self.context_id,
            challenge_id=self.challenge_id,
            epoch=1,
            budget_token=f"budget::{self.challenge_id}",
            externally_generated=True,
        )
        return self.signer.sign(receipt)


def execute_two(runtime, descriptor, hidden_model, signers, verifier, suffix):
    proposal = proposal_from_descriptor(descriptor)
    for index, (_issuer, signer) in enumerate(signers.items()):
        runtime.execute_world_intervention(
            proposal,
            HiddenModelWorld(
                hidden_model,
                signer,
                source_id=f"source-{index}-{suffix}",
                context_id="hidden-intervention-surface",
                challenge_id=f"challenge-{index}-{suffix}",
            ),
            verifier=verifier,
        )


def main(seed_path: str) -> None:
    rng = random.Random(int(Path(seed_path).read_text().strip()))
    suffix = rng.randrange(100000, 999999)
    x, z = f"sensor_x_{suffix}", f"sensor_z_{suffix}"
    variables = [x, z]

    # Evaluator uses the same public capability schema only to construct a hidden
    # challenge. It does not pass any concrete candidate descriptor list to BODY.
    evaluator_surface_engine = InterventionSurfaceGenesisEngine(budget=256)
    full_surface = evaluator_surface_engine.generate(variables)
    assert not evaluator_surface_engine.last_truncated

    hard_pool = [
        row for row in full_surface
        if set(row.targets) == {x, z}
        and len(row.blocked) == 1
        and row.delay_steps == 1
        and row.context_shift
        and row.cost >= 10.0
    ]
    assert hard_pool
    decisive = rng.choice(hard_pool)

    predictions_a = []
    predictions_b = []
    for row in full_surface:
        if row.intervention_id == decisive.intervention_id:
            predictions_a.append((row.intervention_id, "POSITIVE_EFFECT"))
            predictions_b.append((row.intervention_id, "NO_EFFECT"))
        else:
            predictions_a.append((row.intervention_id, "NO_EFFECT"))
            predictions_b.append((row.intervention_id, "NO_EFFECT"))

    model_a = CausalWorldModel(
        "G3_HYPOTHESIS_A", 1.0, tuple(predictions_a),
        origin="GENERATED_PREDICATE", family="SYNTHESIZED_ACTIVATION_PREDICATE",
        structure=("HIDDEN_BRANCH_A",), generation=3,
    )
    model_b = CausalWorldModel(
        "G3_HYPOTHESIS_B", 1.0, tuple(predictions_b),
        origin="GENERATED_PREDICATE", family="SYNTHESIZED_ACTIVATION_PREDICATE",
        structure=("HIDDEN_BRANCH_B",), generation=3,
    )
    hidden_model = rng.choice([model_a, model_b])

    runtime = EpistemicallyDeepPersistentCognitiveRuntime()
    runtime.register_causal_world_models([model_a, model_b])
    initial = runtime.generation_version_space(3)
    assert initial.compatible_model_ids == ("G3_HYPOTHESIS_A", "G3_HYPOTHESIS_B")

    # No concrete intervention candidates are supplied here. BODY generates them.
    choice = runtime.select_synthesized_generation_intervention(
        generation=3,
        variables=variables,
        observed_intervention_ids=(),
    )
    assert choice.status == "SELECTED"
    assert choice.descriptor is not None
    assert choice.descriptor.intervention_id == decisive.intervention_id
    assert choice.expected_information_gain == 1.0
    assert choice.descriptor.cost >= 10.0
    assert set(choice.descriptor.targets) == {x, z}
    assert len(choice.descriptor.blocked) == 1
    assert choice.descriptor.delay_steps == 1
    assert choice.descriptor.context_shift

    keys = {
        f"issuer-a-{suffix}": f"secret-a-{suffix}".encode(),
        f"issuer-b-{suffix}": f"secret-b-{suffix}".encode(),
    }
    signers = {issuer: HMACWorldReceiptSigner(issuer, secret) for issuer, secret in keys.items()}
    verifier = HMACWorldReceiptVerifier(keys, independence_classes={
        f"issuer-a-{suffix}": "independent-A",
        f"issuer-b-{suffix}": "independent-B",
    })
    execute_two(runtime, choice.descriptor, hidden_model, signers, verifier, suffix)

    final_space = runtime.generation_version_space(3)
    assert final_space.identified
    assert final_space.identified_model_id == hidden_model.model_id

    payload = epistemic_checkpoint_dict(runtime)
    no_verify = restore_epistemic_runtime(payload, world_verifier=None)
    reverified = restore_epistemic_runtime(payload, world_verifier=verifier)
    no_verify_space = no_verify.generation_version_space(3)
    reverified_space = reverified.generation_version_space(3)
    assert len(no_verify_space.compatible_model_ids) == 2
    assert reverified_space.identified_model_id == hidden_model.model_id

    # The verifierless descendant reconstructs the same missing experiment from
    # variables alone; the reverified descendant correctly stops because identified.
    replay_choice = no_verify.select_synthesized_generation_intervention(3, variables, ())
    assert replay_choice.status == "SELECTED"
    assert replay_choice.descriptor is not None
    assert replay_choice.descriptor.intervention_id == decisive.intervention_id
    done_choice = reverified.select_synthesized_generation_intervention(3, variables, ())
    assert done_choice.status == "ALREADY_IDENTIFIED"

    print(json.dumps({
        "status": "PASS_BOUNDED_AUTONOMOUS_INTERVENTION_SURFACE_GENESIS_SELECTION_AND_DESCENDANT_RECONSTRUCTION",
        "evaluator_supplied_concrete_intervention_candidates": False,
        "hidden_model_exposed_to_body": False,
        "hidden_model": hidden_model.model_id,
        "generated_surface_size": choice.generated_surface_size,
        "selected_generated_intervention_id": choice.descriptor.intervention_id,
        "selected_targets": list(choice.descriptor.targets),
        "selected_blocked": list(choice.descriptor.blocked),
        "selected_delay_steps": choice.descriptor.delay_steps,
        "selected_context_shift": choice.descriptor.context_shift,
        "selected_cost": choice.descriptor.cost,
        "selected_expected_information_gain": choice.expected_information_gain,
        "initial_version_space": len(initial.compatible_model_ids),
        "final_version_space": len(final_space.compatible_model_ids),
        "verifierless_descendant_version_space": len(no_verify_space.compatible_model_ids),
        "verifierless_descendant_reconstructed_same_intervention": True,
        "reverified_descendant_identified_model": reverified_space.identified_model_id,
        "intervention_capability_schema_human_authored": True,
        "unrestricted_action_operator_genesis": False,
        "physical_world": False,
        "global_recursive_acceleration": False,
        "foundation_weight_change": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
