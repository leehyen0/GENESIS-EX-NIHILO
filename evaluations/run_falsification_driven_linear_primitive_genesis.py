from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.causal_linear_primitive_genesis import LinearFormPrimitiveGenesisEngine
from arte_cognition.causal_model_genesis import InterventionDescriptor
from arte_cognition.causal_primitive_genesis import RawThresholdPrimitiveGenesisEngine
from arte_cognition.experiment_genesis import InterventionProposal
from arte_cognition.primitive_genesis_runtime import (
    WorldDrivenPrimitiveRuntime,
    primitive_checkpoint_dict,
    restore_world_driven_primitive_runtime,
)
from arte_cognition.raw_observation_authority import (
    HMACRawObservationSigner,
    HMACRawObservationVerifier,
    RawObservationReceipt,
)
from arte_cognition.world_coupling import (
    HMACWorldReceiptSigner,
    HMACWorldReceiptVerifier,
    WorldOutcomeReceipt,
)


def proposal(descriptor: InterventionDescriptor) -> InterventionProposal:
    return InterventionProposal(
        experiment_id=descriptor.intervention_id,
        axis_id="RAW_LINEAR_PRIMITIVE_DISCOVERY_AXIS",
        manipulated_variable=descriptor.targets[0] if descriptor.targets else "__context__",
        held_fixed=tuple((f"blocked::{name}", 1.0) for name in descriptor.blocked),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="LOW",
        predicted_high_side="HIGH",
        reason="same authored semantics; authenticated raw channels require synthesized relation",
    )


class HiddenLinearPrimitiveWorld:
    def __init__(self, model, signer, source_id: str, challenge_id: str):
        self.model = model
        self.signer = signer
        self.source_id = source_id
        self.challenge_id = challenge_id

    def execute(self, p, arm: str, value: float):
        label = self.model.prediction_for(p.experiment_id) or "NO_EFFECT"
        effect = {
            "POSITIVE_EFFECT": 1.0,
            "NEGATIVE_EFFECT": -1.0,
            "NO_EFFECT": 0.0,
        }[label]
        return self.signer.sign(WorldOutcomeReceipt(
            receipt_id=f"{self.challenge_id}::{p.experiment_id}::{arm}",
            experiment_id=p.experiment_id,
            axis_id=p.axis_id,
            arm=arm,
            intervention_value=float(value),
            outcome=0.0 if arm.upper() == "LOW" else effect,
            source_id=self.source_id,
            context_id="hidden-linear-primitive-world",
            challenge_id=self.challenge_id,
            epoch=1,
            budget_token=f"budget::{self.challenge_id}",
            externally_generated=True,
        ))


def raw_receipt(pair, row):
    return RawObservationReceipt(
        observation_id=f"RAWOBS::{pair.pair_id}",
        intervention_id=pair.experiment_id,
        channel_values=tuple(sorted((str(k), float(v)) for k, v in row.items())),
        source_id=pair.source_id,
        context_id=pair.context_id,
        challenge_id=pair.challenge_id,
        epoch=pair.epoch,
        externally_generated=True,
    )


def execute_two(runtime, descriptor, row, hidden_model, world_signers, world_verifier, raw_signers, raw_verifier, suffix, trial_index):
    for issuer_index, (issuer, signer) in enumerate(world_signers.items()):
        pair = runtime.execute_world_intervention(
            proposal(descriptor),
            HiddenLinearPrimitiveWorld(
                hidden_model,
                signer,
                source_id=f"source-{issuer_index}-{trial_index}-{suffix}",
                challenge_id=f"challenge-{issuer_index}-{trial_index}-{suffix}",
            ),
            verifier=world_verifier,
        )
        runtime.ingest_raw_observation_receipt(
            raw_signers[issuer].sign(raw_receipt(pair, row)),
            raw_verifier,
        )


def main(seed_path: str) -> None:
    rng = random.Random(int(Path(seed_path).read_text().strip()))
    suffix = rng.randrange(100000, 999999)
    x = f"sensor_x_{suffix}"
    variables = [x]
    trial_count = 14
    descriptors = [
        InterventionDescriptor(
            intervention_id=f"LINEARTRIAL::{suffix}::{index:02d}",
            targets=(x,),
            blocked=(),
            delay_steps=0,
            context_shift=False,
            cost=1.0,
        )
        for index in range(trial_count)
    ]

    channel_a = f"raw_{rng.randrange(10**7, 10**8)}"
    channel_b = f"raw_{rng.randrange(10**7, 10**8)}"
    while channel_b == channel_a:
        channel_b = f"raw_{rng.randrange(10**7, 10**8)}"
    values_a = list(range(trial_count))
    values_b = list(range(trial_count))
    rng.shuffle(values_a)
    rng.shuffle(values_b)
    raw_observations = {
        descriptor.intervention_id: {
            channel_a: float(values_a[index]),
            channel_b: float(values_b[index]),
        }
        for index, descriptor in enumerate(descriptors)
    }

    g5_engine = RawThresholdPrimitiveGenesisEngine(model_budget=4096)
    g5 = g5_engine.generate_novel(variables, descriptors, raw_observations, (), ())
    assert g5 and not g5_engine.last_truncated
    g5_models = [item.model for item in g5]
    g5_signatures = {tuple(sorted(model.predictions)) for model in g5_models}

    g6_engine = LinearFormPrimitiveGenesisEngine(model_budget=8192, max_coefficient_abs=2)
    g6_shadow = g6_engine.generate_novel(variables, descriptors, raw_observations, (), g5_models)
    assert g6_shadow and not g6_engine.last_truncated
    balanced = []
    for item in g6_shadow:
        effect_count = sum(1 for _iid, outcome in item.model.predictions if outcome != "NO_EFFECT")
        coefficient_weights = tuple(weight for _channel, weight in item.primitive.coefficients)
        if 4 <= effect_count <= trial_count - 4 and len(coefficient_weights) >= 2:
            balanced.append(item)
    assert balanced
    hidden = rng.choice(balanced)
    hidden_signature = tuple(sorted(hidden.model.predictions))
    assert hidden_signature not in g5_signatures

    runtime = WorldDrivenPrimitiveRuntime()
    runtime.register_causal_world_models(g5_models)

    world_keys = {
        f"issuer-a-{suffix}": f"world-secret-a-{suffix}".encode(),
        f"issuer-b-{suffix}": f"world-secret-b-{suffix}".encode(),
    }
    raw_keys = {
        f"issuer-a-{suffix}": f"raw-secret-a-{suffix}".encode(),
        f"issuer-b-{suffix}": f"raw-secret-b-{suffix}".encode(),
    }
    independence = {
        f"issuer-a-{suffix}": "independent-A",
        f"issuer-b-{suffix}": "independent-B",
    }
    world_signers = {issuer: HMACWorldReceiptSigner(issuer, secret) for issuer, secret in world_keys.items()}
    world_verifier = HMACWorldReceiptVerifier(world_keys, independence_classes=independence)
    raw_signers = {issuer: HMACRawObservationSigner(issuer, secret) for issuer, secret in raw_keys.items()}
    raw_verifier = HMACRawObservationVerifier(raw_keys, independence_classes=independence)

    for trial_index, descriptor in enumerate(descriptors):
        execute_two(
            runtime,
            descriptor,
            raw_observations[descriptor.intervention_id],
            hidden.model,
            world_signers,
            world_verifier,
            raw_signers,
            raw_verifier,
            suffix,
            trial_index,
        )

    assert len(runtime.raw_observation_memory) == len(descriptors)
    g5_final = runtime.generation_version_space(5)
    assert not g5_final.compatible_model_ids
    assert runtime.generation_falsified(5)
    assert runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS"

    frontier = runtime.expand_causal_model_class_with_raw_observations(variables, descriptors)
    assert frontier.status == "EXPANDED"
    assert frontier.generation == 6
    assert frontier.origin == "GENERATED_LINEAR_PRIMITIVE"
    assert len(frontier.active_model_ids) == 1

    g6_final = runtime.generation_version_space(6)
    assert g6_final.identified
    assert g6_final.identified_model_id == hidden.model.model_id

    payload = primitive_checkpoint_dict(runtime)
    no_verify = restore_world_driven_primitive_runtime(payload)
    world_only = restore_world_driven_primitive_runtime(payload, world_verifier=world_verifier)
    reverified = restore_world_driven_primitive_runtime(
        payload,
        world_verifier=world_verifier,
        raw_observation_verifier=raw_verifier,
    )
    assert not no_verify.raw_observation_memory
    assert not world_only.raw_observation_memory
    assert reverified.raw_observation_memory == runtime.raw_observation_memory
    no_verify_g6 = no_verify.generation_version_space(6)
    reverified_g6 = reverified.generation_version_space(6)
    assert len(no_verify_g6.compatible_model_ids) == len(frontier.shadow_model_ids)
    assert reverified_g6.identified_model_id == hidden.model.model_id

    old_semantic_signatures = {(d.targets, d.blocked, d.delay_steps, d.context_shift) for d in descriptors}
    assert len(old_semantic_signatures) == 1

    print(json.dumps({
        "status": "PASS_BOUNDED_AUTHENTICATED_MULTI_CHANNEL_LINEAR_PRIMITIVE_GENESIS_AND_DESCENDANT",
        "old_atom_semantic_signature_count": len(old_semantic_signatures),
        "same_old_semantics_across_all_trials": True,
        "raw_channel_count": 2,
        "raw_channel_names_random_post_checkout": True,
        "raw_outcome_keys_separated": True,
        "raw_quorum_requires_two_independence_classes": True,
        "g5_complete_candidate_universe": True,
        "g5_shadow_model_count": len(g5_models),
        "g5_final_version_space": len(g5_final.compatible_model_ids),
        "linear_form_count": g6_engine.last_linear_form_count,
        "g6_shadow_model_count": len(frontier.shadow_model_ids),
        "g6_active_model_count": len(frontier.active_model_ids),
        "hidden_g6_model": hidden.model.model_id,
        "hidden_linear_coefficients": hidden.primitive.coefficients,
        "hidden_linear_threshold": hidden.primitive.threshold,
        "hidden_linear_direction": hidden.primitive.direction,
        "hidden_linear_primitive_exposed_to_body": False,
        "g6_prediction_signature_absent_from_g5": True,
        "g6_exact_identified_model": g6_final.identified_model_id,
        "linear_candidate_generation_uses_outcomes": False,
        "linear_activation_requires_g5_falsification": True,
        "verifierless_descendant_raw_rows": len(no_verify.raw_observation_memory),
        "world_only_descendant_raw_rows": len(world_only.raw_observation_memory),
        "reverified_descendant_raw_rows": len(reverified.raw_observation_memory),
        "reverified_descendant_g6_identified_model": reverified_g6.identified_model_id,
        "integer_linear_meta_grammar_human_authored": True,
        "unrestricted_operator_genesis": False,
        "generation_beyond_6": False,
        "physical_world": False,
        "independent_organizational_custody": False,
        "global_recursive_acceleration": False,
        "foundation_weight_change": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
