from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from arte_cognition.sparse_minterm_genesis import SparseMintermCausalGenesisEngine
from arte_cognition.world_coupling import (
    HMACWorldReceiptSigner,
    HMACWorldReceiptVerifier,
    WorldOutcomeReceipt,
)


def proposal(descriptor: InterventionDescriptor) -> InterventionProposal:
    return InterventionProposal(
        experiment_id=descriptor.intervention_id,
        axis_id="RAW_PRIMITIVE_DISCOVERY_AXIS",
        manipulated_variable=descriptor.targets[0] if descriptor.targets else "__context__",
        held_fixed=tuple((f"blocked::{name}", 1.0) for name in descriptor.blocked),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="LOW",
        predicted_high_side="HIGH",
        reason="repeat same authored intervention semantics under authenticated raw observable variation",
    )


class HiddenPrimitiveWorld:
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
        receipt = WorldOutcomeReceipt(
            receipt_id=f"{self.challenge_id}::{p.experiment_id}::{arm}",
            experiment_id=p.experiment_id,
            axis_id=p.axis_id,
            arm=arm,
            intervention_value=float(value),
            outcome=0.0 if arm.upper() == "LOW" else effect,
            source_id=self.source_id,
            context_id="hidden-raw-primitive-world",
            challenge_id=self.challenge_id,
            epoch=1,
            budget_token=f"budget::{self.challenge_id}",
            externally_generated=True,
        )
        return self.signer.sign(receipt)


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


def execute_two(
    runtime,
    descriptor,
    row,
    hidden_model,
    world_signers,
    world_verifier,
    raw_signers,
    raw_verifier,
    suffix,
    trial_index,
):
    pairs = []
    for issuer_index, (issuer, signer) in enumerate(world_signers.items()):
        pair = runtime.execute_world_intervention(
            proposal(descriptor),
            HiddenPrimitiveWorld(
                hidden_model,
                signer,
                source_id=f"source-{issuer_index}-{trial_index}-{suffix}",
                challenge_id=f"challenge-{issuer_index}-{trial_index}-{suffix}",
            ),
            verifier=world_verifier,
        )
        pairs.append(pair)
        signed_raw = raw_signers[issuer].sign(raw_receipt(pair, row))
        runtime.ingest_raw_observation_receipt(signed_raw, raw_verifier)
    return pairs


def main(seed_path: str) -> None:
    rng = random.Random(int(Path(seed_path).read_text().strip()))
    suffix = rng.randrange(100000, 999999)
    x, z = f"sensor_x_{suffix}", f"sensor_z_{suffix}"
    variables = [x, z]

    trial_count = 12
    descriptors = [
        InterventionDescriptor(
            intervention_id=f"RAWTRIAL::{suffix}::{index:02d}",
            targets=(x,),
            blocked=(),
            delay_steps=0,
            context_shift=False,
            cost=1.0,
        )
        for index in range(trial_count)
    ]

    raw_channel_a = f"raw_{rng.randrange(10**7, 10**8)}"
    raw_channel_b = f"raw_{rng.randrange(10**7, 10**8)}"
    while raw_channel_b == raw_channel_a:
        raw_channel_b = f"raw_{rng.randrange(10**7, 10**8)}"
    values_a = list(range(trial_count))
    values_b = list(range(trial_count))
    rng.shuffle(values_a)
    rng.shuffle(values_b)
    raw_observations = {
        descriptor.intervention_id: {
            raw_channel_a: float(values_a[index]),
            raw_channel_b: float(values_b[index]),
        }
        for index, descriptor in enumerate(descriptors)
    }

    g4_engine = SparseMintermCausalGenesisEngine(model_budget=4096, max_minterms=3)
    g4 = g4_engine.generate_novel(variables, descriptors, (), ())
    assert g4 and not g4_engine.last_truncated
    g4_models = [item.model for item in g4]
    g4_signatures = {tuple(sorted(model.predictions)) for model in g4_models}

    primitive_engine = RawThresholdPrimitiveGenesisEngine(model_budget=4096)
    g5_shadow = primitive_engine.generate_novel(
        variables, descriptors, raw_observations, (), g4_models
    )
    assert g5_shadow and not primitive_engine.last_truncated
    balanced = []
    for item in g5_shadow:
        effect_count = sum(
            1 for _intervention_id, outcome in item.model.predictions
            if outcome != "NO_EFFECT"
        )
        if 3 <= effect_count <= trial_count - 3:
            balanced.append(item)
    assert balanced
    hidden = rng.choice(balanced)
    hidden_signature = tuple(sorted(hidden.model.predictions))
    assert hidden_signature not in g4_signatures

    runtime = WorldDrivenPrimitiveRuntime()
    runtime.register_causal_world_models(g4_models)

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

    # Negative control: one authenticated class cannot make raw representation
    # authoritative, and a post-signature mutation cannot supply the second class.
    first = descriptors[0]
    first_row = raw_observations[first.intervention_id]
    issuer_a = f"issuer-a-{suffix}"
    pair_a = runtime.execute_world_intervention(
        proposal(first),
        HiddenPrimitiveWorld(
            hidden.model,
            world_signers[issuer_a],
            source_id=f"source-0-0-{suffix}",
            challenge_id=f"challenge-0-0-{suffix}",
        ),
        verifier=world_verifier,
    )
    signed_a = raw_signers[issuer_a].sign(raw_receipt(pair_a, first_row))
    runtime.ingest_raw_observation_receipt(signed_a, raw_verifier)
    assert first.intervention_id not in runtime.raw_observation_memory

    issuer_b = f"issuer-b-{suffix}"
    pair_b = runtime.execute_world_intervention(
        proposal(first),
        HiddenPrimitiveWorld(
            hidden.model,
            world_signers[issuer_b],
            source_id=f"source-1-0-{suffix}",
            challenge_id=f"challenge-1-0-{suffix}",
        ),
        verifier=world_verifier,
    )
    honest_b = raw_signers[issuer_b].sign(raw_receipt(pair_b, first_row))
    tampered_b = RawObservationReceipt(
        observation_id=honest_b.observation_id + "::tampered",
        intervention_id=honest_b.intervention_id,
        channel_values=((raw_channel_a, first_row[raw_channel_a] + 999.0), (raw_channel_b, first_row[raw_channel_b])),
        source_id=honest_b.source_id,
        context_id=honest_b.context_id,
        challenge_id=honest_b.challenge_id,
        epoch=honest_b.epoch,
        externally_generated=honest_b.externally_generated,
        issuer_id=honest_b.issuer_id,
        signature=honest_b.signature,
    )
    runtime.ingest_raw_observation_receipt(tampered_b, raw_verifier)
    assert first.intervention_id not in runtime.raw_observation_memory
    runtime.ingest_raw_observation_receipt(honest_b, raw_verifier)
    assert runtime.raw_observation_memory[first.intervention_id] == first_row

    for trial_index, descriptor in enumerate(descriptors[1:], start=1):
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
    g4_final = runtime.generation_version_space(4)
    assert not g4_final.compatible_model_ids
    assert runtime.generation_falsified(4)
    assert runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS"

    try:
        runtime.expand_causal_model_class_with_raw_observations(variables, descriptors, raw_observations)
        raise AssertionError("direct raw-map bypass must fail closed")
    except ValueError:
        pass

    frontier = runtime.expand_causal_model_class_with_raw_observations(variables, descriptors)
    assert frontier.status == "EXPANDED"
    assert frontier.generation == 5
    assert frontier.origin == "GENERATED_PRIMITIVE_THRESHOLD"

    g5_final = runtime.generation_version_space(5)
    assert g5_final.identified
    assert g5_final.identified_model_id == hidden.model.model_id
    assert len(frontier.active_model_ids) == 1

    payload = primitive_checkpoint_dict(runtime)
    no_verify = restore_world_driven_primitive_runtime(payload, world_verifier=None, raw_observation_verifier=None)
    world_only = restore_world_driven_primitive_runtime(payload, world_verifier=world_verifier, raw_observation_verifier=None)
    reverified = restore_world_driven_primitive_runtime(
        payload,
        world_verifier=world_verifier,
        raw_observation_verifier=raw_verifier,
    )
    assert not no_verify.raw_observation_memory
    assert not world_only.raw_observation_memory
    assert reverified.raw_observation_memory == runtime.raw_observation_memory
    no_verify_g5 = no_verify.generation_version_space(5)
    reverified_g5 = reverified.generation_version_space(5)
    assert len(no_verify_g5.compatible_model_ids) == len(frontier.shadow_model_ids)
    assert reverified_g5.identified_model_id == hidden.model.model_id

    old_semantic_signatures = {
        (d.targets, d.blocked, d.delay_steps, d.context_shift)
        for d in descriptors
    }
    assert len(old_semantic_signatures) == 1

    print(json.dumps({
        "status": "PASS_BOUNDED_AUTHENTICATED_RAW_PRIMITIVE_GENESIS_AND_DESCENDANT",
        "old_atom_semantic_signature_count": len(old_semantic_signatures),
        "same_old_semantics_across_all_trials": True,
        "raw_channel_count": 2,
        "raw_channel_names_random_post_checkout": True,
        "raw_outcome_keys_separated": True,
        "raw_quorum_requires_two_independence_classes": True,
        "tampered_raw_signature_rejected": True,
        "direct_raw_map_bypass_blocked": True,
        "hidden_primitive_channel": hidden.primitive.channel,
        "hidden_primitive_direction": hidden.primitive.direction,
        "hidden_primitive_threshold": hidden.primitive.threshold,
        "hidden_primitive_exposed_to_body": False,
        "hidden_g5_model": hidden.model.model_id,
        "g4_complete_candidate_universe": True,
        "g4_shadow_model_count": len(g4_models),
        "g4_final_version_space": len(g4_final.compatible_model_ids),
        "g5_shadow_model_count": len(frontier.shadow_model_ids),
        "g5_active_model_count": len(frontier.active_model_ids),
        "g5_exact_identified_model": g5_final.identified_model_id,
        "g5_prediction_signature_absent_from_g4": True,
        "primitive_candidate_generation_uses_outcomes": False,
        "primitive_activation_requires_g4_falsification": True,
        "verifierless_descendant_raw_rows": len(no_verify.raw_observation_memory),
        "world_only_descendant_raw_rows": len(world_only.raw_observation_memory),
        "reverified_descendant_raw_rows": len(reverified.raw_observation_memory),
        "reverified_descendant_g5_identified_model": reverified_g5.identified_model_id,
        "numeric_threshold_meta_rule_human_authored": True,
        "unrestricted_primitive_genesis": False,
        "generation_beyond_5": False,
        "physical_world": False,
        "independent_organizational_custody": False,
        "global_recursive_acceleration": False,
        "foundation_weight_change": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
