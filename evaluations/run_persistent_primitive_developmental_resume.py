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
from arte_cognition.causal_symbolic_primitive_genesis import SymbolicPrimitiveGenesisEngine
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
        axis_id="PERSISTENT_RAW_DEVELOPMENT_AXIS",
        manipulated_variable=descriptor.targets[0],
        held_fixed=(),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="LOW",
        predicted_high_side="HIGH",
        reason="checkpoint-before-symbolic-expansion authenticated developmental continuity probe",
    )


class HiddenDevelopmentWorld:
    def __init__(self, model, signer, source_id: str, challenge_id: str):
        self.model = model
        self.signer = signer
        self.source_id = source_id
        self.challenge_id = challenge_id

    def execute(self, p, arm: str, value: float):
        label = self.model.prediction_for(p.experiment_id) or "NO_EFFECT"
        effect = {"POSITIVE_EFFECT": 1.0, "NEGATIVE_EFFECT": -1.0, "NO_EFFECT": 0.0}[label]
        return self.signer.sign(WorldOutcomeReceipt(
            receipt_id=f"{self.challenge_id}::{p.experiment_id}::{arm}",
            experiment_id=p.experiment_id,
            axis_id=p.axis_id,
            arm=arm,
            intervention_value=float(value),
            outcome=0.0 if arm.upper() == "LOW" else effect,
            source_id=self.source_id,
            context_id="hidden-persistent-development-world",
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
            HiddenDevelopmentWorld(
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
    trial_count = 16
    descriptors = [
        InterventionDescriptor(
            intervention_id=f"RESUMETRIAL::{suffix}::{index:02d}",
            targets=(x,), blocked=(), delay_steps=0, context_shift=False, cost=1.0,
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
        d.intervention_id: {
            channel_a: float(values_a[i] - trial_count // 2),
            channel_b: float(values_b[i] - trial_count // 2),
        }
        for i, d in enumerate(descriptors)
    }

    g6_engine = LinearFormPrimitiveGenesisEngine(model_budget=8192, max_coefficient_abs=2)
    g6 = g6_engine.generate_novel(variables, descriptors, raw_observations, (), ())
    assert g6 and not g6_engine.last_truncated
    g6_models = [item.model for item in g6]

    symbolic_policy = SymbolicPrimitiveGenesisEngine(
        model_budget=16384,
        expression_budget=512,
        max_depth=1,
        operators=("ADD", "SUB", "MUL", "ABS"),
    )
    symbolic_shadow = symbolic_policy.generate_novel(variables, descriptors, raw_observations, (), g6_models)
    assert symbolic_shadow and not symbolic_policy.last_truncated
    nonlinear = []
    for item in symbolic_shadow:
        expression = item.primitive.expression.render()
        effect_count = sum(1 for _iid, outcome in item.model.predictions if outcome != "NO_EFFECT")
        if " * " in expression and 5 <= effect_count <= trial_count - 5:
            nonlinear.append(item)
    assert nonlinear
    hidden = rng.choice(nonlinear)

    runtime = WorldDrivenPrimitiveRuntime(
        symbolic_primitive_genesis=SymbolicPrimitiveGenesisEngine(
            model_budget=16384,
            expression_budget=512,
            max_depth=1,
            operators=("ADD", "SUB", "MUL", "ABS"),
        )
    )
    runtime.register_causal_world_models(g6_models)

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

    assert runtime.raw_observation_memory == raw_observations
    before = runtime.generation_version_space(6)
    assert not before.compatible_model_ids
    assert runtime.generation_falsified(6)
    assert runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS"

    payload = primitive_checkpoint_dict(runtime)
    raw_receipt_payload_length = len(json.dumps(payload["raw_observation_receipts"], sort_keys=True))
    raw_cache_payload_length = len(json.dumps(payload["raw_observation_memory_cache"], sort_keys=True))

    verifierless = restore_world_driven_primitive_runtime(payload)
    world_only = restore_world_driven_primitive_runtime(payload, world_verifier=world_verifier)
    raw_only = restore_world_driven_primitive_runtime(payload, raw_observation_verifier=raw_verifier)
    reverified = restore_world_driven_primitive_runtime(
        payload,
        world_verifier=world_verifier,
        raw_observation_verifier=raw_verifier,
    )

    assert not verifierless.raw_observation_memory
    assert not world_only.raw_observation_memory
    assert not raw_only.raw_observation_memory
    assert reverified.raw_observation_memory == runtime.raw_observation_memory
    assert reverified.symbolic_primitive_genesis.max_depth == 1
    assert reverified.symbolic_primitive_genesis.expression_budget == 512
    assert reverified.symbolic_primitive_genesis.operators == ("ADD", "SUB", "MUL", "ABS")

    verifierless_frontier = verifierless.expand_causal_model_class_with_raw_observations(variables, descriptors)
    world_only_frontier = world_only.expand_causal_model_class_with_raw_observations(variables, descriptors)
    raw_only_frontier = raw_only.expand_causal_model_class_with_raw_observations(variables, descriptors)
    assert verifierless_frontier.status == "NO_EXPANSION_REQUIRED"
    assert world_only_frontier.status == "RAW_OBSERVATION_AUTHORITY_INCOMPLETE"
    assert raw_only_frontier.status == "NO_EXPANSION_REQUIRED"

    reverified_frontier = reverified.expand_causal_model_class_with_raw_observations(variables, descriptors)
    assert reverified_frontier.status == "EXPANDED"
    assert reverified_frontier.generation == 7
    assert len(reverified_frontier.active_model_ids) == 1
    resumed_space = reverified.generation_version_space(7)
    assert resumed_space.identified_model_id == hidden.model.model_id

    treatment = runtime.expand_causal_model_class_with_raw_observations(variables, descriptors)
    assert treatment.status == "EXPANDED"
    assert treatment.shadow_model_ids == reverified_frontier.shadow_model_ids
    assert treatment.active_model_ids == reverified_frontier.active_model_ids

    print(json.dumps({
        "status": "PASS_BOUNDED_AUTHENTICATED_PRIMITIVE_DEVELOPMENTAL_STATE_AND_CHECKPOINT_RESUME",
        "checkpoint_taken_before_g7_expansion": True,
        "signed_raw_receipts_persisted": len(payload["raw_observation_receipts"]),
        "raw_observation_rows_authoritative_before_checkpoint": len(runtime.raw_observation_memory),
        "raw_channels_persisted": sorted({channel for row in runtime.raw_observation_memory.values() for channel in row}),
        "raw_receipt_payload_length": raw_receipt_payload_length,
        "raw_cache_payload_length": raw_cache_payload_length,
        "raw_cache_is_non_authoritative_after_restore": True,
        "world_verifier_secret_persisted": False,
        "raw_verifier_secret_persisted": False,
        "genesis_policy_persisted": True,
        "symbolic_policy_max_depth": reverified.symbolic_primitive_genesis.max_depth,
        "symbolic_policy_expression_budget": reverified.symbolic_primitive_genesis.expression_budget,
        "symbolic_policy_operators": list(reverified.symbolic_primitive_genesis.operators),
        "g6_version_space_before_checkpoint": len(before.compatible_model_ids),
        "verifierless_raw_rows": len(verifierless.raw_observation_memory),
        "world_only_raw_rows": len(world_only.raw_observation_memory),
        "raw_only_raw_rows": len(raw_only.raw_observation_memory),
        "reverified_raw_rows": len(reverified.raw_observation_memory),
        "verifierless_descendant_frontier_status": verifierless_frontier.status,
        "world_only_descendant_frontier_status": world_only_frontier.status,
        "raw_only_descendant_frontier_status": raw_only_frontier.status,
        "reverified_descendant_frontier_status": reverified_frontier.status,
        "raw_argument_resupplied_after_restore": False,
        "resumed_g7_shadow_model_count": len(reverified_frontier.shadow_model_ids),
        "resumed_g7_active_model_count": len(reverified_frontier.active_model_ids),
        "resumed_g7_identified_model": resumed_space.identified_model_id,
        "hidden_model": hidden.model.model_id,
        "matched_treatment_shadow_equal": treatment.shadow_model_ids == reverified_frontier.shadow_model_ids,
        "matched_treatment_active_equal": treatment.active_model_ids == reverified_frontier.active_model_ids,
        "external_outcome_and_raw_authority_both_required": True,
        "independent_organizational_custody": False,
        "physical_world": False,
        "foundation_weight_change": False,
        "global_recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
