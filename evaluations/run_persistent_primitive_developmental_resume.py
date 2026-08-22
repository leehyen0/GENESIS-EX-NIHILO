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
        reason="checkpoint-before-symbolic-expansion developmental continuity probe",
    )


class HiddenDevelopmentWorld:
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
            context_id="hidden-persistent-development-world",
            challenge_id=self.challenge_id,
            epoch=1,
            budget_token=f"budget::{self.challenge_id}",
            externally_generated=True,
        ))


def execute_two(runtime, descriptor, hidden_model, signers, verifier, suffix, trial_index):
    for issuer_index, (_issuer, signer) in enumerate(signers.items()):
        runtime.execute_world_intervention(
            proposal(descriptor),
            HiddenDevelopmentWorld(
                hidden_model,
                signer,
                source_id=f"source-{issuer_index}-{trial_index}-{suffix}",
                challenge_id=f"challenge-{issuer_index}-{trial_index}-{suffix}",
            ),
            verifier=verifier,
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
            channel_a: float(values_a[index] - trial_count // 2),
            channel_b: float(values_b[index] - trial_count // 2),
        }
        for index, descriptor in enumerate(descriptors)
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
    symbolic_shadow = symbolic_policy.generate_novel(
        variables, descriptors, raw_observations, (), g6_models
    )
    assert symbolic_shadow and not symbolic_policy.last_truncated
    nonlinear = []
    for item in symbolic_shadow:
        expression = item.primitive.expression.render()
        effect_count = sum(
            1 for _intervention_id, outcome in item.model.predictions
            if outcome != "NO_EFFECT"
        )
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
    # Raw observation is ingested before structural failure is known. It becomes
    # developmental BODY state rather than an evaluator argument to future search.
    runtime.ingest_raw_observations(raw_observations)

    keys = {
        f"issuer-a-{suffix}": f"secret-a-{suffix}".encode(),
        f"issuer-b-{suffix}": f"secret-b-{suffix}".encode(),
    }
    signers = {issuer: HMACWorldReceiptSigner(issuer, secret) for issuer, secret in keys.items()}
    verifier = HMACWorldReceiptVerifier(keys, independence_classes={
        f"issuer-a-{suffix}": "independent-A",
        f"issuer-b-{suffix}": "independent-B",
    })

    for trial_index, descriptor in enumerate(descriptors):
        execute_two(runtime, descriptor, hidden.model, signers, verifier, suffix, trial_index)

    before = runtime.generation_version_space(6)
    assert not before.compatible_model_ids
    assert runtime.generation_falsified(6)
    assert runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS"

    # Freeze BEFORE symbolic expansion. A real descendant must retain enough body
    # state to continue development without the evaluator re-supplying raw data.
    payload = primitive_checkpoint_dict(runtime)
    raw_hash_material = json.dumps(payload["raw_observation_memory"], sort_keys=True)

    verifierless = restore_world_driven_primitive_runtime(payload, world_verifier=None)
    reverified = restore_world_driven_primitive_runtime(payload, world_verifier=verifier)

    assert verifierless.raw_observation_memory == runtime.raw_observation_memory
    assert reverified.raw_observation_memory == runtime.raw_observation_memory
    assert reverified.symbolic_primitive_genesis.max_depth == 1
    assert reverified.symbolic_primitive_genesis.expression_budget == 512
    assert reverified.symbolic_primitive_genesis.operators == ("ADD", "SUB", "MUL", "ABS")

    # No verifier means serialized world evidence is deauthorized, so raw memory
    # alone must not authorize structural growth.
    verifierless_frontier = verifierless.expand_causal_model_class_with_raw_observations(
        variables, descriptors
    )
    assert verifierless_frontier.status == "NO_EXPANSION_REQUIRED"

    # Reverified descendant receives no raw_observations argument here.
    reverified_frontier = reverified.expand_causal_model_class_with_raw_observations(
        variables, descriptors
    )
    assert reverified_frontier.status == "EXPANDED"
    assert reverified_frontier.generation == 7
    assert len(reverified_frontier.active_model_ids) == 1
    resumed_space = reverified.generation_version_space(7)
    assert resumed_space.identified_model_id == hidden.model.model_id

    # Matched non-checkpoint treatment from the same pre-expansion state should
    # generate exactly the same candidate and active IDs.
    treatment = runtime.expand_causal_model_class_with_raw_observations(
        variables, descriptors
    )
    assert treatment.status == "EXPANDED"
    assert treatment.shadow_model_ids == reverified_frontier.shadow_model_ids
    assert treatment.active_model_ids == reverified_frontier.active_model_ids

    print(json.dumps({
        "status": "PASS_BOUNDED_PERSISTENT_PRIMITIVE_DEVELOPMENTAL_STATE_AND_CHECKPOINT_RESUME",
        "checkpoint_taken_before_g7_expansion": True,
        "raw_observation_rows_persisted": len(runtime.raw_observation_memory),
        "raw_channels_persisted": sorted({
            channel for row in runtime.raw_observation_memory.values() for channel in row
        }),
        "raw_memory_payload_length": len(raw_hash_material),
        "genesis_policy_persisted": True,
        "symbolic_policy_max_depth": reverified.symbolic_primitive_genesis.max_depth,
        "symbolic_policy_expression_budget": reverified.symbolic_primitive_genesis.expression_budget,
        "symbolic_policy_operators": list(reverified.symbolic_primitive_genesis.operators),
        "g6_version_space_before_checkpoint": len(before.compatible_model_ids),
        "verifierless_descendant_frontier_status": verifierless_frontier.status,
        "reverified_descendant_frontier_status": reverified_frontier.status,
        "raw_argument_resupplied_after_restore": False,
        "resumed_g7_shadow_model_count": len(reverified_frontier.shadow_model_ids),
        "resumed_g7_active_model_count": len(reverified_frontier.active_model_ids),
        "resumed_g7_identified_model": resumed_space.identified_model_id,
        "hidden_model": hidden.model.model_id,
        "matched_treatment_shadow_equal": treatment.shadow_model_ids == reverified_frontier.shadow_model_ids,
        "matched_treatment_active_equal": treatment.active_model_ids == reverified_frontier.active_model_ids,
        "external_authority_still_required": True,
        "independent_organizational_custody": False,
        "physical_world": False,
        "foundation_weight_change": False,
        "global_recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
