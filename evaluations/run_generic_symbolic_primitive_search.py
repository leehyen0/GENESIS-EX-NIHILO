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
        axis_id="RAW_SYMBOLIC_PRIMITIVE_DISCOVERY_AXIS",
        manipulated_variable=descriptor.targets[0] if descriptor.targets else "__context__",
        held_fixed=(),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="LOW",
        predicted_high_side="HIGH",
        reason="same authored intervention semantics; discover symbolic relation from authenticated raw observations",
    )


class HiddenSymbolicWorld:
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
            context_id="hidden-symbolic-primitive-world",
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
            HiddenSymbolicWorld(
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
            intervention_id=f"SYMBOLICTRIAL::{suffix}::{index:02d}",
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
    g6_signatures = {tuple(sorted(model.predictions)) for model in g6_models}

    symbolic_engine = SymbolicPrimitiveGenesisEngine(
        model_budget=16384,
        expression_budget=512,
        max_depth=1,
        operators=("ADD", "SUB", "MUL", "ABS"),
    )
    g7_shadow = symbolic_engine.generate_novel(variables, descriptors, raw_observations, (), g6_models)
    assert g7_shadow and not symbolic_engine.last_truncated
    balanced_nonlinear = []
    for item in g7_shadow:
        expression = item.primitive.expression.render()
        effect_count = sum(1 for _iid, outcome in item.model.predictions if outcome != "NO_EFFECT")
        if " * " in expression and 5 <= effect_count <= trial_count - 5:
            balanced_nonlinear.append(item)
    assert balanced_nonlinear
    hidden = rng.choice(balanced_nonlinear)
    assert tuple(sorted(hidden.model.predictions)) not in g6_signatures

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

    assert len(runtime.raw_observation_memory) == len(descriptors)
    g6_final = runtime.generation_version_space(6)
    assert not g6_final.compatible_model_ids
    assert runtime.generation_falsified(6)
    assert runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS"

    frontier = runtime.expand_causal_model_class_with_raw_observations(variables, descriptors)
    assert frontier.status == "EXPANDED"
    assert frontier.generation == 7
    assert frontier.origin == "GENERATED_SYMBOLIC_PRIMITIVE"
    assert len(frontier.active_model_ids) == 1

    g7_final = runtime.generation_version_space(7)
    assert g7_final.identified
    assert g7_final.identified_model_id == hidden.model.model_id

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
    no_verify_g7 = no_verify.generation_version_space(7)
    reverified_g7 = reverified.generation_version_space(7)
    assert len(no_verify_g7.compatible_model_ids) == len(frontier.shadow_model_ids)
    assert reverified_g7.identified_model_id == hidden.model.model_id

    old_semantic_signatures = {(d.targets, d.blocked, d.delay_steps, d.context_shift) for d in descriptors}
    assert len(old_semantic_signatures) == 1

    print(json.dumps({
        "status": "PASS_BOUNDED_AUTHENTICATED_GENERIC_SYMBOLIC_PRIMITIVE_SEARCH",
        "same_old_semantics_across_all_trials": True,
        "old_atom_semantic_signature_count": len(old_semantic_signatures),
        "raw_channel_count": 2,
        "raw_channel_names_random_post_checkout": True,
        "raw_outcome_keys_separated": True,
        "raw_quorum_requires_two_independence_classes": True,
        "g6_complete_candidate_universe": True,
        "g6_shadow_model_count": len(g6_models),
        "g6_final_version_space": len(g6_final.compatible_model_ids),
        "symbolic_expression_count": symbolic_engine.last_expression_count,
        "g7_shadow_model_count": len(frontier.shadow_model_ids),
        "g7_active_model_count": len(frontier.active_model_ids),
        "hidden_symbolic_expression": hidden.primitive.expression.render(),
        "hidden_symbolic_threshold": hidden.primitive.threshold,
        "hidden_symbolic_direction": hidden.primitive.direction,
        "hidden_symbolic_model": hidden.model.model_id,
        "hidden_symbolic_expression_exposed_to_body": False,
        "g7_prediction_signature_absent_from_g6": True,
        "g7_exact_identified_model": g7_final.identified_model_id,
        "symbolic_candidate_generation_uses_outcomes": False,
        "symbolic_activation_requires_g6_falsification": True,
        "external_evaluator_selected_symbolic_operator": False,
        "verifierless_descendant_raw_rows": len(no_verify.raw_observation_memory),
        "world_only_descendant_raw_rows": len(world_only.raw_observation_memory),
        "reverified_descendant_raw_rows": len(reverified.raw_observation_memory),
        "reverified_descendant_g7_identified_model": reverified_g7.identified_model_id,
        "authored_operation_alphabet": list(symbolic_engine.operators),
        "unrestricted_operator_genesis": False,
        "physical_world": False,
        "independent_organizational_custody": False,
        "global_recursive_acceleration": False,
        "foundation_weight_change": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
