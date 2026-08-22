from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.causal_model_genesis import CausalModelGenesisEngine, InterventionDescriptor
from arte_cognition.epistemic_depth_runtime import (
    EpistemicallyDeepPersistentCognitiveRuntime,
    epistemic_checkpoint_dict,
    restore_epistemic_runtime,
)
from arte_cognition.experiment_genesis import InterventionProposal
from arte_cognition.world_coupling import (
    HMACWorldReceiptSigner,
    HMACWorldReceiptVerifier,
    WorldOutcomeReceipt,
)
from arte_cognition.world_model_ecology import CausalWorldModel


def load_seed(path: str) -> int:
    return int(Path(path).read_text().strip())


def proposal(descriptor: InterventionDescriptor) -> InterventionProposal:
    target = descriptor.targets[0] if descriptor.targets else "__context__"
    held_fixed = tuple((f"blocked::{name}", 1.0) for name in descriptor.blocked)
    return InterventionProposal(
        experiment_id=descriptor.intervention_id,
        axis_id="AUTONOMOUS_CAUSAL_MODEL_GENESIS_AXIS",
        manipulated_variable=target,
        held_fixed=held_fixed,
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="LOW",
        predicted_high_side="HIGH",
        reason=(
            f"generated causal-model discrimination targets={descriptor.targets} "
            f"blocked={descriptor.blocked} delay={descriptor.delay_steps} "
            f"context_shift={descriptor.context_shift}"
        ),
    )


class HiddenGeneratedModelWorld:
    def __init__(self, hidden_model: CausalWorldModel, signer, context_id: str, source_id: str, challenge_id: str):
        self.hidden_model = hidden_model
        self.signer = signer
        self.context_id = context_id
        self.source_id = source_id
        self.challenge_id = challenge_id

    def execute(self, p: InterventionProposal, arm: str, value: float):
        label = self.hidden_model.prediction_for(p.experiment_id) or "NO_EFFECT"
        effect = {
            "POSITIVE_EFFECT": 1.0,
            "NEGATIVE_EFFECT": -1.0,
            "NO_EFFECT": 0.0,
        }[label]
        outcome = 0.0 if arm.upper() == "LOW" else effect
        receipt = WorldOutcomeReceipt(
            receipt_id=f"{self.challenge_id}::{p.experiment_id}::{arm}",
            experiment_id=p.experiment_id,
            axis_id=p.axis_id,
            arm=arm,
            intervention_value=float(value),
            outcome=float(outcome),
            source_id=self.source_id,
            context_id=self.context_id,
            challenge_id=self.challenge_id,
            epoch=1,
            budget_token=f"budget::{self.challenge_id}",
            externally_generated=True,
        )
        return self.signer.sign(receipt)


def execute_two_classes(runtime, p, hidden_model, signers, verifier, suffix, stage):
    for index, (issuer, signer) in enumerate(signers.items()):
        runtime.execute_world_intervention(
            p,
            HiddenGeneratedModelWorld(
                hidden_model,
                signer,
                context_id=f"causal-genesis-{stage}",
                source_id=f"source-{stage}-{index}-{suffix}",
                challenge_id=f"challenge-{stage}-{index}-{suffix}",
            ),
            verifier=verifier,
        )


def generated_posterior(runtime):
    posterior = runtime.world_models.posterior()
    generated_ids = {
        model_id for model_id, model in runtime.world_models.models.items()
        if model.origin == "GENERATED"
    }
    values = {mid: posterior.get(mid, 0.0) for mid in generated_ids}
    total = sum(values.values()) or 1.0
    return {mid: value / total for mid, value in values.items()}


def main(seed_path: str) -> None:
    rng = random.Random(load_seed(seed_path))
    suffix = rng.randrange(100000, 999999)
    x = f"sensor_x_{suffix}"
    z = f"sensor_z_{suffix}"

    descriptors = [
        InterventionDescriptor(f"do-x::{suffix}", (x,), cost=1.0),
        InterventionDescriptor(f"do-z::{suffix}", (z,), cost=1.0),
        InterventionDescriptor(f"do-both::{suffix}", (x, z), cost=8.0),
        InterventionDescriptor(f"do-x-block-z::{suffix}", (x,), blocked=(z,), cost=12.0),
        InterventionDescriptor(f"do-z-block-x::{suffix}", (z,), blocked=(x,), cost=12.0),
        InterventionDescriptor(f"delay-x::{suffix}", (x,), delay_steps=1, cost=6.0),
        InterventionDescriptor(f"delay-z::{suffix}", (z,), delay_steps=1, cost=6.0),
        InterventionDescriptor(f"context-shift::{suffix}", (), context_shift=True, cost=20.0),
    ]

    # Evaluator chooses a hidden mechanism only after checkout. BODY sees the
    # descriptor surface but never receives this selected model id/family.
    seed_engine = CausalModelGenesisEngine(model_budget=64)
    seed_candidates = seed_engine.generate([x, z], descriptors)
    hidden_pool = [
        item.model for item in seed_candidates
        if item.model.family in {"MEDIATED", "INTERACTION", "TEMPORAL", "LATENT_COMMON_CAUSE"}
        and len(set(value for _, value in item.model.predictions)) >= 2
    ]
    assert hidden_pool
    hidden_model = rng.choice(hidden_pool)

    # Pick a hidden NO_EFFECT probe so two authored models can both be wrong in
    # opposite directions. This creates genuine model-class failure without
    # revealing which generated structure explains it.
    no_effect_descriptors = [
        d for d in descriptors
        if hidden_model.prediction_for(d.intervention_id) == "NO_EFFECT"
    ]
    assert no_effect_descriptors
    surprise_descriptor = rng.choice(no_effect_descriptors)
    surprise_id = surprise_descriptor.intervention_id

    runtime = EpistemicallyDeepPersistentCognitiveRuntime()
    runtime.register_causal_world_models([
        CausalWorldModel("AUTHORED_POS", 1.0, ((surprise_id, "POSITIVE_EFFECT"),)),
        CausalWorldModel("AUTHORED_NEG", 1.0, ((surprise_id, "NEGATIVE_EFFECT"),)),
    ])

    keys = {
        f"issuer-a-{suffix}": f"secret-a-{suffix}".encode(),
        f"issuer-b-{suffix}": f"secret-b-{suffix}".encode(),
    }
    signers = {issuer: HMACWorldReceiptSigner(issuer, secret) for issuer, secret in keys.items()}
    verifier = HMACWorldReceiptVerifier(
        keys,
        independence_classes={
            f"issuer-a-{suffix}": "independent-A",
            f"issuer-b-{suffix}": "independent-B",
        },
    )

    # One class is insufficient to open structural genesis.
    first_issuer, first_signer = next(iter(signers.items()))
    runtime.execute_world_intervention(
        proposal(surprise_descriptor),
        HiddenGeneratedModelWorld(
            hidden_model, first_signer, "model-class-failure", "single-source", f"single-{suffix}"
        ),
        verifier=verifier,
    )
    assert runtime.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS"

    # The second independent class closes the structural-failure gate.
    second_issuer, second_signer = list(signers.items())[1]
    runtime.execute_world_intervention(
        proposal(surprise_descriptor),
        HiddenGeneratedModelWorld(
            hidden_model, second_signer, "model-class-failure", "second-source", f"second-{suffix}"
        ),
        verifier=verifier,
    )
    assert runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS"

    generated = runtime.generate_replacement_causal_models([x, z], descriptors)
    assert generated
    generated_ids = {item.model.model_id for item in generated}
    assert hidden_model.model_id in generated_ids
    assert all(item.model.origin == "GENERATED" for item in generated)
    signatures = [tuple(sorted(item.model.predictions)) for item in generated]
    assert len(signatures) == len(set(signatures))
    assert runtime.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS"

    executed = {surprise_id}
    chosen_interventions = []
    # Actively discriminate the newly generated causal structures. Each selected
    # experiment is executed by both verifier-bound independence classes.
    for round_index in range(6):
        posterior = generated_posterior(runtime)
        if posterior and max(posterior.values()) >= 0.97:
            break
        queries = [
            q for q in runtime.generated_model_queries(descriptors, generated)
            if q.query_id not in executed
        ]
        assert queries
        ranked = runtime.rank_epistemic_interventions(queries)
        assert ranked
        selected_id = ranked[0].intervention_id
        selected_descriptor = next(d for d in descriptors if d.intervention_id == selected_id)
        execute_two_classes(
            runtime,
            proposal(selected_descriptor),
            hidden_model,
            signers,
            verifier,
            suffix,
            f"discriminate-{round_index}",
        )
        executed.add(selected_id)
        chosen_interventions.append({
            "intervention_id": selected_id,
            "cost": selected_descriptor.cost,
            "expected_information_gain": ranked[0].expected_information_gain,
        })

    final_generated_posterior = generated_posterior(runtime)
    top_generated_id, top_generated_prob = max(final_generated_posterior.items(), key=lambda item: item[1])
    assert top_generated_id == hidden_model.model_id
    assert top_generated_prob >= 0.97

    payload = epistemic_checkpoint_dict(runtime)
    without_verifier = restore_epistemic_runtime(payload, world_verifier=None)
    with_verifier = restore_epistemic_runtime(payload, world_verifier=verifier)

    inherited_generated = {
        model_id: model for model_id, model in with_verifier.world_models.models.items()
        if model.origin == "GENERATED"
    }
    assert hidden_model.model_id in inherited_generated
    assert inherited_generated[hidden_model.model_id].structure
    assert inherited_generated[hidden_model.model_id].family == hidden_model.family

    no_verify_generated_posterior = generated_posterior(without_verifier)
    reverified_generated_posterior = generated_posterior(with_verifier)
    assert max(no_verify_generated_posterior.values()) < 0.90
    reverified_top_id, reverified_top_prob = max(reverified_generated_posterior.items(), key=lambda item: item[1])
    assert reverified_top_id == hidden_model.model_id
    assert reverified_top_prob >= 0.97

    out = {
        "status": "PASS_BOUNDED_AUTONOMOUS_CAUSAL_MODEL_STRUCTURE_GENESIS_DISCRIMINATION_AND_DESCENDANT",
        "hidden_model_exposed_to_body": False,
        "hidden_family": hidden_model.family,
        "hidden_structure": list(hidden_model.structure),
        "model_class_failure_requires_two_independence_classes": True,
        "generated_model_count_after_quotient": len(generated),
        "prediction_signature_duplicates": 0,
        "generated_families": sorted({item.model.family for item in generated}),
        "selected_discriminating_interventions": chosen_interventions,
        "top_generated_model": top_generated_id,
        "top_generated_posterior": top_generated_prob,
        "generated_model_matches_hidden_structure": top_generated_id == hidden_model.model_id,
        "generated_structure_persisted_to_descendant": True,
        "verifierless_descendant_top_probability": max(no_verify_generated_posterior.values()),
        "reverified_descendant_top_model": reverified_top_id,
        "reverified_descendant_top_probability": reverified_top_prob,
        "authority_reverification_required": True,
        "foundation_weight_change": False,
        "global_recursive_acceleration": False,
        "physical_world": False,
        "AGI": False,
        "ASI": False,
    }
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
