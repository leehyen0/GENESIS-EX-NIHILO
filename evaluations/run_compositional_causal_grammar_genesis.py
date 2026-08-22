from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.causal_model_genesis import CausalModelGenesisEngine, InterventionDescriptor
from arte_cognition.causal_program_genesis import CompositionalCausalProgramGenesisEngine
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


def as_proposal(d: InterventionDescriptor) -> InterventionProposal:
    target = d.targets[0] if d.targets else "__context__"
    return InterventionProposal(
        experiment_id=d.intervention_id,
        axis_id="COMPOSITIONAL_CAUSAL_GRAMMAR_AXIS",
        manipulated_variable=target,
        held_fixed=tuple((f"blocked::{name}", 1.0) for name in d.blocked),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="LOW",
        predicted_high_side="HIGH",
        reason=(
            f"causal-program discrimination targets={d.targets} blocked={d.blocked} "
            f"delay={d.delay_steps} context_shift={d.context_shift}"
        ),
    )


class HiddenProgramWorld:
    def __init__(self, hidden_model, signer, context_id: str, source_id: str, challenge_id: str):
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


def execute_two_classes(runtime, descriptor, hidden_model, signers, verifier, suffix, stage):
    p = as_proposal(descriptor)
    for index, (_issuer, signer) in enumerate(signers.items()):
        runtime.execute_world_intervention(
            p,
            HiddenProgramWorld(
                hidden_model,
                signer,
                context_id=f"grammar-{stage}",
                source_id=f"source-{stage}-{index}-{suffix}",
                challenge_id=f"challenge-{stage}-{index}-{suffix}",
            ),
            verifier=verifier,
        )


def restricted_posterior(runtime, ids):
    posterior = runtime.world_models.posterior()
    values = {mid: posterior.get(mid, 0.0) for mid in ids}
    total = sum(values.values()) or 1.0
    return {mid: value / total for mid, value in values.items()}


def main(seed_path: str) -> None:
    rng = random.Random(load_seed(seed_path))
    suffix = rng.randrange(100000, 999999)
    x = f"sensor_x_{suffix}"
    z = f"sensor_z_{suffix}"

    d = {}
    def add(name, targets, blocked=(), delay=0, context=False, cost=1.0):
        item = InterventionDescriptor(
            f"{name}::{suffix}", tuple(targets), tuple(blocked), int(delay), bool(context), float(cost)
        )
        d[name] = item
        return item

    descriptors = [
        add("do-x", (x,), cost=1),
        add("do-z", (z,), cost=1),
        add("do-both", (x, z), cost=4),
        add("do-x-block-z", (x,), blocked=(z,), cost=5),
        add("do-z-block-x", (z,), blocked=(x,), cost=5),
        add("delay-x", (x,), delay=1, cost=5),
        add("delay-z", (z,), delay=1, cost=5),
        add("delay-both", (x, z), delay=1, cost=9),
        add("delay-x-block-z", (x,), blocked=(z,), delay=1, cost=11),
        add("delay-z-block-x", (z,), blocked=(x,), delay=1, cost=11),
        add("context-shift", (), context=True, cost=20),
        add("context-delay-x", (x,), delay=1, context=True, cost=16),
        add("context-delay-z", (z,), delay=1, context=True, cost=16),
        add("context-delay-both", (x, z), delay=1, context=True, cost=24),
    ]
    discovery_names = ["do-x", "do-z", "do-both", "delay-x", "delay-z", "context-shift"]
    discovery = [d[name] for name in discovery_names]
    validation = [item for item in descriptors if item not in discovery]

    base_engine = CausalModelGenesisEngine(model_budget=64)
    program_engine = CompositionalCausalProgramGenesisEngine(model_budget=96, max_extra_primitives=2)
    base_surface = base_engine.generate([x, z], descriptors)
    base_signatures = {tuple(sorted(item.model.predictions)) for item in base_surface}
    composite_surface = program_engine.generate_novel(
        [x, z], descriptors, (), [item.model for item in base_surface]
    )
    hybrid_pool = [
        item for item in composite_surface
        if {p.op for p in item.program.primitives} >= {"VIA", "LAG"}
        and tuple(sorted(item.model.predictions)) not in base_signatures
        and any(item.model.prediction_for(row.intervention_id) == "NO_EFFECT" for row in discovery[:3])
        and len(set(value for _, value in item.model.predictions)) >= 2
    ]
    assert hybrid_pool
    hidden = rng.choice(hybrid_pool)
    hidden_model = hidden.model
    assert hidden_model.family == "COMPOSITIONAL_PROGRAM"
    assert hidden_model.generation == 2

    # BODY is not told the selected hidden program. It starts with two mutually
    # exclusive authored predictions on a probe where the hidden hybrid predicts no effect.
    surprise = rng.choice([
        item for item in discovery[:3]
        if hidden_model.prediction_for(item.intervention_id) == "NO_EFFECT"
    ])
    runtime = EpistemicallyDeepPersistentCognitiveRuntime()
    runtime.register_causal_world_models([
        CausalWorldModel("AUTHORED_POS", 1.0, ((surprise.intervention_id, "POSITIVE_EFFECT"),)),
        CausalWorldModel("AUTHORED_NEG", 1.0, ((surprise.intervention_id, "NEGATIVE_EFFECT"),)),
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

    # First structural failure and first-generation model genesis.
    execute_two_classes(runtime, surprise, hidden_model, signers, verifier, suffix, "authored-failure")
    assert runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS"
    assert runtime.generate_compositional_causal_models([x, z], descriptors) == []
    first_generation = runtime.generate_replacement_causal_models([x, z], descriptors)
    assert first_generation
    first_ids = {item.model.model_id for item in first_generation}
    assert hidden_model.model_id not in first_ids
    assert runtime.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS"

    # Let the BODY choose probes that discriminate the first generated model class.
    executed = {surprise.intervention_id}
    first_generation_probes = []
    for round_index in range(len(discovery)):
        if runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS":
            break
        available = [item for item in discovery if item.intervention_id not in executed]
        if not available:
            break
        queries = runtime.generated_model_queries(available, first_generation)
        ranked = runtime.rank_epistemic_interventions(queries)
        assert ranked
        selected_id = ranked[0].intervention_id
        selected = next(item for item in available if item.intervention_id == selected_id)
        execute_two_classes(runtime, selected, hidden_model, signers, verifier, suffix, f"first-gen-{round_index}")
        executed.add(selected_id)
        first_generation_probes.append({
            "intervention_id": selected_id,
            "cost": selected.cost,
            "expected_information_gain": ranked[0].expected_information_gain,
        })
    assert runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS"

    # Second structural generation: composed causal programs not represented by
    # any existing prediction signature.
    compositional = runtime.generate_compositional_causal_models([x, z], descriptors)
    assert compositional
    compositional_ids = {item.model.model_id for item in compositional}
    assert hidden_model.model_id in compositional_ids
    existing_before_composition = [
        model for model in runtime.world_models.models.values()
        if model.origin != "GENERATED_COMPOSITIONAL"
    ]
    existing_signatures = {tuple(sorted(model.predictions)) for model in existing_before_composition}
    assert all(tuple(sorted(item.model.predictions)) not in existing_signatures for item in compositional)
    comp_signatures = [tuple(sorted(item.model.predictions)) for item in compositional]
    assert len(comp_signatures) == len(set(comp_signatures))
    hidden_live = next(item.model for item in compositional if item.model.model_id == hidden_model.model_id)
    assert any(clause.startswith("VIA(") for clause in hidden_live.structure)
    assert any(clause.startswith("LAG(") for clause in hidden_live.structure)
    assert hidden_live.parent_model_ids

    # Held-out/source-disjoint intervention semantics discriminate the second
    # generation. Cost may rise; EIG, not cheapness, chooses probes.
    second_generation_probes = []
    used_validation = set()
    for round_index in range(len(validation)):
        posterior = restricted_posterior(runtime, compositional_ids)
        if posterior and max(posterior.values()) >= 0.97:
            break
        available = [item for item in validation if item.intervention_id not in used_validation]
        assert available
        queries = runtime.compositional_model_queries(available, compositional)
        ranked = runtime.rank_epistemic_interventions(queries)
        assert ranked
        selected_id = ranked[0].intervention_id
        selected = next(item for item in available if item.intervention_id == selected_id)
        execute_two_classes(runtime, selected, hidden_model, signers, verifier, suffix, f"second-gen-{round_index}")
        used_validation.add(selected_id)
        second_generation_probes.append({
            "intervention_id": selected_id,
            "cost": selected.cost,
            "expected_information_gain": ranked[0].expected_information_gain,
        })

    final_posterior = restricted_posterior(runtime, compositional_ids)
    top_id, top_prob = max(final_posterior.items(), key=lambda item: item[1])
    assert top_id == hidden_model.model_id
    assert top_prob >= 0.97

    payload = epistemic_checkpoint_dict(runtime)
    without_verifier = restore_epistemic_runtime(payload, world_verifier=None)
    with_verifier = restore_epistemic_runtime(payload, world_verifier=verifier)
    restored_hidden = with_verifier.world_models.models[hidden_model.model_id]
    assert restored_hidden.origin == "GENERATED_COMPOSITIONAL"
    assert restored_hidden.generation == 2
    assert restored_hidden.parent_model_ids
    assert restored_hidden.structure == hidden_live.structure

    no_verify_ids = {
        mid for mid, model in without_verifier.world_models.models.items()
        if model.origin == "GENERATED_COMPOSITIONAL"
    }
    verified_ids = {
        mid for mid, model in with_verifier.world_models.models.items()
        if model.origin == "GENERATED_COMPOSITIONAL"
    }
    no_verify_post = restricted_posterior(without_verifier, no_verify_ids)
    verified_post = restricted_posterior(with_verifier, verified_ids)
    assert max(no_verify_post.values()) < 0.90
    verified_top_id, verified_top_prob = max(verified_post.items(), key=lambda item: item[1])
    assert verified_top_id == hidden_model.model_id
    assert verified_top_prob >= 0.97

    out = {
        "status": "PASS_BOUNDED_TWO_GENERATION_COMPOSITIONAL_CAUSAL_GRAMMAR_GENESIS_AND_DESCENDANT",
        "hidden_program_exposed_to_body": False,
        "hidden_program": list(hidden.program.signature),
        "hidden_structure": list(hidden_model.structure),
        "base_named_family_count": len({item.model.family for item in first_generation}),
        "first_generation_model_count": len(first_generation),
        "first_generation_probes": first_generation_probes,
        "second_model_class_failure_observed": True,
        "compositional_model_count_after_novelty_and_quotient": len(compositional),
        "compositional_prediction_signature_duplicates": 0,
        "hidden_prediction_signature_absent_from_base_families": True,
        "hidden_compositional_model_generated": True,
        "second_generation_probes": second_generation_probes,
        "top_compositional_model": top_id,
        "top_compositional_posterior": top_prob,
        "generated_model_generation": restored_hidden.generation,
        "generated_model_parent_count": len(restored_hidden.parent_model_ids),
        "generated_structure_persisted_to_descendant": True,
        "verifierless_descendant_top_probability": max(no_verify_post.values()),
        "reverified_descendant_top_model": verified_top_id,
        "reverified_descendant_top_probability": verified_top_prob,
        "primitive_vocabulary_human_authored": True,
        "unrestricted_operator_genesis": False,
        "foundation_weight_change": False,
        "global_recursive_acceleration": False,
        "physical_world": False,
        "AGI": False,
        "ASI": False,
    }
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
