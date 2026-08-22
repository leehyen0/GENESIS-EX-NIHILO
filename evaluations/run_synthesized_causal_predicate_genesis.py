from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.causal_model_genesis import CausalModelGenesisEngine, InterventionDescriptor
from arte_cognition.causal_predicate_genesis import BooleanCausalPredicateGenesisEngine
from arte_cognition.causal_program_genesis import CompositionalCausalProgramGenesisEngine
from arte_cognition.epistemic_depth_runtime import (
    EpistemicallyDeepPersistentCognitiveRuntime,
    epistemic_checkpoint_dict,
    restore_epistemic_runtime,
)
from arte_cognition.experiment_genesis import InterventionProposal
from arte_cognition.world_coupling import HMACWorldReceiptSigner, HMACWorldReceiptVerifier, WorldOutcomeReceipt
from arte_cognition.world_model_ecology import CausalWorldModel


def load_seed(path: str) -> int:
    return int(Path(path).read_text().strip())


def as_proposal(d: InterventionDescriptor) -> InterventionProposal:
    return InterventionProposal(
        experiment_id=d.intervention_id,
        axis_id="SYNTHESIZED_CAUSAL_PREDICATE_AXIS",
        manipulated_variable=d.targets[0] if d.targets else "__context__",
        held_fixed=tuple((f"blocked::{name}", 1.0) for name in d.blocked),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="LOW",
        predicted_high_side="HIGH",
        reason=(f"predicate-synthesis targets={d.targets} blocked={d.blocked} "
                f"delay={d.delay_steps} context={d.context_shift}"),
    )


class HiddenPredicateWorld:
    def __init__(self, hidden_model, signer, context_id, source_id, challenge_id):
        self.hidden_model = hidden_model
        self.signer = signer
        self.context_id = context_id
        self.source_id = source_id
        self.challenge_id = challenge_id

    def execute(self, proposal, arm: str, value: float):
        label = self.hidden_model.prediction_for(proposal.experiment_id) or "NO_EFFECT"
        effect = {"POSITIVE_EFFECT": 1.0, "NEGATIVE_EFFECT": -1.0, "NO_EFFECT": 0.0}[label]
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


def execute_two(runtime, descriptor, hidden_model, signers, verifier, suffix, stage):
    for index, (_issuer, signer) in enumerate(signers.items()):
        runtime.execute_world_intervention(
            as_proposal(descriptor),
            HiddenPredicateWorld(
                hidden_model, signer, f"predicate-{stage}",
                f"source-{stage}-{index}-{suffix}", f"challenge-{stage}-{index}-{suffix}",
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
    x, z = f"sensor_x_{suffix}", f"sensor_z_{suffix}"

    d = {}
    def add(name, targets, blocked=(), delay=0, context=False, cost=1.0):
        item = InterventionDescriptor(
            f"{name}::{suffix}", tuple(targets), tuple(blocked), int(delay), bool(context), float(cost)
        )
        d[name] = item
        return item

    descriptors = [
        add("do-x", (x,), cost=1), add("do-z", (z,), cost=1),
        add("do-both", (x, z), cost=3),
        add("delay-x", (x,), delay=1, cost=4), add("delay-z", (z,), delay=1, cost=4),
        add("delay-both", (x, z), delay=1, cost=7),
        add("context-x", (x,), context=True, cost=6),
        add("context-z", (z,), context=True, cost=6),
        add("context-both", (x, z), context=True, cost=10),
        add("context-delay-x", (x,), delay=1, context=True, cost=11),
        add("context-delay-z", (z,), delay=1, context=True, cost=11),
        add("context-delay-both", (x, z), delay=1, context=True, cost=18),
        add("delay-both-block-z", (x, z), blocked=(z,), delay=1, cost=14),
        add("context-both-block-z", (x, z), blocked=(z,), context=True, cost=15),
        add("context-delay-both-block-z", (x, z), blocked=(z,), delay=1, context=True, cost=22),
    ]

    base_engine = CausalModelGenesisEngine(model_budget=128)
    program_engine = CompositionalCausalProgramGenesisEngine(model_budget=256, max_extra_primitives=2)
    predicate_engine = BooleanCausalPredicateGenesisEngine(
        model_budget=2048, max_literals_per_term=3, max_terms=2
    )
    base_surface = base_engine.generate([x, z], descriptors)
    comp_surface = program_engine.generate_novel(
        [x, z], descriptors, (), [item.model for item in base_surface]
    )
    prior_surface = [item.model for item in base_surface] + [item.model for item in comp_surface]
    pred_surface = predicate_engine.generate_novel([x, z], descriptors, (), prior_surface)
    assert not predicate_engine.last_truncated
    full_predicate_signature_count = predicate_engine.last_unique_signature_count

    wanted = {
        d["do-both"].intervention_id: "NO_EFFECT",
        d["delay-both"].intervention_id: "POSITIVE_EFFECT",
        d["context-both"].intervention_id: "POSITIVE_EFFECT",
        d["context-delay-both"].intervention_id: "NO_EFFECT",
        d["delay-x"].intervention_id: "NO_EFFECT",
        d["context-x"].intervention_id: "NO_EFFECT",
    }
    hidden_pool = [
        item for item in pred_surface
        if item.cause == x and item.sign == "POS"
        and all(item.model.prediction_for(key) == value for key, value in wanted.items())
        and "!CONTEXT" in item.predicate.render()
        and "!DELAY" in item.predicate.render()
    ]
    assert hidden_pool
    hidden = rng.choice(hidden_pool)
    hidden_model = hidden.model
    prior_signatures = {tuple(sorted(model.predictions)) for model in prior_surface}
    assert tuple(sorted(hidden_model.predictions)) not in prior_signatures

    runtime = EpistemicallyDeepPersistentCognitiveRuntime()
    surprise = d["do-both"]
    runtime.register_causal_world_models([
        CausalWorldModel("AUTH_POS", 1.0, ((surprise.intervention_id, "POSITIVE_EFFECT"),)),
        CausalWorldModel("AUTH_NEG", 1.0, ((surprise.intervention_id, "NEGATIVE_EFFECT"),)),
    ])

    keys = {
        f"issuer-a-{suffix}": f"secret-a-{suffix}".encode(),
        f"issuer-b-{suffix}": f"secret-b-{suffix}".encode(),
    }
    signers = {issuer: HMACWorldReceiptSigner(issuer, secret) for issuer, secret in keys.items()}
    verifier = HMACWorldReceiptVerifier(keys, independence_classes={
        f"issuer-a-{suffix}": "independent-A", f"issuer-b-{suffix}": "independent-B",
    })

    execute_two(runtime, surprise, hidden_model, signers, verifier, suffix, "authored-failure")
    assert runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS"
    first = runtime.generate_replacement_causal_models([x, z], descriptors)
    assert first and runtime.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS"

    first_candidates = [d["delay-both"], d["delay-x"], d["delay-z"]]
    first_probes = []
    used_first = set()
    for round_index in range(len(first_candidates)):
        if runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS":
            break
        available = [item for item in first_candidates if item.intervention_id not in used_first]
        if not available:
            break
        ranked = runtime.rank_epistemic_interventions(runtime.generated_model_queries(available, first))
        assert ranked
        selected = next(item for item in available if item.intervention_id == ranked[0].intervention_id)
        execute_two(runtime, selected, hidden_model, signers, verifier, suffix, f"g1-{round_index}")
        used_first.add(selected.intervention_id)
        first_probes.append({"id": selected.intervention_id, "cost": selected.cost, "eig": ranked[0].expected_information_gain})
    assert runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS"

    second = runtime.generate_compositional_causal_models([x, z], descriptors)
    assert second and runtime.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS"
    second_all = {
        mid for mid, model in runtime.world_models.models.items()
        if model.origin == "GENERATED_COMPOSITIONAL"
    }
    assert second_all

    second_failure_descriptors = [d["context-both"], d["context-delay-both"]]
    second_probes = []
    used_second = set()
    for round_index in range(len(second_failure_descriptors)):
        if runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS":
            break
        available = [item for item in second_failure_descriptors if item.intervention_id not in used_second]
        assert available
        ranked = runtime.rank_epistemic_interventions(runtime.compositional_model_queries(available))
        assert ranked
        selected = next(item for item in available if item.intervention_id == ranked[0].intervention_id)
        execute_two(runtime, selected, hidden_model, signers, verifier, suffix, f"g2-{round_index}")
        used_second.add(selected.intervention_id)
        second_probes.append({"id": selected.intervention_id, "cost": selected.cost, "eig": ranked[0].expected_information_gain})
    assert runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS"

    third_active = runtime.generate_predicate_causal_models([x, z], descriptors)
    assert third_active
    assert not runtime.predicate_genesis.last_truncated
    third_all_models = [
        model for model in runtime.world_models.models.values()
        if model.origin == "GENERATED_PREDICATE"
    ]
    third_ids = {model.model_id for model in third_all_models}
    assert hidden_model.model_id in third_ids
    assert len(third_ids) >= len(third_active)
    old_signatures = {
        tuple(sorted(model.predictions)) for model in runtime.world_models.models.values()
        if model.origin != "GENERATED_PREDICATE"
    }
    assert all(tuple(sorted(model.predictions)) not in old_signatures for model in third_all_models)
    assert len({tuple(sorted(model.predictions)) for model in third_all_models}) == len(third_all_models)

    validation = [
        d["context-delay-both"], d["delay-both-block-z"],
        d["context-both-block-z"], d["context-delay-both-block-z"],
        d["context-delay-x"], d["context-delay-z"],
    ]
    third_probes = []
    used = set()
    for round_index in range(len(validation)):
        post = restricted_posterior(runtime, third_ids)
        if third_probes and max(post.values()) >= 0.97:
            break
        available = [item for item in validation if item.intervention_id not in used]
        assert available
        ranked = runtime.rank_epistemic_interventions(runtime.predicate_model_queries(available))
        assert ranked
        selected = next(item for item in available if item.intervention_id == ranked[0].intervention_id)
        execute_two(runtime, selected, hidden_model, signers, verifier, suffix, f"g3-{round_index}")
        used.add(selected.intervention_id)
        third_probes.append({"id": selected.intervention_id, "cost": selected.cost, "eig": ranked[0].expected_information_gain})

    final_post = restricted_posterior(runtime, third_ids)
    top_id, top_prob = max(final_post.items(), key=lambda item: item[1])
    assert third_probes
    assert top_id == hidden_model.model_id and top_prob >= 0.97

    payload = epistemic_checkpoint_dict(runtime)
    without_verifier = restore_epistemic_runtime(payload, world_verifier=None)
    with_verifier = restore_epistemic_runtime(payload, world_verifier=verifier)
    no_verify_ids = {
        mid for mid, model in without_verifier.world_models.models.items()
        if model.origin == "GENERATED_PREDICATE"
    }
    verified_ids = {
        mid for mid, model in with_verifier.world_models.models.items()
        if model.origin == "GENERATED_PREDICATE"
    }
    assert no_verify_ids == verified_ids == third_ids
    no_verify_post = restricted_posterior(without_verifier, no_verify_ids)
    verified_post = restricted_posterior(with_verifier, verified_ids)
    assert max(no_verify_post.values()) < 0.90
    verified_top_id, verified_top_prob = max(verified_post.items(), key=lambda item: item[1])
    assert verified_top_id == hidden_model.model_id and verified_top_prob >= 0.97
    restored = with_verifier.world_models.models[hidden_model.model_id]
    assert restored.generation == 3 and restored.parent_model_ids

    print(json.dumps({
        "status": "PASS_BOUNDED_THREE_GENERATION_SYNTHESIZED_CAUSAL_PREDICATE_GENESIS_AND_DESCENDANT",
        "hidden_predicate_exposed_to_body": False,
        "hidden_predicate": hidden.predicate.render(),
        "hidden_structure": list(hidden_model.structure),
        "hidden_signature_absent_from_generation_1_and_2": True,
        "full_predicate_equivalence_signature_count": full_predicate_signature_count,
        "predicate_search_truncated": False,
        "generation_1_probes": first_probes,
        "generation_2_probes": second_probes,
        "generation_3_active_count": len(third_active),
        "generation_3_shadow_pool_count": len(third_ids),
        "generation_3_heldout_probes": third_probes,
        "top_generation_3_model": top_id,
        "top_generation_3_posterior": top_prob,
        "verifierless_descendant_top_probability": max(no_verify_post.values()),
        "reverified_descendant_top_model": verified_top_id,
        "reverified_descendant_top_probability": verified_top_prob,
        "generation_3_parent_count": len(restored.parent_model_ids),
        "candidate_presence_evidence_independent": True,
        "boolean_metalanguage_human_authored": True,
        "unrestricted_logic_operator_genesis": False,
        "foundation_weight_change": False,
        "global_recursive_acceleration": False,
        "physical_world": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
