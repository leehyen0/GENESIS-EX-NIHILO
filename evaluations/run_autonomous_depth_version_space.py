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


def as_proposal(d: InterventionDescriptor) -> InterventionProposal:
    return InterventionProposal(
        experiment_id=d.intervention_id,
        axis_id="AUTONOMOUS_STRUCTURAL_DEPTH_AXIS",
        manipulated_variable=d.targets[0] if d.targets else "__context__",
        held_fixed=tuple((f"blocked::{name}", 1.0) for name in d.blocked),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="LOW",
        predicted_high_side="HIGH",
        reason=(f"autonomous-depth targets={d.targets} blocked={d.blocked} "
                f"delay={d.delay_steps} context={d.context_shift}"),
    )


class HiddenWorld:
    def __init__(self, model, signer, source_id, context_id, challenge_id):
        self.model = model
        self.signer = signer
        self.source_id = source_id
        self.context_id = context_id
        self.challenge_id = challenge_id

    def execute(self, proposal, arm: str, value: float):
        label = self.model.prediction_for(proposal.experiment_id) or "NO_EFFECT"
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
    proposal = as_proposal(descriptor)
    for index, (_issuer, signer) in enumerate(signers.items()):
        runtime.execute_world_intervention(
            proposal,
            HiddenWorld(
                hidden_model,
                signer,
                source_id=f"source-{stage}-{index}-{suffix}",
                context_id=f"ctx-{stage}",
                challenge_id=f"challenge-{stage}-{index}-{suffix}",
            ),
            verifier=verifier,
        )


def main(seed_path: str) -> None:
    rng = random.Random(int(Path(seed_path).read_text().strip()))
    suffix = rng.randrange(100000, 999999)
    x, z = f"sensor_x_{suffix}", f"sensor_z_{suffix}"

    d = {}
    def add(name, targets, blocked=(), delay=0, context=False, cost=1.0):
        row = InterventionDescriptor(
            f"{name}::{suffix}", tuple(targets), tuple(blocked), int(delay), bool(context), float(cost)
        )
        d[name] = row
        return row

    descriptors = [
        add("do-x", (x,), cost=1), add("do-z", (z,), cost=1), add("do-both", (x, z), cost=3),
        add("delay-x", (x,), delay=1, cost=4), add("delay-z", (z,), delay=1, cost=4),
        add("delay-both", (x, z), delay=1, cost=7),
        add("context-x", (x,), context=True, cost=6), add("context-z", (z,), context=True, cost=6),
        add("context-both", (x, z), context=True, cost=10),
        add("context-delay-x", (x,), delay=1, context=True, cost=11),
        add("context-delay-z", (z,), delay=1, context=True, cost=11),
        add("context-delay-both", (x, z), delay=1, context=True, cost=18),
        add("delay-both-block-z", (x, z), blocked=(z,), delay=1, cost=14),
        add("context-both-block-z", (x, z), blocked=(z,), context=True, cost=15),
        add("context-delay-both-block-z", (x, z), blocked=(z,), delay=1, context=True, cost=22),
    ]

    # Evaluator chooses hidden truth only after checkout. BODY receives descriptors,
    # never the selected predicate or its model id.
    base = CausalModelGenesisEngine(model_budget=128).generate([x, z], descriptors)
    comp = CompositionalCausalProgramGenesisEngine(model_budget=256, max_extra_primitives=2).generate_novel(
        [x, z], descriptors, (), [item.model for item in base]
    )
    prior = [item.model for item in base] + [item.model for item in comp]
    pred_engine = BooleanCausalPredicateGenesisEngine(model_budget=2048, max_literals_per_term=3, max_terms=2)
    pred = pred_engine.generate_novel([x, z], descriptors, (), prior)
    assert not pred_engine.last_truncated

    wanted = {
        d["do-both"].intervention_id: "NO_EFFECT",
        d["delay-both"].intervention_id: "POSITIVE_EFFECT",
        d["context-both"].intervention_id: "POSITIVE_EFFECT",
        d["context-delay-both"].intervention_id: "NO_EFFECT",
        d["delay-x"].intervention_id: "NO_EFFECT",
        d["context-x"].intervention_id: "NO_EFFECT",
    }
    hidden_pool = [
        item for item in pred
        if item.cause == x and item.sign == "POS"
        and all(item.model.prediction_for(key) == value for key, value in wanted.items())
        and "!CONTEXT" in item.predicate.render() and "!DELAY" in item.predicate.render()
    ]
    assert hidden_pool
    hidden = rng.choice(hidden_pool)
    hidden_model = hidden.model
    prior_signatures = {tuple(sorted(model.predictions)) for model in prior}
    assert tuple(sorted(hidden_model.predictions)) not in prior_signatures

    keys = {
        f"issuer-a-{suffix}": f"secret-a-{suffix}".encode(),
        f"issuer-b-{suffix}": f"secret-b-{suffix}".encode(),
    }
    signers = {issuer: HMACWorldReceiptSigner(issuer, secret) for issuer, secret in keys.items()}
    verifier = HMACWorldReceiptVerifier(keys, independence_classes={
        f"issuer-a-{suffix}": "independent-A",
        f"issuer-b-{suffix}": "independent-B",
    })

    runtime = EpistemicallyDeepPersistentCognitiveRuntime()
    surprise = d["do-both"]
    runtime.register_causal_world_models([
        CausalWorldModel("AUTH_POS", 1.0, ((surprise.intervention_id, "POSITIVE_EFFECT"),)),
        CausalWorldModel("AUTH_NEG", 1.0, ((surprise.intervention_id, "NEGATIVE_EFFECT"),)),
    ])
    executed = {surprise.intervention_id}
    execute_two(runtime, surprise, hidden_model, signers, verifier, suffix, "g0-failure")
    assert runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS"

    decisions = []
    g1 = runtime.expand_causal_model_class([x, z], descriptors)
    decisions.append(g1)
    assert g1.status == "EXPANDED" and g1.generation == 1

    # Independent world outcomes defeat G1. Query ranking is generation-scoped;
    # the external evaluator only supplies outcomes for selected descriptors.
    for i, descriptor in enumerate([d["delay-both"], d["delay-x"], d["delay-z"]]):
        if runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS":
            break
        queries = runtime.model_genesis.query_candidates(
            [row for row in [d["delay-both"], d["delay-x"], d["delay-z"]] if row.intervention_id not in executed],
            runtime.structural_models(1),
        )
        selected = runtime.select_generation_intervention(1, queries)
        chosen = next((row for row in descriptors if selected and row.intervention_id == selected.intervention_id), descriptor)
        execute_two(runtime, chosen, hidden_model, signers, verifier, suffix, f"g1-{i}")
        executed.add(chosen.intervention_id)
    assert runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS"

    g2 = runtime.expand_causal_model_class([x, z], descriptors)
    decisions.append(g2)
    assert g2.status == "EXPANDED" and g2.generation == 2

    for i, descriptor in enumerate([d["context-both"], d["context-delay-both"]]):
        if runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS":
            break
        execute_two(runtime, descriptor, hidden_model, signers, verifier, suffix, f"g2-{i}")
        executed.add(descriptor.intervention_id)
    assert runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS"

    g3 = runtime.expand_causal_model_class([x, z], descriptors)
    decisions.append(g3)
    assert g3.status == "EXPANDED" and g3.generation == 3
    assert hidden_model.model_id in set(g3.shadow_model_ids)

    initial_vs = runtime.generation_version_space(3)
    assert len(initial_vs.compatible_model_ids) > 1
    identification_trace = [len(initial_vs.compatible_model_ids)]
    probe_trace = []

    # Exact identification: continue until one deterministic G3 hypothesis remains.
    for round_index in range(len(descriptors)):
        snapshot = runtime.generation_version_space(3)
        if snapshot.identified:
            break
        available = [row for row in descriptors if row.intervention_id not in executed]
        assert available
        queries = runtime.model_genesis.query_candidates(available, runtime.structural_models(3))
        selected = runtime.select_generation_intervention(3, queries)
        assert selected is not None and selected.expected_information_gain > 0.0
        chosen = next(row for row in available if row.intervention_id == selected.intervention_id)
        execute_two(runtime, chosen, hidden_model, signers, verifier, suffix, f"g3-{round_index}")
        executed.add(chosen.intervention_id)
        after = runtime.generation_version_space(3)
        identification_trace.append(len(after.compatible_model_ids))
        probe_trace.append({
            "intervention_id": chosen.intervention_id,
            "cost": chosen.cost,
            "expected_information_gain": selected.expected_information_gain,
            "version_space_after": len(after.compatible_model_ids),
        })

    final_vs = runtime.generation_version_space(3)
    assert final_vs.identified
    assert final_vs.identified_model_id == hidden_model.model_id
    assert all(b < a for a, b in zip(identification_trace, identification_trace[1:]))

    payload = epistemic_checkpoint_dict(runtime)
    no_verify = restore_epistemic_runtime(payload, world_verifier=None)
    reverified = restore_epistemic_runtime(payload, world_verifier=verifier)
    no_verify_vs = no_verify.generation_version_space(3)
    reverified_vs = reverified.generation_version_space(3)
    assert len(no_verify_vs.compatible_model_ids) == len(no_verify_vs.model_ids)
    assert reverified_vs.identified_model_id == hidden_model.model_id

    print(json.dumps({
        "status": "PASS_BOUNDED_AUTONOMOUS_DEPTH_ESCALATION_AND_EXACT_GENERATION_SCOPED_CAUSAL_IDENTIFICATION",
        "hidden_predicate_exposed_to_body": False,
        "hidden_predicate": hidden.predicate.render(),
        "autonomous_generation_sequence": [item.generation for item in decisions],
        "autonomous_expansion_statuses": [item.status for item in decisions],
        "external_evaluator_selected_generator": False,
        "generation_3_shadow_model_count": len(g3.shadow_model_ids),
        "generation_3_active_model_count": len(g3.active_model_ids),
        "version_space_trajectory": identification_trace,
        "generation_3_probe_trace": probe_trace,
        "exact_identified_model": final_vs.identified_model_id,
        "verifierless_descendant_version_space": len(no_verify_vs.compatible_model_ids),
        "reverified_descendant_identified_model": reverified_vs.identified_model_id,
        "candidate_absence_authority_leak_blocked": True,
        "boolean_metalanguage_human_authored": True,
        "generation_beyond_3": False,
        "foundation_weight_change": False,
        "global_recursive_acceleration": False,
        "physical_world": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
