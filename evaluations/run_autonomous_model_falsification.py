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


def proposal(d):
    return InterventionProposal(
        experiment_id=d.intervention_id,
        axis_id="POST_IDENTIFICATION_FALSIFICATION_AXIS",
        manipulated_variable=d.targets[0] if d.targets else "__context__",
        held_fixed=tuple((f"blocked::{name}", 1.0) for name in d.blocked),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="LOW",
        predicted_high_side="HIGH",
        reason=(f"falsification targets={d.targets} blocked={d.blocked} "
                f"delay={d.delay_steps} context={d.context_shift}"),
    )


class HiddenCounterexampleWorld:
    def __init__(self, counterexample_id, signer, source_id, challenge_id):
        self.counterexample_id = counterexample_id
        self.signer = signer
        self.source_id = source_id
        self.challenge_id = challenge_id

    def execute(self, p, arm: str, value: float):
        effect = 1.0 if p.experiment_id == self.counterexample_id else 0.0
        receipt = WorldOutcomeReceipt(
            receipt_id=f"{self.challenge_id}::{p.experiment_id}::{arm}",
            experiment_id=p.experiment_id,
            axis_id=p.axis_id,
            arm=arm,
            intervention_value=float(value),
            outcome=0.0 if arm.upper() == "LOW" else effect,
            source_id=self.source_id,
            context_id="hidden-post-identification-world",
            challenge_id=self.challenge_id,
            epoch=1,
            budget_token=f"budget::{self.challenge_id}",
            externally_generated=True,
        )
        return self.signer.sign(receipt)


def execute_two(runtime, descriptor, counterexample_id, signers, verifier, suffix, stage):
    for index, (_issuer, signer) in enumerate(signers.items()):
        runtime.execute_world_intervention(
            proposal(descriptor),
            HiddenCounterexampleWorld(
                counterexample_id,
                signer,
                source_id=f"source-{stage}-{index}-{suffix}",
                challenge_id=f"challenge-{stage}-{index}-{suffix}",
            ),
            verifier=verifier,
        )


def main(seed_path: str) -> None:
    rng = random.Random(int(Path(seed_path).read_text().strip()))
    suffix = rng.randrange(100000, 999999)
    x, z = f"sensor_x_{suffix}", f"sensor_z_{suffix}"
    variables = [x, z]

    surface_engine = InterventionSurfaceGenesisEngine(budget=256)
    full_surface = surface_engine.generate(variables)
    assert not surface_engine.last_truncated
    confirm = full_surface[0]
    counterexample_pool = [row for row in full_surface if row.intervention_id != confirm.intervention_id]
    assert counterexample_pool
    counterexample = rng.choice(counterexample_pool)

    model = CausalWorldModel(
        "IDENTIFIED_G3_MODEL", 1.0,
        tuple((row.intervention_id, "NO_EFFECT") for row in full_surface),
        origin="GENERATED_PREDICATE",
        family="SYNTHESIZED_ACTIVATION_PREDICATE",
        structure=("SOLE_SURVIVING_MODEL",),
        generation=3,
    )
    runtime = EpistemicallyDeepPersistentCognitiveRuntime()
    runtime.register_causal_world_models([model])

    keys = {
        f"issuer-a-{suffix}": f"secret-a-{suffix}".encode(),
        f"issuer-b-{suffix}": f"secret-b-{suffix}".encode(),
    }
    signers = {issuer: HMACWorldReceiptSigner(issuer, secret) for issuer, secret in keys.items()}
    verifier = HMACWorldReceiptVerifier(keys, independence_classes={
        f"issuer-a-{suffix}": "independent-A",
        f"issuer-b-{suffix}": "independent-B",
    })

    execute_two(runtime, confirm, counterexample.intervention_id, signers, verifier, suffix, "confirm")
    initial = runtime.generation_version_space(3)
    assert initial.identified and initial.identified_model_id == model.model_id

    observed = {confirm.intervention_id}
    challenge_trace = []
    found = False
    for round_index in range(len(full_surface) - 1):
        decision = runtime.select_falsification_intervention(3, variables, tuple(sorted(observed)))
        assert decision.status == "SELECTED"
        assert decision.descriptor is not None
        chosen = decision.descriptor
        assert chosen.intervention_id not in observed
        execute_two(runtime, chosen, counterexample.intervention_id, signers, verifier, suffix, f"challenge-{round_index}")
        observed.add(chosen.intervention_id)
        falsified = runtime.generation_falsified(3)
        challenge_trace.append({
            "intervention_id": chosen.intervention_id,
            "cost": chosen.cost,
            "semantic_novelty": decision.semantic_novelty,
            "structural_stress": decision.structural_stress,
            "falsified_after": falsified,
        })
        if falsified:
            found = True
            break

    assert found
    assert challenge_trace[-1]["intervention_id"] == counterexample.intervention_id
    final_space = runtime.generation_version_space(3)
    assert len(final_space.compatible_model_ids) == 0
    assert runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS"

    # Reopening a frontier is distinct from proving that the *next* bounded grammar
    # can explain an arbitrary random counterexample. BODY must attempt G4 rather
    # than remain falsely confident or report the old G3 maximum. The dedicated G4
    # hidden evaluation separately requires an expressible G4 world to be recovered.
    frontier = runtime.expand_causal_model_class(variables, full_surface)
    assert frontier.generation == 4
    assert frontier.status != "MAX_GENERATION_REACHED"
    assert frontier.status in {
        "EXPANDED",
        "NO_EVIDENCE_COMPATIBLE_CANDIDATES",
        "NO_STRUCTURAL_CANDIDATES",
    }

    payload = epistemic_checkpoint_dict(runtime)
    no_verify = restore_epistemic_runtime(payload, world_verifier=None)
    reverified = restore_epistemic_runtime(payload, world_verifier=verifier)
    no_verify_space = no_verify.generation_version_space(3)
    reverified_space = reverified.generation_version_space(3)
    assert no_verify_space.identified
    assert not reverified_space.compatible_model_ids
    assert reverified.generation_falsified(3)

    print(json.dumps({
        "status": "PASS_BOUNDED_POST_IDENTIFICATION_AUTONOMOUS_MODEL_FALSIFICATION_AND_FRONTIER_REOPEN",
        "hidden_counterexample_exposed_to_body": False,
        "hidden_counterexample_id": counterexample.intervention_id,
        "generated_surface_size": len(full_surface),
        "initial_identified_model": initial.identified_model_id,
        "challenge_count_until_counterexample": len(challenge_trace),
        "challenge_trace": challenge_trace,
        "final_generation_3_version_space": len(final_space.compatible_model_ids),
        "model_class_reopened": True,
        "next_structural_frontier_status": frontier.status,
        "next_structural_frontier_generation": frontier.generation,
        "next_structural_frontier_origin": frontier.origin,
        "next_grammar_adequacy_not_assumed": True,
        "verifierless_descendant_identified_model": no_verify_space.identified_model_id,
        "reverified_descendant_generation_falsified": reverified.generation_falsified(3),
        "identified_equals_true_assumption": False,
        "bounded_surface_exhaustive_if_needed": True,
        "intervention_capability_schema_human_authored": True,
        "generation_beyond_4": False,
        "physical_world": False,
        "global_recursive_acceleration": False,
        "foundation_weight_change": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
