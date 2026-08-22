from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.causal_predicate_genesis import BooleanCausalPredicateGenesisEngine
from arte_cognition.epistemic_depth_runtime import (
    EpistemicallyDeepPersistentCognitiveRuntime,
    epistemic_checkpoint_dict,
    restore_epistemic_runtime,
)
from arte_cognition.experiment_genesis import InterventionProposal
from arte_cognition.intervention_surface_genesis import InterventionSurfaceGenesisEngine
from arte_cognition.sparse_minterm_genesis import SparseMintermCausalGenesisEngine
from arte_cognition.world_coupling import HMACWorldReceiptSigner, HMACWorldReceiptVerifier, WorldOutcomeReceipt


def proposal(d):
    return InterventionProposal(
        experiment_id=d.intervention_id,
        axis_id="FALSIFICATION_DRIVEN_G4_AXIS",
        manipulated_variable=d.targets[0] if d.targets else "__context__",
        held_fixed=tuple((f"blocked::{name}", 1.0) for name in d.blocked),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="LOW",
        predicted_high_side="HIGH",
        reason=(f"g4-grammar targets={d.targets} blocked={d.blocked} "
                f"delay={d.delay_steps} context={d.context_shift}"),
    )


class HiddenG4World:
    def __init__(self, model, signer, source_id, challenge_id):
        self.model = model
        self.signer = signer
        self.source_id = source_id
        self.challenge_id = challenge_id

    def execute(self, p, arm: str, value: float):
        label = self.model.prediction_for(p.experiment_id) or "NO_EFFECT"
        effect = {"POSITIVE_EFFECT": 1.0, "NEGATIVE_EFFECT": -1.0, "NO_EFFECT": 0.0}[label]
        receipt = WorldOutcomeReceipt(
            receipt_id=f"{self.challenge_id}::{p.experiment_id}::{arm}",
            experiment_id=p.experiment_id,
            axis_id=p.axis_id,
            arm=arm,
            intervention_value=float(value),
            outcome=0.0 if arm.upper() == "LOW" else effect,
            source_id=self.source_id,
            context_id="hidden-g4-world",
            challenge_id=self.challenge_id,
            epoch=1,
            budget_token=f"budget::{self.challenge_id}",
            externally_generated=True,
        )
        return self.signer.sign(receipt)


def execute_two(runtime, descriptor, hidden_model, signers, verifier, suffix, stage):
    for index, (_issuer, signer) in enumerate(signers.items()):
        runtime.execute_world_intervention(
            proposal(descriptor),
            HiddenG4World(
                hidden_model,
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
    descriptors = InterventionSurfaceGenesisEngine(budget=256).generate(variables)

    g3_engine = BooleanCausalPredicateGenesisEngine(model_budget=2048, max_literals_per_term=3, max_terms=2)
    g3 = g3_engine.generate_novel(variables, descriptors, (), ())
    assert g3 and not g3_engine.last_truncated
    g3_models = [item.model for item in g3]
    g3_signatures = {tuple(sorted(model.predictions)) for model in g3_models}

    g4_engine = SparseMintermCausalGenesisEngine(model_budget=4096, max_minterms=3)
    g4_surface = g4_engine.generate_novel(variables, descriptors, (), g3_models)
    assert g4_surface and not g4_engine.last_truncated
    hidden_pool = [
        item for item in g4_surface
        if item.cause == x and item.sign == "POS" and len(item.minterms) == 3
        and tuple(sorted(item.model.predictions)) not in g3_signatures
        and sum(1 for _, value in item.model.predictions if value == "POSITIVE_EFFECT") >= 3
    ]
    assert hidden_pool
    hidden = rng.choice(hidden_pool)
    hidden_model = hidden.model

    runtime = EpistemicallyDeepPersistentCognitiveRuntime()
    runtime.register_causal_world_models(g3_models)

    keys = {
        f"issuer-a-{suffix}": f"secret-a-{suffix}".encode(),
        f"issuer-b-{suffix}": f"secret-b-{suffix}".encode(),
    }
    signers = {issuer: HMACWorldReceiptSigner(issuer, secret) for issuer, secret in keys.items()}
    verifier = HMACWorldReceiptVerifier(keys, independence_classes={
        f"issuer-a-{suffix}": "independent-A",
        f"issuer-b-{suffix}": "independent-B",
    })

    # Exhaustively expose the bounded intervention surface. Because the selected
    # three-minterm truth signature is absent from G3, all G3 hypotheses must die.
    for index, descriptor in enumerate(descriptors):
        execute_two(runtime, descriptor, hidden_model, signers, verifier, suffix, f"g3-refute-{index}")
    g3_space = runtime.generation_version_space(3)
    assert not g3_space.compatible_model_ids
    assert runtime.generation_falsified(3)
    assert runtime.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS"

    # BODY chooses the next generator from ancestry/failure; evaluator does not call
    # the sparse-minterm generator directly.
    expansion = runtime.expand_causal_model_class(variables, descriptors)
    assert expansion.status == "EXPANDED"
    assert expansion.generation == 4
    assert expansion.origin == "GENERATED_SPARSE_MINTERM"
    assert hidden_model.model_id in set(expansion.shadow_model_ids)
    assert hidden_model.model_id in set(expansion.active_model_ids)

    g4_space = runtime.generation_version_space(4)
    assert g4_space.identified
    assert g4_space.identified_model_id == hidden_model.model_id
    restored_hidden = runtime.world_models.models[hidden_model.model_id]
    assert restored_hidden.generation == 4
    assert len([part for part in restored_hidden.structure if part.startswith("MINTERM(")]) == 3
    assert restored_hidden.parent_model_ids

    payload = epistemic_checkpoint_dict(runtime)
    no_verify = restore_epistemic_runtime(payload, world_verifier=None)
    reverified = restore_epistemic_runtime(payload, world_verifier=verifier)
    no_verify_space = no_verify.generation_version_space(4)
    reverified_space = reverified.generation_version_space(4)
    assert len(no_verify_space.compatible_model_ids) == len(no_verify_space.model_ids)
    assert len(no_verify_space.compatible_model_ids) > 1
    assert reverified_space.identified_model_id == hidden_model.model_id

    print(json.dumps({
        "status": "PASS_BOUNDED_FALSIFICATION_DRIVEN_G4_SPARSE_MINTERM_GRAMMAR_AND_DESCENDANT",
        "hidden_g4_model_exposed_to_body": False,
        "hidden_g4_model": hidden_model.model_id,
        "hidden_g4_minterm_count": len(hidden.minterms),
        "g3_shadow_model_count": len(g3_models),
        "g3_final_version_space": len(g3_space.compatible_model_ids),
        "g3_falsified_by_two_independence_classes": True,
        "external_evaluator_selected_g4_generator": False,
        "g4_expansion_status": expansion.status,
        "g4_shadow_model_count": len(expansion.shadow_model_ids),
        "g4_active_model_count": len(expansion.active_model_ids),
        "g4_exact_identified_model": g4_space.identified_model_id,
        "g4_three_minterm_structure": True,
        "g4_prediction_signature_absent_from_g3": True,
        "verifierless_descendant_g4_version_space": len(no_verify_space.compatible_model_ids),
        "reverified_descendant_g4_identified_model": reverified_space.identified_model_id,
        "boolean_metalanguage_human_authored": True,
        "complexity_escalation_world_falsification_driven": True,
        "generation_beyond_4": False,
        "physical_world": False,
        "global_recursive_acceleration": False,
        "foundation_weight_change": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
