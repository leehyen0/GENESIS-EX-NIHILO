from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.adaptive_cognition import Hypothesis, QueryCandidate, TaskState
from arte_cognition.epistemic_depth_runtime import (
    EpistemicallyDeepPersistentCognitiveRuntime,
    epistemic_checkpoint_dict,
    restore_epistemic_runtime,
)
from arte_cognition.experiment_genesis import InterventionProposal
from arte_cognition.possibility_space import Fact, OperatorSpec
from arte_cognition.world_coupling import (
    HMACWorldReceiptSigner,
    HMACWorldReceiptVerifier,
    WorldOutcomeReceipt,
)
from arte_cognition.world_model_ecology import CausalWorldModel


def load_seed(path: str) -> int:
    return int(Path(path).read_text().strip())


class SignedEffectExecutor:
    def __init__(self, signer, context_id: str, source_id: str, challenge_id: str, effect: float):
        self.signer = signer
        self.context_id = context_id
        self.source_id = source_id
        self.challenge_id = challenge_id
        self.effect = float(effect)

    def execute(self, proposal, arm: str, value: float):
        outcome = 0.0 if arm.upper() == "LOW" else self.effect
        receipt = WorldOutcomeReceipt(
            receipt_id=f"{self.challenge_id}::{proposal.experiment_id}::{arm}",
            experiment_id=proposal.experiment_id,
            axis_id=proposal.axis_id,
            arm=arm,
            intervention_value=float(value),
            outcome=outcome,
            source_id=self.source_id,
            context_id=self.context_id,
            challenge_id=self.challenge_id,
            epoch=1,
            budget_token=f"budget::{self.challenge_id}",
            externally_generated=True,
        )
        return self.signer.sign(receipt)


def proposal(experiment_id: str):
    return InterventionProposal(
        experiment_id=experiment_id,
        axis_id="MODEL_DISCRIMINATION_AXIS",
        manipulated_variable="probe",
        held_fixed=(),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="LOW",
        predicted_high_side="HIGH",
        reason="hidden causal-model discrimination probe",
    )


def main(seed_path: str) -> None:
    rng = random.Random(load_seed(seed_path))
    suffix = rng.randrange(100000, 999999)
    expensive_id = f"EXPENSIVE::{suffix}"
    cheap_id = f"CHEAP::{suffix}"
    surprise_id = f"SURPRISE::{suffix}"

    models = [
        CausalWorldModel(
            "MODEL_A", 1.0,
            ((expensive_id, "POSITIVE_EFFECT"), (cheap_id, "NO_EFFECT"), (surprise_id, "POSITIVE_EFFECT")),
        ),
        CausalWorldModel(
            "MODEL_B", 1.0,
            ((expensive_id, "NEGATIVE_EFFECT"), (cheap_id, "NO_EFFECT"), (surprise_id, "POSITIVE_EFFECT")),
        ),
    ]

    runtime = EpistemicallyDeepPersistentCognitiveRuntime()
    runtime.register_causal_world_models(models)
    candidates = [
        QueryCandidate(cheap_id, {"MODEL_A": "NO_EFFECT", "MODEL_B": "NO_EFFECT"}, cost=1.0, intervention=True),
        QueryCandidate(expensive_id, {"MODEL_A": "POSITIVE_EFFECT", "MODEL_B": "NEGATIVE_EFFECT"}, cost=20.0, intervention=True),
    ]
    ranked = runtime.rank_epistemic_interventions(candidates)
    assert ranked and ranked[0].intervention_id == expensive_id
    assert ranked[0].cost > candidates[0].cost
    assert runtime.epistemic_depth_plan().mode == "DEEP_DISCRIMINATION"

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

    expensive = proposal(expensive_id)
    for index, (issuer, signer) in enumerate(signers.items()):
        runtime.execute_world_intervention(
            expensive,
            SignedEffectExecutor(signer, "model-discrimination", f"source-{index}", f"expensive-{index}-{suffix}", 1.0),
            verifier=verifier,
        )
    concentrated = runtime.world_models.posterior()
    assert concentrated["MODEL_A"] > 0.99
    assert runtime.epistemic_depth_plan().mode == "COMPACT"

    facts = [Fact(f"system-{i}", "state", f"value-{i}") for i in range(6)]
    spec = OperatorSpec(
        relation_opposites={"state": "not-state"},
        object_complements={f"value-{i}": f"other-{i}" for i in range(6)},
    )
    task = TaskState(
        goal="resolve model-class residual",
        hypotheses=[Hypothesis("h1"), Hypothesis("h2")],
        residuals=["family:r1", "family:r2", "family:r3", "family:r4"],
        stakes=1.0,
        novelty=0.9,
        action_required=True,
        external_world=True,
    )
    compact_cycle = runtime.cycle(task, facts=facts, operator_spec=spec, possibility_budget=4)
    assert len(compact_cycle.possibilities) == 4

    surprise = proposal(surprise_id)
    for index, (issuer, signer) in enumerate(signers.items()):
        runtime.execute_world_intervention(
            surprise,
            SignedEffectExecutor(signer, "model-class-shift", f"surprise-source-{index}", f"surprise-{index}-{suffix}", 0.0),
            verifier=verifier,
        )
    expanded_plan = runtime.epistemic_depth_plan()
    assert expanded_plan.mode == "EXPAND_MODEL_CLASS"
    assert expanded_plan.model_class_inadequate
    expanded_cycle = runtime.cycle(task, facts=facts, operator_spec=spec, possibility_budget=4)
    assert len(expanded_cycle.possibilities) > len(compact_cycle.possibilities)

    payload = epistemic_checkpoint_dict(runtime)
    without_verifier = restore_epistemic_runtime(payload, world_verifier=None)
    with_verifier = restore_epistemic_runtime(payload, world_verifier=verifier)
    assert without_verifier.epistemic_depth_plan().mode != "EXPAND_MODEL_CLASS"
    assert with_verifier.epistemic_depth_plan().mode == "EXPAND_MODEL_CLASS"
    descendant_cycle = with_verifier.cycle(task, facts=facts, operator_spec=spec, possibility_budget=4)
    assert len(descendant_cycle.possibilities) == len(expanded_cycle.possibilities)

    out = {
        "status": "PASS_BOUNDED_COST_TOLERANT_CAUSAL_MODEL_ECOLOGY_AND_EPISTEMIC_DEPTH_EXPANSION",
        "selected_intervention": ranked[0].intervention_id,
        "selected_intervention_cost": ranked[0].cost,
        "cheap_intervention_cost": candidates[0].cost,
        "selected_expected_information_gain": ranked[0].expected_information_gain,
        "posterior_after_discrimination": concentrated,
        "compact_possibility_count": len(compact_cycle.possibilities),
        "expanded_possibility_count": len(expanded_cycle.possibilities),
        "depth_mode_after_world_surprise": expanded_plan.mode,
        "expanded_possibility_budget": expanded_plan.possibility_budget,
        "expanded_representation_axis_budget": expanded_plan.representation_axis_budget,
        "expanded_intervention_budget": expanded_plan.intervention_budget,
        "cost_exponent_after_surprise": expanded_plan.cost_exponent,
        "verifierless_descendant_mode": without_verifier.epistemic_depth_plan().mode,
        "reverified_descendant_mode": with_verifier.epistemic_depth_plan().mode,
        "descendant_expanded_possibility_count": len(descendant_cycle.possibilities),
        "model_class_inadequacy_rederived_after_external_reverification": True,
        "foundation_weight_change": False,
        "global_recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
    }
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
