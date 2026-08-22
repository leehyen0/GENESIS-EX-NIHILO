from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.body_checkpoint import checkpoint_json, restore_json
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.experiment_genesis import InterventionProposal
from arte_cognition.world_coupling import WorldOutcomeReceipt


class HiddenWorldExecutor:
    """Evaluator-owned software world.

    The runtime receives only receipts. Hidden coefficients stay inside this
    evaluator object and are never passed into the cognition BODY.
    """

    def __init__(self, coefficients, source_id, context_id, challenge_id, epoch):
        self._coefficients = dict(coefficients)
        self.source_id = source_id
        self.context_id = context_id
        self.challenge_id = challenge_id
        self.epoch = epoch

    def execute(self, proposal, arm, value):
        coefficient = float(self._coefficients.get(proposal.axis_id, 0.0))
        outcome = coefficient * float(value)
        return WorldOutcomeReceipt(
            receipt_id=f"hidden::{self.context_id}::{self.source_id}::{self.challenge_id}::{proposal.axis_id}::{arm}",
            experiment_id=proposal.experiment_id,
            axis_id=proposal.axis_id,
            arm=arm,
            intervention_value=float(value),
            outcome=outcome,
            source_id=self.source_id,
            context_id=self.context_id,
            challenge_id=self.challenge_id,
            epoch=self.epoch,
            budget_token=f"matched::{self.context_id}::{self.source_id}::{self.challenge_id}",
            externally_generated=True,
        )


def proposal(axis_id):
    return InterventionProposal(
        experiment_id=f"EXPERIMENT::{axis_id}::x",
        axis_id=axis_id,
        manipulated_variable="x",
        held_fixed=(),
        low_value=0.0,
        high_value=1.0,
        predicted_low_side="LE_THRESHOLD",
        predicted_high_side="GT_THRESHOLD",
        reason="evaluator-owned hidden software-world challenge",
    )


def hidden_coefficients(rng, proposals, active_axis):
    coefficients = {
        item.axis_id: rng.uniform(-0.04, 0.04)
        for item in proposals
    }
    coefficients[active_axis] = rng.choice([-1.0, 1.0]) * rng.uniform(1.25, 2.75)
    return coefficients


def main(seed_path):
    seed = int(Path(seed_path).read_text().strip())
    rng = random.Random(seed)

    runtime = PersistentCognitiveRuntime()
    proposals = [proposal("AXIS::A"), proposal("AXIS::B"), proposal("AXIS::C")]
    initial_top = runtime.rank_intervention_proposals(proposals)[0].axis_id

    # Two hidden regimes require different intervention preferences. At least one
    # active axis differs from the BODY's initial default, so a static order cannot
    # satisfy both regimes. Their disagreement must also block contextless transport.
    axis_ids = [item.axis_id for item in proposals]
    active_regime_1 = rng.choice([axis for axis in axis_ids if axis != initial_top])
    active_regime_2 = rng.choice([axis for axis in axis_ids if axis != active_regime_1])
    regimes = {
        "hidden-regime-1": (active_regime_1, hidden_coefficients(rng, proposals, active_regime_1)),
        "hidden-regime-2": (active_regime_2, hidden_coefficients(rng, proposals, active_regime_2)),
    }

    for context_id, (_, coefficients) in regimes.items():
        for index in (1, 2):
            executor = HiddenWorldExecutor(
                coefficients=coefficients,
                source_id=f"{context_id}-source-{index}",
                context_id=context_id,
                challenge_id=f"{context_id}-challenge-{index}",
                epoch=index,
            )
            for item in proposals:
                runtime.execute_world_intervention(item, executor)

    learned = {
        context_id: runtime.rank_intervention_proposals(proposals, context_id=context_id)[0].axis_id
        for context_id in regimes
    }
    for context_id, (active_axis, _) in regimes.items():
        if learned[context_id] != active_axis:
            raise AssertionError(f"world outcomes did not learn the correct intervention preference in {context_id}")

    transport = runtime.assess_world_transport(proposals)
    if transport.status != "REGIME_CONFLICT_BLOCK_GLOBAL_TRANSPORT":
        raise AssertionError("conflicting hidden regimes did not block unsafe global transport")
    if transport.safe_for_global_transport:
        raise AssertionError("unsafe global transport was incorrectly authorized")
    contextless = runtime.rank_intervention_proposals(proposals)
    if [item.axis_id for item in contextless] != [item.axis_id for item in proposals]:
        raise AssertionError("contextless policy should abstain and preserve proposal order under regime conflict")

    descendant = restore_json(checkpoint_json(runtime))
    descendant_learned = {
        context_id: descendant.rank_intervention_proposals(proposals, context_id=context_id)[0].axis_id
        for context_id in regimes
    }
    if descendant_learned != learned:
        raise AssertionError("context-conditioned world policy did not survive checkpoint/restore")

    descendant_transport = descendant.assess_world_transport(proposals)
    if descendant_transport != transport:
        raise AssertionError("transport abstention did not survive checkpoint/restore")
    descendant_contextless = descendant.rank_intervention_proposals(proposals)
    if [item.axis_id for item in descendant_contextless] != [item.axis_id for item in proposals]:
        raise AssertionError("descendant lost contextless abstention under regime conflict")

    evidence = {
        context_id: descendant.world_axis_summary(active_axis, context_id=context_id).independent_evidence_classes
        for context_id, (active_axis, _) in regimes.items()
    }
    if min(evidence.values()) < 2:
        raise AssertionError("hidden regimes did not provide two independent evidence classes each")

    print(json.dumps({
        "status": "PASS_BOUNDED_HIDDEN_MULTI_REGIME_WORLD_WITH_TRANSPORT_ABSTENTION",
        "initial_top_axis": initial_top,
        "learned_top_by_regime": learned,
        "descendant_top_by_regime": descendant_learned,
        "independent_evidence_classes_by_regime": evidence,
        "global_transport_status": transport.status,
        "global_transport_safe": transport.safe_for_global_transport,
        "contextless_top_axis": contextless[0].axis_id,
        "descendant_contextless_top_axis": descendant_contextless[0].axis_id,
        "world_pair_count": len(descendant.world_coupling.pairs),
        "hidden_mechanism_exposed_to_body": False,
        "independent_organizational_custody": False,
        "physical_world": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_hidden_world_coupling.py <evaluator-owned-seed-file>")
    main(sys.argv[1])
