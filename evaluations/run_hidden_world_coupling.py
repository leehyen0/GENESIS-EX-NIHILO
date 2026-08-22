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

    def __init__(self, coefficients, source_id, challenge_id, epoch):
        self._coefficients = dict(coefficients)
        self.source_id = source_id
        self.challenge_id = challenge_id
        self.epoch = epoch

    def execute(self, proposal, arm, value):
        coefficient = float(self._coefficients.get(proposal.axis_id, 0.0))
        outcome = coefficient * float(value)
        return WorldOutcomeReceipt(
            receipt_id=f"hidden::{self.source_id}::{self.challenge_id}::{proposal.axis_id}::{arm}",
            experiment_id=proposal.experiment_id,
            axis_id=proposal.axis_id,
            arm=arm,
            intervention_value=float(value),
            outcome=outcome,
            source_id=self.source_id,
            context_id="hidden-software-world",
            challenge_id=self.challenge_id,
            epoch=self.epoch,
            budget_token=f"matched::{self.source_id}::{self.challenge_id}",
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


def main(seed_path):
    seed = int(Path(seed_path).read_text().strip())
    rng = random.Random(seed)

    runtime = PersistentCognitiveRuntime()
    proposals = [proposal("AXIS::A"), proposal("AXIS::B"), proposal("AXIS::C")]
    before = runtime.rank_intervention_proposals(proposals)

    # The evaluator deliberately chooses an axis that is not initially ranked
    # first, preventing a hard-coded default order from passing without learning.
    active_axis = rng.choice([item.axis_id for item in before[1:]])
    active_magnitude = rng.uniform(1.25, 2.75)
    active_sign = rng.choice([-1.0, 1.0])
    coefficients = {
        item.axis_id: rng.uniform(-0.04, 0.04)
        for item in proposals
    }
    coefficients[active_axis] = active_sign * active_magnitude

    for index, source_id in enumerate(("hidden-source-1", "hidden-source-2"), start=1):
        executor = HiddenWorldExecutor(
            coefficients=coefficients,
            source_id=source_id,
            challenge_id=f"hidden-challenge-{index}",
            epoch=index,
        )
        for item in proposals:
            runtime.execute_world_intervention(item, executor)

    after = runtime.rank_intervention_proposals(proposals)
    if after[0].axis_id != active_axis:
        raise AssertionError("world outcomes did not change future intervention choice to the consequence-bearing axis")

    descendant = restore_json(checkpoint_json(runtime))
    descendant_ranked = descendant.rank_intervention_proposals(proposals)
    if descendant_ranked[0].axis_id != active_axis:
        raise AssertionError("world-caused intervention policy did not survive checkpoint/restore")

    summary = descendant.world_axis_summary(active_axis)
    if summary.independent_evidence_classes < 2:
        raise AssertionError("hidden challenge did not provide two independent evidence classes")

    print(json.dumps({
        "status": "PASS_BOUNDED_HIDDEN_SOFTWARE_WORLD_TO_DESCENDANT_BEHAVIOR",
        "initial_top_axis": before[0].axis_id,
        "learned_top_axis": after[0].axis_id,
        "descendant_top_axis": descendant_ranked[0].axis_id,
        "independent_evidence_classes": summary.independent_evidence_classes,
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
