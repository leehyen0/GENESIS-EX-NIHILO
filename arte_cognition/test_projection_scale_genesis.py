from __future__ import annotations

import unittest

from arte_cognition.experiment_genesis import InterventionProposal
from arte_cognition.projection_scale_genesis import (
    derive_projection_scale_frontier,
    validated_generated_projection_scales,
)
from arte_cognition.world_coupling import WorldOutcomePair


AUTHORED = (1.0, 2.0, 4.0)


def proposal(scale: float, suffix: str = "x") -> InterventionProposal:
    return InterventionProposal(
        experiment_id=f"probe-{scale:g}-{suffix}",
        axis_id="axis",
        manipulated_variable="x",
        held_fixed=(),
        low_value=-scale,
        high_value=scale,
        predicted_low_side="LOW",
        predicted_high_side="HIGH",
        reason=f"test probe_scale={scale:g}",
    )


def pair(p, cls, effect, authority=True):
    return WorldOutcomePair(
        pair_id=f"pair-{p.experiment_id}-{cls}",
        experiment_id=p.experiment_id,
        axis_id=p.axis_id,
        source_id=f"source-{cls}",
        context_id="ctx",
        challenge_id=f"challenge-{cls}",
        epoch=1,
        low_outcome=0.0,
        high_outcome=float(effect),
        low_value=p.low_value,
        high_value=p.high_value,
        matched_budget=True,
        externally_generated=True,
        issuer_id=f"issuer-{cls}",
        independence_class_id=cls,
        authority_verified=authority,
    )


def scale_of(p):
    return float(p.reason.split("probe_scale=", 1)[1])


class ProjectionScaleGenesisTests(unittest.TestCase):
    def test_weak_authored_vocabulary_generates_midpoint_atoms(self):
        ps = [proposal(scale) for scale in AUTHORED]
        effects = {1.0: 0.0, 2.0: 0.5, 4.0: 0.5}
        pairs = []
        for p in ps:
            effect = effects[scale_of(p)]
            pairs.extend([pair(p, "A", effect), pair(p, "B", effect)])
        frontier = derive_projection_scale_frontier(
            AUTHORED, ps, pairs, 2, scale_of, context_id="ctx"
        )
        self.assertEqual(frontier.status, "GENERATED_NUMERIC_REFINEMENT")
        self.assertIn(3.0, frontier.candidate_scales)
        self.assertIn(1.5, frontier.candidate_scales)
        self.assertLess(frontier.best_verified_effect, frontier.strong_effect_threshold)

    def test_strong_existing_scale_stops_refinement(self):
        ps = [proposal(scale) for scale in AUTHORED]
        pairs = []
        for p in ps:
            effect = 1.0 if scale_of(p) == 4.0 else 0.0
            pairs.extend([pair(p, "A", effect), pair(p, "B", effect)])
        frontier = derive_projection_scale_frontier(
            AUTHORED, ps, pairs, 2, scale_of, context_id="ctx"
        )
        self.assertEqual(frontier.status, "STRONG_SCALE_ALREADY_AVAILABLE")
        self.assertEqual(frontier.candidate_scales, ())

    def test_generated_strong_scale_enters_descendant_vocabulary(self):
        ps = [proposal(scale) for scale in (*AUTHORED, 3.0)]
        pairs = []
        for p in ps:
            effect = 1.0 if scale_of(p) == 3.0 else 0.5
            pairs.extend([pair(p, "A", effect), pair(p, "B", effect)])
        generated = validated_generated_projection_scales(
            AUTHORED, ps, pairs, 2, scale_of, strong_effect_threshold=0.9
        )
        self.assertEqual(generated, (3.0,))

    def test_one_independence_class_cannot_generate_or_validate_scale(self):
        ps = [proposal(scale) for scale in (*AUTHORED, 3.0)]
        pairs = []
        for p in ps:
            effect = 1.0 if scale_of(p) == 3.0 else 0.5
            pairs.append(pair(p, "A", effect))
        frontier = derive_projection_scale_frontier(
            AUTHORED, ps, pairs, 2, scale_of, context_id="ctx"
        )
        generated = validated_generated_projection_scales(
            AUTHORED, ps, pairs, 2, scale_of
        )
        self.assertEqual(frontier.status, "NO_AUTHENTICATED_SCALE_EVIDENCE")
        self.assertEqual(generated, ())

    def test_unverified_pairs_cannot_train_scale_genesis(self):
        ps = [proposal(scale) for scale in AUTHORED]
        pairs = []
        for p in ps:
            pairs.extend([
                pair(p, "A", 0.5, authority=False),
                pair(p, "B", 0.5, authority=False),
            ])
        frontier = derive_projection_scale_frontier(
            AUTHORED, ps, pairs, 2, scale_of, context_id="ctx"
        )
        self.assertEqual(frontier.status, "NO_AUTHENTICATED_SCALE_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
