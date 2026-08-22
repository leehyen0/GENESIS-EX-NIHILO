from __future__ import annotations

import unittest

from arte_cognition.experiment_genesis import InterventionProposal
from arte_cognition.projection_search_metapolicy import derive_projection_search_metapolicy
from arte_cognition.world_coupling import WorldOutcomePair


BASE = (1.0, 2.0, 4.0)


def proposal(scale: float, suffix: str) -> InterventionProposal:
    return InterventionProposal(
        experiment_id=f"exp-{scale:g}-{suffix}",
        axis_id="axis",
        manipulated_variable="x",
        held_fixed=(),
        low_value=-scale,
        high_value=scale,
        predicted_low_side="LOW",
        predicted_high_side="HIGH",
        reason=f"unit probe_scale={scale:g}",
    )


def pair(p: InterventionProposal, context: str, cls: str, effect: float, authority: bool = True):
    return WorldOutcomePair(
        pair_id=f"pair-{p.experiment_id}-{context}-{cls}",
        experiment_id=p.experiment_id,
        axis_id=p.axis_id,
        source_id=f"source-{cls}",
        context_id=context,
        challenge_id=f"challenge-{context}-{cls}",
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


def parse_scale(p: InterventionProposal):
    return float(p.reason.split("probe_scale=", 1)[1])


class ProjectionSearchMetaPolicyTests(unittest.TestCase):
    def derive(self, proposals, pairs):
        return derive_projection_search_metapolicy(
            BASE,
            proposals,
            pairs,
            min_independent_classes=2,
            probe_scale=parse_scale,
        )

    def test_one_context_dominant_4x_keeps_two_scale_exploration(self):
        ps = [proposal(s, "c1") for s in BASE]
        pairs = []
        for p in ps:
            effect = 1.0 if parse_scale(p) == 4.0 else 0.0
            pairs.extend([pair(p, "c1", "A", effect), pair(p, "c1", "B", effect)])
        policy = self.derive(ps, pairs)
        self.assertEqual(policy.schedule, (4.0, 1.0))

    def test_two_contexts_reproduced_4x_can_collapse_to_singleton(self):
        ps = [proposal(s, c) for c in ("c1", "c2") for s in BASE]
        pairs = []
        for p in ps:
            effect = 1.0 if parse_scale(p) == 4.0 else 0.0
            context = p.experiment_id.rsplit("-", 1)[1]
            pairs.extend([pair(p, context, "A", effect), pair(p, context, "B", effect)])
        policy = self.derive(ps, pairs)
        self.assertEqual(policy.schedule, (4.0,))

    def test_heterogeneous_contexts_generate_non_prefix_1x_4x_policy(self):
        ps = [proposal(s, c) for c in ("needs4", "needs1") for s in BASE]
        pairs = []
        for p in ps:
            context = p.experiment_id.rsplit("-", 1)[1]
            scale = parse_scale(p)
            effect = float(
                (context == "needs4" and scale == 4.0)
                or (context == "needs1" and scale == 1.0)
            )
            pairs.extend([pair(p, context, "A", effect), pair(p, context, "B", effect)])
        policy = self.derive(ps, pairs)
        self.assertEqual(set(policy.schedule), {1.0, 4.0})
        self.assertEqual(len(policy.schedule), 2)
        self.assertNotIn(2.0, policy.schedule)

    def test_missing_base_scale_blocks_contraction(self):
        ps = [proposal(s, "c1") for s in (1.0, 4.0)]
        pairs = []
        for p in ps:
            effect = 1.0 if parse_scale(p) == 4.0 else 0.0
            pairs.extend([pair(p, "c1", "A", effect), pair(p, "c1", "B", effect)])
        policy = self.derive(ps, pairs)
        self.assertEqual(policy.schedule, BASE)

    def test_unverified_pairs_do_not_train_policy(self):
        ps = [proposal(s, "c1") for s in BASE]
        pairs = []
        for p in ps:
            effect = 1.0 if parse_scale(p) == 4.0 else 0.0
            pairs.extend([
                pair(p, "c1", "A", effect, authority=False),
                pair(p, "c1", "B", effect, authority=False),
            ])
        policy = self.derive(ps, pairs)
        self.assertEqual(policy.schedule, BASE)


if __name__ == "__main__":
    unittest.main()
