from __future__ import annotations

import unittest

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.experiment_genesis import InterventionProposal
from arte_cognition.projection_scale_genesis import (
    derive_projection_scale_frontier,
    validated_generated_projection_scales,
)
from arte_cognition.representation_genesis import RepresentationAxis
from arte_cognition.world_coupling import (
    HMACWorldReceiptSigner,
    HMACWorldReceiptVerifier,
    WorldOutcomePair,
    WorldOutcomeReceipt,
)


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


def axis(names, label):
    a, b = names
    return RepresentationAxis(
        axis_id=f"AXIS::PROJECTION::{a}|{b}",
        family="PROJECTION",
        inputs=(a, b),
        threshold=0.0,
        direction="GT",
        information_gain=1.0,
        train_support=8,
        positive_partition=(f"{label}-positive",),
        formula=f"(1)*{a} + (1)*{b}",
        coefficients=((a, 1.0), (b, 1.0)),
        bias=0.0,
        status="PROPOSAL_ONLY",
    )


class SignedSmoothWorld:
    def __init__(self, names, target_scale, signer, source_id, challenge_id, context_id):
        self.names = tuple(names)
        self.target_scale = float(target_scale)
        self.signer = signer
        self.source_id = source_id
        self.challenge_id = challenge_id
        self.context_id = context_id

    def execute(self, p, arm, value):
        state = {name: 0.0 for name in self.names}
        state.update({name: float(v) for name, v in p.held_fixed})
        state[p.manipulated_variable] = float(value)
        latent = sum(state[name] for name in self.names)
        target = 0.15 * self.target_scale
        response = max(0.0, 1.0 - abs(latent - target) / 0.30)
        return self.signer.sign(WorldOutcomeReceipt(
            receipt_id=f"{self.challenge_id}::{p.experiment_id}::{arm}",
            experiment_id=p.experiment_id,
            axis_id=p.axis_id,
            arm=arm,
            intervention_value=float(value),
            outcome=float(response),
            source_id=self.source_id,
            context_id=self.context_id,
            challenge_id=self.challenge_id,
            epoch=1,
            budget_token=f"budget::{self.challenge_id}",
            externally_generated=True,
        ))


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

    def test_signed_world_generated_3x_reconstructs_in_fresh_descendant(self):
        target = 3.0
        keys = {"issuer-a": b"key-a", "issuer-b": b"key-b"}
        signers = {name: HMACWorldReceiptSigner(name, secret) for name, secret in keys.items()}
        verifier = HMACWorldReceiptVerifier(
            keys,
            independence_classes={"issuer-a": "A", "issuer-b": "B"},
        )
        runtime = PersistentCognitiveRuntime()
        names = ("train-x", "train-z")
        train_axis = axis(names, "train")
        runtime.memory.remember_representation(train_axis)
        reference = {names[0]: 0.0, names[1]: 0.0}

        base = runtime.generate_interventions(train_axis, reference)
        for index, p in enumerate(base):
            for issuer, signer in signers.items():
                runtime.execute_world_intervention(
                    p,
                    SignedSmoothWorld(
                        names, target, signer,
                        f"base-source-{index}-{issuer}",
                        f"base-challenge-{index}-{issuer}",
                        "train-context",
                    ),
                    verifier=verifier,
                )
        frontier = runtime.projection_scale_frontier(context_id="train-context")
        self.assertEqual(frontier.status, "GENERATED_NUMERIC_REFINEMENT")
        self.assertIn(target, frontier.candidate_scales)

        generated = runtime.generate_projection_scale_frontier_interventions(
            train_axis, reference, context_id="train-context"
        )
        for index, p in enumerate(generated):
            for issuer, signer in signers.items():
                runtime.execute_world_intervention(
                    p,
                    SignedSmoothWorld(
                        names, target, signer,
                        f"gen-source-{index}-{issuer}",
                        f"gen-challenge-{index}-{issuer}",
                        "train-context",
                    ),
                    verifier=verifier,
                )
        self.assertIn(target, runtime.projection_probe_vocabulary())

        payload = checkpoint_dict(runtime)
        verifierless = restore_runtime(payload)
        self.assertNotIn(target, verifierless.projection_probe_vocabulary())
        restored = restore_runtime(payload, world_verifier=verifier)
        self.assertIn(target, restored.projection_probe_vocabulary())

        fresh_names = ("fresh-x", "fresh-z")
        fresh_axis = axis(fresh_names, "fresh")
        restored.memory.remember_representation(fresh_axis)
        fresh = restored.generate_interventions(
            fresh_axis, {fresh_names[0]: 0.0, fresh_names[1]: 0.0}
        )
        fresh_scales = {restored._proposal_probe_scale(p) for p in fresh}
        self.assertIn(target, fresh_scales)
        self.assertLess(len(fresh_scales), len(restored.projection_probe_vocabulary()))


if __name__ == "__main__":
    unittest.main()