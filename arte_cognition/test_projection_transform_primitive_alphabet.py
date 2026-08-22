from __future__ import annotations

import unittest

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.experiment_genesis import ExperimentGenesisEngine
from arte_cognition.projection_generator_transform_grammar import (
    DEEP_TRANSFORM_SIGNATURE_ANCHORS,
    generate_projection_transform_programs,
)
from arte_cognition.projection_scale_genesis import projection_scale_scores
from arte_cognition.projection_transform_primitive_genesis import (
    generate_projection_power_primitive_programs,
)
from arte_cognition.projection_transform_primitive_runtime import TransformPrimitiveAlphabetOrgan
from arte_cognition.representation_genesis import RepresentationAxis
from arte_cognition.world_coupling import (
    HMACWorldReceiptSigner,
    HMACWorldReceiptVerifier,
    WorldOutcomeReceipt,
)


class ExactPrimitiveWorld:
    def __init__(self, target, signer, source_id, challenge_id, context_id, epoch):
        self.target = float(target)
        self.signer = signer
        self.source_id = str(source_id)
        self.challenge_id = str(challenge_id)
        self.context_id = str(context_id)
        self.epoch = int(epoch)

    @staticmethod
    def scale_of(proposal):
        marker = "probe_scale="
        return float(str(proposal.reason).split(marker, 1)[1].split()[0].rstrip(",;)") )

    def execute(self, proposal, arm, value):
        scale = self.scale_of(proposal)
        high = 1.0 if abs(scale - self.target) <= 1e-9 else 0.25
        outcome = 0.0 if str(arm).upper() == "LOW" else high
        return self.signer.sign(WorldOutcomeReceipt(
            receipt_id=f"{self.challenge_id}::{proposal.experiment_id}::{arm}",
            experiment_id=proposal.experiment_id,
            axis_id=proposal.axis_id,
            arm=arm,
            intervention_value=float(value),
            outcome=float(outcome),
            source_id=self.source_id,
            context_id=self.context_id,
            challenge_id=self.challenge_id,
            epoch=self.epoch,
            budget_token=f"budget::{self.challenge_id}",
            externally_generated=True,
        ))


class TransformPrimitiveAlphabetTests(unittest.TestCase):
    def setUp(self):
        self.keys = {"primitive-a": b"primitive-key-a", "primitive-b": b"primitive-key-b"}
        self.signers = {
            issuer: HMACWorldReceiptSigner(issuer, secret)
            for issuer, secret in self.keys.items()
        }
        self.verifier = HMACWorldReceiptVerifier(
            self.keys,
            independence_classes={"primitive-a": "A", "primitive-b": "B"},
        )
        self.programs = generate_projection_power_primitive_programs()
        self.hidden = next(
            item for item in self.programs
            if item.exponent == 0.5 and item.alpha == 0.25
        )

    @staticmethod
    def axis(label):
        x, z = f"{label}-x", f"{label}-z"
        return RepresentationAxis(
            axis_id=f"AXIS::PROJECTION::{x}|{z}",
            family="PROJECTION",
            inputs=(x, z),
            threshold=0.0,
            direction="GT",
            information_gain=1.0,
            train_support=8,
            positive_partition=(f"{label}-positive",),
            formula=f"(1)*{x} + (1)*{z}",
            coefficients=((x, 1.0), (z, 1.0)),
            bias=0.0,
            status="PROPOSAL_ONLY",
        )

    def endpoints(self, ax, left, right):
        return ExperimentGenesisEngine(
            projection_margin_multipliers=(float(left), float(right)),
            max_proposals=64,
        ).propose(ax, {ax.inputs[0]: 0.0, ax.inputs[1]: 0.0})

    def execute(self, body, proposals, target, context, epoch_base):
        for proposal_index, proposal in enumerate(proposals):
            body.memory.remember_experiment(proposal)
            for issuer_index, (issuer, signer) in enumerate(self.signers.items()):
                pair = body.execute_world_intervention(
                    proposal,
                    ExactPrimitiveWorld(
                        target,
                        signer,
                        f"{context}-{epoch_base}-source-{proposal_index}-{issuer}",
                        f"{context}-{epoch_base}-challenge-{proposal_index}-{issuer}",
                        context,
                        epoch_base + proposal_index * 10 + issuer_index,
                    ),
                    verifier=self.verifier,
                )
                self.assertTrue(pair.authority_verified)

    @staticmethod
    def scale_of(proposal):
        return ExactPrimitiveWorld.scale_of(proposal)

    def capability(self, body, context, target):
        scores = projection_scale_scores(
            (record.proposal for record in body.memory.experiments.values()),
            body.world_coupling.pairs,
            body.world_coupling.min_independent_classes,
            self.scale_of,
            context_id=context,
        )
        return float(scores.get(round(float(target), 12), 0.0) >= 0.9)

    def assert_outside_old_depth4(self, hidden, left, right):
        target = hidden.apply(left, right)
        old_depth4 = generate_projection_transform_programs(
            max_transform_depth=4,
            signature_anchors=DEEP_TRANSFORM_SIGNATURE_ANCHORS,
        )
        values = {
            program.apply(left, right)
            for program in old_depth4
            if program.apply(left, right) is not None
        }
        self.assertTrue(all(abs(float(value) - target) > 1e-9 for value in values))

    def falsify_old_alphabet(self, body, brackets, hidden, epoch_base=1000):
        for index, (context, (left, right)) in enumerate(brackets.items()):
            self.assert_outside_old_depth4(hidden, left, right)
            ax = self.axis(context)
            body.memory.remember_representation(ax)
            target = hidden.apply(left, right)
            self.assertIsNotNone(target)
            self.execute(body, self.endpoints(ax, left, right), target, context, epoch_base + index * 10000)
            generated = body.generate_projection_transform_adaptive_interventions(
                ax,
                {ax.inputs[0]: 0.0, ax.inputs[1]: 0.0},
                context,
                left,
                right,
                brackets,
                current_depth=3,
                next_depth=4,
                max_candidates=128,
                allow_depth_expansion=False,
                apply_learned_program=False,
            )
            self.execute(body, generated, target, context, epoch_base + 1000 + index * 10000)
        assessment = body.projection_transform_depth_assessment(
            brackets, current_depth=3, next_depth=4
        )
        self.assertEqual(assessment.status, "TRANSFORM_GRAMMAR_DEPTH_FALSIFIED_OPEN_NEXT")
        self.assertEqual(assessment.authorized_depth, 4)
        self.assertFalse(any(item.missing_program_ids for item in assessment.context_assessments))
        return assessment

    def train_primitive(self, body, failure_brackets, hidden, contexts, epoch_base):
        organ = TransformPrimitiveAlphabetOrgan(body)
        for index, (context, left, right) in enumerate(contexts):
            self.assert_outside_old_depth4(hidden, left, right)
            ax = self.axis(context)
            body.memory.remember_representation(ax)
            target = hidden.apply(left, right)
            frontier = organ.frontier(
                context, left, right, failure_brackets,
                current_depth=3, max_candidates=64,
                apply_learned_primitive=False,
            )
            self.assertEqual(frontier.status, "SHADOW_TRANSFORM_PRIMITIVE_GENESIS")
            self.assertIn(round(float(target), 12), {item.scale for item in frontier.candidates})
            generated = organ.generate_interventions(
                ax,
                {ax.inputs[0]: 0.0, ax.inputs[1]: 0.0},
                context, left, right, failure_brackets,
                current_depth=3, max_candidates=64,
                apply_learned_primitive=False,
            )
            self.execute(body, generated, target, context, epoch_base + index * 10000)
        return organ.policy()

    def test_world_failure_opens_new_power_primitive_and_descendant_needs_it(self):
        # These geometries are chosen without outcome evidence and are explicitly
        # checked to keep the generated POWER target outside old LOG/INV depth 4.
        failure_brackets = {"alphabet-f1": (1.7, 5.61), "alphabet-f2": (2.3, 7.59)}
        body = PersistentCognitiveRuntime()
        self.falsify_old_alphabet(body, failure_brackets, self.hidden)

        training = (("primitive-t1", 3.7, 12.21), ("primitive-t2", 5.2, 17.16))
        learned = self.train_primitive(body, failure_brackets, self.hidden, training, 30000)
        self.assertEqual(learned.status, "REPRODUCED_TRANSFORM_PRIMITIVE")
        self.assertEqual(learned.exponent, 0.5)
        self.assertEqual(learned.alpha, 0.25)
        self.assertEqual(len(learned.supporting_contexts), 2)

        checkpoint = checkpoint_dict(body)
        verifierless = restore_runtime(checkpoint)
        verifierless_organ = TransformPrimitiveAlphabetOrgan(verifierless)
        self.assertIsNone(verifierless_organ.policy().primitive_id)
        blocked = verifierless_organ.frontier(
            "verifierless", 7.1, 23.43, failure_brackets, current_depth=3
        )
        self.assertEqual(blocked.status, "CURRENT_TRANSFORM_ALPHABET_NOT_EXHAUSTIVELY_FALSIFIED")

        treatment = restore_runtime(checkpoint, world_verifier=self.verifier)
        remove = restore_runtime(checkpoint, world_verifier=self.verifier)
        heldout_context = "primitive-heldout"
        heldout_left, heldout_right = 7.1, 23.43
        heldout_target = self.hidden.apply(heldout_left, heldout_right)
        heldout_axis = self.axis(heldout_context)
        self.assert_outside_old_depth4(self.hidden, heldout_left, heldout_right)

        for candidate_body in (treatment, remove):
            candidate_body.memory.remember_representation(heldout_axis)
            self.execute(
                candidate_body,
                self.endpoints(heldout_axis, heldout_left, heldout_right),
                heldout_target,
                heldout_context,
                70000,
            )
            self.assertEqual(self.capability(candidate_body, heldout_context, heldout_target), 0.0)

        treatment_organ = TransformPrimitiveAlphabetOrgan(treatment)
        treatment_frontier = treatment_organ.frontier(
            heldout_context, heldout_left, heldout_right, failure_brackets,
            current_depth=3, max_candidates=1, apply_learned_primitive=True,
        )
        self.assertEqual(treatment_frontier.status, "LEARNED_TRANSFORM_PRIMITIVE_TRANSFER")
        self.assertEqual(len(treatment_frontier.candidates), 1)
        self.assertAlmostEqual(treatment_frontier.candidates[0].scale, heldout_target, places=9)
        generated = treatment_organ.generate_interventions(
            heldout_axis,
            {heldout_axis.inputs[0]: 0.0, heldout_axis.inputs[1]: 0.0},
            heldout_context, heldout_left, heldout_right, failure_brackets,
            current_depth=3, max_candidates=1, apply_learned_primitive=True,
        )
        self.execute(treatment, generated, heldout_target, heldout_context, 80000)
        self.assertEqual(self.capability(treatment, heldout_context, heldout_target), 1.0)

        remove_organ = TransformPrimitiveAlphabetOrgan(remove)
        remove_frontier = remove_organ.frontier(
            heldout_context, heldout_left, heldout_right, failure_brackets,
            current_depth=3, max_candidates=1, apply_learned_primitive=False,
        )
        self.assertEqual(remove_frontier.status, "SHADOW_TRANSFORM_PRIMITIVE_GENESIS")
        self.assertNotAlmostEqual(remove_frontier.candidates[0].scale, heldout_target, places=9)
        remove_generated = remove_organ.generate_interventions(
            heldout_axis,
            {heldout_axis.inputs[0]: 0.0, heldout_axis.inputs[1]: 0.0},
            heldout_context, heldout_left, heldout_right, failure_brackets,
            current_depth=3, max_candidates=1, apply_learned_primitive=False,
        )
        self.execute(remove, remove_generated, heldout_target, heldout_context, 80000)
        self.assertEqual(self.capability(remove, heldout_context, heldout_target), 0.0)

        # WRONG is a separately developed BODY, not an injected policy.
        wrong_hidden = next(
            item for item in self.programs if item.exponent == 2.0 and item.alpha == 0.25
        )
        wrong = PersistentCognitiveRuntime()
        wrong_failure = {"wrong-f1": (1.7, 5.61), "wrong-f2": (2.3, 7.59)}
        self.falsify_old_alphabet(wrong, wrong_failure, wrong_hidden, epoch_base=100000)
        wrong_policy = self.train_primitive(
            wrong, wrong_failure, wrong_hidden,
            (("wrong-t1", 3.7, 12.21), ("wrong-t2", 5.2, 17.16)),
            130000,
        )
        self.assertEqual(wrong_policy.exponent, 2.0)
        wrong = restore_runtime(checkpoint_dict(wrong), world_verifier=self.verifier)
        wrong_context = "wrong-heldout"
        wrong_axis = self.axis(wrong_context)
        wrong.memory.remember_representation(wrong_axis)
        self.execute(wrong, self.endpoints(wrong_axis, heldout_left, heldout_right), heldout_target, wrong_context, 170000)
        wrong_organ = TransformPrimitiveAlphabetOrgan(wrong)
        wrong_generated = wrong_organ.generate_interventions(
            wrong_axis,
            {wrong_axis.inputs[0]: 0.0, wrong_axis.inputs[1]: 0.0},
            wrong_context, heldout_left, heldout_right, wrong_failure,
            current_depth=3, max_candidates=1, apply_learned_primitive=True,
        )
        self.assertNotAlmostEqual(self.scale_of(wrong_generated[0]), heldout_target, places=9)
        self.execute(wrong, wrong_generated, heldout_target, wrong_context, 180000)
        self.assertEqual(self.capability(wrong, wrong_context, heldout_target), 0.0)


if __name__ == "__main__":
    unittest.main()
