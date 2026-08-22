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
from arte_cognition.representation_genesis import RepresentationAxis
from arte_cognition.world_coupling import (
    HMACWorldReceiptSigner,
    HMACWorldReceiptVerifier,
    WorldOutcomeReceipt,
)


class ExactDepthWorld:
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


class ProjectionTransformDepthGenesisTests(unittest.TestCase):
    def setUp(self):
        self.keys = {"depth-a": b"depth-key-a", "depth-b": b"depth-key-b"}
        self.signers = {
            issuer: HMACWorldReceiptSigner(issuer, secret)
            for issuer, secret in self.keys.items()
        }
        self.verifier = HMACWorldReceiptVerifier(
            self.keys,
            independence_classes={"depth-a": "A", "depth-b": "B"},
        )
        programs = generate_projection_transform_programs(
            max_transform_depth=3,
            signature_anchors=DEEP_TRANSFORM_SIGNATURE_ANCHORS,
        )
        self.hidden = next(
            program for program in programs
            if program.operations == ("LOG", "LOG", "LOG") and program.alpha == 0.25
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

    def endpoint_proposals(self, axis, left, right):
        return ExperimentGenesisEngine(
            projection_margin_multipliers=(float(left), float(right)),
            max_proposals=64,
        ).propose(axis, {axis.inputs[0]: 0.0, axis.inputs[1]: 0.0})

    def execute(self, runtime, proposals, target, context, epoch_base):
        for proposal_index, proposal in enumerate(proposals):
            runtime.memory.remember_experiment(proposal)
            for issuer_index, (issuer, signer) in enumerate(self.signers.items()):
                pair = runtime.execute_world_intervention(
                    proposal,
                    ExactDepthWorld(
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
        marker = "probe_scale="
        return float(str(proposal.reason).split(marker, 1)[1].split()[0].rstrip(",;)") )

    def capability(self, runtime, context, target):
        scores = projection_scale_scores(
            (record.proposal for record in runtime.memory.experiments.values()),
            runtime.world_coupling.pairs,
            runtime.world_coupling.min_independent_classes,
            self.scale_of,
            context_id=context,
        )
        return float(scores.get(round(float(target), 12), 0.0) >= 0.9)

    def evaluate_current_depth(self, runtime, context, left, right, depth_brackets, epoch, omit_last_scale=False):
        ax = self.axis(context)
        runtime.memory.remember_representation(ax)
        target = self.hidden.apply(left, right)
        self.assertIsNotNone(target)
        self.execute(runtime, self.endpoint_proposals(ax, left, right), target, context, epoch)
        frontier = runtime.projection_transform_adaptive_frontier(
            context,
            left,
            right,
            depth_brackets,
            max_candidates=64,
            allow_depth_expansion=False,
            apply_learned_program=False,
        )
        self.assertEqual(frontier.status, "SHADOW_TRANSFORM_PROGRAM_GENESIS")
        generated = runtime.generate_projection_transform_adaptive_interventions(
            ax,
            {ax.inputs[0]: 0.0, ax.inputs[1]: 0.0},
            context,
            left,
            right,
            depth_brackets,
            max_candidates=64,
            allow_depth_expansion=False,
            apply_learned_program=False,
        )
        if omit_last_scale:
            omitted = frontier.candidates[-1].scale
            generated = [p for p in generated if abs(self.scale_of(p) - omitted) > 1e-12]
        self.execute(runtime, generated, target, context, epoch + 1000)
        return target

    def test_missing_current_depth_candidate_cannot_authorize_expansion(self):
        runtime = PersistentCognitiveRuntime()
        brackets = {"f1": (32.0, 4096.0), "f2": (64.0, 65536.0)}
        self.evaluate_current_depth(runtime, "f1", *brackets["f1"], brackets, 1000)
        self.evaluate_current_depth(
            runtime, "f2", *brackets["f2"], brackets, 5000, omit_last_scale=True
        )
        assessment = runtime.projection_transform_depth_assessment(brackets)
        self.assertEqual(assessment.status, "TRANSFORM_DEPTH_EVIDENCE_INCOMPLETE")
        self.assertEqual(assessment.authorized_depth, 2)
        self.assertIn("f2", assessment.incomplete_contexts)
        f2 = next(item for item in assessment.context_assessments if item.context_id == "f2")
        self.assertGreater(len(f2.missing_program_ids), 0)

    def test_exhaustive_depth2_failure_opens_depth3_and_causes_fresh_capability(self):
        runtime = PersistentCognitiveRuntime()
        depth_brackets = {"f1": (32.0, 4096.0), "f2": (64.0, 65536.0)}
        self.evaluate_current_depth(runtime, "f1", *depth_brackets["f1"], depth_brackets, 1000)
        before = runtime.projection_transform_depth_assessment(depth_brackets)
        self.assertEqual(before.authorized_depth, 2)
        self.evaluate_current_depth(runtime, "f2", *depth_brackets["f2"], depth_brackets, 5000)
        opened = runtime.projection_transform_depth_assessment(depth_brackets)
        self.assertEqual(opened.status, "TRANSFORM_GRAMMAR_DEPTH_FALSIFIED_OPEN_NEXT")
        self.assertEqual(opened.authorized_depth, 3)
        self.assertEqual(set(opened.falsified_contexts), {"f1", "f2"})

        # Once depth 3 is world-authorized, two new contexts train the newly
        # reachable LOG>LOG>LOG program. Candidate generation still precedes outcomes.
        training = (("t1", 32.0, 32768.0, 12000), ("t2", 64.0, 1048576.0, 20000))
        for context, left, right, epoch in training:
            ax = self.axis(context)
            runtime.memory.remember_representation(ax)
            target = self.hidden.apply(left, right)
            self.execute(runtime, self.endpoint_proposals(ax, left, right), target, context, epoch)
            frontier = runtime.projection_transform_adaptive_frontier(
                context, left, right, depth_brackets,
                max_candidates=64,
                allow_depth_expansion=True,
                apply_learned_program=False,
            )
            self.assertEqual(frontier.status, "SHADOW_TRANSFORM_PROGRAM_GENESIS")
            target_candidates = [
                c for c in frontier.candidates if abs(c.scale - target) <= 1e-9
            ]
            self.assertEqual(len(target_candidates), 1)
            generated = runtime.generate_projection_transform_adaptive_interventions(
                ax,
                {ax.inputs[0]: 0.0, ax.inputs[1]: 0.0},
                context,
                left,
                right,
                depth_brackets,
                max_candidates=64,
                allow_depth_expansion=True,
                apply_learned_program=False,
            )
            self.execute(runtime, generated, target, context, epoch + 1000)

        policy = runtime.projection_transform_adaptive_policy(depth_brackets)
        self.assertEqual(policy.status, "REPRODUCED_TRANSFORM_PROGRAM")
        self.assertEqual(policy.operations, ("LOG", "LOG", "LOG"))
        self.assertEqual(policy.alpha, 0.25)

        checkpoint = checkpoint_dict(runtime)
        verifierless = restore_runtime(checkpoint)
        self.assertEqual(
            verifierless.projection_transform_depth_assessment(depth_brackets).authorized_depth, 2
        )
        self.assertIsNone(
            verifierless.projection_transform_adaptive_policy(depth_brackets).program_id
        )

        treatment = restore_runtime(checkpoint, world_verifier=self.verifier)
        remove_depth = restore_runtime(checkpoint, world_verifier=self.verifier)
        remove_policy = restore_runtime(checkpoint, world_verifier=self.verifier)
        heldout_context = "heldout"
        heldout_left, heldout_right = 32.0, 32768.0
        heldout_target = self.hidden.apply(heldout_left, heldout_right)
        heldout_axis = self.axis(heldout_context)

        for body in (treatment, remove_depth, remove_policy):
            body.memory.remember_representation(heldout_axis)
            self.execute(
                body,
                self.endpoint_proposals(heldout_axis, heldout_left, heldout_right),
                heldout_target,
                heldout_context,
                30000,
            )
            self.assertEqual(self.capability(body, heldout_context, heldout_target), 0.0)

        treatment_generated = treatment.generate_projection_transform_adaptive_interventions(
            heldout_axis,
            {heldout_axis.inputs[0]: 0.0, heldout_axis.inputs[1]: 0.0},
            heldout_context,
            heldout_left,
            heldout_right,
            depth_brackets,
            max_candidates=1,
            allow_depth_expansion=True,
            apply_learned_program=True,
        )
        self.execute(treatment, treatment_generated, heldout_target, heldout_context, 40000)
        self.assertEqual(self.capability(treatment, heldout_context, heldout_target), 1.0)

        remove_depth_generated = remove_depth.generate_projection_transform_adaptive_interventions(
            heldout_axis,
            {heldout_axis.inputs[0]: 0.0, heldout_axis.inputs[1]: 0.0},
            heldout_context,
            heldout_left,
            heldout_right,
            depth_brackets,
            max_candidates=1,
            allow_depth_expansion=False,
            apply_learned_program=True,
        )
        self.assertTrue(remove_depth_generated)
        self.assertNotAlmostEqual(self.scale_of(remove_depth_generated[0]), heldout_target, places=9)
        self.execute(remove_depth, remove_depth_generated, heldout_target, heldout_context, 40000)
        self.assertEqual(self.capability(remove_depth, heldout_context, heldout_target), 0.0)

        remove_policy_generated = remove_policy.generate_projection_transform_adaptive_interventions(
            heldout_axis,
            {heldout_axis.inputs[0]: 0.0, heldout_axis.inputs[1]: 0.0},
            heldout_context,
            heldout_left,
            heldout_right,
            depth_brackets,
            max_candidates=1,
            allow_depth_expansion=True,
            apply_learned_program=False,
        )
        self.assertTrue(remove_policy_generated)
        self.assertNotAlmostEqual(self.scale_of(remove_policy_generated[0]), heldout_target, places=9)
        self.execute(remove_policy, remove_policy_generated, heldout_target, heldout_context, 40000)
        self.assertEqual(self.capability(remove_policy, heldout_context, heldout_target), 0.0)


if __name__ == "__main__":
    unittest.main()
