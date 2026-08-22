from __future__ import annotations

import unittest

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.experiment_genesis import ExperimentGenesisEngine
from arte_cognition.projection_generator_transform_grammar import (
    ProjectionTransformProgram,
    generate_projection_transform_programs,
)
from arte_cognition.projection_scale_genesis import projection_scale_scores
from arte_cognition.representation_genesis import RepresentationAxis
from arte_cognition.world_coupling import (
    HMACWorldReceiptSigner,
    HMACWorldReceiptVerifier,
    WorldOutcomeReceipt,
)


class ExactTransformScaleWorld:
    def __init__(self, target_scale, signer, source_id, challenge_id, context_id, epoch):
        self.target_scale = float(target_scale)
        self.signer = signer
        self.source_id = str(source_id)
        self.challenge_id = str(challenge_id)
        self.context_id = str(context_id)
        self.epoch = int(epoch)

    @staticmethod
    def _scale(proposal):
        marker = "probe_scale="
        return float(str(proposal.reason).split(marker, 1)[1].split()[0].rstrip(",;)") )

    def execute(self, proposal, arm, value):
        scale = self._scale(proposal)
        high = 1.0 if abs(scale - self.target_scale) <= 1e-9 else 0.25
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


class ProjectionTransformGrammarTests(unittest.TestCase):
    def setUp(self):
        self.keys = {"issuer-a": b"transform-a", "issuer-b": b"transform-b"}
        self.signers = {
            issuer: HMACWorldReceiptSigner(issuer, secret)
            for issuer, secret in self.keys.items()
        }
        self.verifier = HMACWorldReceiptVerifier(
            self.keys,
            independence_classes={"issuer-a": "A", "issuer-b": "B"},
        )
        self.programs = generate_projection_transform_programs()

    @staticmethod
    def _axis(label):
        x = f"{label}-x"
        z = f"{label}-z"
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

    def _program(self, operations, alpha):
        for program in self.programs:
            if program.operations == tuple(operations) and abs(program.alpha - float(alpha)) <= 1e-12:
                return program
        raise AssertionError(f"program not generated: {operations}, {alpha}")

    def _endpoint_proposals(self, axis, left, right):
        engine = ExperimentGenesisEngine(
            projection_margin_multipliers=(float(left), float(right)),
            max_proposals=64,
        )
        return engine.propose(axis, {axis.inputs[0]: 0.0, axis.inputs[1]: 0.0})

    def _execute(self, runtime, proposals, target, context, epoch_base):
        for proposal_index, proposal in enumerate(proposals):
            runtime.memory.remember_experiment(proposal)
            for issuer_index, (issuer, signer) in enumerate(self.signers.items()):
                pair = runtime.execute_world_intervention(
                    proposal,
                    ExactTransformScaleWorld(
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
    def _scale_of(proposal):
        marker = "probe_scale="
        return float(str(proposal.reason).split(marker, 1)[1].split()[0].rstrip(",;)") )

    def _capability(self, runtime, context, target):
        scores = projection_scale_scores(
            (record.proposal for record in runtime.memory.experiments.values()),
            runtime.world_coupling.pairs,
            runtime.world_coupling.min_independent_classes,
            self._scale_of,
            context_id=context,
        )
        return float(scores.get(round(float(target), 12), 0.0) >= 0.9)

    def _train_program(self, runtime, operations, alpha, training, epoch_offset=0):
        hidden_program = self._program(operations, alpha)
        for index, (context, left, right) in enumerate(training):
            axis = self._axis(context)
            runtime.memory.remember_representation(axis)
            target = hidden_program.apply(left, right)
            self.assertIsNotNone(target)
            endpoints = self._endpoint_proposals(axis, left, right)
            self._execute(runtime, endpoints, target, context, epoch_offset + 1000 + index * 3000)
            frontier = runtime.projection_transform_frontier(
                context, left, right, max_candidates=64, apply_learned_program=False
            )
            self.assertEqual(frontier.status, "SHADOW_TRANSFORM_PROGRAM_GENESIS")
            self.assertGreater(frontier.shadow_program_count, 3)
            self.assertIn(round(float(target), 12), {candidate.scale for candidate in frontier.candidates})
            generated = runtime.generate_projection_transform_interventions(
                axis,
                {axis.inputs[0]: 0.0, axis.inputs[1]: 0.0},
                context,
                left,
                right,
                max_candidates=64,
                apply_learned_program=False,
            )
            self._execute(runtime, generated, target, context, epoch_offset + 2000 + index * 3000)
        return hidden_program

    def test_log_log_ast_is_learned_reverified_transferred_and_causally_required(self):
        training = (
            ("ast-train-a", 2.0, 16.0),
            ("ast-train-b", 4.0, 256.0),
        )
        parent = PersistentCognitiveRuntime()
        hidden_program = self._train_program(parent, ("LOG", "LOG"), 0.25, training)

        learned = parent.projection_transform_program_policy()
        self.assertEqual(learned.status, "REPRODUCED_TRANSFORM_PROGRAM")
        self.assertEqual(learned.operations, ("LOG", "LOG"))
        self.assertEqual(learned.alpha, 0.25)
        self.assertEqual(len(learned.supporting_contexts), 2)

        checkpoint = checkpoint_dict(parent)
        verifierless = restore_runtime(checkpoint)
        self.assertIsNone(verifierless.projection_transform_program_policy().program_id)

        # Treatment and REMOVE start from the exact same trained checkpoint and
        # receive exactly the same fresh authored endpoint evidence. The only
        # difference is whether the reverified learned AST is applied.
        treatment = restore_runtime(checkpoint, world_verifier=self.verifier)
        remove = restore_runtime(checkpoint, world_verifier=self.verifier)
        heldout_context = "ast-heldout-3-to-243"
        heldout_axis = self._axis(heldout_context)
        heldout_target = hidden_program.apply(3.0, 243.0)
        self.assertIsNotNone(heldout_target)

        for runtime in (treatment, remove):
            runtime.memory.remember_representation(heldout_axis)
            endpoints = self._endpoint_proposals(heldout_axis, 3.0, 243.0)
            self._execute(runtime, endpoints, heldout_target, heldout_context, 20000)
            self.assertEqual(self._capability(runtime, heldout_context, heldout_target), 0.0)

        treatment_frontier = treatment.projection_transform_frontier(
            heldout_context, 3.0, 243.0, max_candidates=1, apply_learned_program=True
        )
        self.assertEqual(treatment_frontier.status, "LEARNED_TRANSFORM_PROGRAM_TRANSFER")
        self.assertEqual(len(treatment_frontier.candidates), 1)
        self.assertAlmostEqual(treatment_frontier.candidates[0].scale, heldout_target, places=9)
        treatment_generated = treatment.generate_projection_transform_interventions(
            heldout_axis,
            {heldout_axis.inputs[0]: 0.0, heldout_axis.inputs[1]: 0.0},
            heldout_context,
            3.0,
            243.0,
            max_candidates=1,
            apply_learned_program=True,
        )
        self._execute(treatment, treatment_generated, heldout_target, heldout_context, 23000)
        self.assertEqual(self._capability(treatment, heldout_context, heldout_target), 1.0)

        remove_frontier = remove.projection_transform_frontier(
            heldout_context, 3.0, 243.0, max_candidates=1, apply_learned_program=False
        )
        self.assertEqual(remove_frontier.status, "SHADOW_TRANSFORM_PROGRAM_GENESIS")
        self.assertEqual(len(remove_frontier.candidates), 1)
        self.assertNotAlmostEqual(remove_frontier.candidates[0].scale, heldout_target, places=9)
        remove_generated = remove.generate_projection_transform_interventions(
            heldout_axis,
            {heldout_axis.inputs[0]: 0.0, heldout_axis.inputs[1]: 0.0},
            heldout_context,
            3.0,
            243.0,
            max_candidates=1,
            apply_learned_program=False,
        )
        self._execute(remove, remove_generated, heldout_target, heldout_context, 23000)
        self.assertEqual(self._capability(remove, heldout_context, heldout_target), 0.0)

        # WRONG is not injected. A separate BODY really learns LOG(alpha=.25)
        # from two independent training worlds, then receives the same one-candidate
        # fresh-resource budget. Its valid but wrong AST must fail LOG>LOG heldout.
        wrong_parent = PersistentCognitiveRuntime()
        wrong_program = self._train_program(
            wrong_parent,
            ("LOG",),
            0.25,
            (("wrong-train-a", 2.0, 16.0), ("wrong-train-b", 4.0, 256.0)),
            epoch_offset=30000,
        )
        wrong_policy = wrong_parent.projection_transform_program_policy()
        self.assertEqual(wrong_policy.operations, ("LOG",))
        wrong = restore_runtime(checkpoint_dict(wrong_parent), world_verifier=self.verifier)
        wrong_context = "ast-wrong-heldout"
        wrong_axis = self._axis(wrong_context)
        wrong.memory.remember_representation(wrong_axis)
        endpoints = self._endpoint_proposals(wrong_axis, 3.0, 243.0)
        self._execute(wrong, endpoints, heldout_target, wrong_context, 50000)
        wrong_frontier = wrong.projection_transform_frontier(
            wrong_context, 3.0, 243.0, max_candidates=1, apply_learned_program=True
        )
        self.assertEqual(wrong_frontier.status, "LEARNED_TRANSFORM_PROGRAM_TRANSFER")
        self.assertAlmostEqual(wrong_frontier.candidates[0].scale, wrong_program.apply(3.0, 243.0), places=9)
        self.assertNotAlmostEqual(wrong_frontier.candidates[0].scale, heldout_target, places=9)
        wrong_generated = wrong.generate_projection_transform_interventions(
            wrong_axis,
            {wrong_axis.inputs[0]: 0.0, wrong_axis.inputs[1]: 0.0},
            wrong_context,
            3.0,
            243.0,
            max_candidates=1,
            apply_learned_program=True,
        )
        self._execute(wrong, wrong_generated, heldout_target, wrong_context, 53000)
        self.assertEqual(self._capability(wrong, wrong_context, heldout_target), 0.0)

        # This exact target is outside the original three named interpolation
        # families at alpha=.25 on the heldout bracket.
        named = [
            ProjectionTransformProgram("affine", (), 0.25, 1),
            ProjectionTransformProgram("geometric", ("LOG",), 0.25, 2),
            ProjectionTransformProgram("harmonic", ("INV",), 0.25, 2),
        ]
        self.assertTrue(all(abs(program.apply(3.0, 243.0) - heldout_target) > 1e-9 for program in named))


if __name__ == "__main__":
    unittest.main()
