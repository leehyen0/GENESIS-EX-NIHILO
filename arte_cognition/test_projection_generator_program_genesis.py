from __future__ import annotations

import math
import unittest

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.experiment_genesis import ExperimentGenesisEngine
from arte_cognition.projection_scale_genesis import projection_scale_scores
from arte_cognition.representation_genesis import RepresentationAxis
from arte_cognition.world_coupling import (
    HMACWorldReceiptSigner,
    HMACWorldReceiptVerifier,
    WorldOutcomeReceipt,
)


AUTHORED = (1.0, 2.0, 4.0)


def axis(label: str) -> RepresentationAxis:
    a = f"{label}-x"
    b = f"{label}-z"
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


def scale_of(proposal):
    marker = "probe_scale="
    return float(str(proposal.reason).split(marker, 1)[1].split()[0].rstrip(",;)") )


class ExactScaleWorld:
    def __init__(self, target_scale, signer, source_id, challenge_id, context_id, epoch):
        self.target_scale = float(target_scale)
        self.signer = signer
        self.source_id = str(source_id)
        self.challenge_id = str(challenge_id)
        self.context_id = str(context_id)
        self.epoch = int(epoch)

    def execute(self, proposal, arm, value):
        scale = scale_of(proposal)
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


class ProjectionGeneratorProgramGenesisTests(unittest.TestCase):
    def setUp(self):
        self.keys = {"issuer-a": b"program-key-a", "issuer-b": b"program-key-b"}
        self.signers = {
            issuer: HMACWorldReceiptSigner(issuer, secret)
            for issuer, secret in self.keys.items()
        }
        self.verifier = HMACWorldReceiptVerifier(
            self.keys,
            independence_classes={"issuer-a": "A", "issuer-b": "B"},
        )

    def _proposals(self, ax, scales):
        engine = ExperimentGenesisEngine(
            projection_margin_multipliers=tuple(scales),
            max_proposals=128,
        )
        return engine.propose(ax, {ax.inputs[0]: 0.0, ax.inputs[1]: 0.0})

    def _execute(self, runtime, proposals, target, context, epoch_base):
        for proposal_index, proposal in enumerate(proposals):
            runtime.memory.remember_experiment(proposal)
            for issuer_index, (issuer, signer) in enumerate(self.signers.items()):
                pair = runtime.execute_world_intervention(
                    proposal,
                    ExactScaleWorld(
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

    def _capability(self, runtime, context, target):
        scores = projection_scale_scores(
            (record.proposal for record in runtime.memory.experiments.values()),
            runtime.world_coupling.pairs,
            runtime.world_coupling.min_independent_classes,
            scale_of,
            context_id=context,
        )
        return float(scores.get(round(float(target), 12), 0.0) >= 0.9)

    def _train_program(self, runtime, family: str):
        if family == "GEOMETRIC":
            training = (
                ("program-train-a", 1.0, 2.0, math.pow(2.0, 0.25), 1000),
                ("program-train-b", 2.0, 4.0, 2.0 * math.pow(2.0, 0.25), 3000),
            )
        elif family == "AFFINE":
            training = (
                ("wrong-train-a", 1.0, 2.0, 1.25, 13000),
                ("wrong-train-b", 2.0, 4.0, 2.5, 15000),
            )
        else:
            raise AssertionError(f"unsupported test family {family}")

        for context, left, right, target, epoch in training:
            ax = axis(context)
            runtime.memory.remember_representation(ax)
            base = self._proposals(ax, AUTHORED)
            self._execute(runtime, base, target, context, epoch)
            frontier = runtime.projection_generator_program_frontier(
                context, left, right, max_candidates=32
            )
            self.assertEqual(frontier.status, "SHADOW_GENERATOR_PROGRAM_GENESIS")
            self.assertEqual(frontier.shadow_program_count, 9)
            candidate_scales = {candidate.scale for candidate in frontier.candidates}
            self.assertIn(round(target, 12), candidate_scales)
            generated = runtime.generate_projection_generator_program_interventions(
                ax,
                {ax.inputs[0]: 0.0, ax.inputs[1]: 0.0},
                context,
                left,
                right,
                max_candidates=32,
            )
            self._execute(runtime, generated, target, context, epoch + 1000)
        return runtime.projection_generator_program_policy()

    def test_geometric_generator_program_transfers_across_unseen_interval_geometry(self):
        parent = PersistentCognitiveRuntime()
        learned = self._train_program(parent, "GEOMETRIC")
        self.assertEqual(learned.status, "REPRODUCED_GENERATOR_PROGRAM")
        self.assertEqual(learned.family, "GEOMETRIC")
        self.assertEqual(learned.alpha, 0.25)
        self.assertEqual(len(learned.supporting_contexts), 2)

        checkpoint = checkpoint_dict(parent)
        verifierless = restore_runtime(checkpoint)
        self.assertIsNone(verifierless.projection_generator_program_policy().program_id)

        # Treatment and REMOVE begin from the exact same trained checkpoint and are
        # both reverified by the same external verifier. Fresh base evidence is also
        # identical. The only causal difference is learned-program application.
        treatment = restore_runtime(checkpoint, world_verifier=self.verifier)
        remove = restore_runtime(checkpoint, world_verifier=self.verifier)
        treatment_policy = treatment.projection_generator_program_policy()
        remove_policy = remove.projection_generator_program_policy()
        self.assertEqual(treatment_policy.program_id, remove_policy.program_id)
        self.assertEqual(treatment_policy.family, "GEOMETRIC")

        heldout_context = "program-heldout-1-to-4"
        heldout_target = math.sqrt(2.0)
        treatment_axis = axis(heldout_context)
        remove_axis = axis(heldout_context)
        treatment.memory.remember_representation(treatment_axis)
        remove.memory.remember_representation(remove_axis)
        base = self._proposals(treatment_axis, AUTHORED)
        self._execute(treatment, base, heldout_target, heldout_context, 7000)
        self._execute(remove, base, heldout_target, heldout_context, 7000)
        self.assertEqual(self._capability(treatment, heldout_context, heldout_target), 0.0)
        self.assertEqual(self._capability(remove, heldout_context, heldout_target), 0.0)

        treatment_frontier = treatment.projection_generator_program_frontier(
            heldout_context, 1.0, 4.0, max_candidates=1, apply_learned_program=True
        )
        remove_frontier = remove.projection_generator_program_frontier(
            heldout_context, 1.0, 4.0, max_candidates=1, apply_learned_program=False
        )
        self.assertEqual(treatment_frontier.status, "LEARNED_GENERATOR_PROGRAM_TRANSFER")
        self.assertEqual(len(treatment_frontier.candidates), 1)
        self.assertAlmostEqual(treatment_frontier.candidates[0].scale, heldout_target, places=9)
        self.assertEqual(len(remove_frontier.candidates), 1)
        self.assertNotAlmostEqual(remove_frontier.candidates[0].scale, heldout_target, places=9)

        treatment_generated = treatment.generate_projection_generator_program_interventions(
            treatment_axis,
            {treatment_axis.inputs[0]: 0.0, treatment_axis.inputs[1]: 0.0},
            heldout_context,
            1.0,
            4.0,
            max_candidates=1,
            apply_learned_program=True,
        )
        remove_generated = remove.generate_projection_generator_program_interventions(
            remove_axis,
            {remove_axis.inputs[0]: 0.0, remove_axis.inputs[1]: 0.0},
            heldout_context,
            1.0,
            4.0,
            max_candidates=1,
            apply_learned_program=False,
        )
        self.assertEqual(len(treatment_generated), len(remove_generated))
        self._execute(treatment, treatment_generated, heldout_target, heldout_context, 8000)
        self._execute(remove, remove_generated, heldout_target, heldout_context, 8000)
        self.assertEqual(self._capability(treatment, heldout_context, heldout_target), 1.0)
        self.assertEqual(self._capability(remove, heldout_context, heldout_target), 0.0)

        # RESET has the same one-candidate refinement budget but no inherited BODY
        # evidence or learned program; it must not solve the heldout target.
        reset = PersistentCognitiveRuntime()
        reset_context = "program-reset-heldout"
        reset_axis = axis(reset_context)
        reset.memory.remember_representation(reset_axis)
        reset_base = self._proposals(reset_axis, AUTHORED)
        self._execute(reset, reset_base, heldout_target, reset_context, 9000)
        reset_generated = reset.generate_projection_generator_program_interventions(
            reset_axis,
            {reset_axis.inputs[0]: 0.0, reset_axis.inputs[1]: 0.0},
            reset_context,
            1.0,
            4.0,
            max_candidates=1,
        )
        self.assertEqual(len(reset_generated), len(treatment_generated))
        self._execute(reset, reset_generated, heldout_target, reset_context, 10000)
        self.assertEqual(self._capability(reset, reset_context, heldout_target), 0.0)

        # WRONG-SWAP is not injected. A separate BODY actually learns AFFINE,.25
        # from two independent contexts, then receives the same one-candidate fresh
        # budget. Its transferred program predicts 1.75 instead of sqrt(2), so it
        # must fail despite having a valid learned generator program of its own.
        wrong = PersistentCognitiveRuntime()
        wrong_policy = self._train_program(wrong, "AFFINE")
        self.assertEqual(wrong_policy.status, "REPRODUCED_GENERATOR_PROGRAM")
        self.assertEqual(wrong_policy.family, "AFFINE")
        self.assertEqual(wrong_policy.alpha, 0.25)
        wrong_context = "program-wrong-heldout"
        wrong_axis = axis(wrong_context)
        wrong.memory.remember_representation(wrong_axis)
        wrong_base = self._proposals(wrong_axis, AUTHORED)
        self._execute(wrong, wrong_base, heldout_target, wrong_context, 17000)
        wrong_frontier = wrong.projection_generator_program_frontier(
            wrong_context, 1.0, 4.0, max_candidates=1
        )
        self.assertEqual(wrong_frontier.status, "LEARNED_GENERATOR_PROGRAM_TRANSFER")
        self.assertEqual(wrong_frontier.candidates[0].scale, 1.75)
        wrong_generated = wrong.generate_projection_generator_program_interventions(
            wrong_axis,
            {wrong_axis.inputs[0]: 0.0, wrong_axis.inputs[1]: 0.0},
            wrong_context,
            1.0,
            4.0,
            max_candidates=1,
        )
        self.assertEqual(len(wrong_generated), len(treatment_generated))
        self._execute(wrong, wrong_generated, heldout_target, wrong_context, 18000)
        self.assertEqual(self._capability(wrong, wrong_context, heldout_target), 0.0)


if __name__ == "__main__":
    unittest.main()
