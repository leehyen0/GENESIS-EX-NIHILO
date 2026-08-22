from __future__ import annotations

from dataclasses import replace
import math
import unittest

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.experiment_genesis import ExperimentGenesisEngine
from arte_cognition.projection_generator_program_genesis import (
    PROGRAM_MARKER,
    ProjectionGeneratorProgramPolicy,
    derive_projection_generator_program_frontier,
    derive_projection_generator_program_policy,
    generate_projection_generator_programs,
)
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
        self.programs = generate_projection_generator_programs()

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

    def _program_proposals(self, ax, frontier):
        out = []
        for candidate in frontier.candidates:
            for proposal in self._proposals(ax, (candidate.scale,)):
                reason = f"{proposal.reason} {PROGRAM_MARKER}{'|'.join(candidate.program_ids)}"
                out.append(replace(proposal, reason=reason))
        return out

    def _policy(self, runtime):
        return derive_projection_generator_program_policy(
            (record.proposal for record in runtime.memory.experiments.values()),
            runtime.world_coupling.pairs,
            runtime.world_coupling.min_independent_classes,
            programs=self.programs,
            strong_effect_threshold=0.9,
            min_contexts=2,
        )

    def _frontier(self, runtime, context, left, right, policy=None, max_candidates=32):
        return derive_projection_generator_program_frontier(
            (record.proposal for record in runtime.memory.experiments.values()),
            runtime.world_coupling.pairs,
            runtime.world_coupling.min_independent_classes,
            scale_of,
            context_id=context,
            left=left,
            right=right,
            policy=policy,
            programs=self.programs,
            strong_effect_threshold=0.9,
            max_candidates=max_candidates,
        )

    def _capability(self, runtime, context, target):
        scores = projection_scale_scores(
            (record.proposal for record in runtime.memory.experiments.values()),
            runtime.world_coupling.pairs,
            runtime.world_coupling.min_independent_classes,
            scale_of,
            context_id=context,
        )
        return float(scores.get(round(float(target), 12), 0.0) >= 0.9)

    def test_geometric_generator_program_transfers_across_unseen_interval_geometry(self):
        runtime = PersistentCognitiveRuntime()
        training = (
            ("program-train-a", 1.0, 2.0, math.pow(2.0, 0.25), 1000),
            ("program-train-b", 2.0, 4.0, 2.0 * math.pow(2.0, 0.25), 3000),
        )

        for context, left, right, target, epoch in training:
            ax = axis(context)
            runtime.memory.remember_representation(ax)
            base = self._proposals(ax, AUTHORED)
            self._execute(runtime, base, target, context, epoch)
            frontier = self._frontier(runtime, context, left, right)
            self.assertEqual(frontier.status, "SHADOW_GENERATOR_PROGRAM_GENESIS")
            self.assertEqual(frontier.shadow_program_count, 9)
            candidate_scales = {candidate.scale for candidate in frontier.candidates}
            self.assertIn(round(target, 12), candidate_scales)
            self._execute(runtime, self._program_proposals(ax, frontier), target, context, epoch + 1000)

        learned = self._policy(runtime)
        self.assertEqual(learned.status, "REPRODUCED_GENERATOR_PROGRAM")
        self.assertEqual(learned.family, "GEOMETRIC")
        self.assertEqual(learned.alpha, 0.25)
        self.assertEqual(len(learned.supporting_contexts), 2)

        checkpoint = checkpoint_dict(runtime)
        verifierless = restore_runtime(checkpoint)
        self.assertIsNone(self._policy(verifierless).program_id)
        descendant = restore_runtime(checkpoint, world_verifier=self.verifier)
        descendant_policy = self._policy(descendant)
        self.assertEqual(descendant_policy.family, "GEOMETRIC")
        self.assertEqual(descendant_policy.alpha, 0.25)

        # Heldout interval [1, 4] was never used as a training bracket. Its target
        # is the geometric quarter point sqrt(2), which differs from affine and
        # harmonic quarter points and from every authored scale.
        heldout_context = "program-heldout-1-to-4"
        heldout_axis = axis(heldout_context)
        descendant.memory.remember_representation(heldout_axis)
        heldout_target = math.sqrt(2.0)
        base = self._proposals(heldout_axis, AUTHORED)
        self._execute(descendant, base, heldout_target, heldout_context, 7000)
        self.assertEqual(self._capability(descendant, heldout_context, heldout_target), 0.0)

        treatment_frontier = self._frontier(
            descendant,
            heldout_context,
            1.0,
            4.0,
            policy=descendant_policy,
            max_candidates=1,
        )
        self.assertEqual(treatment_frontier.status, "LEARNED_GENERATOR_PROGRAM_TRANSFER")
        self.assertEqual(len(treatment_frontier.candidates), 1)
        self.assertAlmostEqual(treatment_frontier.candidates[0].scale, heldout_target, places=9)
        treatment = self._program_proposals(heldout_axis, treatment_frontier)
        self._execute(descendant, treatment, heldout_target, heldout_context, 8000)
        self.assertEqual(self._capability(descendant, heldout_context, heldout_target), 1.0)

        # RESET has exactly one refinement candidate but no learned program. The
        # deterministic bounded shadow ordering begins with AFFINE alpha=.25 -> 1.75.
        reset = PersistentCognitiveRuntime()
        reset_context = "program-reset-heldout"
        reset_axis = axis(reset_context)
        reset.memory.remember_representation(reset_axis)
        reset_base = self._proposals(reset_axis, AUTHORED)
        self._execute(reset, reset_base, heldout_target, reset_context, 9000)
        reset_frontier = self._frontier(reset, reset_context, 1.0, 4.0, max_candidates=1)
        self.assertEqual(len(reset_frontier.candidates), 1)
        self.assertEqual(reset_frontier.candidates[0].scale, 1.75)
        self._execute(reset, self._program_proposals(reset_axis, reset_frontier), heldout_target, reset_context, 10000)
        self.assertEqual(self._capability(reset, reset_context, heldout_target), 0.0)

        # WRONG-SWAP receives the same one-candidate resource budget but uses the
        # wrong affine generator. It must fail the fresh target.
        wrong = PersistentCognitiveRuntime()
        wrong_context = "program-wrong-heldout"
        wrong_axis = axis(wrong_context)
        wrong.memory.remember_representation(wrong_axis)
        wrong_base = self._proposals(wrong_axis, AUTHORED)
        self._execute(wrong, wrong_base, heldout_target, wrong_context, 11000)
        wrong_policy = ProjectionGeneratorProgramPolicy(
            status="REPRODUCED_GENERATOR_PROGRAM",
            program_id="GENERATOR::AFFINE::ALPHA::0.25",
            family="AFFINE",
            alpha=0.25,
            supporting_contexts=("wrong-a", "wrong-b"),
            candidate_program_count=9,
            reason="matched wrong-swap",
        )
        wrong_frontier = self._frontier(
            wrong, wrong_context, 1.0, 4.0, policy=wrong_policy, max_candidates=1
        )
        self.assertEqual(wrong_frontier.candidates[0].scale, 1.75)
        self._execute(wrong, self._program_proposals(wrong_axis, wrong_frontier), heldout_target, wrong_context, 12000)
        self.assertEqual(self._capability(wrong, wrong_context, heldout_target), 0.0)


if __name__ == "__main__":
    unittest.main()
