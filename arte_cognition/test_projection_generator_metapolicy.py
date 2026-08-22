from __future__ import annotations

import unittest

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.experiment_genesis import ExperimentGenesisEngine
from arte_cognition.projection_generator_metapolicy import (
    ProjectionGeneratorPolicy,
    derive_projection_generator_frontier,
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


class PeakByProbeScaleWorld:
    """Evaluator-owned world whose hidden optimum is an intervention scale."""

    def __init__(self, target_scale, signer, source_id, challenge_id, context_id, epoch):
        self.target_scale = float(target_scale)
        self.signer = signer
        self.source_id = str(source_id)
        self.challenge_id = str(challenge_id)
        self.context_id = str(context_id)
        self.epoch = int(epoch)

    def execute(self, proposal, arm, value):
        scale = scale_of(proposal)
        high = 1.0 if abs(scale - self.target_scale) <= 1e-12 else 0.25
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


class ProjectionGeneratorMetaPolicyTests(unittest.TestCase):
    def setUp(self):
        self.keys = {"issuer-a": b"generator-key-a", "issuer-b": b"generator-key-b"}
        self.signers = {
            issuer: HMACWorldReceiptSigner(issuer, secret)
            for issuer, secret in self.keys.items()
        }
        self.verifier = HMACWorldReceiptVerifier(
            self.keys,
            independence_classes={"issuer-a": "A", "issuer-b": "B"},
        )

    def _remember_and_execute(self, runtime, proposals, target, context, epoch_base):
        for proposal_index, proposal in enumerate(proposals):
            runtime.memory.remember_experiment(proposal)
            for issuer_index, (issuer, signer) in enumerate(self.signers.items()):
                pair = runtime.execute_world_intervention(
                    proposal,
                    PeakByProbeScaleWorld(
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

    def _proposals(self, ax, scales):
        names = tuple(ax.inputs)
        engine = ExperimentGenesisEngine(
            projection_margin_multipliers=tuple(scales),
            max_proposals=64,
        )
        return engine.propose(ax, {names[0]: 0.0, names[1]: 0.0})

    def _context_capability(self, runtime, context, target):
        scores = projection_scale_scores(
            (record.proposal for record in runtime.memory.experiments.values()),
            runtime.world_coupling.pairs,
            runtime.world_coupling.min_independent_classes,
            scale_of,
            context_id=context,
        )
        return float(scores.get(float(target), 0.0) >= 0.9)

    def test_generator_program_is_learned_then_transferred_to_unseen_bracket(self):
        runtime = PersistentCognitiveRuntime()

        # Two disjoint training contexts share only the latent refinement rule:
        # the strong off-grid atom is 25% into authored bracket [1, 2].
        for index, context in enumerate(("generator-train-a", "generator-train-b")):
            ax = axis(context)
            runtime.memory.remember_representation(ax)
            base = self._proposals(ax, AUTHORED)
            self._remember_and_execute(runtime, base, 1.25, context, 1000 + index * 1000)

            shadow = runtime.projection_generator_frontier(context, max_candidates=16)
            self.assertEqual(shadow.status, "SHADOW_GENERATOR_PROGRAM_SEARCH")
            self.assertEqual(shadow.generator_alphas, (0.25, 0.5, 0.75))
            self.assertIn(1.25, shadow.candidate_scales)
            generated = runtime.generate_projection_generator_interventions(
                ax,
                {ax.inputs[0]: 0.0, ax.inputs[1]: 0.0},
                context,
                max_candidates=16,
            )
            self._remember_and_execute(runtime, generated, 1.25, context, 1500 + index * 1000)

        learned = runtime.projection_generator_policy()
        self.assertEqual(learned.status, "REPRODUCED_GENERATOR_POLICY")
        self.assertEqual(learned.alpha, 0.25)
        self.assertEqual(len(learned.supporting_contexts), 2)

        # Authority must be re-derived after checkpoint. Serialized evidence alone
        # cannot recreate a generator policy without the external verifier.
        payload = checkpoint_dict(runtime)
        verifierless = restore_runtime(payload)
        self.assertIsNone(verifierless.projection_generator_policy().alpha)
        descendant = restore_runtime(payload, world_verifier=self.verifier)
        descendant_policy = descendant.projection_generator_policy()
        self.assertEqual(descendant_policy.alpha, 0.25)

        # Fresh context moves the hidden optimum to a different authored bracket.
        # The previously successful atom 1.25 itself is useless; the learned rule
        # must transfer alpha=.25 to [2, 4], generating the novel 2.5 atom.
        fresh_context = "generator-heldout-new-bracket"
        fresh_axis = axis(fresh_context)
        descendant.memory.remember_representation(fresh_axis)
        base = self._proposals(fresh_axis, AUTHORED)
        self._remember_and_execute(descendant, base, 2.5, fresh_context, 5000)
        self.assertEqual(self._context_capability(descendant, fresh_context, 2.5), 0.0)

        learned_frontier = descendant.projection_generator_frontier(
            fresh_context, max_candidates=2
        )
        self.assertEqual(learned_frontier.status, "LEARNED_GENERATOR_TRANSFER")
        self.assertEqual(learned_frontier.learned_alpha, 0.25)
        self.assertEqual(learned_frontier.candidate_scales, (1.25, 2.5))
        treatment = descendant.generate_projection_generator_interventions(
            fresh_axis,
            {fresh_axis.inputs[0]: 0.0, fresh_axis.inputs[1]: 0.0},
            fresh_context,
            max_candidates=2,
        )
        self._remember_and_execute(descendant, treatment, 2.5, fresh_context, 6000)
        self.assertEqual(self._context_capability(descendant, fresh_context, 2.5), 1.0)

        # RESET gets the same two-scale candidate budget but has no learned generator.
        # Its bounded shadow ordering cannot reach 2.5 under that matched budget.
        reset = PersistentCognitiveRuntime()
        reset_axis = axis("generator-reset-heldout")
        reset.memory.remember_representation(reset_axis)
        reset_context = "generator-reset-heldout"
        reset_base = self._proposals(reset_axis, AUTHORED)
        self._remember_and_execute(reset, reset_base, 2.5, reset_context, 7000)
        reset_frontier = reset.projection_generator_frontier(reset_context, max_candidates=2)
        self.assertEqual(reset_frontier.candidate_scales, (1.25, 1.5))
        reset_generated = reset.generate_projection_generator_interventions(
            reset_axis,
            {reset_axis.inputs[0]: 0.0, reset_axis.inputs[1]: 0.0},
            reset_context,
            max_candidates=2,
        )
        self._remember_and_execute(reset, reset_generated, 2.5, reset_context, 8000)
        self.assertEqual(self._context_capability(reset, reset_context, 2.5), 0.0)

        # WRONG-SWAP receives the same candidate count but the wrong generator.
        wrong = PersistentCognitiveRuntime()
        wrong_axis = axis("generator-wrong-heldout")
        wrong.memory.remember_representation(wrong_axis)
        wrong_context = "generator-wrong-heldout"
        wrong_base = self._proposals(wrong_axis, AUTHORED)
        self._remember_and_execute(wrong, wrong_base, 2.5, wrong_context, 9000)
        wrong_policy = ProjectionGeneratorPolicy(
            status="REPRODUCED_GENERATOR_POLICY",
            alpha=0.75,
            supporting_contexts=("wrong-a", "wrong-b"),
            candidate_alpha_count=1,
            strong_effect_threshold=0.9,
            reason="matched wrong-swap",
        )
        wrong_frontier = derive_projection_generator_frontier(
            AUTHORED,
            (record.proposal for record in wrong.memory.experiments.values()),
            wrong.world_coupling.pairs,
            wrong.world_coupling.min_independent_classes,
            scale_of,
            context_id=wrong_context,
            learned_policy=wrong_policy,
            strong_effect_threshold=0.9,
            max_candidates=2,
        )
        self.assertEqual(wrong_frontier.candidate_scales, (1.75, 3.5))
        wrong_generated = self._proposals(wrong_axis, wrong_frontier.candidate_scales)
        self._remember_and_execute(wrong, wrong_generated, 2.5, wrong_context, 10000)
        self.assertEqual(self._context_capability(wrong, wrong_context, 2.5), 0.0)


if __name__ == "__main__":
    unittest.main()
