from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def probe_scale(proposal):
    marker = "probe_scale="
    return float(str(proposal.reason).split(marker, 1)[1].split()[0].rstrip(",;)") )


def axis(label: str, rng: random.Random) -> RepresentationAxis:
    x = f"sensor_{rng.randrange(10000000, 99999999)}"
    z = f"sensor_{rng.randrange(10000000, 99999999)}"
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


def endpoints(ax, left, right):
    return ExperimentGenesisEngine(
        projection_margin_multipliers=(float(left), float(right)),
        max_proposals=64,
    ).propose(ax, {ax.inputs[0]: 0.0, ax.inputs[1]: 0.0})


class HiddenScaleWorld:
    def __init__(self, target, signer, source_id, challenge_id, context_id, epoch):
        self.target = float(target)
        self.signer = signer
        self.source_id = str(source_id)
        self.challenge_id = str(challenge_id)
        self.context_id = str(context_id)
        self.epoch = int(epoch)

    def execute(self, proposal, arm, value):
        scale = probe_scale(proposal)
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


def execute(runtime, proposals, target, context, epoch_base, signers, verifier):
    for proposal_index, proposal in enumerate(proposals):
        runtime.memory.remember_experiment(proposal)
        for issuer_index, (issuer, signer) in enumerate(signers.items()):
            pair = runtime.execute_world_intervention(
                proposal,
                HiddenScaleWorld(
                    target,
                    signer,
                    f"{context}-{epoch_base}-source-{proposal_index}-{issuer}",
                    f"{context}-{epoch_base}-challenge-{proposal_index}-{issuer}",
                    context,
                    epoch_base + proposal_index * 10 + issuer_index,
                ),
                verifier=verifier,
            )
            if not pair.authority_verified:
                raise AssertionError("hidden pair lost authenticated authority")


def capability(runtime, context, target):
    scores = projection_scale_scores(
        (record.proposal for record in runtime.memory.experiments.values()),
        runtime.world_coupling.pairs,
        runtime.world_coupling.min_independent_classes,
        probe_scale,
        context_id=context,
    )
    return float(scores.get(round(float(target), 12), 0.0) >= 0.9)


def select_program(programs, operations, alpha):
    for program in programs:
        if program.operations == tuple(operations) and abs(program.alpha - float(alpha)) <= 1e-12:
            return program
    raise AssertionError(f"missing generated transform AST {operations} alpha={alpha}")


def choose_bracket(rng, programs, hidden, require_not_first=False):
    for _ in range(2000):
        left = round(rng.uniform(20.0, 80.0), 6)
        right = round(left * rng.uniform(80.0, 2000.0), 6)
        target = hidden.apply(left, right)
        if target is None:
            continue
        values = {}
        collision = False
        for program in programs:
            value = program.apply(left, right)
            if value is None:
                continue
            values.setdefault(value, []).append(program.program_id)
            if abs(value - target) <= 1e-9 and program.program_id != hidden.program_id:
                collision = True
        if collision:
            continue
        if require_not_first and values and abs(min(values) - target) <= 1e-9:
            continue
        return left, right, target
    raise AssertionError("failed to sample a nondegenerate depth-expansion bracket")


def evaluate_depth2_context(
    runtime, context, bracket, target, depth_brackets, epoch, signers, verifier
):
    left, right = bracket
    ax = axis(context, random.Random(epoch))
    runtime.memory.remember_representation(ax)
    execute(runtime, endpoints(ax, left, right), target, context, epoch, signers, verifier)
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
    execute(runtime, generated, target, context, epoch + 1000, signers, verifier)
    return ax, len(generated)


def train_depth3_context(
    runtime, context, bracket, target, depth_brackets, epoch, signers, verifier, rng
):
    left, right = bracket
    ax = axis(context, rng)
    runtime.memory.remember_representation(ax)
    execute(runtime, endpoints(ax, left, right), target, context, epoch, signers, verifier)
    frontier = runtime.projection_transform_adaptive_frontier(
        context,
        left,
        right,
        depth_brackets,
        max_candidates=64,
        allow_depth_expansion=True,
        apply_learned_program=False,
    )
    if round(float(target), 12) not in {candidate.scale for candidate in frontier.candidates}:
        raise AssertionError("new depth failed to generate the hidden depth-3 target")
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
    execute(runtime, generated, target, context, epoch + 1000, signers, verifier)
    return ax, frontier, len(generated)


def build_wrong_body(
    rng, depth3_programs, wrong_hidden, signers, verifier, epoch_base
):
    body = PersistentCognitiveRuntime()
    f_samples = [choose_bracket(rng, depth3_programs, wrong_hidden) for _ in range(2)]
    f_contexts = [f"wrong-f-{rng.randrange(10000000, 99999999)}" for _ in range(2)]
    depth_brackets = {
        context: (sample[0], sample[1])
        for context, sample in zip(f_contexts, f_samples)
    }
    for index, (context, sample) in enumerate(zip(f_contexts, f_samples)):
        evaluate_depth2_context(
            body,
            context,
            (sample[0], sample[1]),
            sample[2],
            depth_brackets,
            epoch_base + index * 5000,
            signers,
            verifier,
        )
    opened = body.projection_transform_depth_assessment(depth_brackets)
    if opened.authorized_depth != 3:
        raise AssertionError("wrong-control BODY did not independently earn depth 3")

    t_samples = [choose_bracket(rng, depth3_programs, wrong_hidden) for _ in range(2)]
    for index, sample in enumerate(t_samples):
        context = f"wrong-t-{rng.randrange(10000000, 99999999)}"
        train_depth3_context(
            body,
            context,
            (sample[0], sample[1]),
            sample[2],
            depth_brackets,
            epoch_base + 20000 + index * 5000,
            signers,
            verifier,
            rng,
        )
    policy = body.projection_transform_adaptive_policy(depth_brackets)
    if policy.operations != wrong_hidden.operations or policy.alpha != wrong_hidden.alpha:
        raise AssertionError(f"wrong-control BODY learned unexpected AST: {policy}")
    return body, depth_brackets, policy


def main(seed_path: str) -> None:
    seed = int(Path(seed_path).read_text().strip())
    rng = random.Random(seed)
    depth2_programs = generate_projection_transform_programs(
        max_transform_depth=2,
        signature_anchors=DEEP_TRANSFORM_SIGNATURE_ANCHORS,
    )
    depth3_programs = generate_projection_transform_programs(
        max_transform_depth=3,
        signature_anchors=DEEP_TRANSFORM_SIGNATURE_ANCHORS,
    )
    hidden = select_program(depth3_programs, ("LOG", "LOG", "LOG"), 0.25)
    wrong_hidden = select_program(depth3_programs, ("LOG", "LOG", "INV"), 0.25)

    issuer_a = f"issuer-{rng.randrange(10**7, 10**8)}"
    issuer_b = f"issuer-{rng.randrange(10**7, 10**8)}"
    key_a = hashlib.sha256(f"{seed}:depth:a".encode()).digest()
    key_b = hashlib.sha256(f"{seed}:depth:b".encode()).digest()
    signers = {
        issuer_a: HMACWorldReceiptSigner(issuer_a, key_a),
        issuer_b: HMACWorldReceiptSigner(issuer_b, key_b),
    }
    verifier = HMACWorldReceiptVerifier(
        {issuer_a: key_a, issuer_b: key_b},
        independence_classes={issuer_a: "LAB_A", issuer_b: "LAB_B"},
    )

    body = PersistentCognitiveRuntime()
    falsification_samples = [choose_bracket(rng, depth3_programs, hidden) for _ in range(2)]
    falsification_contexts = [f"depth-f-{rng.randrange(10000000, 99999999)}" for _ in range(2)]
    depth_brackets = {
        context: (sample[0], sample[1])
        for context, sample in zip(falsification_contexts, falsification_samples)
    }

    depth2_generated_counts = []
    first_assessment = None
    for index, (context, sample) in enumerate(zip(falsification_contexts, falsification_samples)):
        _, generated_count = evaluate_depth2_context(
            body,
            context,
            (sample[0], sample[1]),
            sample[2],
            depth_brackets,
            10000 + index * 5000,
            signers,
            verifier,
        )
        depth2_generated_counts.append(generated_count)
        if index == 0:
            first_assessment = body.projection_transform_depth_assessment(depth_brackets)

    opened = body.projection_transform_depth_assessment(depth_brackets)
    if first_assessment is None or first_assessment.authorized_depth != 2:
        raise AssertionError("one complete context incorrectly opened a deeper grammar")
    if opened.status != "TRANSFORM_GRAMMAR_DEPTH_FALSIFIED_OPEN_NEXT" or opened.authorized_depth != 3:
        raise AssertionError(f"complete repeated depth-2 failure did not open depth 3: {opened}")
    if any(item.missing_program_ids for item in opened.context_assessments):
        raise AssertionError("depth expansion used absence as refutation")

    training_samples = [choose_bracket(rng, depth3_programs, hidden) for _ in range(2)]
    depth3_frontier_counts = []
    for index, sample in enumerate(training_samples):
        context = f"depth-t-{rng.randrange(10000000, 99999999)}"
        _, frontier, generated_count = train_depth3_context(
            body,
            context,
            (sample[0], sample[1]),
            sample[2],
            depth_brackets,
            30000 + index * 5000,
            signers,
            verifier,
            rng,
        )
        depth3_frontier_counts.append((len(frontier.candidates), generated_count))

    learned = body.projection_transform_adaptive_policy(depth_brackets)
    if learned.operations != hidden.operations or learned.alpha != hidden.alpha:
        raise AssertionError(f"BODY failed to learn depth-3 hidden AST: {learned}")

    checkpoint = checkpoint_dict(body)
    verifierless = restore_runtime(checkpoint)
    verifierless_depth = verifierless.projection_transform_depth_assessment(depth_brackets)
    verifierless_policy = verifierless.projection_transform_adaptive_policy(depth_brackets)
    if verifierless_depth.authorized_depth != 2 or verifierless_policy.program_id is not None:
        raise AssertionError("checkpoint restored depth/policy authority without external verifier")

    heldout = choose_bracket(rng, depth3_programs, hidden, require_not_first=True)
    heldout_left, heldout_right, heldout_target = heldout
    heldout_context = f"depth-heldout-{rng.randrange(10000000, 99999999)}"
    heldout_axis = axis(heldout_context, rng)

    treatment = restore_runtime(checkpoint, world_verifier=verifier)
    remove_depth = restore_runtime(checkpoint, world_verifier=verifier)
    remove_policy = restore_runtime(checkpoint, world_verifier=verifier)
    for runtime in (treatment, remove_depth, remove_policy):
        runtime.memory.remember_representation(heldout_axis)
        execute(
            runtime,
            endpoints(heldout_axis, heldout_left, heldout_right),
            heldout_target,
            heldout_context,
            60000,
            signers,
            verifier,
        )
        if capability(runtime, heldout_context, heldout_target) != 0.0:
            raise AssertionError("fresh target already solved before adaptive-depth transfer")

    treatment_frontier = treatment.projection_transform_adaptive_frontier(
        heldout_context, heldout_left, heldout_right, depth_brackets,
        max_candidates=1, allow_depth_expansion=True, apply_learned_program=True,
    )
    treatment_generated = treatment.generate_projection_transform_adaptive_interventions(
        heldout_axis,
        {heldout_axis.inputs[0]: 0.0, heldout_axis.inputs[1]: 0.0},
        heldout_context, heldout_left, heldout_right, depth_brackets,
        max_candidates=1, allow_depth_expansion=True, apply_learned_program=True,
    )
    execute(treatment, treatment_generated, heldout_target, heldout_context, 70000, signers, verifier)
    treatment_capability = capability(treatment, heldout_context, heldout_target)

    remove_depth_frontier = remove_depth.projection_transform_adaptive_frontier(
        heldout_context, heldout_left, heldout_right, depth_brackets,
        max_candidates=1, allow_depth_expansion=False, apply_learned_program=True,
    )
    remove_depth_generated = remove_depth.generate_projection_transform_adaptive_interventions(
        heldout_axis,
        {heldout_axis.inputs[0]: 0.0, heldout_axis.inputs[1]: 0.0},
        heldout_context, heldout_left, heldout_right, depth_brackets,
        max_candidates=1, allow_depth_expansion=False, apply_learned_program=True,
    )
    execute(remove_depth, remove_depth_generated, heldout_target, heldout_context, 70000, signers, verifier)
    remove_depth_capability = capability(remove_depth, heldout_context, heldout_target)

    remove_policy_frontier = remove_policy.projection_transform_adaptive_frontier(
        heldout_context, heldout_left, heldout_right, depth_brackets,
        max_candidates=1, allow_depth_expansion=True, apply_learned_program=False,
    )
    remove_policy_generated = remove_policy.generate_projection_transform_adaptive_interventions(
        heldout_axis,
        {heldout_axis.inputs[0]: 0.0, heldout_axis.inputs[1]: 0.0},
        heldout_context, heldout_left, heldout_right, depth_brackets,
        max_candidates=1, allow_depth_expansion=True, apply_learned_program=False,
    )
    execute(remove_policy, remove_policy_generated, heldout_target, heldout_context, 70000, signers, verifier)
    remove_policy_capability = capability(remove_policy, heldout_context, heldout_target)

    reset = PersistentCognitiveRuntime()
    reset.memory.remember_representation(heldout_axis)
    execute(reset, endpoints(heldout_axis, heldout_left, heldout_right), heldout_target, heldout_context, 60000, signers, verifier)
    reset_generated = reset.generate_projection_transform_adaptive_interventions(
        heldout_axis,
        {heldout_axis.inputs[0]: 0.0, heldout_axis.inputs[1]: 0.0},
        heldout_context, heldout_left, heldout_right, depth_brackets,
        max_candidates=1, allow_depth_expansion=True, apply_learned_program=True,
    )
    execute(reset, reset_generated, heldout_target, heldout_context, 70000, signers, verifier)
    reset_capability = capability(reset, heldout_context, heldout_target)

    wrong_parent, wrong_depth_brackets, wrong_policy = build_wrong_body(
        rng, depth3_programs, wrong_hidden, signers, verifier, 100000
    )
    wrong = restore_runtime(checkpoint_dict(wrong_parent), world_verifier=verifier)
    wrong_context = f"depth-wrong-heldout-{rng.randrange(10000000, 99999999)}"
    wrong_axis = axis(wrong_context, rng)
    wrong.memory.remember_representation(wrong_axis)
    execute(wrong, endpoints(wrong_axis, heldout_left, heldout_right), heldout_target, wrong_context, 150000, signers, verifier)
    wrong_frontier = wrong.projection_transform_adaptive_frontier(
        wrong_context, heldout_left, heldout_right, wrong_depth_brackets,
        max_candidates=1, allow_depth_expansion=True, apply_learned_program=True,
    )
    wrong_generated = wrong.generate_projection_transform_adaptive_interventions(
        wrong_axis,
        {wrong_axis.inputs[0]: 0.0, wrong_axis.inputs[1]: 0.0},
        wrong_context, heldout_left, heldout_right, wrong_depth_brackets,
        max_candidates=1, allow_depth_expansion=True, apply_learned_program=True,
    )
    execute(wrong, wrong_generated, heldout_target, wrong_context, 160000, signers, verifier)
    wrong_capability = capability(wrong, wrong_context, heldout_target)

    if treatment_capability != 1.0:
        raise AssertionError("depth-3 treatment failed fresh capability")
    if any(value != 0.0 for value in (
        remove_depth_capability, remove_policy_capability, reset_capability, wrong_capability
    )):
        raise AssertionError("causal control retained capability without the correct depth-3 developmental state")

    result = {
        "status": "PASS_BOUNDED_WORLD_FALSIFICATION_DRIVEN_TRANSFORM_DEPTH3_AND_DESCENDANT_CAUSAL_TRANSFER",
        "initial_transform_depth": 2,
        "authorized_transform_depth": opened.authorized_depth,
        "depth2_program_count": len(depth2_programs),
        "depth3_program_count": len(depth3_programs),
        "depth2_falsified_contexts": len(opened.falsified_contexts),
        "depth2_missing_programs_after_gate": sum(len(item.missing_program_ids) for item in opened.context_assessments),
        "depth2_generated_proposal_counts": depth2_generated_counts,
        "depth3_frontier_counts": depth3_frontier_counts,
        "learned_depth3_operations": list(learned.operations),
        "learned_alpha": learned.alpha,
        "heldout_bracket": [heldout_left, heldout_right],
        "heldout_target": heldout_target,
        "treatment_candidate_count": len(treatment_frontier.candidates),
        "remove_depth_candidate_count": len(remove_depth_frontier.candidates),
        "remove_policy_candidate_count": len(remove_policy_frontier.candidates),
        "wrong_candidate_count": len(wrong_frontier.candidates),
        "treatment_capability": treatment_capability,
        "remove_depth_same_checkpoint_capability": remove_depth_capability,
        "remove_policy_same_checkpoint_capability": remove_policy_capability,
        "reset_capability": reset_capability,
        "wrong_learned_operations": list(wrong_policy.operations),
        "wrong_capability": wrong_capability,
        "same_checkpoint_treatment_remove_depth_remove_policy": True,
        "absence_is_not_refutation": True,
        "candidate_generation_uses_hidden_target": False,
        "depth_authority_rederived_after_external_verification": True,
        "verifierless_authorized_depth": verifierless_depth.authorized_depth,
        "verifierless_policy_authority": False,
        "primitive_transform_alphabet": ["LOG", "INV"],
        "primitive_transform_alphabet_human_authored": True,
        "depth_expansion_rule_human_authored": True,
        "unrestricted_transform_operator_genesis": False,
        "foundation_weight_change": False,
        "physical_world": False,
        "independent_organizational_custody": False,
        "global_recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_world_driven_transform_depth_expansion.py <seed_path>")
    main(sys.argv[1])
