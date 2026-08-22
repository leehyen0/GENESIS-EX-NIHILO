from __future__ import annotations

import json
import random
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.adaptive_cognition import TaskState
from arte_cognition.body_checkpoint import checkpoint_json, restore_json
from arte_cognition.causal_credit import OutcomeAblationCreditEngine
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.world_action_policy import EvidenceBoundWorldActionPolicy
from arte_cognition.world_coupling import HMACWorldReceiptSigner, HMACWorldReceiptVerifier
from arte_cognition.world_revision import AuthenticatedWorldCognitionReviser
from evaluations.run_world_caused_cognition_rewrite import (
    HiddenRegimeWorld,
    NEW_HELDOUT,
    NEW_TRAIN,
    OLD_HELDOUT,
    OLD_TRAIN,
    best_incremental,
    build_observations,
    execute_many,
    remember_reference_experiments,
)


MODULE = "WORLD_CAUSED_COGNITION_REWRITE"


def build_trial(seed: int):
    rng = random.Random(seed)
    scale = rng.choice((-1.0, 1.0)) * rng.uniform(0.65, 2.75)
    swap = bool(rng.getrandbits(1))
    feature_names = (
        f"credit_sensor_{rng.randrange(10_000, 99_999)}",
        f"credit_sensor_{rng.randrange(100_000, 999_999)}",
    )
    label_flip = bool(rng.getrandbits(1))
    issuer_ids = (
        f"credit-evaluator-a-{rng.randrange(1_000_000, 9_999_999)}",
        f"credit-evaluator-b-{rng.randrange(1_000_000, 9_999_999)}",
    )
    secrets_by_issuer = {
        issuer_ids[0]: secrets.token_bytes(32),
        issuer_ids[1]: secrets.token_bytes(32),
    }
    signers = {
        issuer: HMACWorldReceiptSigner(issuer, secret)
        for issuer, secret in secrets_by_issuer.items()
    }
    verifier = HMACWorldReceiptVerifier(
        secrets_by_issuer,
        independence_classes={
            issuer_ids[0]: "credit-independent-a",
            issuer_ids[1]: "credit-independent-b",
        },
    )
    return rng, scale, swap, feature_names, label_flip, issuer_ids, secrets_by_issuer, signers, verifier


def learn_old_world(runtime, scale, swap, feature_names, label_flip, issuer_ids, signers, verifier):
    old_context = "credit-old-sum"
    world = HiddenRegimeWorld(
        scale, swap, feature_names, "SUM", old_context,
        "credit-old-observation-source", "credit-old-observation-challenge", 0,
        signers[issuer_ids[0]],
    )
    measurements, residuals = build_observations(world, OLD_TRAIN, OLD_HELDOUT, label_flip)
    cycle = runtime.cycle(
        TaskState(
            goal="learn pre-shift world model",
            novelty=0.95,
            residuals=[row.residual_id for row in residuals if not row.heldout],
            external_world=True,
            action_required=True,
        ),
        residuals=residuals,
        measurements=measurements,
        world_context_id=old_context,
    )
    axis, assessments = best_incremental(cycle, "PROJECTION")
    if assessments[axis.axis_id].heldout_accuracy != 1.0:
        raise AssertionError("pre-shift representation failed held-out gate")
    proposals = remember_reference_experiments(runtime, axis, measurements)
    effects = execute_many(
        runtime, proposals, scale, swap, feature_names, "SUM", old_context,
        issuer_ids, signers, verifier, 1000,
    )
    supported = [
        proposal for proposal in proposals
        if min(abs(v) for v in effects[proposal.experiment_id]) >= 0.5
    ]
    if len(supported) < 2:
        raise AssertionError("pre-shift world supplied fewer than two material exact experiments")
    decision = EvidenceBoundWorldActionPolicy().select(
        proposals, runtime.world_coupling, context_id=old_context
    )
    if decision.status != "WORLD_SUPPORTED_ACTION":
        raise AssertionError("pre-shift BODY did not earn action authority")
    return old_context, axis.axis_id, supported


def add_changed_world_counterevidence(runtime, old_supported, scale, swap, feature_names, issuer_ids, signers, verifier):
    new_context = "credit-new-diff"
    execute_many(
        runtime, old_supported, scale, swap, feature_names, "DIFF", new_context,
        issuer_ids, signers, verifier, 5000,
    )
    return new_context


def immediate_contradicted_action_score(runtime, contradicted_ids, new_context):
    candidates = [
        proposal for proposal in runtime.persisted_intervention_proposals()
        if proposal.experiment_id in contradicted_ids
    ]
    decision = EvidenceBoundWorldActionPolicy().select(
        candidates, runtime.world_coupling, context_id=new_context
    )
    # The challenge is deliberately restricted to exact experiments already
    # independently contradicted by the changed world. A correct rewritten BODY
    # must have removed all of them from its active phenotype surface.
    safe = 1.0 if decision.proposal is None else 0.0
    if decision.proposal is not None and decision.proposal.experiment_id not in contradicted_ids:
        raise AssertionError("control selected an experiment outside the counterevidence challenge")
    return decision, safe


def adapt_after_shift(runtime, revision_residual, scale, swap, feature_names, label_flip, issuer_ids, signers, verifier, epoch_base):
    new_context = "credit-new-diff"
    world = HiddenRegimeWorld(
        scale, swap, feature_names, "DIFF", new_context,
        f"fresh-source-{epoch_base}", f"fresh-challenge-{epoch_base}", epoch_base,
        signers[issuer_ids[1]],
    )
    measurements, residuals = build_observations(world, NEW_TRAIN, NEW_HELDOUT, label_flip)
    cycle_residuals = ([revision_residual] if revision_residual is not None else []) + residuals
    cycle = runtime.cycle(
        TaskState(
            goal="adapt to fresh post-shift observations",
            novelty=0.99,
            residuals=[row.residual_id for row in cycle_residuals if not row.heldout],
            external_world=True,
            action_required=True,
        ),
        residuals=cycle_residuals,
        measurements=measurements,
        world_context_id=new_context,
    )
    axis, assessments = best_incremental(cycle, "DIFFERENCE")
    if assessments[axis.axis_id].heldout_accuracy != 1.0:
        raise AssertionError("post-shift replacement representation failed held-out gate")
    proposals = remember_reference_experiments(runtime, axis, measurements)
    effects = execute_many(
        runtime, proposals, scale, swap, feature_names, "DIFF", new_context,
        issuer_ids, signers, verifier, epoch_base + 1000,
    )
    decision = EvidenceBoundWorldActionPolicy().select(
        proposals, runtime.world_coupling, context_id=new_context
    )
    if decision.status != "WORLD_SUPPORTED_ACTION" or decision.proposal is None:
        return 0.0, axis.axis_id, decision
    material = min(abs(v) for v in effects[decision.proposal.experiment_id]) >= 0.5
    return (1.0 if material else 0.0), axis.axis_id, decision


def main(seed_path: str) -> None:
    seed = int(Path(seed_path).read_text().strip())
    (
        _, scale, swap, feature_names, label_flip, issuer_ids,
        secrets_by_issuer, signers, verifier,
    ) = build_trial(seed)

    base = PersistentCognitiveRuntime()
    old_context, old_axis_id, old_supported = learn_old_world(
        base, scale, swap, feature_names, label_flip, issuer_ids, signers, verifier
    )
    new_context = add_changed_world_counterevidence(
        base, old_supported, scale, swap, feature_names, issuer_ids, signers, verifier
    )
    frozen = checkpoint_json(base)
    if any(secret.hex() in frozen for secret in secrets_by_issuer.values()):
        raise AssertionError("external verifier secret leaked before ablation split")

    # All arms begin from the exact same post-counterevidence BODY checkpoint.
    treatment = restore_json(frozen, world_verifier=verifier)
    remove = restore_json(frozen, world_verifier=verifier)
    wrong = restore_json(frozen, world_verifier=verifier)

    reviser = AuthenticatedWorldCognitionReviser()
    treatment_revision = reviser.assess_and_apply(
        treatment.memory, treatment.world_coupling, old_axis_id, old_context, new_context
    )
    if treatment_revision.status != "PASS_BOUNDED_WORLD_CAUSED_COGNITION_DEMOTION":
        raise AssertionError("treatment rewrite gate did not close")
    contradicted_ids = {item.experiment_id for item in treatment_revision.counterevidence}
    if len(contradicted_ids) < 2:
        raise AssertionError("treatment had insufficient exact counterevidence")

    # REMOVE control executes the same revision computation on a shadow copy, but
    # discards its BODY delta. This keeps the expensive evidence scan present while
    # ablating only the state mutation consumed by future behavior.
    remove_shadow = restore_json(frozen, world_verifier=verifier)
    shadow_revision = reviser.assess_and_apply(
        remove_shadow.memory, remove_shadow.world_coupling, old_axis_id, old_context, new_context
    )
    if shadow_revision.status != treatment_revision.status:
        raise AssertionError("REMOVE shadow computation did not reproduce treatment revision evidence")

    # WRONG-EVIDENCE control runs the same revision operator against a context for
    # which no authenticated world receipts exist. It must leave the BODY unchanged.
    wrong_revision = reviser.assess_and_apply(
        wrong.memory,
        wrong.world_coupling,
        old_axis_id,
        old_context,
        "credit-unobserved-wrong-context",
    )
    if wrong_revision.mutations:
        raise AssertionError("wrong-evidence control rewrote cognition without qualifying evidence")

    treatment_decision, treatment_safe = immediate_contradicted_action_score(
        treatment, contradicted_ids, new_context
    )
    remove_decision, remove_safe = immediate_contradicted_action_score(
        remove, contradicted_ids, new_context
    )
    wrong_decision, wrong_safe = immediate_contradicted_action_score(
        wrong, contradicted_ids, new_context
    )
    if treatment_safe != 1.0:
        raise AssertionError("treatment retained a known contradicted exact action")
    if remove_safe != 0.0 or remove_decision.status != "WORLD_SUPPORTED_ACTION":
        raise AssertionError("REMOVE control did not expose stale contradicted-action behavior")
    if wrong_safe != 0.0 or wrong_decision.status != "WORLD_SUPPORTED_ACTION":
        raise AssertionError("wrong-evidence control did not preserve stale contradicted-action behavior")

    # Give all arms the same fresh source-disjoint post-shift observations and
    # equivalent new-world experiment schedule. Controls are allowed to adapt;
    # causal credit is therefore not manufactured by withholding information.
    treatment_adapt, treatment_new_axis, treatment_new_decision = adapt_after_shift(
        treatment, treatment_revision.residual,
        scale, swap, feature_names, label_flip, issuer_ids, signers, verifier, 10000,
    )
    remove_adapt, remove_new_axis, remove_new_decision = adapt_after_shift(
        remove, None,
        scale, swap, feature_names, label_flip, issuer_ids, signers, verifier, 20000,
    )
    wrong_adapt, wrong_new_axis, wrong_new_decision = adapt_after_shift(
        wrong, None,
        scale, swap, feature_names, label_flip, issuer_ids, signers, verifier, 30000,
    )
    if min(treatment_adapt, remove_adapt, wrong_adapt) != 1.0:
        raise AssertionError("one arm failed the common fresh-data adaptation challenge")

    # Composite realized outcome: immediate rejection of already-refuted actions
    # plus later ability to adapt once genuinely fresh observations arrive.
    treatment_outcome = 0.5 * treatment_safe + 0.5 * treatment_adapt
    remove_outcome = 0.5 * remove_safe + 0.5 * remove_adapt
    wrong_outcome = 0.5 * wrong_safe + 0.5 * wrong_adapt

    credit = OutcomeAblationCreditEngine().assign(
        full_outcome=treatment_outcome,
        ablation_outcomes={MODULE: remove_outcome},
        active_modules=[MODULE],
        matched_compute={MODULE: True},
    )
    if len(credit) != 1 or credit[0].causal_credit <= 0.0:
        raise AssertionError("REMOVE ablation did not assign positive realized-outcome causal credit")
    if abs(credit[0].marginal_contribution - 0.5) > 1e-12:
        raise AssertionError("unexpected rewrite marginal contribution")

    encoded = checkpoint_json(treatment)
    descendant = restore_json(encoded, world_verifier=verifier)
    if descendant.memory.representations[old_axis_id].status != "SHADOW_WORLD_REFUTED":
        raise AssertionError("descendant lost the causally credited refutation state")
    if descendant.memory.representations[treatment_new_axis].status != "ACTIVE_VALIDATED":
        raise AssertionError("descendant lost the causally credited replacement representation")

    print(json.dumps({
        "status": "PASS_BOUNDED_WORLD_REWRITE_REMOVE_ABLATION_CAUSAL_CREDIT",
        "module": MODULE,
        "same_pre_ablation_checkpoint": True,
        "matched_world_evidence": True,
        "remove_shadow_revision_computed_and_delta_discarded": True,
        "contradicted_exact_experiments": len(contradicted_ids),
        "treatment_immediate_status": treatment_decision.status,
        "remove_immediate_status": remove_decision.status,
        "wrong_evidence_immediate_status": wrong_decision.status,
        "treatment_immediate_safe_score": treatment_safe,
        "remove_immediate_safe_score": remove_safe,
        "wrong_evidence_immediate_safe_score": wrong_safe,
        "treatment_fresh_adaptation": treatment_adapt,
        "remove_fresh_adaptation": remove_adapt,
        "wrong_evidence_fresh_adaptation": wrong_adapt,
        "treatment_new_axis": treatment_new_axis,
        "remove_new_axis": remove_new_axis,
        "wrong_evidence_new_axis": wrong_new_axis,
        "treatment_new_action": treatment_new_decision.proposal.experiment_id,
        "remove_new_action": remove_new_decision.proposal.experiment_id,
        "wrong_evidence_new_action": wrong_new_decision.proposal.experiment_id,
        "treatment_outcome": treatment_outcome,
        "remove_outcome": remove_outcome,
        "wrong_evidence_outcome": wrong_outcome,
        "marginal_contribution_vs_remove": credit[0].marginal_contribution,
        "causal_credit_vs_remove": credit[0].causal_credit,
        "matched_structural_compute": credit[0].matched_compute,
        "refutation_state_inherited_by_descendant": True,
        "independent_organizational_custody": False,
        "physical_world": False,
        "recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
    }, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_world_revision_causal_credit.py <evaluator-owned-seed-file>")
    main(sys.argv[1])
