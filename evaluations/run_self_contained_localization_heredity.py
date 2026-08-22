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
from arte_cognition.repository_localization_representation_genesis import (
    RepositoryLocalizationRepresentationOrgan,
    parse_graph_localization_signature,
)
from arte_cognition.world_coupling import HMACWorldReceiptSigner, HMACWorldReceiptVerifier
from evaluations.run_world_driven_localization_representation_escape import (
    REPAIR_FAMILIES,
    _execute_one,
    _make_task,
    _train_complete,
)


def _assessment_tuple(assessment):
    return (
        assessment.status,
        assessment.ambiguous_contexts,
        assessment.complete_contexts,
        assessment.missing_experiment_ids,
        assessment.evaluated_candidate_count,
    )


def _policy_tuple(policy):
    return (
        policy.status,
        policy.fingerprint,
        policy.operator_id,
        policy.fingerprint_depth,
        policy.supporting_contexts,
        policy.candidate_signature_count,
    )


def main(seed_path):
    seed = int(Path(seed_path).read_text().strip())
    rng = random.Random(seed)
    hidden_operator = rng.choice(tuple(sorted(REPAIR_FAMILIES)))

    issuer_a = f"heredity-lab-{rng.randrange(10_000_000, 99_999_999)}"
    issuer_b = f"heredity-lab-{rng.randrange(10_000_000, 99_999_999)}"
    key_a = hashlib.sha256(f"{seed}:heredity:a".encode()).digest()
    key_b = hashlib.sha256(f"{seed}:heredity:b".encode()).digest()
    signers = {
        issuer_a: HMACWorldReceiptSigner(issuer_a, key_a),
        issuer_b: HMACWorldReceiptSigner(issuer_b, key_b),
    }
    verifier = HMACWorldReceiptVerifier(
        {issuer_a: key_a, issuer_b: key_b},
        independence_classes={issuer_a: "HEREDITY_LAB_A", issuer_b: "HEREDITY_LAB_B"},
    )

    parent = PersistentCognitiveRuntime()
    train_a = _make_task(rng, hidden_operator, "numeric-scalar", "heredity-train-a", "TARGET")
    train_b = _make_task(rng, hidden_operator, "lexical-scalar", "heredity-train-b", "TARGET")
    candidates_a, strong_a = _train_complete(parent, train_a, signers, verifier, 10000)
    candidates_b, strong_b = _train_complete(parent, train_b, signers, verifier, 20000)

    parent_organ = RepositoryLocalizationRepresentationOrgan(parent)
    parent_assessment = parent_organ.assess_persisted_old_language()
    parent_policy = parent_organ.policy(parent_assessment)
    if parent_assessment.status != "NAMED_ROLE_LOCALIZATION_NON_IDENTIFYING_OPEN_GRAPH_REPRESENTATION":
        raise AssertionError(f"parent persisted ambiguity did not close: {parent_assessment}")
    if parent_assessment.evaluated_candidate_count != 4 or parent_assessment.missing_experiment_ids:
        raise AssertionError("parent persisted candidate universe was not complete")
    if parent_policy.status != "REPRODUCED_GENERATED_GRAPH_LOCALIZATION":
        raise AssertionError(f"parent graph policy not reconstructed from BODY: {parent_policy}")
    if parent_policy.operator_id != hidden_operator or parent_policy.fingerprint_depth != 2:
        raise AssertionError("parent persisted policy drifted from hidden repair/depth")
    if parse_graph_localization_signature(strong_a.proposal) != parse_graph_localization_signature(strong_b.proposal):
        raise AssertionError("training target graph phenotype did not reproduce")

    checkpoint = checkpoint_dict(parent)

    # From this point onward the training source repositories and generated candidate
    # objects are deliberately discarded. Descendant authority must come only from
    # checkpointed proposal lineage plus externally reverified receipts.
    train_a = None
    train_b = None
    candidates_a = None
    candidates_b = None
    strong_a = None
    strong_b = None

    verifierless = restore_runtime(checkpoint)
    verifierless_organ = RepositoryLocalizationRepresentationOrgan(verifierless)
    verifierless_assessment = verifierless_organ.assess_persisted_old_language()
    verifierless_policy = verifierless_organ.policy(verifierless_assessment)
    if verifierless_policy.fingerprint is not None:
        raise AssertionError("checkpointed proposal lineage self-authorized without external verifier")

    descendant = restore_runtime(checkpoint, world_verifier=verifier)
    descendant_organ = RepositoryLocalizationRepresentationOrgan(descendant)
    descendant_assessment = descendant_organ.assess_persisted_old_language()
    descendant_policy = descendant_organ.policy(descendant_assessment)
    if _assessment_tuple(descendant_assessment) != _assessment_tuple(parent_assessment):
        raise AssertionError("descendant ambiguity assessment was not reconstructed exactly from BODY")
    if _policy_tuple(descendant_policy) != _policy_tuple(parent_policy):
        raise AssertionError("descendant graph policy was not reconstructed exactly from BODY")

    heldout = _make_task(rng, hidden_operator, "record-structure", "heredity-heldout", "TARGET")
    heldout_candidates, heldout_depth = descendant_organ.propose(heldout.task_id, heldout.files)
    selection = descendant_organ.select(heldout_candidates, descendant_policy, max_candidates=1)
    effects = _execute_one(descendant, heldout, selection, signers, verifier, 50000)
    capability = float(min(effects) >= 0.9)
    if capability != 1.0:
        raise AssertionError("self-contained descendant graph policy failed fresh repository transfer")
    if heldout_depth != 2 or len(selection.candidates) != 1:
        raise AssertionError("heldout self-contained localization did not retain minimal depth/budget")
    if selection.candidates[0].file_path != heldout.target_mid_path:
        raise AssertionError("self-contained descendant selected wrong heldout file")

    result = {
        "status": "PASS_BOUNDED_SELF_CONTAINED_REPOSITORY_LOCALIZATION_PHENOTYPE_HEREDITY",
        "hidden_repair_operator": hidden_operator,
        "training_source_resupplied_after_restore": False,
        "parent_candidate_count": 4,
        "parent_complete_contexts": len(parent_assessment.complete_contexts),
        "parent_ambiguous_contexts": len(parent_assessment.ambiguous_contexts),
        "parent_missing_candidate_count": len(parent_assessment.missing_experiment_ids),
        "parent_graph_fingerprint": parent_policy.fingerprint,
        "parent_graph_fingerprint_depth": parent_policy.fingerprint_depth,
        "descendant_assessment_exact_match": True,
        "descendant_policy_exact_match": True,
        "descendant_reconstruction_inputs": "checkpointed_proposal_lineage_plus_externally_reverified_world_receipts",
        "authoritative_boolean_checkpointed": False,
        "verifierless_graph_localization_authority": False,
        "heldout_domain": heldout.domain,
        "heldout_candidate_count": len(heldout_candidates),
        "heldout_selected_candidate_count": len(selection.candidates),
        "heldout_external_pair_count": 2,
        "heldout_capability": capability,
        "heldout_minimal_graph_depth": heldout_depth,
        "hidden_tests_exposed_to_body_before_execution": False,
        "training_repository_hashes_needed_after_restore": False,
        "parent_process_candidate_objects_reused_after_restore": False,
        "real_repository_autonomous_repair": False,
        "unrestricted_localization_representation_genesis": False,
        "foundation_weight_change": False,
        "independent_organizational_custody": False,
        "physical_world": False,
        "global_recursive_acceleration": False,
        "AGI": False,
        "ASI": False,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_self_contained_localization_heredity.py <seed_path>")
    main(sys.argv[1])
