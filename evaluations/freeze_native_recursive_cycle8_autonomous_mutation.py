from __future__ import annotations

import argparse
import json
from pathlib import Path

from arte_cognition.autonomous_generator_mutation import (
    AutonomousMetaMutationCompiler,
    MetaFailureCertificate,
    apply_autonomous_selection,
)
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.executable_morphology import ExperienceArchive, MorphologyGenome, OrganKind, OrganSpec
from arte_cognition.meta_acceleration import MutationProgramDevelopmentState, MutationStrategyState
from arte_cognition.self_evolving_body_checkpoint import SelfEvolvingResearchBody, checkpoint_dict, restore_body


def parent_genome() -> MorphologyGenome:
    return MorphologyGenome(
        organs=(
            OrganSpec("source", OrganKind.SOURCE, produces=("raw_observation",), implementation_ref="bootstrap://source"),
            OrganSpec("generator", OrganKind.GENERATOR, implementation_ref="bootstrap://generator"),
            OrganSpec("mutator", OrganKind.MUTATOR, implementation_ref="bootstrap://mutator"),
            OrganSpec("governor", OrganKind.GOVERNOR),
            OrganSpec("archive", OrganKind.ARCHIVE),
        ),
        edges=(),
        event_order=(),
    )


def certificate_from_prior(cycle6: dict, cycle7: dict, *, wrong_layer: bool = False) -> MetaFailureCertificate:
    parent = parent_genome()
    exhausted = bool(
        cycle6.get("old_language_more_compute_resistant")
        and cycle7.get("parent_more_compute_expression_unreachable")
    )
    return MetaFailureCertificate(
        certificate_id="cycle8-native-cert-wrong" if wrong_layer else "cycle8-native-cert",
        parent_body_hash=parent.fingerprint(),
        failure_layer="MUTATOR_SEARCH_POLICY" if wrong_layer else "REPRESENTATION_GENERATOR_LANGUAGE",
        more_compute_exhausted=exhausted,
        independent_contexts=("cycle6-representation-escape", "cycle7-generator-language-transfer"),
        prior_generator_language_gain=float(cycle7.get("full_hidden_useful_rate", 0.0))
        - float(cycle7.get("parent_fixed_family_success_count", 0.0)) / max(1, int(cycle7.get("hidden_task_count", 1))),
        source_receipt_hashes=(str(cycle6["outcome_receipt_sha256"]), str(cycle7["outcome_receipt_sha256"])),
        current_hidden_task_information_present=False,
    )


def _body(genome: MorphologyGenome) -> SelfEvolvingResearchBody:
    return SelfEvolvingResearchBody(
        runtime=PersistentCognitiveRuntime(),
        morphology=genome,
        mutation_strategy=MutationStrategyState(),
        mutation_program_state=MutationProgramDevelopmentState(),
        experience_archive=ExperienceArchive(),
    )


def _selection_row(selection) -> dict:
    return {
        "certificate_fingerprint": selection.certificate_fingerprint,
        "selected_proposal_id": selection.selected_proposal_id,
        "selected_family": selection.selected.family,
        "selection_trace_hash": selection.trace_hash(),
        "generation_uses_current_outcomes": selection.generation_uses_current_outcomes,
        "proposal_count": len(selection.proposals),
        "proposals": [
            {
                "proposal_id": row.proposal_id,
                "family": row.family,
                "mutation_id": row.mutation.mutation_id,
                "mutation_level": int(row.mutation.level),
                "operation": row.mutation.operation,
                "score": row.score,
            }
            for row in selection.proposals
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", required=True)
    parser.add_argument("--cycle6-receipt", required=True)
    parser.add_argument("--cycle7-receipt", required=True)
    parser.add_argument("--candidate-head", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    precommit = json.loads(Path(args.precommit).read_text(encoding="utf-8"))
    cycle6 = json.loads(Path(args.cycle6_receipt).read_text(encoding="utf-8"))
    cycle7 = json.loads(Path(args.cycle7_receipt).read_text(encoding="utf-8"))
    if precommit.get("parent_outcome_receipt_sha256") != cycle7.get("outcome_receipt_sha256"):
        raise SystemExit("cycle8 parent receipt mismatch")

    parent = parent_genome()
    compiler = AutonomousMetaMutationCompiler()
    full_cert = certificate_from_prior(cycle6, cycle7, wrong_layer=False)
    wrong_cert = certificate_from_prior(cycle6, cycle7, wrong_layer=True)
    full_selection = compiler.propose(parent, full_cert)
    wrong_selection = compiler.propose(parent, wrong_cert)
    if len(full_selection.proposals) < 3 or len(wrong_selection.proposals) < 3:
        raise SystemExit("cycle8 autonomous proposal ecology too small")
    if full_selection.selected_proposal_id == wrong_selection.selected_proposal_id:
        raise SystemExit("cycle8 correct and wrong certificate selected same proposal")

    full_child = apply_autonomous_selection(parent, full_selection)
    wrong_child = apply_autonomous_selection(parent, wrong_selection)
    full_checkpoint = checkpoint_dict(_body(full_child))
    wrong_checkpoint = checkpoint_dict(_body(wrong_child))
    if restore_body(full_checkpoint).morphology.fingerprint() != full_child.fingerprint():
        raise SystemExit("cycle8 full autonomous child checkpoint mismatch")
    if restore_body(wrong_checkpoint).morphology.fingerprint() != wrong_child.fingerprint():
        raise SystemExit("cycle8 wrong autonomous child checkpoint mismatch")

    payload = {
        "schema": "arte.native_recursive_cycle8_autonomous_mutation_freeze/v1",
        "candidate_head_sha": args.candidate_head,
        "hidden_task_information_consumed": False,
        "current_outcomes_consumed": False,
        "full_certificate": full_cert.__dict__,
        "wrong_certificate": wrong_cert.__dict__,
        "full_selection": _selection_row(full_selection),
        "wrong_selection": _selection_row(wrong_selection),
        "full_child_morphology_fingerprint": full_child.fingerprint(),
        "wrong_child_morphology_fingerprint": wrong_child.fingerprint(),
        "full_checkpoint": full_checkpoint,
        "wrong_checkpoint": wrong_checkpoint,
    }
    Path(args.output).write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    summary = dict(payload)
    summary.pop("full_checkpoint")
    summary.pop("wrong_checkpoint")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
