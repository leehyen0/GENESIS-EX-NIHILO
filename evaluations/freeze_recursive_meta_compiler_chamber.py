from __future__ import annotations

import argparse
import json
from pathlib import Path

from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.executable_morphology import ExperienceArchive, MorphologyGenome, OrganKind, OrganSpec
from arte_cognition.meta_acceleration import MutationProgramDevelopmentState, MutationStrategyState
from arte_cognition.mutable_meta_compiler import initial_meta_compiler_ref, meta_compiler_policy_from_body
from arte_cognition.self_evolving_body_checkpoint import SelfEvolvingResearchBody, checkpoint_dict, restore_body


def initial_body() -> SelfEvolvingResearchBody:
    morphology = MorphologyGenome(
        organs=(
            OrganSpec("source", OrganKind.SOURCE, produces=("raw_observation",), implementation_ref="bootstrap://source"),
            OrganSpec("generator", OrganKind.GENERATOR, implementation_ref="bootstrap://generator"),
            OrganSpec("mutator", OrganKind.MUTATOR, implementation_ref="bootstrap://mutator"),
            OrganSpec("compiler", OrganKind.COMPILER, implementation_ref=initial_meta_compiler_ref()),
            OrganSpec("governor", OrganKind.GOVERNOR),
            OrganSpec("archive", OrganKind.ARCHIVE),
        ),
        edges=(),
        event_order=(),
    )
    return SelfEvolvingResearchBody(
        runtime=PersistentCognitiveRuntime(),
        morphology=morphology,
        mutation_strategy=MutationStrategyState(),
        mutation_program_state=MutationProgramDevelopmentState(),
        experience_archive=ExperienceArchive(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", required=True)
    parser.add_argument("--cycle8-receipt", required=True)
    parser.add_argument("--candidate-head", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    precommit = json.loads(Path(args.precommit).read_text(encoding="utf-8"))
    cycle8 = json.loads(Path(args.cycle8_receipt).read_text(encoding="utf-8"))
    if precommit.get("parent_outcome_receipt_sha256") != cycle8.get("outcome_receipt_sha256"):
        raise SystemExit("cycle9 chamber parent receipt mismatch")
    if precommit.get("parent_evidence_head_sha") != "9e23db38aad20f85cc7db935005079d371746087":
        raise SystemExit("cycle9 chamber parent evidence mismatch")

    body = initial_body()
    checkpoint = checkpoint_dict(body)
    restored = restore_body(checkpoint)
    policy = meta_compiler_policy_from_body(restored)
    if policy.rules or policy.generation != 0:
        raise SystemExit("cycle9 initial compiler policy is not empty")

    payload = {
        "schema": "arte.recursive_meta_compiler_chamber_freeze/v1",
        "candidate_source_head_sha": args.candidate_head,
        "parent_evidence_head_sha": precommit["parent_evidence_head_sha"],
        "parent_outcome_receipt_sha256": precommit["parent_outcome_receipt_sha256"],
        "initial_body_hash": body.morphology.fingerprint(),
        "initial_policy_fingerprint": policy.fingerprint(),
        "source_frozen_before_hidden_seed": True,
        "future_generation_source_edits_allowed": False,
        "checkpoint": checkpoint,
    }
    Path(args.output).write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    summary = dict(payload)
    summary.pop("checkpoint")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
