from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.executable_morphology import ExperienceArchive, MorphologyGenome, MorphologyMutator, OrganKind, OrganSpec
from arte_cognition.meta_acceleration import MutationProgramDevelopmentState, MutationStrategyState
from arte_cognition.native_representation_generator_language import (
    derive_generator_language_mutation,
    generator_policies,
)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", required=True)
    parser.add_argument("--cycle6-receipt", required=True)
    parser.add_argument("--candidate-head", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    precommit = json.loads(Path(args.precommit).read_text(encoding="utf-8"))
    cycle6 = json.loads(Path(args.cycle6_receipt).read_text(encoding="utf-8"))
    if precommit.get("parent_outcome_receipt_sha256") != cycle6.get("outcome_receipt_sha256"):
        raise SystemExit("cycle7 parent receipt mismatch")
    if precommit.get("parent_core_body_sha256") != cycle6.get("candidate_core_body_hash"):
        raise SystemExit("cycle7 parent core BODY mismatch")

    origin = "cycle7-generator-language-origin"
    fossil = "cycle6::fixed-family-representation-generator-inadequacy"
    parent = parent_genome()
    mutation = derive_generator_language_mutation(parent, origin_residual_id=origin, failure_fossil=fossil)
    child = MorphologyMutator().apply(parent, mutation)
    body = SelfEvolvingResearchBody(
        runtime=PersistentCognitiveRuntime(),
        morphology=child,
        mutation_strategy=MutationStrategyState(),
        mutation_program_state=MutationProgramDevelopmentState(),
        experience_archive=ExperienceArchive(),
    )
    checkpoint = checkpoint_dict(body)
    restored = restore_body(checkpoint)
    policies = generator_policies(restored.morphology, expected_origin_residual_id=origin)
    if len(policies) != 1:
        raise SystemExit("cycle7 generator policy did not survive checkpoint")

    checkpoint_sha = hashlib.sha256(
        json.dumps(checkpoint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema": "arte.native_recursive_cycle7_generator_freeze/v1",
        "candidate_head_sha": args.candidate_head,
        "generator_policy_origin_residual_id": origin,
        "failure_fossil": fossil,
        "mutation_id": mutation.mutation_id,
        "mutation_level": int(mutation.level),
        "parent_morphology_fingerprint": parent.fingerprint(),
        "generator_child_morphology_fingerprint": child.fingerprint(),
        "generator_policy_fingerprint": policies[0].fingerprint(),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint": checkpoint,
        "hidden_task_information_consumed": False,
        "current_outcomes_consumed": False,
    }
    Path(args.output).write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "checkpoint"}, sort_keys=True))


if __name__ == "__main__":
    main()
