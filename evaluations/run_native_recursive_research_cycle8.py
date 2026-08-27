from __future__ import annotations

import argparse
import json
from pathlib import Path

from arte_cognition.autonomous_generator_mutation import (
    AutonomousMetaMutationCompiler,
    MetaFailureCertificate,
    credit_autonomous_selection,
)
from arte_cognition.executable_morphology import MorphologyMutator
from arte_cognition.native_representation_genesis import NativeRepresentationGenesisEngine
from arte_cognition.native_representation_generator_language import (
    CompositionalRepresentationGenesisEngine,
    expression_representation_programs,
    generator_policies,
)
from arte_cognition.self_evolving_body_checkpoint import checkpoint_dict, restore_body
from evaluations.freeze_native_recursive_cycle8_autonomous_mutation import (
    certificate_from_prior,
    parent_genome,
)
from evaluations.run_native_recursive_research_cycle7 import _hidden_tasks, _residual


def _read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-head", required=True)
    parser.add_argument("--candidate-head", required=True)
    parser.add_argument("--precommit", required=True)
    parser.add_argument("--cycle6-receipt", required=True)
    parser.add_argument("--cycle7-receipt", required=True)
    parser.add_argument("--mutation-freeze", required=True)
    parser.add_argument("--hidden-seed-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    precommit = _read(args.precommit)
    cycle6 = _read(args.cycle6_receipt)
    cycle7 = _read(args.cycle7_receipt)
    freeze = _read(args.mutation_freeze)
    if precommit.get("parent_evidence_head_sha") != args.parent_head:
        raise SystemExit("cycle8 parent head mismatch")
    if precommit.get("parent_outcome_receipt_sha256") != cycle7.get("outcome_receipt_sha256"):
        raise SystemExit("cycle8 parent outcome mismatch")
    if freeze.get("candidate_head_sha") != args.candidate_head:
        raise SystemExit("cycle8 candidate freeze head mismatch")
    if freeze.get("hidden_task_information_consumed") or freeze.get("current_outcomes_consumed"):
        raise SystemExit("cycle8 proposal freeze contaminated")

    parent = parent_genome()
    compiler = AutonomousMetaMutationCompiler()
    full_cert = certificate_from_prior(cycle6, cycle7, wrong_layer=False)
    wrong_cert = certificate_from_prior(cycle6, cycle7, wrong_layer=True)
    full_selection = compiler.propose(parent, full_cert)
    wrong_selection = compiler.propose(parent, wrong_cert)
    if full_selection.trace_hash() != freeze["full_selection"]["selection_trace_hash"]:
        raise SystemExit("cycle8 full autonomous selection replay mismatch")
    if wrong_selection.trace_hash() != freeze["wrong_selection"]["selection_trace_hash"]:
        raise SystemExit("cycle8 wrong autonomous selection replay mismatch")

    full_body = restore_body(dict(freeze["full_checkpoint"]))
    wrong_body = restore_body(dict(freeze["wrong_checkpoint"]))
    full_origin = full_cert.certificate_id
    full_policies = generator_policies(full_body.morphology, expected_origin_residual_id=full_origin)
    if len(full_policies) != 1:
        raise SystemExit("cycle8 autonomously selected full child lacks generator policy")

    seed = int(Path(args.hidden_seed_file).read_text(encoding="utf-8").strip())
    count = int(precommit["resource_contract"]["hidden_task_count"])
    support_count = int(precommit["resource_contract"]["support_examples_per_task"])
    tasks = _hidden_tasks(seed, count, support_count)
    engine = CompositionalRepresentationGenesisEngine(candidate_budget=1)

    full = 0
    remove = 0
    wrong = 0
    restart_equal = True
    parent = parent_genome()

    for task in tasks:
        residual = _residual(task)
        candidate = engine.generate(
            full_body.morphology,
            residual,
            task.support,
            expected_generator_origin_residual_id=full_origin,
        )[0]
        child = MorphologyMutator().apply(full_body.morphology, candidate.mutation)
        task_body = restore_body(dict(freeze["full_checkpoint"]))
        task_body.morphology = child
        restored_task = restore_body(checkpoint_dict(task_body))
        program = expression_representation_programs(
            restored_task.morphology,
            expected_artifact_type=task.artifact_type,
            expected_residual_id=task.task_id,
        )[0]
        prediction = program.execute(task.query)
        full += int(prediction == task.hidden_output)
        restarted = restore_body(checkpoint_dict(restored_task))
        restart_program = expression_representation_programs(
            restarted.morphology,
            expected_artifact_type=task.artifact_type,
            expected_residual_id=task.task_id,
        )[0]
        restart_equal = restart_equal and (
            restart_program.fingerprint() == program.fingerprint()
            and restart_program.execute(task.query) == prediction
        )

        try:
            parent_candidate = NativeRepresentationGenesisEngine(candidate_budget=4096).generate(
                parent, residual, task.support
            )[0]
            parent_child = MorphologyMutator().apply(parent, parent_candidate.mutation)
            if expression_representation_programs(
                parent_child,
                expected_artifact_type=task.artifact_type,
                expected_residual_id=task.task_id,
            ):
                remove += 1
        except ValueError:
            pass

        try:
            wrong_candidate = engine.generate(
                wrong_body.morphology,
                residual,
                task.support,
                expected_generator_origin_residual_id=wrong_cert.certificate_id,
            )[0]
            wrong_child = MorphologyMutator().apply(wrong_body.morphology, wrong_candidate.mutation)
            wrong_program = expression_representation_programs(
                wrong_child,
                expected_artifact_type=task.artifact_type,
                expected_residual_id=task.task_id,
            )[0]
            wrong += int(wrong_program.execute(task.query) == task.hidden_output)
        except ValueError:
            pass

    full_rate = full / count
    remove_rate = remove / count
    wrong_rate = wrong / count
    if not (full == count and full_rate > remove_rate and full_rate > wrong_rate and restart_equal):
        raise SystemExit(
            f"cycle8 hidden transfer failed full={full} remove={remove} wrong={wrong} restart={restart_equal}"
        )

    # Only now, after hidden outcomes, may credit enter inherited BODY state.
    credit_effect = credit_autonomous_selection(
        full_body,
        full_selection,
        full_useful_rate=full_rate,
        remove_useful_rate=remove_rate,
        wrong_useful_rate=wrong_rate,
        task_ref="cycle8-hidden-fresh-expression-suite",
    )
    credited_score = full_body.mutation_strategy.score(full_selection.selected.family)
    credited_support = full_body.mutation_strategy.support_map().get(full_selection.selected.family, 0)
    credited_episode_count = len(full_body.experience_archive.episodes)
    post_credit = restore_body(checkpoint_dict(full_body))
    credit_restart_equal = bool(
        post_credit.morphology.fingerprint() == full_body.morphology.fingerprint()
        and post_credit.mutation_strategy == full_body.mutation_strategy
        and post_credit.experience_archive.episodes == full_body.experience_archive.episodes
        and len(generator_policies(post_credit.morphology, expected_origin_residual_id=full_origin)) == 1
    )

    wrong_credit_effect = credit_autonomous_selection(
        wrong_body,
        wrong_selection,
        full_useful_rate=wrong_rate,
        remove_useful_rate=full_rate,
        wrong_useful_rate=remove_rate,
        task_ref="cycle8-wrong-certificate-control",
    )
    wrong_positive_score = wrong_body.mutation_strategy.score(wrong_selection.selected.family)

    pass_credit = bool(
        credit_effect > 0.0
        and credited_score > 0.0
        and credited_support == 1
        and credited_episode_count == 1
        and credit_restart_equal
        and wrong_credit_effect == 0.0
        and wrong_positive_score == 0.0
    )
    if not pass_credit:
        raise SystemExit("cycle8 native causal credit contract failed")

    receipt = {
        "schema": "arte.native_recursive_research_cycle8/v1",
        "generation": 8,
        "problem_detector": "AUTONOMOUS_GENERATOR_MUTATION_CREDIT_UNPROVEN",
        "parent_evidence_head_sha": args.parent_head,
        "candidate_implementation_head_sha": args.candidate_head,
        "proposal_count": len(full_selection.proposals),
        "autonomously_selected_family": full_selection.selected.family,
        "wrong_certificate_selected_family": wrong_selection.selected.family,
        "selection_trace_hash": full_selection.trace_hash(),
        "proposal_selected_before_hidden_seed": True,
        "proposal_generation_uses_current_outcomes": False,
        "hidden_task_count": count,
        "full_hidden_useful_count": full,
        "full_hidden_useful_rate": full_rate,
        "remove_hidden_useful_count": remove,
        "remove_hidden_useful_rate": remove_rate,
        "wrong_hidden_useful_count": wrong,
        "wrong_hidden_useful_rate": wrong_rate,
        "fresh_transfer_of_autonomous_mutation": True,
        "causal_credit_effect": credit_effect,
        "credited_strategy_score": credited_score,
        "credited_strategy_support": credited_support,
        "credited_experience_count": credited_episode_count,
        "credit_checkpoint_restart_equal": credit_restart_equal,
        "wrong_control_credit_effect": wrong_credit_effect,
        "wrong_control_positive_strategy_score": wrong_positive_score,
        "native_structural_proposal_selection_established": True,
        "native_causal_credit_established": True,
        "source_code_autonomous_self_modification_established": False,
        "external_structural_intervention_still_required": True,
        "eligible_claim": "BOUNDED_NATIVE_AUTONOMOUS_GENERATOR_MUTATION_SELECTION_AND_CREDIT",
        "recursive_acceleration_established": False,
        "broad_capability_improvement_established": False,
        "official_benchmark_used": False,
        "external_claim_authority": False,
        "next_problem_detectors": ["META_MUTATION_COMPILER_SELF_IMPROVEMENT_UNPROVEN"],
        "AGI": False,
        "ASI": False,
    }
    import hashlib
    material = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    receipt["outcome_receipt_sha256"] = hashlib.sha256(material).hexdigest()
    Path(args.output).write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
