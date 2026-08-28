from __future__ import annotations

import argparse
import json
from pathlib import Path

from arte_cognition.executable_morphology import MorphologyGenome, MorphologyMutator, MutationLevel, OrganKind, OrganSpec
from arte_cognition.native_recursive_research import NativeMetaMorphologyGenesisEngine
from arte_cognition.native_representation_genesis import NativeRepresentationGenesisEngine
from evaluations.run_native_recursive_research_cycle7 import _hidden_tasks, _residual


def cycle6_parent_genome() -> MorphologyGenome:
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
    parser.add_argument("--hidden-seed-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    precommit = json.loads(Path(args.precommit).read_text(encoding="utf-8"))
    seed = int(Path(args.hidden_seed_file).read_text(encoding="utf-8").strip())
    task_count = int(precommit["resource_contract"]["hidden_task_count"])
    support_count = int(precommit["resource_contract"]["support_examples_per_task"])
    tasks = _hidden_tasks(seed, task_count, support_count)
    parent = cycle6_parent_genome()

    fixed_family_success = 0
    l1_count = 0
    native_expression_count = 0
    ordinary_candidate_count = 0
    for task in tasks:
        residual = _residual(task)
        try:
            NativeRepresentationGenesisEngine(candidate_budget=4096).generate(parent, residual, task.support)
            fixed_family_success += 1
        except ValueError:
            pass
        rows = NativeMetaMorphologyGenesisEngine(candidate_budget=4096).generate(parent, (residual,))
        ordinary_candidate_count += len(rows)
        for row in rows:
            l1_count += int(row.mutation.level == MutationLevel.REPRESENTATION_MEMORY_TOOL)
            child = MorphologyMutator().apply(parent, row.mutation)
            native_expression_count += int(
                any(str(organ.implementation_ref).startswith("native-repr-expr://") for organ in child.organs)
            )

    passed = fixed_family_success == 0 and l1_count == 0 and native_expression_count == 0
    payload = {
        "schema": "arte.cycle7_parent_unreachability/v1",
        "hidden_task_count": task_count,
        "parent_more_compute_candidate_budget": int(precommit["resource_contract"]["parent_more_compute_candidate_budget"]),
        "parent_fixed_family_success_count": fixed_family_success,
        "parent_ordinary_candidate_count": ordinary_candidate_count,
        "parent_l1_expression_candidate_count": l1_count,
        "parent_native_expression_count": native_expression_count,
        "parent_expression_unreachable": passed,
    }
    Path(args.output).write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if not passed:
        raise SystemExit("cycle7 cycle6-parent expression reachability unexpectedly nonzero")


if __name__ == "__main__":
    main()
