from __future__ import annotations

import unittest
from dataclasses import replace

from arte_cognition.epistemic_memory import EpistemicMemory
from arte_cognition.experiment_genesis import InterventionProposal
from arte_cognition.projection_generator_transform_grammar import parse_transform_program_ids


class TransformExperimentAliasProvenanceTests(unittest.TestCase):
    def test_same_exact_intervention_unions_transform_program_ancestry(self):
        base = InterventionProposal(
            experiment_id="EXPERIMENT::ALIAS",
            axis_id="AXIS::P",
            manipulated_variable="x",
            held_fixed=(("z", 0.0),),
            low_value=-2057.4569594629756,
            high_value=2057.4569594629756,
            predicted_low_side="LE_THRESHOLD",
            predicted_high_side="GT_THRESHOLD",
            reason=(
                "probe_scale=13716.379729753171 "
                "generator_transform_programs=GENERATOR_AST::LOG::ALPHA::0.75"
            ),
        )
        alias = replace(
            base,
            reason=(
                "probe_scale=13716.379729753173 "
                "generator_transform_programs=GENERATOR_AST::INV>LOG::ALPHA::0.75"
            ),
        )
        memory = EpistemicMemory()
        memory.remember_experiment(base)
        memory.remember_experiment(alias)
        stored = memory.experiments[base.experiment_id].proposal
        self.assertEqual(
            set(parse_transform_program_ids(stored)),
            {
                "GENERATOR_AST::LOG::ALPHA::0.75",
                "GENERATOR_AST::INV>LOG::ALPHA::0.75",
            },
        )
        self.assertEqual(stored.low_value, base.low_value)
        self.assertEqual(stored.high_value, base.high_value)

    def test_different_exact_action_does_not_merge_transform_ancestry(self):
        first = InterventionProposal(
            experiment_id="EXPERIMENT::SYNTHETIC_COLLISION",
            axis_id="AXIS::P",
            manipulated_variable="x",
            held_fixed=(),
            low_value=-1.0,
            high_value=1.0,
            predicted_low_side="LE_THRESHOLD",
            predicted_high_side="GT_THRESHOLD",
            reason="probe_scale=1 generator_transform_programs=A",
        )
        changed = replace(
            first,
            high_value=2.0,
            reason="probe_scale=2 generator_transform_programs=B",
        )
        memory = EpistemicMemory()
        memory.remember_experiment(first)
        memory.remember_experiment(changed)
        stored = memory.experiments[first.experiment_id].proposal
        self.assertEqual(stored.high_value, 2.0)
        self.assertEqual(parse_transform_program_ids(stored), ("B",))


if __name__ == "__main__":
    unittest.main()
