from __future__ import annotations

import unittest

from arte_cognition.executable_morphology import (
    EdgeSpec,
    ExperienceArchive,
    ExperienceUnit,
    GoalField,
    MorphologyCompiler,
    MorphologyGenome,
    MorphologyMutation,
    MorphologyMutator,
    MutationLevel,
    OrganKind,
    OrganSpec,
    PressureVector,
    meta_productivity,
    quotient_equivalent,
    split_pressure,
)


def base_genome() -> MorphologyGenome:
    organs = (
        OrganSpec("source", OrganKind.SOURCE, produces=("evidence",)),
        OrganSpec("question", OrganKind.QUESTION, consumes=("evidence",), produces=("probe",)),
        OrganSpec("world", OrganKind.WORLD, consumes=("probe",), produces=("outcome",)),
        OrganSpec("verifier", OrganKind.VERIFIER, consumes=("outcome",), produces=("authorized_outcome",)),
        OrganSpec("governor", OrganKind.GOVERNOR, consumes=("authorized_outcome",), produces=("decision",)),
        OrganSpec("archive", OrganKind.ARCHIVE, consumes=("decision",), produces=("experience",)),
    )
    edges = (
        EdgeSpec("e0", "source", "question", "evidence"),
        EdgeSpec("e1", "question", "world", "probe"),
        EdgeSpec("e2", "world", "verifier", "outcome"),
        EdgeSpec("e3", "verifier", "governor", "authorized_outcome", authority_required=True),
        EdgeSpec("e4", "governor", "archive", "decision"),
    )
    return MorphologyGenome(organs, edges, tuple(o.organ_id for o in organs))


class ExecutableMorphologyTests(unittest.TestCase):
    def test_compiler_has_no_generation_number_ladder(self):
        genome = base_genome()
        schedule = MorphologyCompiler.compile(genome)
        self.assertEqual(schedule[0], "source")
        self.assertNotIn("G7", repr(genome))
        self.assertNotIn("G8", repr(genome))

    def test_causal_alias_opens_split_only_after_more_compute_failure(self):
        self.assertFalse(split_pressure(True, False))
        self.assertFalse(split_pressure(False, True))
        self.assertTrue(split_pressure(True, True))

    def test_structural_mutation_can_add_new_organ_and_rewire_without_generation_case(self):
        genome = base_genome()
        mutator = MorphologyMutator()
        mutation = MorphologyMutation(
            "m1",
            MutationLevel.TOPOLOGY,
            "ADD_ORGAN",
            {
                "organ": {
                    "organ_id": "representation",
                    "kind": "REPRESENTATION",
                    "consumes": ["evidence"],
                    "produces": ["evidence2"],
                    "implementation_ref": "shadow://generated-representation",
                    "provenance": ["residual::alias-collision"],
                }
            },
            genome.fingerprint(),
        )
        expanded = mutator.apply(genome, mutation)
        self.assertIn("representation", expanded.organ_map())
        self.assertEqual(expanded.validate(), ())

        mutation2 = MorphologyMutation(
            "m2",
            MutationLevel.TOPOLOGY,
            "ADD_EDGE",
            {
                "edge": {
                    "edge_id": "e5",
                    "source": "source",
                    "target": "representation",
                    "artifact_type": "evidence",
                }
            },
            expanded.fingerprint(),
        )
        expanded2 = mutator.apply(expanded, mutation2)
        self.assertIn("e5", {edge.edge_id for edge in expanded2.edges})

    def test_mutation_parent_mismatch_fails_closed(self):
        genome = base_genome()
        mutation = MorphologyMutation(
            "bad", MutationLevel.TOPOLOGY, "REMOVE_EDGE", {"edge_id": "e0"}, "wrong-parent"
        )
        with self.assertRaisesRegex(ValueError, "MUTATION_PARENT_MISMATCH"):
            MorphologyMutator().apply(genome, mutation)

    def test_constitution_change_requires_highest_level(self):
        genome = base_genome()
        low = MorphologyMutation(
            "c0", MutationLevel.GOAL_GOVERNANCE, "CONSTITUTION_EPOCH", {}, genome.fingerprint()
        )
        with self.assertRaisesRegex(ValueError, "CONSTITUTION_CHANGE_REQUIRES_LEVEL_7"):
            MorphologyMutator().apply(genome, low)

        high = MorphologyMutation(
            "c1", MutationLevel.CONSTITUTION, "CONSTITUTION_EPOCH", {}, genome.fingerprint()
        )
        amended = MorphologyMutator().apply(genome, high)
        self.assertEqual(amended.constitution_epoch, 1)

    def test_goal_field_is_pressure_driven_and_can_target_meta_improvement(self):
        goals = GoalField().generate(
            PressureVector(human_dependency=1.0, identifiability_deficit=0.8, capability_residual=0.4)
        )
        surfaces = {goal.target_surface for goal in goals}
        self.assertIn("GENERATOR_OR_COMPILER", surfaces)
        self.assertIn("EVIDENCE_SOURCE_OR_QUESTION", surfaces)

    def test_archive_preserves_failure_as_experience_not_only_success(self):
        archive = ExperienceArchive()
        episode = ExperienceUnit(
            episode_id="failure-1",
            pre_body_hash=base_genome().fingerprint(),
            source_refs=("external://benchmark",),
            task_ref="task-1",
            benchmark_family="EXTERNAL",
            precommitted_hypotheses=("h0", "h1"),
            selected_goal_id="g",
            action_trace_hash="trace",
            outcome_summary="failed under hidden regime",
            success=False,
            uncertainty_before=0.4,
            uncertainty_after=0.8,
            mutation_ids=("m0",),
            notes=("retain-counterexample",),
        )
        self.assertTrue(archive.append(episode))
        self.assertFalse(archive.append(episode))
        self.assertFalse(archive.episodes["failure-1"].success)

    def test_behavioral_quotient_preserves_equivalence_class(self):
        groups = quotient_equivalent({"a": (1, 0, 1), "b": (1, 0, 1), "c": (0, 1, 0)})
        self.assertIn(("a", ("a", "b")), groups)
        self.assertIn(("c", ("c",)), groups)

    def test_meta_productivity_penalizes_human_structural_intervention(self):
        low_human = meta_productivity(2.0, 1.0, 1.0, 0.0)
        high_human = meta_productivity(2.0, 1.0, 1.0, 2.0)
        self.assertGreater(low_human, high_human)


if __name__ == "__main__":
    unittest.main()
