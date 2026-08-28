from __future__ import annotations

import unittest

from arte_cognition.body_policy_generation import generate_body_candidates
from arte_cognition.executable_morphology import (
    MorphologyGenome,
    MorphologyMutator,
    OrganKind,
    OrganSpec,
    PressureVector,
)
from arte_cognition.morphology_genesis import MorphologyResidual
from arte_cognition.native_recursive_research import NativeMetaMorphologyGenesisEngine


def parent_genome() -> MorphologyGenome:
    organs = (
        OrganSpec("generator", OrganKind.GENERATOR, produces=("candidate",), implementation_ref="bootstrap://generator"),
        OrganSpec("mutator", OrganKind.MUTATOR, consumes=("candidate",), produces=("mutation",), implementation_ref="bootstrap://mutator"),
        OrganSpec("governor", OrganKind.GOVERNOR),
        OrganSpec("archive", OrganKind.ARCHIVE),
    )
    return MorphologyGenome(organs=organs, edges=(), event_order=("generator", "mutator", "governor", "archive"))


def inherited_child(origin: str, family: str) -> MorphologyGenome:
    genome = parent_genome()
    residual = MorphologyResidual(origin, PressureVector(human_dependency=1.0, theory_blindspot=0.25))
    rows = NativeMetaMorphologyGenesisEngine(candidate_budget=16).generate(genome, (residual,))
    candidate = next(row for row in rows if row.operation_family == family)
    return MorphologyMutator().apply(genome, candidate.mutation)


def fresh(residual_id: str = "fresh-B") -> MorphologyResidual:
    return MorphologyResidual(residual_id, PressureVector(human_dependency=1.0, theory_blindspot=0.25))


class BodyPolicyGenerationTests(unittest.TestCase):
    def test_remove_parent_has_no_policy_and_nominal_budget(self):
        result = generate_body_candidates(parent_genome(), (fresh(),), nominal_budget=1)
        self.assertEqual(result.effective_budget, 1)
        self.assertEqual(result.policy_fingerprints, ())
        self.assertEqual(len(result.candidate_ids), 1)
        self.assertFalse(result.current_outcomes_consumed)

    def test_generator_child_transfers_origin_A_policy_to_fresh_B(self):
        child = inherited_child("origin-A", "CHANGE_GENERATOR_POLICY")
        result = generate_body_candidates(
            child,
            (fresh("fresh-B"),),
            nominal_budget=1,
            expected_policy_origin_residual_id="origin-A",
        )
        self.assertEqual(result.fresh_residual_ids, ("fresh-B",))
        self.assertGreater(result.effective_budget, 1)
        self.assertGreater(len(result.candidate_ids), 1)
        self.assertEqual(len(result.policy_fingerprints), 1)
        self.assertFalse(result.current_outcomes_consumed)

    def test_mutator_child_changes_fresh_candidate_family_priority(self):
        parent = generate_body_candidates(parent_genome(), (fresh("fresh-B"),), nominal_budget=1)
        child = inherited_child("origin-A", "CHANGE_MUTATOR_POLICY")
        result = generate_body_candidates(
            child,
            (fresh("fresh-B"),),
            nominal_budget=1,
            expected_policy_origin_residual_id="origin-A",
        )
        self.assertEqual(parent.operation_families, ("CHANGE_GENERATOR_POLICY",))
        self.assertEqual(result.operation_families, ("CHANGE_MUTATOR_POLICY",))
        self.assertNotEqual(result.candidate_ids, parent.candidate_ids)

    def test_policy_origin_shuffle_fails_closed(self):
        child = inherited_child("origin-C", "CHANGE_GENERATOR_POLICY")
        with self.assertRaisesRegex(ValueError, "RESIDUAL_MISMATCH"):
            generate_body_candidates(
                child,
                (fresh("fresh-B"),),
                nominal_budget=1,
                expected_policy_origin_residual_id="origin-A",
            )

    def test_restart_is_deterministic_on_fresh_residual(self):
        child_a = inherited_child("origin-A", "CHANGE_MUTATOR_POLICY")
        child_b = inherited_child("origin-A", "CHANGE_MUTATOR_POLICY")
        a = generate_body_candidates(
            child_a,
            (fresh("fresh-B"),),
            nominal_budget=1,
            expected_policy_origin_residual_id="origin-A",
        )
        b = generate_body_candidates(
            child_b,
            (fresh("fresh-B"),),
            nominal_budget=1,
            expected_policy_origin_residual_id="origin-A",
        )
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
