from __future__ import annotations

import unittest

from arte_cognition.executable_morphology import EdgeSpec, MorphologyGenome, OrganKind, OrganSpec, PressureVector
from arte_cognition.morphology_genesis import (
    MorphologyEvaluation,
    MorphologyGenesisEngine,
    MorphologyResidual,
    derive_morphology_policy,
    select_authorized_morphology_candidate,
)


def genome() -> MorphologyGenome:
    organs = (
        OrganSpec("source", OrganKind.SOURCE, produces=("evidence",)),
        OrganSpec("rep_a", OrganKind.REPRESENTATION, consumes=("evidence",), produces=("feature",)),
        OrganSpec("rep_b", OrganKind.REPRESENTATION, consumes=("evidence",), produces=("feature",)),
        OrganSpec("planner", OrganKind.PLANNER, consumes=("feature",), produces=("action",)),
        OrganSpec("world", OrganKind.WORLD, consumes=("action",), produces=("outcome",)),
        OrganSpec("verifier", OrganKind.VERIFIER, consumes=("outcome",), produces=("authorized_outcome",)),
        OrganSpec("governor", OrganKind.GOVERNOR, consumes=("authorized_outcome",), produces=("decision",)),
        OrganSpec("archive", OrganKind.ARCHIVE, consumes=("decision",), produces=("experience",)),
    )
    edges = (
        EdgeSpec("e0", "source", "rep_a", "evidence"),
        EdgeSpec("e1", "rep_a", "planner", "feature"),
        EdgeSpec("e2", "planner", "world", "action"),
        EdgeSpec("e3", "world", "verifier", "outcome"),
        EdgeSpec("e4", "verifier", "governor", "authorized_outcome", authority_required=True),
        EdgeSpec("e5", "governor", "archive", "decision"),
    )
    return MorphologyGenome(organs, edges, tuple(o.organ_id for o in organs))


class MorphologyGenesisTests(unittest.TestCase):
    def test_rewire_candidates_are_generated_from_typed_failure_before_outcomes(self):
        body = genome()
        residual = MorphologyResidual(
            "r1",
            PressureVector(transfer_failure=1.0),
            failed_edge_ids=("e0",),
            implicated_organ_ids=("rep_a", "rep_b"),
        )
        engine = MorphologyGenesisEngine()
        candidates = engine.generate(body, (residual,))
        self.assertGreater(len(candidates), 0)
        self.assertTrue(all(not c.generation_uses_outcomes for c in candidates))
        self.assertIn("REWIRE_EDGE", {c.operation_family for c in candidates})

    def test_split_requires_alias_collision_and_more_compute_failure(self):
        body = genome()
        no_gate = MorphologyResidual(
            "r0", PressureVector(capability_residual=1.0),
            same_frozen_phenotype_different_outcome=True,
            more_compute_still_aliased=False,
            implicated_organ_ids=("rep_a",),
        )
        yes_gate = MorphologyResidual(
            "r1", PressureVector(capability_residual=1.0),
            same_frozen_phenotype_different_outcome=True,
            more_compute_still_aliased=True,
            implicated_organ_ids=("rep_a",),
        )
        engine = MorphologyGenesisEngine()
        self.assertNotIn("ADD_ORGAN", {c.operation_family for c in engine.generate(body, (no_gate,))})
        self.assertIn("ADD_ORGAN", {c.operation_family for c in engine.generate(body, (yes_gate,))})

    def test_efficiency_pressure_can_generate_schedule_mutation(self):
        body = genome()
        residual = MorphologyResidual(
            "r2", PressureVector(efficiency_pressure=1.0), implicated_organ_ids=("rep_a", "planner")
        )
        candidates = MorphologyGenesisEngine().generate(body, (residual,))
        self.assertIn("SET_EVENT_ORDER", {c.operation_family for c in candidates})

    def test_policy_requires_repeated_disjoint_external_authority(self):
        body = genome()
        residual = MorphologyResidual("r3", PressureVector(transfer_failure=1.0), failed_edge_ids=("e0",))
        candidates = MorphologyGenesisEngine().generate(body, (residual,))
        candidate = candidates[0]
        one = MorphologyEvaluation("x1", candidate.candidate_id, "ctx-a", "class-a", 1.0, 0.0, 0.0, 0.2, True, True, True)
        policy_one = derive_morphology_policy(candidates, (one,))
        self.assertEqual(policy_one.allowed_candidate_ids, ())

        two = MorphologyEvaluation("x2", candidate.candidate_id, "ctx-b", "class-b", 1.0, 0.0, 0.0, 0.1, True, True, True)
        policy_two = derive_morphology_policy(candidates, (one, two))
        self.assertIn(candidate.candidate_id, policy_two.allowed_candidate_ids)
        self.assertEqual(select_authorized_morphology_candidate(candidates, policy_two), candidate)

    def test_unverified_or_non_disjoint_evidence_cannot_authorize(self):
        body = genome()
        residual = MorphologyResidual("r4", PressureVector(transfer_failure=1.0), failed_edge_ids=("e0",))
        candidates = MorphologyGenesisEngine().generate(body, (residual,))
        candidate = candidates[0]
        rows = (
            MorphologyEvaluation("x1", candidate.candidate_id, "ctx-a", "class-a", 1.0, 0.0, 0.0, 0.2, True, False, True),
            MorphologyEvaluation("x2", candidate.candidate_id, "ctx-b", "class-b", 1.0, 0.0, 0.0, 0.2, True, True, False),
        )
        policy = derive_morphology_policy(candidates, rows)
        self.assertEqual(policy.allowed_candidate_ids, ())


if __name__ == "__main__":
    unittest.main()
