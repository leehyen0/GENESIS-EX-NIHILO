from __future__ import annotations

import unittest

from arte_cognition.equivalence_criterion_genesis import (
    WorldDerivedEquivalenceCriterionInducer,
    derive_equivalence_policy,
    select_authorized_equivalence,
)
from arte_cognition.latent_relation_ontology_genesis import OpaqueInterventionalWorld, contrast
from arte_cognition.response_partition_genesis import WorldDerivedResponsePartitionInducer
from arte_cognition.world_coupling import WorldOutcomePair
from evaluations.run_equivalence_criterion_transfer import main as run_external_equivalence_transfer


def _world(context_id: str, prefix: str, exponent: int) -> OpaqueInterventionalWorld:
    root, down, up, target, decoy = [f"{prefix}_{x}" for x in ("root", "down", "up", "target", "decoy")]
    nodes = (root, down, up, target, decoy)
    rows = []
    measure = lambda x: float(x) ** int(exponent)

    def add(source, effects):
        for repeat in range(2):
            low = [{node: 0.0 for node in nodes} for _ in range(3)]
            high = [{node: 0.0 for node in nodes} for _ in range(3)]
            high[0][source] = 1.0
            for destination, curve in effects:
                high[1][destination] = measure(curve[0])
                high[2][destination] = measure(curve[1])
            low[2][decoy] = high[2][decoy] = float(repeat + 1)
            rows.append(contrast(f"{context_id}:{source}:{repeat}", source, low, high))

    add(root, ((down, (4.0, 1.0)), (up, (1.0, 4.0))))
    add(down, ((target, (4.0, 1.0)),))
    add(up, ((target, (1.0, 4.0)),))
    return OpaqueInterventionalWorld(context_id, "OPAQUE", root, target, tuple(rows))


def _pair(cid, context, cls, effect, verified=True):
    return WorldOutcomePair(
        pair_id=f"{cid}:{context}:{cls}", experiment_id=cid, axis_id="EQ",
        source_id=f"s::{cls}", context_id=context, challenge_id=f"c::{context}", epoch=1,
        low_outcome=0.0, high_outcome=float(effect), low_value=0.0, high_value=1.0,
        matched_budget=True, externally_generated=True, issuer_id=f"i::{cls}",
        independence_class_id=cls if verified else "UNVERIFIED", authority_verified=verified,
    )


class EquivalenceCriterionGenesisTests(unittest.TestCase):
    def test_exact_numeric_response_partition_fails_across_measurement_systems(self):
        worlds = (_world("a", "alpha", 1), _world("b", "beta", 2))
        predecessor = WorldDerivedResponsePartitionInducer(min_repeats=2)
        assessment = predecessor.assess_residual(worlds, (2, 2), 2)
        self.assertEqual(predecessor.generate_candidates(assessment, worlds), ())
        for _ in range(16):
            self.assertEqual(predecessor.generate_candidates(assessment, worlds), ())

    def test_world_intersection_generates_two_measurement_invariant_criteria(self):
        worlds = (_world("a", "alpha", 1), _world("b", "beta", 2))
        inducer = WorldDerivedEquivalenceCriterionInducer(min_repeats=2)
        assessment = inducer.assess_residual(worlds, (0, 0), 2)
        criteria = inducer.generate_candidates(assessment, worlds)
        self.assertEqual(len(criteria), 2)
        self.assertNotEqual(criteria[0].constraints, criteria[1].constraints)

    def test_generated_criteria_transfer_to_unseen_measurement_exponent(self):
        train = (_world("a", "alpha", 1), _world("b", "beta", 2))
        heldout = _world("c", "omega", 3)
        inducer = WorldDerivedEquivalenceCriterionInducer(min_repeats=2)
        criteria = inducer.generate_candidates(inducer.assess_residual(train, (0, 0), 2), train)
        self.assertEqual(sum(int(inducer.matches(c, heldout)) for c in criteria), 2)

    def test_outcome_authority_is_not_part_of_candidate_generation(self):
        train = (_world("a", "alpha", 1), _world("b", "beta", 2))
        inducer = WorldDerivedEquivalenceCriterionInducer(min_repeats=2)
        criteria = inducer.generate_candidates(inducer.assess_residual(train, (0, 0), 2), train)
        criterion = criteria[0]
        one = (_pair(criterion.criterion_id, "a", "A", 1.0), _pair(criterion.criterion_id, "a", "B", 1.0))
        self.assertIsNone(select_authorized_equivalence(criteria, derive_equivalence_policy(criteria, one, 2, 2)))
        two = one + (_pair(criterion.criterion_id, "b", "A", 1.0), _pair(criterion.criterion_id, "b", "B", 1.0))
        selected = select_authorized_equivalence(criteria, derive_equivalence_policy(criteria, two, 2, 2))
        self.assertIsNotNone(selected)
        verifierless = tuple(_pair(criterion.criterion_id, p.context_id, "X", 1.0, False) for p in two)
        self.assertIsNone(select_authorized_equivalence(criteria, derive_equivalence_policy(criteria, verifierless, 2, 2)))

    def test_external_causal_measurement_equivalence_transfer(self):
        report = run_external_equivalence_transfer()
        self.assertEqual(report["status"], "PASS_BOUNDED_WORLD_DERIVED_MEASUREMENT_EQUIVALENCE_AND_PREOUTCOME_TRANSFER")
        self.assertEqual(report["predecessor_exact_profile_candidate_count"], 0)
        self.assertEqual(report["generated_equivalence_candidate_count"], 2)
        self.assertEqual(report["treatment_capability"], 1.0)
        self.assertEqual(report["remove_same_checkpoint_capability"], 0.0)
        self.assertEqual(report["structurally_valid_wrong_capability"], 0.0)


if __name__ == "__main__":
    unittest.main()
