from __future__ import annotations

import unittest

from arte_cognition.comparison_functional_genesis import (
    WorldDerivedComparisonFunctionalInducer,
    derive_comparison_policy,
    select_authorized_comparison,
)
from evaluations.run_comparison_functional_transfer import _world


class ComparisonFunctionalRestartTests(unittest.TestCase):
    def _training(self):
        return (
            _world("restart-train-a", "alpha", "SOFTWARE", 1.0, 7.0),
            _world("restart-train-b", "beta", "SOFTWARE", 5.0, -11.0),
        )

    def _generate(self):
        worlds = self._training()
        inducer = WorldDerivedComparisonFunctionalInducer(coefficient_bound=2, min_repeats=2)
        assessment = inducer.assess_residual(worlds, (1, 1), (2, 2), 2)
        return inducer, worlds, inducer.generate_candidates(assessment, worlds)

    def test_fresh_inducer_reconstructs_exact_functional_partition_without_object_reuse(self):
        inducer_a, _, schemas_a = self._generate()
        inducer_b, _, schemas_b = self._generate()

        self.assertIsNot(inducer_a, inducer_b)
        self.assertEqual(len(schemas_a), 2)
        self.assertEqual(len(schemas_b), 2)
        self.assertTrue(all(a is not b for a, b in zip(schemas_a, schemas_b)))
        self.assertEqual(
            [(s.schema_id, s.coefficients, s.path_signs) for s in schemas_a],
            [(s.schema_id, s.coefficients, s.path_signs) for s in schemas_b],
        )
        self.assertEqual({s.coefficients for s in schemas_b}, {(1, -2, 1)})

    def test_fresh_reconstruction_has_no_authority_without_external_receipts(self):
        _, _, schemas = self._generate()
        policy = derive_comparison_policy(schemas, ())

        self.assertEqual(policy.allowed_schema_ids, ())
        self.assertIsNone(select_authorized_comparison(schemas, policy))

    def test_fresh_heldout_world_can_match_structure_but_cannot_self_authorize(self):
        inducer, _, schemas = self._generate()
        heldout = _world("restart-heldout", "omega", "CAUSAL_WORLD", 11.0, 23.0)

        self.assertEqual(sum(int(inducer.matches(schema, heldout)) for schema in schemas), 2)
        policy = derive_comparison_policy(schemas, ())
        self.assertIsNone(select_authorized_comparison(schemas, policy))


if __name__ == "__main__":
    unittest.main()
