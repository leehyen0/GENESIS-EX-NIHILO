from __future__ import annotations

import unittest

from arte_cognition.polynomial_relation_genesis import (
    WorldDerivedPolynomialRelationInducer,
    derive_polynomial_relation_policy,
    select_authorized_polynomial_relation,
)
from evaluations.run_polynomial_relation_transfer import _world


class PolynomialRelationRestartTests(unittest.TestCase):
    def _training(self):
        return (
            _world("restart-train-a", "alpha", "SOFTWARE", 1.0, 9.0),
            _world("restart-train-b", "beta", "SOFTWARE", 7.0, -13.0),
        )

    def _generate(self):
        worlds = self._training()
        inducer = WorldDerivedPolynomialRelationInducer(degree=2, min_repeats=2)
        assessment = inducer.assess_residual(worlds, (0, 0), 2)
        return inducer, worlds, inducer.generate_candidates(assessment, worlds)

    def test_fresh_inducer_reconstructs_exact_nonlinear_partition_without_object_reuse(self):
        inducer_a, _, schemas_a = self._generate()
        inducer_b, _, schemas_b = self._generate()

        self.assertIsNot(inducer_a, inducer_b)
        self.assertEqual(len(schemas_a), 2)
        self.assertEqual(len(schemas_b), 2)
        self.assertTrue(all(a is not b for a, b in zip(schemas_a, schemas_b)))
        self.assertEqual(
            [
                (s.schema_id, s.positive_monomial, s.negative_monomial, s.path_signs)
                for s in schemas_a
            ],
            [
                (s.schema_id, s.positive_monomial, s.negative_monomial, s.path_signs)
                for s in schemas_b
            ],
        )
        self.assertEqual(
            {(s.positive_monomial, s.negative_monomial) for s in schemas_b},
            {((0, 2, 0), (1, 0, 1))},
        )

    def test_fresh_reconstruction_has_no_authority_without_external_receipts(self):
        _, _, schemas = self._generate()
        policy = derive_polynomial_relation_policy(schemas, ())

        self.assertEqual(policy.allowed_schema_ids, ())
        self.assertIsNone(select_authorized_polynomial_relation(schemas, policy))

    def test_fresh_heldout_world_can_match_nonlinear_structure_but_cannot_self_authorize(self):
        inducer, _, schemas = self._generate()
        heldout = _world("restart-heldout", "omega", "CAUSAL_WORLD", 11.0, 31.0)

        self.assertEqual(sum(int(inducer.matches(schema, heldout)) for schema in schemas), 2)
        policy = derive_polynomial_relation_policy(schemas, ())
        self.assertIsNone(select_authorized_polynomial_relation(schemas, policy))


if __name__ == "__main__":
    unittest.main()
