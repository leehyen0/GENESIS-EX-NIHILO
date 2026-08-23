from __future__ import annotations

import unittest

from arte_cognition.response_partition_genesis import (
    WorldDerivedResponsePartitionInducer,
    derive_response_partition_policy,
    select_authorized_response_schema,
)
from arte_cognition.test_response_partition_genesis import _pair, _parallel_profile_world


class ResponsePartitionRestartTests(unittest.TestCase):
    def _training(self):
        return (
            _parallel_profile_world("restart-a", "alpha_restart", "SOFTWARE", 1.0),
            _parallel_profile_world("restart-b", "beta_restart", "SOFTWARE", 9.0),
        )

    def test_fresh_inducer_reconstructs_exact_partition_without_candidate_object_reuse(self):
        worlds = self._training()
        parent = WorldDerivedResponsePartitionInducer(min_repeats=2)
        parent_assessment = parent.assess_residual(worlds, (2, 2), 2)
        parent_partition = parent.derive_partition(worlds[0])
        parent_schemas = parent.generate_candidates(parent_assessment, worlds)

        descendant = WorldDerivedResponsePartitionInducer(min_repeats=2)
        descendant_assessment = descendant.assess_residual(worlds, (2, 2), 2)
        descendant_partition = descendant.derive_partition(worlds[0])
        descendant_schemas = descendant.generate_candidates(descendant_assessment, worlds)

        self.assertIsNot(parent, descendant)
        self.assertEqual(parent_partition.partition_id, descendant_partition.partition_id)
        self.assertEqual(parent_partition.profile_shapes, descendant_partition.profile_shapes)
        self.assertEqual(
            tuple(schema.schema_id for schema in parent_schemas),
            tuple(schema.schema_id for schema in descendant_schemas),
        )
        self.assertTrue(all(a is not b for a, b in zip(parent_schemas, descendant_schemas)))

    def test_fresh_reconstruction_is_deauthorized_until_external_reverification(self):
        worlds = self._training()
        descendant = WorldDerivedResponsePartitionInducer(min_repeats=2)
        schemas = descendant.generate_candidates(
            descendant.assess_residual(worlds, (2, 2), 2), worlds
        )
        self.assertEqual(len(schemas), 2)

        no_receipts = derive_response_partition_policy(schemas, (), 2, 2)
        self.assertIsNone(select_authorized_response_schema(schemas, no_receipts))

        selected = schemas[0]
        receipts = tuple(
            _pair(selected.schema_id, context, cls, 1.0)
            for context in ("restart-a", "restart-b")
            for cls in ("A", "B")
        )
        reverified = derive_response_partition_policy(schemas, receipts, 2, 2)
        restored = select_authorized_response_schema(schemas, reverified)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.schema_id, selected.schema_id)


if __name__ == "__main__":
    unittest.main()
