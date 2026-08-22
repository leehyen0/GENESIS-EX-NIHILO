from __future__ import annotations

import unittest

from arte_cognition.intervention_surface_genesis import InterventionSurfaceGenesisEngine


class InterventionSurfaceGenesisTests(unittest.TestCase):
    def test_surface_contains_expensive_block_delay_context_combinations(self):
        engine = InterventionSurfaceGenesisEngine(budget=256)
        rows = engine.generate(["x", "z"])
        self.assertFalse(engine.last_truncated)
        self.assertGreater(len(rows), 10)
        self.assertTrue(any(
            set(row.targets) == {"x", "z"}
            and row.blocked == ("z",)
            and row.delay_steps == 1
            and row.context_shift
            for row in rows
        ))

    def test_ids_are_semantic_not_evaluator_named(self):
        engine = InterventionSurfaceGenesisEngine()
        rows_a = engine.generate(["z", "x"])
        rows_b = engine.generate(["x", "z"])
        self.assertEqual([row.intervention_id for row in rows_a], [row.intervention_id for row in rows_b])
        self.assertTrue(all(row.intervention_id.startswith("GENINT::") for row in rows_a))

    def test_novel_removes_observed_without_shrinking_underlying_surface(self):
        engine = InterventionSurfaceGenesisEngine()
        rows = engine.generate(["x", "z"])
        novel = engine.novel(["x", "z"], [rows[0].intervention_id])
        self.assertEqual(len(novel), len(rows) - 1)
        self.assertNotIn(rows[0].intervention_id, {row.intervention_id for row in novel})


if __name__ == "__main__":
    unittest.main()
