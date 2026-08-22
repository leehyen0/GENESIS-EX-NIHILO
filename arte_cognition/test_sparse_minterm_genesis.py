from __future__ import annotations

import unittest

from arte_cognition.causal_predicate_genesis import BooleanCausalPredicateGenesisEngine
from arte_cognition.intervention_surface_genesis import InterventionSurfaceGenesisEngine
from arte_cognition.sparse_minterm_genesis import SparseMintermCausalGenesisEngine


class SparseMintermCausalGenesisTests(unittest.TestCase):
    def test_generation_four_contains_three_minterm_signatures_absent_from_complete_g3(self):
        variables = ["x", "z"]
        descriptors = InterventionSurfaceGenesisEngine(budget=256).generate(variables)
        g3_engine = BooleanCausalPredicateGenesisEngine(
            model_budget=16384, max_literals_per_term=3, max_terms=2
        )
        g3 = g3_engine.generate_novel(variables, descriptors, (), ())
        self.assertFalse(g3_engine.last_truncated)
        self.assertGreater(g3_engine.last_unique_signature_count, 2048)

        g4_engine = SparseMintermCausalGenesisEngine(model_budget=4096, max_minterms=3)
        g4 = g4_engine.generate_novel(variables, descriptors, (), [item.model for item in g3])
        self.assertFalse(g4_engine.last_truncated)
        self.assertTrue(g4)
        self.assertTrue(any(
            item.model.generation == 4
            and item.model.origin == "GENERATED_SPARSE_MINTERM"
            and len(item.minterms) == 3
            for item in g4
        ))
        g3_signatures = {tuple(sorted(item.model.predictions)) for item in g3}
        self.assertTrue(all(tuple(sorted(item.model.predictions)) not in g3_signatures for item in g4))


if __name__ == "__main__":
    unittest.main()
