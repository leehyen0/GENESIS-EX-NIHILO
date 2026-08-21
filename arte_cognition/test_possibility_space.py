import unittest

from arte_cognition.possibility_space import Fact, OperatorSpec, PossibilitySpaceGenerator


class PossibilitySpaceTests(unittest.TestCase):
    def setUp(self):
        self.g = PossibilitySpaceGenerator()
        self.facts = [Fact("servo", "moves", "up")]
        self.spec = OperatorSpec(
            relation_opposites={"moves": "does_not_move", "does_not_move": "moves"},
            object_complements={"up": "down", "down": "up"},
        )

    def test_unknown_opposite_is_not_invented(self):
        out = self.g.expand(self.facts, ["OPPOSITE"], OperatorSpec(), budget=8)
        self.assertEqual(out, [])

    def test_modal_modes_generate_distinct_structures(self):
        out = self.g.expand(
            self.facts,
            ["EXPLICIT", "OPPOSITE", "COMPLEMENT", "ABSENCE", "COUNTERFACTUAL"],
            self.spec,
            budget=16,
        )
        modes = {x.mode for x in out}
        self.assertTrue({"EXPLICIT", "OPPOSITE", "COMPLEMENT", "ABSENCE", "COUNTERFACTUAL"} <= modes)

    def test_imaginary_possibility_is_query_not_evidence(self):
        out = self.g.expand(self.facts, ["IMAGINARY"], self.spec, budget=8)
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0].asserted)
        self.assertTrue(out[0].query_targets[0].startswith("LATENT::"))
        self.assertEqual(out[0].facts, tuple(self.facts))

    def test_budget_prevents_combinatorial_all_active_explosion(self):
        facts = [Fact(f"n{i}", "moves", "up") for i in range(10)]
        out = self.g.expand(
            facts,
            ["EXPLICIT", "OPPOSITE", "COMPLEMENT", "ABSENCE", "COUNTERFACTUAL", "IMAGINARY"],
            self.spec,
            budget=7,
        )
        self.assertEqual(len(out), 7)

    def test_generation_deduplicates_candidates(self):
        facts = [Fact("servo", "moves", "up"), Fact("servo", "moves", "up")]
        out = self.g.expand(facts, ["ABSENCE"], self.spec, budget=10)
        signatures = [x.signature for x in out]
        self.assertEqual(len(signatures), len(set(signatures)))


if __name__ == "__main__":
    unittest.main()
