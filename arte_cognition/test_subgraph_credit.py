import unittest

from arte_cognition.subgraph_credit import MinimumCausalSubgraphFinder, SubgraphEvaluation


class MinimumSubgraphTests(unittest.TestCase):
    def test_finds_smallest_outcome_sufficient_subgraph(self):
        finder = MinimumCausalSubgraphFinder(tolerance=0.02)
        full = ("A", "B", "C")
        evaluations = {
            tuple(sorted(("A", "B", "C"))): SubgraphEvaluation(("A", "B", "C"), 1.00, 3.0),
            tuple(sorted(("A", "B"))): SubgraphEvaluation(("A", "B"), 0.99, 2.0),
            tuple(sorted(("A", "C"))): SubgraphEvaluation(("A", "C"), 0.70, 2.0),
            tuple(sorted(("B", "C"))): SubgraphEvaluation(("B", "C"), 0.75, 2.0),
            tuple(sorted(("A",))): SubgraphEvaluation(("A",), 0.60, 1.0),
        }
        result = finder.find(full, evaluations)
        self.assertEqual(result.status, "MINIMUM_CAUSALLY_SUFFICIENT_SUBGRAPH")
        self.assertEqual(result.modules, ("A", "B"))
        self.assertAlmostEqual(result.outcome_gap_from_full, 0.01)

    def test_missing_full_intervention_is_not_imputed(self):
        finder = MinimumCausalSubgraphFinder()
        with self.assertRaises(ValueError):
            finder.find(("A", "B"), {})

    def test_sparse_plan_contains_full_and_single_removals(self):
        finder = MinimumCausalSubgraphFinder()
        plan = finder.required_sparse_ablation_plan(("A", "B", "C"), include_pairs=False)
        self.assertIn(("A", "B", "C"), plan)
        self.assertIn(("B", "C"), plan)
        self.assertIn(("A", "C"), plan)
        self.assertIn(("A", "B"), plan)
        self.assertEqual(len(plan), 4)


if __name__ == "__main__":
    unittest.main()
