import unittest

from arte_cognition.semantic_genesis import (
    ResidualObservation,
    SemanticGenesisEngine,
)


class SemanticGenesisTests(unittest.TestCase):
    def setUp(self):
        self.engine = SemanticGenesisEngine()

    def test_repeated_residual_structure_generates_concept(self):
        rows = [
            ResidualObservation("r1", ("hot", "vibration"), "FAIL"),
            ResidualObservation("r2", ("hot", "vibration"), "FAIL"),
            ResidualObservation("r3", ("cold", "stable"), "PASS"),
            ResidualObservation("r4", ("cold", "stable"), "PASS"),
        ]
        concepts = self.engine.propose_concepts(rows)
        self.assertTrue(concepts)
        self.assertEqual(concepts[0].status, "PROPOSAL_ONLY")
        self.assertGreater(concepts[0].information_gain, 0)

    def test_one_off_feature_is_not_promoted_to_concept(self):
        rows = [
            ResidualObservation("r1", ("rare", "shared"), "FAIL"),
            ResidualObservation("r2", ("shared",), "FAIL"),
            ResidualObservation("r3", ("other",), "PASS"),
            ResidualObservation("r4", ("other",), "PASS"),
        ]
        concepts = self.engine.propose_concepts(rows)
        self.assertFalse(any("rare" in c.defining_features for c in concepts))

    def test_heldout_counterexample_blocks_law(self):
        rows = [
            ResidualObservation("r1", ("hot",), "FAIL"),
            ResidualObservation("r2", ("hot",), "FAIL"),
            ResidualObservation("r3", ("hot",), "FAIL"),
            ResidualObservation("r4", ("cold",), "PASS"),
            ResidualObservation("h1", ("hot",), "PASS", heldout=True),
        ]
        concept = self.engine.propose_concepts(rows)[0]
        law = self.engine.induce_law(concept, rows)
        self.assertEqual(law.status, "HELDOUT_REFUTED")
        self.assertIn("h1", law.counterexamples)

    def test_reproduced_law_can_be_bounded_not_universal(self):
        rows = [
            ResidualObservation("r1", ("hot",), "FAIL"),
            ResidualObservation("r2", ("hot",), "FAIL"),
            ResidualObservation("r3", ("hot",), "FAIL"),
            ResidualObservation("r4", ("cold",), "PASS"),
            ResidualObservation("h1", ("hot",), "FAIL", heldout=True),
        ]
        concept = self.engine.propose_concepts(rows)[0]
        law = self.engine.induce_law(concept, rows)
        self.assertEqual(law.status, "BOUNDED_LAW")
        self.assertEqual(law.heldout_accuracy, 1.0)

    def test_missing_heldout_never_becomes_law(self):
        rows = [
            ResidualObservation("r1", ("hot",), "FAIL"),
            ResidualObservation("r2", ("hot",), "FAIL"),
            ResidualObservation("r3", ("hot",), "FAIL"),
            ResidualObservation("r4", ("cold",), "PASS"),
        ]
        concept = self.engine.propose_concepts(rows)[0]
        self.assertEqual(self.engine.induce_law(concept, rows).status, "HELDOUT_REQUIRED")

    def test_concept_budget_prevents_semantic_explosion(self):
        rows = []
        for i in range(12):
            rows.append(ResidualObservation(f"r{i}", (f"f{i%4}", f"g{i%3}"), "A" if i % 2 else "B"))
        engine = SemanticGenesisEngine(min_support=1, concept_budget=3)
        self.assertLessEqual(len(engine.propose_concepts(rows)), 3)

    def test_mixed_concept_generates_discriminating_query(self):
        rows = [
            ResidualObservation("r1", ("shared", "x"), "A"),
            ResidualObservation("r2", ("shared", "x"), "A"),
            ResidualObservation("r3", ("shared", "y"), "B"),
            ResidualObservation("r4", ("shared", "y"), "B"),
            ResidualObservation("r5", ("other", "z"), "C"),
        ]
        # Manually define a coarse concept to simulate a representation that still collapses regimes.
        from arte_cognition.semantic_genesis import ConceptCandidate
        coarse = ConceptCandidate("CONCEPT::shared", ("shared",), 4, 0.1, ("r1", "r2", "r3", "r4"))
        queries = self.engine.propose_queries(rows, [coarse])
        self.assertTrue(any(q.target_feature in {"x", "y"} for q in queries))


if __name__ == "__main__":
    unittest.main()
