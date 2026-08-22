from __future__ import annotations

import unittest

from arte_cognition.adaptive_cognition import QueryCandidate
from arte_cognition.causal_identification import GenerationScopedIdentifier
from arte_cognition.world_model_ecology import CausalWorldModel, ModelEvidence


class GenerationScopedIdentifierTests(unittest.TestCase):
    def test_older_generations_do_not_dilute_current_generation_eig(self):
        models = [
            CausalWorldModel("OLD_A", 1.0, (("q", "POSITIVE_EFFECT"),), generation=1),
            CausalWorldModel("OLD_B", 1.0, (("q", "NEGATIVE_EFFECT"),), generation=2),
            CausalWorldModel("G3_A", 1.0, (("q", "POSITIVE_EFFECT"),), generation=3),
            CausalWorldModel("G3_B", 1.0, (("q", "NEGATIVE_EFFECT"),), generation=3),
        ]
        snapshot = GenerationScopedIdentifier.snapshot(3, models, ())
        candidate = QueryCandidate(
            query_id="q",
            distinguishes={"G3_A": "POSITIVE_EFFECT", "G3_B": "NEGATIVE_EFFECT"},
            cost=1.0,
            reason="split generation 3",
        )
        ranked = GenerationScopedIdentifier.rank_interventions(
            [candidate], snapshot.compatible_model_ids, cost_exponent=0.0
        )
        self.assertEqual(snapshot.compatible_model_ids, ("G3_A", "G3_B"))
        self.assertEqual(len(ranked), 1)
        self.assertAlmostEqual(ranked[0].expected_information_gain, 1.0)

    def test_authoritative_evidence_collapses_exact_version_space(self):
        models = [
            CausalWorldModel("G3_A", 1.0, (("q", "POSITIVE_EFFECT"),), generation=3),
            CausalWorldModel("G3_B", 1.0, (("q", "NEGATIVE_EFFECT"),), generation=3),
        ]
        evidence = [
            ModelEvidence("e1", "q", "POSITIVE_EFFECT", "A", "ctx", True),
            ModelEvidence("e2", "q", "POSITIVE_EFFECT", "B", "ctx", True),
        ]
        snapshot = GenerationScopedIdentifier.snapshot(3, models, evidence)
        self.assertTrue(snapshot.identified)
        self.assertEqual(snapshot.identified_model_id, "G3_A")

    def test_unverified_evidence_cannot_remove_hypotheses(self):
        models = [
            CausalWorldModel("G3_A", 1.0, (("q", "POSITIVE_EFFECT"),), generation=3),
            CausalWorldModel("G3_B", 1.0, (("q", "NEGATIVE_EFFECT"),), generation=3),
        ]
        evidence = [ModelEvidence("e1", "q", "POSITIVE_EFFECT", "UNVERIFIED", "ctx", False)]
        snapshot = GenerationScopedIdentifier.snapshot(3, models, evidence)
        self.assertEqual(snapshot.compatible_model_ids, ("G3_A", "G3_B"))


if __name__ == "__main__":
    unittest.main()
