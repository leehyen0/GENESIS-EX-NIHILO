from __future__ import annotations

import unittest

from arte_cognition.external_experience import (
    BenchmarkAuthorityGate,
    BenchmarkEpisode,
    ExposureClass,
    KnowledgeAssimilationEpisode,
    KnowledgeAssimilationGate,
    PreReadCommitment,
    SourceRecord,
    request_source_from_identifiability_deficit,
)


class ExternalExperienceTests(unittest.TestCase):
    def test_source_request_excludes_prior_family(self):
        request = request_source_from_identifiability_deficit(
            "same phenotype, different outcome",
            ("paper-family-a", "paper-family-b"),
            "independent experiment",
        )
        self.assertTrue(request.source_disjoint_required)
        self.assertEqual(request.excluded_source_families, ("paper-family-a", "paper-family-b"))

    def test_external_knowledge_requires_behavior_transfer_and_causal_controls(self):
        source = SourceRecord.from_content(
            "s1", "https://example.invalid/source", b"content", "paper", "2026-08-23T17:14:00+09:00"
        )
        commitment = PreReadCommitment("c1", "r1", ("p",), ("q",), 0.8)
        positive = KnowledgeAssimilationEpisode(
            "k1", ("s1",), "c1", ("claim",), "fresh-task", True, True, 0.4, 0.3, True
        )
        self.assertTrue(KnowledgeAssimilationGate.promoteable(positive, (source,), commitment))

        no_transfer = KnowledgeAssimilationEpisode(
            "k2", ("s1",), "c1", ("claim",), "fresh-task", True, False, 0.4, 0.3, True
        )
        self.assertFalse(KnowledgeAssimilationGate.promoteable(no_transfer, (source,), commitment))

    def test_public_dev_score_is_experience_not_heldout_authority(self):
        episode = BenchmarkEpisode(
            "ARC-AGI-3", "public-game", ExposureClass.PUBLIC_DEV, "hash", "parent", "desc",
            1.0, None, 1.0, 0.0, 0.0, True, False,
        )
        self.assertFalse(BenchmarkAuthorityGate.authoritative_for_promotion(episode))

    def test_unseen_independent_heldout_can_supply_promotion_evidence(self):
        episode = BenchmarkEpisode(
            "EXTERNAL", "sealed-task", ExposureClass.FROZEN_HELDOUT, "hash", "parent", "desc",
            1.0, 0.9, 1.0, 1.0, 0.0, True, False,
        )
        self.assertTrue(BenchmarkAuthorityGate.authoritative_for_promotion(episode))

    def test_seen_answer_voids_even_heldout_label(self):
        episode = BenchmarkEpisode(
            "EXTERNAL", "sealed-task", ExposureClass.FROZEN_HELDOUT, "hash", "parent", "desc",
            1.0, 0.9, 1.0, 1.0, 0.0, True, True,
        )
        self.assertFalse(BenchmarkAuthorityGate.authoritative_for_promotion(episode))


if __name__ == "__main__":
    unittest.main()
