import unittest

from arte_cognition.validation_matrix import RobustPromotionGate, ValidationObservation


class RobustPromotionGateTests(unittest.TestCase):
    def _good_rows(self):
        return [
            ValidationObservation("tr1", "train-A", "core", 1, "TRAIN", "FULL", 0.80),
            ValidationObservation("tr2", "train-B", "core", 1, "TRAIN", "FULL", 0.82),
            ValidationObservation("h-f1", "held-C", "core", 2, "HELDOUT", "FULL", 0.90),
            ValidationObservation("h-f2", "held-D", "core", 2, "HELDOUT", "FULL", 0.88),
            ValidationObservation("h-r1", "held-C", "core", 2, "HELDOUT", "REMOVE", 0.60),
            ValidationObservation("h-r2", "held-D", "core", 2, "HELDOUT", "REMOVE", 0.61),
            ValidationObservation("h-w1", "held-C", "core", 2, "HELDOUT", "WRONG_SWAP", 0.55),
            ValidationObservation("h-w2", "held-D", "core", 2, "HELDOUT", "WRONG_SWAP", 0.54),
            ValidationObservation("d-f1", "delayed-E", "protected", 3, "DELAYED", "FULL", 0.81),
            ValidationObservation("d-f2", "delayed-F", "protected", 3, "DELAYED", "FULL", 0.82),
            ValidationObservation("d-b1", "delayed-E", "protected", 3, "DELAYED", "BASELINE", 0.80),
            ValidationObservation("d-b2", "delayed-F", "protected", 3, "DELAYED", "BASELINE", 0.81),
        ]

    def test_full_robust_matrix_can_be_eligible(self):
        result = RobustPromotionGate().assess(self._good_rows(), protected_contexts=["protected"])
        self.assertEqual(result.status, "ROBUST_PROMOTION_ELIGIBLE")
        self.assertTrue(result.source_disjoint)
        self.assertTrue(result.delayed_present)
        self.assertTrue(result.negative_transfer_pass)

    def test_source_overlap_blocks_promotion(self):
        rows = self._good_rows() + [
            ValidationObservation("tr-overlap", "held-C", "core", 1, "TRAIN", "FULL", 0.8)
        ]
        result = RobustPromotionGate().assess(rows, protected_contexts=["protected"])
        self.assertEqual(result.status, "ROBUST_PROMOTION_BLOCKED")
        self.assertFalse(result.source_disjoint)

    def test_wrong_swap_equivalence_blocks_promotion(self):
        rows = [
            row if row.variant != "WRONG_SWAP" else ValidationObservation(
                row.observation_id, row.source_id, row.context_id, row.epoch,
                row.split, row.variant, 0.895
            )
            for row in self._good_rows()
        ]
        result = RobustPromotionGate(min_advantage=0.02).assess(rows, protected_contexts=["protected"])
        self.assertEqual(result.status, "ROBUST_PROMOTION_BLOCKED")
        self.assertTrue(any("WRONG_SWAP" in reason for reason in result.reasons))

    def test_negative_transfer_blocks_promotion(self):
        rows = [
            row if not (row.split == "DELAYED" and row.variant == "FULL") else ValidationObservation(
                row.observation_id, row.source_id, row.context_id, row.epoch,
                row.split, row.variant, 0.60
            )
            for row in self._good_rows()
        ]
        result = RobustPromotionGate(max_negative_transfer=0.02).assess(rows, protected_contexts=["protected"])
        self.assertEqual(result.status, "ROBUST_PROMOTION_BLOCKED")
        self.assertFalse(result.negative_transfer_pass)


if __name__ == "__main__":
    unittest.main()
