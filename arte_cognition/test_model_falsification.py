from __future__ import annotations

import unittest

from arte_cognition.causal_model_genesis import InterventionDescriptor
from arte_cognition.model_falsification import ModelFalsificationPolicy


class ModelFalsificationPolicyTests(unittest.TestCase):
    def test_high_novelty_structural_stress_can_beat_cheap_repeat(self):
        cheap = InterventionDescriptor("cheap", ("x",), (), 0, False, 1.0)
        stressed = InterventionDescriptor("stress", ("x", "z"), ("z",), 1, True, 13.5)
        observed = [InterventionDescriptor("obs", ("x",), (), 0, False, 1.0)]
        ranked = ModelFalsificationPolicy.rank([cheap, stressed], observed)
        self.assertEqual(ranked[0].intervention_id, "stress")

    def test_selection_is_hidden_outcome_independent(self):
        a = InterventionDescriptor("a", ("x",), (), 0, False, 1.0)
        b = InterventionDescriptor("b", ("x", "z"), ("z",), 1, True, 13.5)
        first = ModelFalsificationPolicy.select([a, b], ())
        second = ModelFalsificationPolicy.select([a, b], ())
        self.assertIsNotNone(first)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
