from __future__ import annotations

import unittest

from evaluations.run_natural_repair_class_third_transfer import main


class NaturalRepairClassThirdTransferTests(unittest.TestCase):
    def test_unused_natural_transform_evaluator_transfer(self):
        main()


if __name__ == "__main__":
    unittest.main()
