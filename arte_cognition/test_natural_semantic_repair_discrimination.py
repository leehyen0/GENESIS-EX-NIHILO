import unittest

from evaluations.run_natural_semantic_repair_discrimination import main


class NaturalSemanticRepairDiscriminationTests(unittest.TestCase):
    def test_world_authorized_semantic_discriminator_selects_unique_g7_repair(self):
        main()


if __name__ == "__main__":
    unittest.main()
