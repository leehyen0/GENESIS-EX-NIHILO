import unittest

from evaluations.run_natural_repair_class_genesis import main


class NaturalRepairClassGenesisTests(unittest.TestCase):
    def test_natural_fixed_class_failure_generates_transferable_new_repair_class(self):
        main()


if __name__ == "__main__":
    unittest.main()
