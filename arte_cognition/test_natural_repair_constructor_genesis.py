import unittest

from evaluations.run_natural_repair_constructor_genesis import main


class NaturalRepairConstructorGenesisTests(unittest.TestCase):
    def test_natural_relational_constructor_genesis_and_fresh_exception_transfer(self):
        main()


if __name__ == "__main__":
    unittest.main()
