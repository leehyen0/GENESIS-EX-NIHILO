import unittest

from evaluations.run_natural_failure_extractor_program_genesis import main


class NaturalFailureExtractorProgramGenesisTests(unittest.TestCase):
    def test_natural_failure_extractor_program_genesis_and_g7_transfer(self):
        main()


if __name__ == "__main__":
    unittest.main()
