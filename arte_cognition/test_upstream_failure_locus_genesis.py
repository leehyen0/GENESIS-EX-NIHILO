import unittest

from evaluations.run_upstream_failure_locus_genesis import main


class UpstreamFailureLocusGenesisTests(unittest.TestCase):
    def test_natural_failure_opens_upstream_locus_and_transfers(self):
        main()


if __name__ == "__main__":
    unittest.main()
