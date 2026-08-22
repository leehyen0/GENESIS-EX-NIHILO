from __future__ import annotations

import unittest

from evaluations.run_real_repository_synthetic_bug_acquisition import main


class RealRepositorySyntheticBugAcquisitionTests(unittest.TestCase):
    def test_post_checkout_actual_repository_hidden_synthetic_bug_transfer(self):
        main()


if __name__ == "__main__":
    unittest.main()
