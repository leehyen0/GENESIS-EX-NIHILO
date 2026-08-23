from __future__ import annotations

import unittest

from evaluations.run_selector_representation_program_genesis import main


class SelectorRepresentationProgramGenesisTests(unittest.TestCase):
    def test_composed_normalizer_program_genesis_and_preoutcome_transfer(self):
        main()


if __name__ == "__main__":
    unittest.main()
