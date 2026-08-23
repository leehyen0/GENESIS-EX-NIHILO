import unittest

from evaluations.run_upstream_patch_semantic_selection import main


class UpstreamPatchSemanticSelectionTests(unittest.TestCase):
    def test_world_authorized_selector_freezes_one_patch_before_heldout_outcomes(self):
        main()


if __name__ == "__main__":
    unittest.main()
