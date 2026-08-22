from __future__ import annotations

import secrets
import tempfile
import unittest
from pathlib import Path

from evaluations.run_coordinated_multifile_patch_composition import main


class CoordinatedMultiFilePatchCompositionTests(unittest.TestCase):
    def test_post_checkout_randomized_two_file_patch_composition(self):
        seed = secrets.randbits(128)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.txt"
            path.write_text(str(seed))
            main(str(path))


if __name__ == "__main__":
    unittest.main()
