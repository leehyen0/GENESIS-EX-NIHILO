from __future__ import annotations

import secrets
import tempfile
import unittest
from pathlib import Path

from evaluations.run_multifile_repository_repair_acquisition import main


class MultiFileRepositoryRepairAcquisitionTests(unittest.TestCase):
    def test_post_checkout_randomized_multifile_repository_localization(self):
        seed = secrets.randbits(128)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.txt"
            path.write_text(str(seed))
            main(str(path))


if __name__ == "__main__":
    unittest.main()
