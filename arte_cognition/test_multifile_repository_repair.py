from __future__ import annotations

import secrets
import tempfile
import unittest
from pathlib import Path

from evaluations.run_multifile_repository_repair import main


class MultiFileRepositoryRepairTests(unittest.TestCase):
    def test_post_checkout_randomized_multifile_repository_repair(self):
        seed = secrets.randbits(128)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed"
            path.write_text(str(seed))
            main(path)


if __name__ == "__main__":
    unittest.main()
