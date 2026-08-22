from __future__ import annotations

import secrets
import tempfile
import unittest
from pathlib import Path

from evaluations.run_world_driven_transform_primitive_alphabet import main


class HiddenTransformPrimitiveAlphabetEvaluationTests(unittest.TestCase):
    def test_post_checkout_randomized_world_driven_primitive_alphabet(self):
        seed = secrets.randbits(128)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.txt"
            path.write_text(str(seed))
            main(str(path))


if __name__ == "__main__":
    unittest.main()
