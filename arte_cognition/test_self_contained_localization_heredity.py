from __future__ import annotations

import secrets
import tempfile
import unittest
from pathlib import Path

from evaluations.run_self_contained_localization_heredity import main


class SelfContainedLocalizationHeredityTests(unittest.TestCase):
    def test_post_checkout_self_contained_localization_heredity(self):
        seed = secrets.randbits(128)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed"
            path.write_text(str(seed))
            main(path)


if __name__ == "__main__":
    unittest.main()
