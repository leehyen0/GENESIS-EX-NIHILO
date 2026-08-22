from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluations.run_actual_repository_unknown_file_localization import main


class ActualRepositoryUnknownFileLocalizationTests(unittest.TestCase):
    def test_post_checkout_actual_repository_unknown_file_localization(self):
        main()


if __name__ == "__main__":
    unittest.main()
