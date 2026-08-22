from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluations.run_natural_historical_provenance_repair import main


class NaturalHistoricalProvenanceRepairTests(unittest.TestCase):
    def test_frozen_historical_alias_bug_requires_state_conflict_repair_language(self):
        main()


if __name__ == "__main__":
    unittest.main()
