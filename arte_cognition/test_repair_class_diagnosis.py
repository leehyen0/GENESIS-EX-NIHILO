from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluations.run_repair_class_diagnosis import main


class RepairClassDiagnosisTests(unittest.TestCase):
    def test_world_evidence_routes_two_natural_historical_repair_families(self):
        main()


if __name__ == "__main__":
    unittest.main()
