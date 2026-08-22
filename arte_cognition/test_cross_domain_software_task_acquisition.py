from __future__ import annotations

import secrets
import tempfile
import unittest
from pathlib import Path

from evaluations.run_cross_domain_software_task_acquisition import main


class CrossDomainSoftwareTaskAcquisitionTests(unittest.TestCase):
    def test_post_checkout_source_disjoint_executable_software_repair(self):
        seed=secrets.randbits(128)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"seed.txt"
            path.write_text(str(seed))
            main(str(path))


if __name__=="__main__":
    unittest.main()
