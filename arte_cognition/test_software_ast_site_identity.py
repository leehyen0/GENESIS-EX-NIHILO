from __future__ import annotations

import ast
import unittest

from arte_cognition.software_task_acquisition import PythonASTRepairGenerator


class SoftwareASTSiteIdentityTests(unittest.TestCase):
    def test_nested_bool_and_compare_sites_regenerate_without_index_drift(self):
        source = '''
def decision(a, b, c):
    return (a == b and (b != c or c >= a)) or (a < c)
'''
        generator = PythonASTRepairGenerator()
        candidates = generator.generate("nested-production-shape", source)
        self.assertGreaterEqual(len(candidates), 6)
        self.assertEqual(
            tuple(candidate.operator_id for candidate in candidates),
            generator._site_operator_ids(source),
        )
        for candidate in candidates:
            ast.parse(candidate.patched_source)
            self.assertNotEqual(candidate.patched_source, ast.unparse(ast.parse(source)) + "\n")


if __name__ == "__main__":
    unittest.main()
