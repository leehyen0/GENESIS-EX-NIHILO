import unittest

from arte_cognition.software_repair_class_diagnosis import assess_repair_class_applicability
from arte_cognition.software_repair_class_genesis import (
    GeneratedRepairClassPhenotype,
    generate_repair_mechanisms,
)
from evaluations.run_natural_repair_class_genesis import (
    HELDOUT_FIXTURE,
    HELDOUT_PATH,
    TRAINING_FIXTURE,
    HistoricalExecutionEnvironment,
    _repository_python_paths,
)


def transplant_training_search_context(source: str) -> str:
    lines = str(source).splitlines()
    insertion = 1 if lines and lines[0].startswith("import ") else 0
    block = [
        "import sys",
        "from pathlib import Path",
        "_arte_generated_root = Path(__file__).resolve().parents[1]",
        "if str(_arte_generated_root) not in sys.path:",
        "    sys.path.insert(0, str(_arte_generated_root))",
        "",
    ]
    return "\n".join(lines[:insertion] + block + lines[insertion:]) + "\n"


class NaturalRepairClassGenesisControlTests(unittest.TestCase):
    def test_fixed_class_applicability_is_explicit_not_silent_refutation(self):
        for fixture in (TRAINING_FIXTURE, HELDOUT_FIXTURE):
            source = fixture.read_text(encoding="utf-8")
            states = assess_repair_class_applicability(source)
            self.assertEqual(states["CONTENT"].status, "APPLICABLE")
            self.assertEqual(states["TRAVERSAL"].status, "INAPPLICABLE")
            self.assertEqual(states["STATE_CONFLICT"].status, "INAPPLICABLE")

    def test_exact_training_search_context_mechanism_does_not_solve_heldout_import_topology(self):
        source = HELDOUT_FIXTURE.read_text(encoding="utf-8")
        environment = HistoricalExecutionEnvironment(source, HELDOUT_PATH, "PACKAGE_UNITTEST")
        baseline, _, stderr = environment.run()
        self.assertEqual(baseline, 0.0)
        self.assertIn("ModuleNotFoundError", stderr)

        transplanted = transplant_training_search_context(source)
        old_mechanism_capability, _, old_stderr = environment.run(transplanted)
        self.assertEqual(old_mechanism_capability, 0.0, msg=old_stderr[-500:])

        phenotype = GeneratedRepairClassPhenotype(
            failure_phase="MODULE_IMPORT",
            resource_relation="LOCAL_MODULE_UNRESOLVED",
            repair_goal="RESTORE_MODULE_REACHABILITY",
        )
        mechanisms = generate_repair_mechanisms(
            phenotype.class_id,
            source,
            HELDOUT_PATH,
            stderr,
            _repository_python_paths(),
        )
        self.assertEqual(
            [item.mechanism_id for item in mechanisms],
            ["IMPORT_REFERENCE::QUALIFY_LOCAL_PACKAGE"],
        )
        new_mechanism_capability, _, new_stderr = environment.run(mechanisms[0].patched_source)
        self.assertEqual(new_mechanism_capability, 1.0, msg=new_stderr[-500:])


if __name__ == "__main__":
    unittest.main()
