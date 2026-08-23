import unittest

from evaluations.run_call_binding_selector_representation import main


class CallBindingSelectorRepresentationTests(unittest.TestCase):
    def test_syntax_shift_opens_binding_representation_and_preoutcome_transfer(self):
        main()


if __name__ == "__main__":
    unittest.main()
