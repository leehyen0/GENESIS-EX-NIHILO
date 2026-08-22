import unittest

from arte_cognition.topology_learning import CognitionTopologyLearner


class TopologyLearningTests(unittest.TestCase):
    def test_no_synergy_evidence_preserves_order(self):
        learner = CognitionTopologyLearner()
        modules = ["REPRESENTATION_ESCAPE", "MODAL_EXPANSION", "QUESTION_FIELD"]
        self.assertEqual(learner.reorder(modules), modules)

    def test_repeated_pair_synergy_changes_execution_order(self):
        learner = CognitionTopologyLearner()
        sequence = ["QUESTION_FIELD", "MODAL_EXPANSION", "REPRESENTATION_ESCAPE"]
        synergy = {
            ("QUESTION_FIELD", "MODAL_EXPANSION"): 0.8,
            ("MODAL_EXPANSION", "REPRESENTATION_ESCAPE"): 0.8,
        }
        for _ in range(3):
            learner.observe_sequence(sequence, synergy)
        reordered = learner.reorder(list(reversed(sequence)))
        self.assertEqual(reordered, sequence)

    def test_repeated_positive_chain_proposes_macro_but_does_not_activate_it(self):
        learner = CognitionTopologyLearner()
        sequence = ["QUESTION_FIELD", "MODAL_EXPANSION", "REPRESENTATION_ESCAPE"]
        synergy = {
            ("QUESTION_FIELD", "MODAL_EXPANSION"): 0.8,
            ("MODAL_EXPANSION", "REPRESENTATION_ESCAPE"): 0.8,
        }
        for _ in range(3):
            learner.observe_sequence(sequence, synergy)
        macros = learner.propose_macros()
        self.assertEqual(len(macros), 1)
        self.assertEqual(macros[0].sequence, tuple(sequence))
        self.assertEqual(macros[0].status, "PROPOSAL_ONLY")

    def test_cooccurrence_without_pair_ablation_synergy_does_not_train_edges(self):
        learner = CognitionTopologyLearner()
        sequence = ["QUESTION_FIELD", "MODAL_EXPANSION"]
        for _ in range(10):
            learner.observe_sequence(sequence, {})
        self.assertEqual(learner.edge_shift(*sequence), 0.0)
        self.assertFalse(learner.propose_macros())


if __name__ == "__main__":
    unittest.main()
