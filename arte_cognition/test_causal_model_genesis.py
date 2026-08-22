import unittest

from causal_model_genesis import CausalModelGenesisEngine, InterventionDescriptor
from epistemic_depth_runtime import EpistemicallyDeepPersistentCognitiveRuntime
from world_model_ecology import CausalWorldModel, ModelEvidence


class CausalModelGenesisTests(unittest.TestCase):
    def setUp(self):
        self.engine = CausalModelGenesisEngine(model_budget=64)
        self.descriptors = [
            InterventionDescriptor("do-x", ("x",), cost=1.0),
            InterventionDescriptor("do-z", ("z",), cost=1.0),
            InterventionDescriptor("do-both", ("x", "z"), cost=4.0),
            InterventionDescriptor("do-x-block-z", ("x",), blocked=("z",), cost=5.0),
            InterventionDescriptor("delay-x", ("x",), delay_steps=1, cost=3.0),
            InterventionDescriptor("context-shift", (), context_shift=True, cost=8.0),
        ]

    def test_prediction_equivalent_structures_are_quotiented(self):
        generated = self.engine.generate(["x", "z"], [InterventionDescriptor("do-x", ("x",))])
        signatures = [tuple(sorted(item.model.predictions)) for item in generated]
        self.assertEqual(len(signatures), len(set(signatures)))
        self.assertTrue(any(item.equivalent_structures for item in generated))

    def test_authoritative_residual_filters_incompatible_generated_models(self):
        residual = [ModelEvidence("e1", "do-x-block-z", "NO_EFFECT", "ind-A", "ctx", True)]
        generated = self.engine.generate(["x", "z"], self.descriptors, residual)
        self.assertTrue(generated)
        self.assertTrue(all(item.model.prediction_for("do-x-block-z") == "NO_EFFECT" for item in generated))

    def test_runtime_does_not_generate_new_model_class_without_class_failure(self):
        runtime = EpistemicallyDeepPersistentCognitiveRuntime()
        runtime.register_causal_world_models([
            CausalWorldModel("M1", 1.0, (("do-x", "POSITIVE_EFFECT"),)),
            CausalWorldModel("M2", 1.0, (("do-x", "NEGATIVE_EFFECT"),)),
        ])
        self.assertNotEqual(runtime.epistemic_depth_plan().mode, "EXPAND_MODEL_CLASS")
        self.assertEqual(runtime.generate_replacement_causal_models(["x", "z"], self.descriptors), [])

    def test_generated_model_can_resolve_previous_model_class_failure(self):
        runtime = EpistemicallyDeepPersistentCognitiveRuntime()
        runtime.register_causal_world_models([
            CausalWorldModel("OLD_A", 1.0, (("do-x-block-z", "POSITIVE_EFFECT"),)),
            CausalWorldModel("OLD_B", 1.0, (("do-x-block-z", "NEGATIVE_EFFECT"),)),
        ])
        runtime.world_models.observe(ModelEvidence(
            "surprise", "do-x-block-z", "NO_EFFECT", "ind-A", "ctx", True
        ))
        self.assertEqual(runtime.epistemic_depth_plan().mode, "EXPAND_MODEL_CLASS")
        generated = runtime.generate_replacement_causal_models(["x", "z"], self.descriptors)
        self.assertTrue(generated)
        self.assertTrue(any(item.model.origin == "GENERATED" for item in generated))
        self.assertNotEqual(runtime.epistemic_depth_plan().mode, "EXPAND_MODEL_CLASS")

    def test_generated_queries_derive_predictions_from_generated_models(self):
        generated = self.engine.generate(["x", "z"], self.descriptors)
        queries = self.engine.query_candidates(self.descriptors, [item.model for item in generated])
        by_id = {query.query_id: query for query in queries}
        self.assertIn("do-both", by_id)
        self.assertGreaterEqual(len(set(by_id["do-both"].distinguishes.values())), 2)
        self.assertTrue(by_id["do-both"].intervention)


if __name__ == "__main__":
    unittest.main()
