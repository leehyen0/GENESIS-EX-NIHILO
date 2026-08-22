import unittest

from arte_cognition.causal_model_genesis import CausalModelGenesisEngine, InterventionDescriptor
from arte_cognition.causal_program_genesis import CompositionalCausalProgramGenesisEngine
from arte_cognition.epistemic_depth_runtime import EpistemicallyDeepPersistentCognitiveRuntime
from arte_cognition.world_model_ecology import CausalWorldModel, ModelEvidence


class CompositionalCausalProgramGenesisTests(unittest.TestCase):
    def setUp(self):
        self.variables = ["x", "z"]
        self.descriptors = [
            InterventionDescriptor("do-x", ("x",), cost=1.0),
            InterventionDescriptor("do-z", ("z",), cost=1.0),
            InterventionDescriptor("do-both", ("x", "z"), cost=4.0),
            InterventionDescriptor("delay-x", ("x",), delay_steps=1, cost=3.0),
            InterventionDescriptor("delay-z", ("z",), delay_steps=1, cost=3.0),
            InterventionDescriptor("delay-both", ("x", "z"), delay_steps=1, cost=6.0),
            InterventionDescriptor("delay-x-block-z", ("x",), blocked=("z",), delay_steps=1, cost=7.0),
            InterventionDescriptor("context-x", ("x",), context_shift=True, cost=5.0),
        ]
        self.base_engine = CausalModelGenesisEngine(model_budget=64)
        self.program_engine = CompositionalCausalProgramGenesisEngine(model_budget=96, max_extra_primitives=2)

    def test_composition_creates_prediction_class_outside_named_base_families(self):
        base = self.base_engine.generate(self.variables, self.descriptors)
        base_signatures = {tuple(sorted(item.model.predictions)) for item in base}
        generated = self.program_engine.generate_novel(
            self.variables, self.descriptors, (), [item.model for item in base]
        )
        self.assertTrue(generated)
        self.assertTrue(all(tuple(sorted(item.model.predictions)) not in base_signatures for item in generated))
        self.assertTrue(any(
            {primitive.op for primitive in item.program.primitives} >= {"VIA", "LAG"}
            for item in generated
        ))

    def test_prediction_equivalent_programs_are_quotiented(self):
        base = self.base_engine.generate(self.variables, self.descriptors)
        generated = self.program_engine.generate_novel(
            self.variables, self.descriptors, (), [item.model for item in base]
        )
        signatures = [tuple(sorted(item.model.predictions)) for item in generated]
        self.assertEqual(len(signatures), len(set(signatures)))

    def test_authoritative_residual_filters_compositional_programs(self):
        base = self.base_engine.generate(self.variables, self.descriptors)
        residual = [ModelEvidence("e", "delay-x-block-z", "NO_EFFECT", "A", "ctx", True)]
        generated = self.program_engine.generate_novel(
            self.variables, self.descriptors, residual, [item.model for item in base]
        )
        self.assertTrue(generated)
        self.assertTrue(all(
            item.model.prediction_for("delay-x-block-z") == "NO_EFFECT"
            for item in generated
        ))

    def test_second_generation_requires_first_generation_parent(self):
        runtime = EpistemicallyDeepPersistentCognitiveRuntime()
        runtime.register_causal_world_models([
            CausalWorldModel("A", 1.0, (("do-x", "POSITIVE_EFFECT"),)),
            CausalWorldModel("B", 1.0, (("do-x", "NEGATIVE_EFFECT"),)),
        ])
        for source in ("ind-A", "ind-B"):
            runtime.world_models.observe(ModelEvidence(
                f"failure-{source}", "do-x", "NO_EFFECT", source, "ctx", True
            ))
        self.assertEqual(runtime.epistemic_depth_plan().mode, "EXPAND_MODEL_CLASS")
        self.assertEqual(runtime.generate_compositional_causal_models(self.variables, self.descriptors), [])

    def test_second_generation_records_first_generation_ancestry(self):
        runtime = EpistemicallyDeepPersistentCognitiveRuntime()
        first_parent = CausalWorldModel(
            "FIRST_PARENT", 1.0,
            tuple((d.intervention_id, "POSITIVE_EFFECT") for d in self.descriptors),
            origin="GENERATED", family="DIRECT", structure=("x->OUTCOME",), generation=1,
        )
        runtime.register_causal_world_models([first_parent])
        for source in ("ind-A", "ind-B"):
            runtime.world_models.observe(ModelEvidence(
                f"failure-{source}", "do-x", "NO_EFFECT", source, "ctx", True
            ))
        self.assertEqual(runtime.epistemic_depth_plan().mode, "EXPAND_MODEL_CLASS")
        generated = runtime.generate_compositional_causal_models(self.variables, self.descriptors)
        self.assertTrue(generated)
        self.assertTrue(all(item.model.generation == 2 for item in generated))
        self.assertTrue(all("FIRST_PARENT" in item.model.parent_model_ids for item in generated))
        self.assertTrue(all(item.model.origin == "GENERATED_COMPOSITIONAL" for item in generated))


if __name__ == "__main__":
    unittest.main()
