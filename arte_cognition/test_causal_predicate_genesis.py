import unittest

from arte_cognition.causal_model_genesis import CausalModelGenesisEngine, InterventionDescriptor
from arte_cognition.causal_predicate_genesis import BooleanCausalPredicateGenesisEngine
from arte_cognition.causal_program_genesis import CompositionalCausalProgramGenesisEngine
from arte_cognition.epistemic_depth_runtime import EpistemicallyDeepPersistentCognitiveRuntime
from arte_cognition.world_model_ecology import CausalWorldModel, ModelEvidence


class BooleanCausalPredicateGenesisTests(unittest.TestCase):
    def setUp(self):
        self.variables = ["x", "z"]
        self.descriptors = [
            InterventionDescriptor("do-both", ("x", "z"), cost=2),
            InterventionDescriptor("delay-both", ("x", "z"), delay_steps=1, cost=4),
            InterventionDescriptor("context-both", ("x", "z"), context_shift=True, cost=5),
            InterventionDescriptor("context-delay-both", ("x", "z"), delay_steps=1, context_shift=True, cost=8),
            InterventionDescriptor("delay-x", ("x",), delay_steps=1, cost=3),
            InterventionDescriptor("context-x", ("x",), context_shift=True, cost=3),
            InterventionDescriptor("context-delay-x", ("x",), delay_steps=1, context_shift=True, cost=6),
            InterventionDescriptor("delay-both-block-z", ("x", "z"), blocked=("z",), delay_steps=1, cost=7),
        ]
        self.base = CausalModelGenesisEngine(model_budget=128)
        self.programs = CompositionalCausalProgramGenesisEngine(model_budget=256, max_extra_primitives=2)
        self.predicates = BooleanCausalPredicateGenesisEngine(model_budget=512, max_literals_per_term=3, max_terms=2)

    def prior_models(self):
        base = self.base.generate(self.variables, self.descriptors)
        comp = self.programs.generate_novel(
            self.variables, self.descriptors, (), [item.model for item in base]
        )
        return [item.model for item in base] + [item.model for item in comp]

    def test_xor_activation_signature_is_generated_outside_prior_grammars(self):
        existing = self.prior_models()
        existing_signatures = {tuple(sorted(model.predictions)) for model in existing}
        generated = self.predicates.generate_novel(self.variables, self.descriptors, (), existing)
        self.assertTrue(generated)
        wanted = {
            "do-both": "NO_EFFECT",
            "delay-both": "POSITIVE_EFFECT",
            "context-both": "POSITIVE_EFFECT",
            "context-delay-both": "NO_EFFECT",
            "delay-x": "NO_EFFECT",
        }
        matching = [
            item for item in generated
            if all(item.model.prediction_for(key) == value for key, value in wanted.items())
        ]
        self.assertTrue(matching)
        self.assertTrue(all(tuple(sorted(item.model.predictions)) not in existing_signatures for item in matching))
        self.assertTrue(any("!CONTEXT" in item.predicate.render() and "!DELAY" in item.predicate.render() for item in matching))

    def test_prediction_equivalent_predicates_are_quotiented(self):
        generated = self.predicates.generate_novel(self.variables, self.descriptors, (), self.prior_models())
        signatures = [tuple(sorted(item.model.predictions)) for item in generated]
        self.assertEqual(len(signatures), len(set(signatures)))

    def test_authoritative_evidence_filters_predicate_models(self):
        evidence = [ModelEvidence("e", "context-delay-both", "NO_EFFECT", "A", "ctx", True)]
        generated = self.predicates.generate_novel(
            self.variables, self.descriptors, evidence, self.prior_models()
        )
        self.assertTrue(generated)
        self.assertTrue(all(item.model.prediction_for("context-delay-both") == "NO_EFFECT" for item in generated))

    def test_generation_three_requires_compositional_parent(self):
        runtime = EpistemicallyDeepPersistentCognitiveRuntime()
        runtime.register_causal_world_models([
            CausalWorldModel(
                "FIRST", 1.0, (("do-both", "POSITIVE_EFFECT"),),
                origin="GENERATED", family="INTERACTION", generation=1,
            )
        ])
        for source in ("A", "B"):
            runtime.world_models.observe(ModelEvidence(
                f"failure-{source}", "do-both", "NO_EFFECT", source, "ctx", True
            ))
        self.assertEqual(runtime.epistemic_depth_plan().mode, "EXPAND_MODEL_CLASS")
        self.assertEqual(runtime.generate_predicate_causal_models(self.variables, self.descriptors), [])

    def test_generation_three_records_compositional_ancestry(self):
        runtime = EpistemicallyDeepPersistentCognitiveRuntime()
        parent = CausalWorldModel(
            "SECOND", 1.0, tuple((d.intervention_id, "POSITIVE_EFFECT") for d in self.descriptors),
            origin="GENERATED_COMPOSITIONAL", family="COMPOSITIONAL_PROGRAM",
            structure=("CAUSE(x)", "REQUIRE(z)", "LAG(1)"), generation=2,
        )
        runtime.register_causal_world_models([parent])
        for source in ("A", "B"):
            runtime.world_models.observe(ModelEvidence(
                f"failure-{source}", "do-both", "NO_EFFECT", source, "ctx", True
            ))
        self.assertEqual(runtime.epistemic_depth_plan().mode, "EXPAND_MODEL_CLASS")
        generated = runtime.generate_predicate_causal_models(self.variables, self.descriptors)
        self.assertTrue(generated)
        self.assertTrue(all(item.model.generation == 3 for item in generated))
        self.assertTrue(all("SECOND" in item.model.parent_model_ids for item in generated))
        self.assertTrue(all(item.model.origin == "GENERATED_PREDICATE" for item in generated))


if __name__ == "__main__":
    unittest.main()
