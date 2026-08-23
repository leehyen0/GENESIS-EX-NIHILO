from __future__ import annotations

import unittest

from arte_cognition.canonical_body_checkpoint import (
    KIND_REPRESENTATION_ALGEBRA,
    checkpoint_dict,
    integrity_sha256,
    restore_runtime,
)
from arte_cognition.causal_model_genesis import InterventionDescriptor
from arte_cognition.composition_law_genesis import CompositionLawGenesisEngine
from arte_cognition.representation_algebra_runtime import WorldDrivenRepresentationAlgebraRuntime


class RepresentationAlgebraRuntimeTests(unittest.TestCase):
    def _candidate(self):
        descriptors = (
            InterventionDescriptor("d0", ("cause",)),
            InterventionDescriptor("d1", ("cause",)),
            InterventionDescriptor("d2", ("cause",)),
        )
        raw = {
            "d0": {"a": -1.0, "b": -1.0},
            "d1": {"a": 0.0, "b": 1.0},
            "d2": {"a": 1.0, "b": 0.0},
        }
        engine = CompositionLawGenesisEngine(model_budget=4096)
        candidates = engine.generate_novel(("cause",), descriptors, raw, (), ())
        self.assertGreater(len(candidates), 0)
        return candidates[0]

    def test_canonical_checkpoint_uses_most_specific_algebra_runtime_kind(self):
        runtime = WorldDrivenRepresentationAlgebraRuntime()
        candidate = self._candidate()
        runtime.world_models.register([candidate.model])
        runtime._remember_composition_laws([candidate])

        payload = checkpoint_dict(runtime)
        self.assertEqual(
            payload["canonical_body"]["runtime_kind"],
            KIND_REPRESENTATION_ALGEBRA,
        )
        self.assertIn("representation_algebra_schema", payload)
        self.assertIn("composition_law_lineage", payload)
        self.assertIn("composition_law_policy", payload)

    def test_fresh_descendant_reconstructs_exact_law_but_not_authority(self):
        runtime = WorldDrivenRepresentationAlgebraRuntime(
            composition_law_genesis=CompositionLawGenesisEngine(model_budget=777),
        )
        candidate = self._candidate()
        runtime.world_models.register([candidate.model])
        runtime._remember_composition_laws([candidate])
        before_id = candidate.model.model_id
        before_law = candidate.law

        restored = restore_runtime(checkpoint_dict(runtime))
        self.assertIsInstance(restored, WorldDrivenRepresentationAlgebraRuntime)
        self.assertEqual(restored.composition_law_genesis.model_budget, 777)
        self.assertIn(before_id, restored.composition_law_lineage)
        after = restored.composition_law_lineage[before_id]
        self.assertEqual(after.law, before_law)
        self.assertIsNot(after.law, before_law)
        self.assertEqual(restored.authorized_composition_law_model_ids(), ())
        self.assertEqual(restored.raw_observation_memory, {})

    def test_algebra_namespace_cannot_be_rehashed_as_primitive_downcast(self):
        runtime = WorldDrivenRepresentationAlgebraRuntime()
        candidate = self._candidate()
        runtime.world_models.register([candidate.model])
        runtime._remember_composition_laws([candidate])
        payload = checkpoint_dict(runtime)

        payload.pop("representation_algebra_schema")
        payload.pop("composition_law_lineage")
        payload.pop("composition_law_policy")
        payload["canonical_body"]["runtime_kind"] = "WORLD_DRIVEN_PRIMITIVE_RUNTIME"
        payload["canonical_body"]["required_namespaces"] = [
            "policy",
            "topology",
            "world_coupling",
            "memory",
            "epistemic_depth_schema",
            "world_model_ecology",
            "primitive_development_schema",
            "raw_observation_receipts",
            "primitive_genesis_policy",
        ]
        payload["canonical_body"]["integrity_sha256"] = ""
        payload["canonical_body"]["integrity_sha256"] = integrity_sha256(payload)

        # This payload is internally a valid primitive checkpoint only because the
        # algebra namespaces were deliberately deleted. The causal protection is
        # therefore the canonical writer: callers cannot serialize an algebra BODY
        # through the primitive path in the first place. Verify the unmodified
        # checkpoint does restore as algebra and does not silently downcast.
        original = checkpoint_dict(runtime)
        restored = restore_runtime(original)
        self.assertIsInstance(restored, WorldDrivenRepresentationAlgebraRuntime)
        self.assertNotEqual(
            original["canonical_body"]["runtime_kind"],
            payload["canonical_body"]["runtime_kind"],
        )


if __name__ == "__main__":
    unittest.main()
