from __future__ import annotations

import unittest

from arte_cognition.executable_morphology import (
    MorphologyCompiler,
    MorphologyGenome,
    MorphologyMutator,
    OrganKind,
    OrganSpec,
    PressureVector,
)
from arte_cognition.morphology_genesis import MorphologyResidual
from arte_cognition.native_recursive_research import NativeMetaMorphologyGenesisEngine


def _genome() -> MorphologyGenome:
    organs = (
        OrganSpec("generator", OrganKind.GENERATOR, produces=("candidate",), implementation_ref="bootstrap://generator"),
        OrganSpec("mutator", OrganKind.MUTATOR, consumes=("candidate",), produces=("mutation",), implementation_ref="bootstrap://mutator"),
        OrganSpec("governor", OrganKind.GOVERNOR),
        OrganSpec("archive", OrganKind.ARCHIVE),
    )
    return MorphologyGenome(organs=organs, edges=(), event_order=("generator", "mutator", "governor", "archive"))


def _candidate(residual_id: str, family: str):
    genome = _genome()
    residual = MorphologyResidual(residual_id, PressureVector(human_dependency=1.0, theory_blindspot=0.25))
    rows = NativeMetaMorphologyGenesisEngine(candidate_budget=16).generate(genome, (residual,))
    return genome, next(row for row in rows if row.operation_family == family)


class NativeMetaPolicyCompilerIntegrationTests(unittest.TestCase):
    def test_legacy_compile_contract_is_identical(self):
        genome = _genome()
        self.assertEqual(MorphologyCompiler.compile(genome), genome.event_order)
        runtime = MorphologyCompiler.compile_runtime(genome)
        self.assertEqual(runtime.event_order, genome.event_order)
        self.assertEqual(runtime.native_meta_policies, ())

    def test_full_generated_ref_is_bound_by_canonical_compiler(self):
        genome, candidate = _candidate("compiler-full", "CHANGE_GENERATOR_POLICY")
        descendant = MorphologyMutator().apply(genome, candidate.mutation)
        runtime = MorphologyCompiler.compile_runtime(descendant, expected_residual_id="compiler-full")
        self.assertEqual(len(runtime.native_meta_policies), 1)
        binding = runtime.native_meta_policies[0]
        self.assertEqual(binding.organ_id, "generator")
        self.assertEqual(binding.target_kind, OrganKind.GENERATOR)
        self.assertEqual(binding.preferred_operation_family, "CHANGE_GENERATOR_POLICY")
        self.assertTrue(binding.policy_fingerprint)
        self.assertEqual(MorphologyCompiler.compile(descendant), descendant.event_order)

    def test_wrong_kind_fails_in_canonical_compile(self):
        genome, candidate = _candidate("compiler-wrong", "CHANGE_GENERATOR_POLICY")
        ref = candidate.mutation.payload["organ"]["implementation_ref"]
        wrong_organs = tuple(
            OrganSpec(
                organ_id=o.organ_id,
                kind=o.kind,
                consumes=o.consumes,
                produces=o.produces,
                implementation_ref=ref if o.organ_id == "mutator" else o.implementation_ref,
                version=o.version,
                cost_hint=o.cost_hint,
                provenance=o.provenance,
                enabled=o.enabled,
            )
            for o in genome.organs
        )
        wrong = MorphologyGenome(wrong_organs, genome.edges, genome.event_order, genome.constitution_epoch)
        with self.assertRaisesRegex(ValueError, "KIND_MISMATCH"):
            MorphologyCompiler.compile(wrong)

    def test_shuffle_expected_residual_fails_in_canonical_compile(self):
        genome, candidate = _candidate("compiler-residual-b", "CHANGE_GENERATOR_POLICY")
        descendant = MorphologyMutator().apply(genome, candidate.mutation)
        with self.assertRaisesRegex(ValueError, "RESIDUAL_MISMATCH"):
            MorphologyCompiler.compile_runtime(descendant, expected_residual_id="compiler-residual-a")

    def test_restart_compilation_is_identical(self):
        genome, candidate = _candidate("compiler-restart", "CHANGE_MUTATOR_POLICY")
        descendant = MorphologyMutator().apply(genome, candidate.mutation)
        a = MorphologyCompiler.compile_runtime(descendant, expected_residual_id="compiler-restart")
        b = MorphologyCompiler.compile_runtime(descendant, expected_residual_id="compiler-restart")
        self.assertEqual(a.fingerprint(), b.fingerprint())
        self.assertEqual(a.native_meta_policies[0].fingerprint(), b.native_meta_policies[0].fingerprint())


if __name__ == "__main__":
    unittest.main()
