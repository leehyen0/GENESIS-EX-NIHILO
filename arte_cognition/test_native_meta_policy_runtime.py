from __future__ import annotations

import unittest

from arte_cognition.executable_morphology import (
    MorphologyGenome,
    MorphologyMutator,
    OrganKind,
    OrganSpec,
    PressureVector,
)
from arte_cognition.morphology_genesis import MorphologyCandidate, MorphologyResidual
from arte_cognition.native_recursive_research import NativeMetaMorphologyGenesisEngine
from arte_cognition.native_meta_policy_runtime import (
    compile_genome_native_meta_policies,
    compile_native_meta_policy,
    execute_native_meta_policy,
    parent_candidate_selection,
)


def _genome() -> MorphologyGenome:
    organs = (
        OrganSpec("generator", OrganKind.GENERATOR, produces=("candidate",), implementation_ref="bootstrap://generator"),
        OrganSpec("mutator", OrganKind.MUTATOR, consumes=("candidate",), produces=("mutation",), implementation_ref="bootstrap://mutator"),
        OrganSpec("governor", OrganKind.GOVERNOR),
        OrganSpec("archive", OrganKind.ARCHIVE),
    )
    return MorphologyGenome(organs=organs, edges=(), event_order=tuple(row.organ_id for row in organs))


def _l3(residual_id: str):
    genome = _genome()
    residual = MorphologyResidual(
        residual_id,
        PressureVector(human_dependency=1.0, theory_blindspot=0.25),
    )
    rows = NativeMetaMorphologyGenesisEngine(candidate_budget=16).generate(genome, (residual,))
    by_family = {row.operation_family: row for row in rows if row.operation_family.startswith("CHANGE_")}
    return genome, residual, by_family


def _probe_rows(by_family):
    generator = by_family["CHANGE_GENERATOR_POLICY"]
    mutator = by_family["CHANGE_MUTATOR_POLICY"]
    # IDs are deliberately frozen so the no-policy parent picks generator first.
    return (
        MorphologyCandidate(
            candidate_id="A_PARENT_DEFAULT_GENERATOR",
            mutation=generator.mutation,
            descendant_fingerprint=generator.descendant_fingerprint,
            origin_residual_ids=generator.origin_residual_ids,
            operation_family=generator.operation_family,
            generation_uses_outcomes=False,
        ),
        MorphologyCandidate(
            candidate_id="B_MUTATOR_TARGET",
            mutation=mutator.mutation,
            descendant_fingerprint=mutator.descendant_fingerprint,
            origin_residual_ids=mutator.origin_residual_ids,
            operation_family=mutator.operation_family,
            generation_uses_outcomes=False,
        ),
    )


class NativeMetaPolicyRuntimeTests(unittest.TestCase):
    def test_generated_refs_compile_from_descendant_organs(self):
        genome, residual, by_family = _l3("runtime-bind")
        for family, candidate in by_family.items():
            descendant = MorphologyMutator().apply(genome, candidate.mutation)
            programs = compile_genome_native_meta_policies(descendant, expected_residual_id=residual.residual_id)
            self.assertEqual(len(programs), 1)
            self.assertEqual(programs[0].preferred_operation_family, family)
            self.assertFalse(programs[0].current_outcomes_required)

    def test_generator_policy_changes_future_frontier_budget(self):
        _, residual, by_family = _l3("generator-probe")
        ref = by_family["CHANGE_GENERATOR_POLICY"].mutation.payload["organ"]["implementation_ref"]
        program = compile_native_meta_policy(ref, target_kind=OrganKind.GENERATOR, expected_residual_id=residual.residual_id)
        probe = _probe_rows(by_family)
        parent = parent_candidate_selection(probe, candidate_budget=1)
        full = execute_native_meta_policy(program, probe, parent_candidate_budget=1)
        self.assertEqual(len(parent), 1)
        self.assertGreater(len(full.selected_candidate_ids), len(parent))
        self.assertFalse(full.current_outcomes_consumed)

    def test_mutator_policy_changes_future_family_priority(self):
        _, residual, by_family = _l3("mutator-probe")
        ref = by_family["CHANGE_MUTATOR_POLICY"].mutation.payload["organ"]["implementation_ref"]
        program = compile_native_meta_policy(ref, target_kind=OrganKind.MUTATOR, expected_residual_id=residual.residual_id)
        probe = _probe_rows(by_family)
        parent = parent_candidate_selection(probe, candidate_budget=1)
        full = execute_native_meta_policy(program, probe, parent_candidate_budget=1)
        self.assertEqual(parent, ("A_PARENT_DEFAULT_GENERATOR",))
        self.assertEqual(full.selected_candidate_ids, ("B_MUTATOR_TARGET",))
        self.assertFalse(full.current_outcomes_consumed)

    def test_wrong_family_binding_fails_closed(self):
        _, residual, by_family = _l3("wrong-family")
        generator_ref = by_family["CHANGE_GENERATOR_POLICY"].mutation.payload["organ"]["implementation_ref"]
        with self.assertRaisesRegex(ValueError, "KIND_MISMATCH"):
            compile_native_meta_policy(
                generator_ref,
                target_kind=OrganKind.MUTATOR,
                expected_residual_id=residual.residual_id,
            )

    def test_shuffle_residual_binding_fails_closed(self):
        _, _, first = _l3("residual-a")
        ref = first["CHANGE_GENERATOR_POLICY"].mutation.payload["organ"]["implementation_ref"]
        with self.assertRaisesRegex(ValueError, "RESIDUAL_MISMATCH"):
            compile_native_meta_policy(ref, target_kind=OrganKind.GENERATOR, expected_residual_id="residual-b")

    def test_malformed_ref_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "INVALID_NATIVE_META_POLICY_REF"):
            compile_native_meta_policy("native-meta://generator/missing-suffix", target_kind=OrganKind.GENERATOR)

    def test_restart_compile_and_execution_are_identical(self):
        _, residual, by_family = _l3("restart-probe")
        ref = by_family["CHANGE_GENERATOR_POLICY"].mutation.payload["organ"]["implementation_ref"]
        a = compile_native_meta_policy(ref, target_kind=OrganKind.GENERATOR, expected_residual_id=residual.residual_id)
        b = compile_native_meta_policy(ref, target_kind=OrganKind.GENERATOR, expected_residual_id=residual.residual_id)
        self.assertEqual(a.fingerprint(), b.fingerprint())
        probe = _probe_rows(by_family)
        ax = execute_native_meta_policy(a, probe, parent_candidate_budget=1)
        bx = execute_native_meta_policy(b, probe, parent_candidate_budget=1)
        self.assertEqual(ax.fingerprint(), bx.fingerprint())


if __name__ == "__main__":
    unittest.main()
