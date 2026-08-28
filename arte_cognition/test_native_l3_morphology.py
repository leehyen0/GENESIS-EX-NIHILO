from __future__ import annotations

import unittest

from arte_cognition.executable_morphology import (
    MorphologyGenome,
    MorphologyMutator,
    MutationLevel,
    OrganKind,
    OrganSpec,
    PressureVector,
)
from arte_cognition.morphology_genesis import MorphologyGenesisEngine, MorphologyResidual
from arte_cognition.native_recursive_research import NativeMetaMorphologyGenesisEngine


def body() -> MorphologyGenome:
    organs = (
        OrganSpec("generator", OrganKind.GENERATOR, produces=("candidate",), implementation_ref="bootstrap://generator"),
        OrganSpec("mutator", OrganKind.MUTATOR, consumes=("candidate",), produces=("mutation",), implementation_ref="bootstrap://mutator"),
        OrganSpec("governor", OrganKind.GOVERNOR),
        OrganSpec("archive", OrganKind.ARCHIVE),
    )
    return MorphologyGenome(organs=organs, edges=(), event_order=tuple(o.organ_id for o in organs))


class NativeL3MorphologyTests(unittest.TestCase):
    def test_human_dependency_opens_preoutcome_generator_and_mutator_candidates(self):
        genome = body()
        residual = MorphologyResidual("native-l3", PressureVector(human_dependency=1.0, theory_blindspot=0.5))
        rows = NativeMetaMorphologyGenesisEngine().generate(genome, (residual,))
        l3 = [row for row in rows if row.mutation.level == MutationLevel.GENERATOR_MUTATOR]
        self.assertEqual({row.operation_family for row in l3}, {"CHANGE_GENERATOR_POLICY", "CHANGE_MUTATOR_POLICY"})
        self.assertTrue(all(not row.generation_uses_outcomes for row in l3))
        self.assertTrue(all("shadow_only_until_runtime_semantics_verified" in row.mutation.rationale for row in l3))
        for row in l3:
            descendant = MorphologyMutator().apply(genome, row.mutation)
            changed = [o for o in descendant.organs if o.implementation_ref.startswith("native-meta://")]
            self.assertEqual(len(changed), 1)
            self.assertNotEqual(descendant.fingerprint(), genome.fingerprint())

    def test_zero_meta_pressure_does_not_open_l3(self):
        genome = body()
        residual = MorphologyResidual("no-l3", PressureVector())
        rows = NativeMetaMorphologyGenesisEngine().generate(genome, (residual,))
        self.assertFalse(any(row.mutation.level == MutationLevel.GENERATOR_MUTATOR for row in rows))

    def test_base_engine_is_remove_control_for_l3_reachability(self):
        genome = body()
        residual = MorphologyResidual("remove-l3", PressureVector(human_dependency=1.0))
        rows = MorphologyGenesisEngine().generate(genome, (residual,))
        self.assertFalse(any(row.mutation.level == MutationLevel.GENERATOR_MUTATOR for row in rows))


if __name__ == "__main__":
    unittest.main()
