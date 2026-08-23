from __future__ import annotations

import unittest

from arte_cognition.executable_morphology import EdgeSpec, MorphologyGenome, OrganKind, OrganSpec, PressureVector
from arte_cognition.meta_acceleration import MutationProgramDevelopmentState, MutationStrategyState, apply_mutation_program
from arte_cognition.morphology_genesis import MorphologyGenesisEngine, MorphologyResidual
from arte_cognition.structural_failure_certificate import (
    StructuralDiagnosticReceipt,
    compile_program_from_certificate,
    derive_structural_failure_certificate,
    open_program_depth_from_certificate,
)


def world():
    organs = []
    edges = []
    primary = []
    alternate = []
    for index in range(3):
        source = f"s{index}"
        p = f"p{index}"
        a = f"a{index}"
        artifact = f"evidence{index}"
        primary.append(p)
        alternate.append(a)
        organs.extend(
            (
                OrganSpec(source, OrganKind.SOURCE, produces=(artifact,)),
                OrganSpec(p, OrganKind.REPRESENTATION, consumes=(artifact,)),
                OrganSpec(a, OrganKind.REPRESENTATION, consumes=(artifact,)),
            )
        )
        edges.append(EdgeSpec(f"e{index}", source, p, artifact))
    organs.extend((OrganSpec("governor", OrganKind.GOVERNOR), OrganSpec("archive", OrganKind.ARCHIVE)))
    genome = MorphologyGenome(tuple(organs), tuple(edges), tuple(o.organ_id for o in organs))
    residual = MorphologyResidual(
        "r",
        PressureVector(transfer_failure=1.0),
        failed_edge_ids=("e0", "e1", "e2"),
        implicated_organ_ids=tuple(primary + alternate),
    )
    pool = MorphologyGenesisEngine(candidate_budget=64).generate(genome, (residual,))
    return genome, pool


class StructuralFailureCertificateTests(unittest.TestCase):
    def test_two_independent_receipts_derive_depth_lower_bound(self):
        receipts = (
            StructuralDiagnosticReceipt("r1", "ctx", "class-a", ("e0", "e1", "e2"), True, True),
            StructuralDiagnosticReceipt("r2", "ctx", "class-b", ("e2", "e1", "e0"), True, True),
        )
        certificate = derive_structural_failure_certificate(receipts)
        self.assertIsNotNone(certificate)
        self.assertEqual(certificate.lower_bound_program_depth, 3)
        self.assertEqual(certificate.failed_locus_ids, ("e0", "e1", "e2"))

    def test_conflicting_diagnostics_fail_closed(self):
        receipts = (
            StructuralDiagnosticReceipt("r1", "ctx", "class-a", ("e0", "e1"), True, True),
            StructuralDiagnosticReceipt("r2", "ctx", "class-b", ("e0", "e2"), True, True),
        )
        self.assertIsNone(derive_structural_failure_certificate(receipts))

    def test_verifierless_receipt_cannot_form_certificate(self):
        receipts = (
            StructuralDiagnosticReceipt("r1", "ctx", "class-a", ("e0",), False, True),
            StructuralDiagnosticReceipt("r2", "ctx", "class-b", ("e0",), True, True),
        )
        self.assertIsNone(derive_structural_failure_certificate(receipts))

    def test_certificate_can_jump_mutator_depth_without_enumerating_lower_depths(self):
        receipts = (
            StructuralDiagnosticReceipt("r1", "ctx", "class-a", ("e0", "e1", "e2"), True, True),
            StructuralDiagnosticReceipt("r2", "ctx", "class-b", ("e0", "e1", "e2"), True, True),
        )
        certificate = derive_structural_failure_certificate(receipts)
        state = open_program_depth_from_certificate(MutationProgramDevelopmentState(max_depth=1), certificate)
        self.assertEqual(state.max_depth, 3)
        self.assertTrue(state.lineage_hash)

    def test_certificate_compiler_uses_inherited_prior_not_current_outcomes(self):
        genome, pool = world()
        receipts = (
            StructuralDiagnosticReceipt("r1", "ctx", "class-a", ("e0", "e1", "e2"), True, True),
            StructuralDiagnosticReceipt("r2", "ctx", "class-b", ("e0", "e1", "e2"), True, True),
        )
        certificate = derive_structural_failure_certificate(receipts)
        strategy = MutationStrategyState(
            operation_scores=(("ADD_EDGE", 0.0), ("REWIRE_EDGE", 3.0)), lineage_hash="past"
        )
        compilation = compile_program_from_certificate(genome, pool, strategy, certificate)
        self.assertIsNotNone(compilation.program)
        self.assertEqual(compilation.unresolved_locus_ids, ())
        self.assertEqual(compilation.program.depth, 3)
        self.assertTrue(all(template.operation == "REWIRE_EDGE" for template in compilation.program.templates))
        self.assertFalse(compilation.program.generation_uses_current_outcomes)
        descendant = apply_mutation_program(genome, compilation.program)
        edge_map = {edge.edge_id: edge for edge in descendant.edges}
        self.assertEqual(edge_map["e0"].target, "a0")
        self.assertEqual(edge_map["e1"].target, "a1")
        self.assertEqual(edge_map["e2"].target, "a2")


if __name__ == "__main__":
    unittest.main()
