from __future__ import annotations

import unittest

from arte_cognition.executable_morphology import EdgeSpec, MorphologyGenome, OrganKind, OrganSpec, PressureVector
from arte_cognition.meta_acceleration import MutationStrategyState, apply_mutation_program
from arte_cognition.morphology_genesis import MorphologyGenesisEngine, MorphologyResidual
from arte_cognition.parametric_morphology_macro import (
    MacroTrainingExample,
    ParametricMorphologyMacro,
    apply_parametric_macro,
    derive_parametric_rewire_macro,
)
from arte_cognition.structural_failure_certificate import (
    StructuralDiagnosticReceipt,
    compile_program_from_certificate,
    derive_structural_failure_certificate,
)


def make_world(prefix: str, depth: int):
    organs = []
    edges = []
    alternates = []
    primaries = []
    for index in range(depth):
        source = f"{prefix}_source_{index}"
        primary = f"{prefix}_primary_{index}"
        alternate = f"{prefix}_alternate_{index}"
        artifact = f"{prefix}_evidence_{index}"
        edge_id = f"{prefix}_edge_{index}"
        organs.extend(
            (
                OrganSpec(source, OrganKind.SOURCE, produces=(artifact,)),
                OrganSpec(primary, OrganKind.REPRESENTATION, consumes=(artifact,)),
                OrganSpec(alternate, OrganKind.REPRESENTATION, consumes=(artifact,)),
            )
        )
        edges.append(EdgeSpec(edge_id, source, primary, artifact))
        alternates.append(alternate)
        primaries.append(primary)
    organs.extend((OrganSpec(f"{prefix}_governor", OrganKind.GOVERNOR), OrganSpec(f"{prefix}_archive", OrganKind.ARCHIVE)))
    genome = MorphologyGenome(tuple(organs), tuple(edges), tuple(o.organ_id for o in organs))
    failed = tuple(edge.edge_id for edge in edges)
    residual = MorphologyResidual(
        f"{prefix}_residual",
        PressureVector(transfer_failure=1.0),
        failed_edge_ids=failed,
        implicated_organ_ids=tuple(primaries + alternates),
    )
    receipts = (
        StructuralDiagnosticReceipt(f"{prefix}_r1", f"{prefix}_ctx", f"{prefix}_class_a", failed, True, True),
        StructuralDiagnosticReceipt(f"{prefix}_r2", f"{prefix}_ctx", f"{prefix}_class_b", tuple(reversed(failed)), True, True),
    )
    certificate = derive_structural_failure_certificate(receipts)
    return genome, residual, certificate, tuple(alternates)


def compile_training_example(prefix: str, depth: int) -> MacroTrainingExample:
    genome, residual, certificate, _ = make_world(prefix, depth)
    candidates = MorphologyGenesisEngine(candidate_budget=128).generate(genome, (residual,))
    strategy = MutationStrategyState(operation_scores=(("REWIRE_EDGE", 3.0), ("ADD_EDGE", 0.0)), lineage_hash=f"{prefix}_prior")
    compilation = compile_program_from_certificate(genome, candidates, strategy, certificate)
    if compilation.program is None:
        raise AssertionError("training compilation failed")
    descendant = apply_mutation_program(genome, compilation.program)
    return MacroTrainingExample(
        context_id=f"{prefix}_context",
        source_class=f"{prefix}_source_class",
        genome=genome,
        certificate=certificate,
        successful_program=compilation.program,
        external_capability=1.0 if descendant.fingerprint() != genome.fingerprint() else 0.0,
        authority_verified=True,
        benchmark_disjoint=True,
    )


class ParametricMorphologyMacroTests(unittest.TestCase):
    def test_two_disjoint_successes_generate_identifier_free_macro(self):
        examples = (compile_training_example("alpha", 2), compile_training_example("beta", 3))
        macro = derive_parametric_rewire_macro(examples)
        self.assertIsNotNone(macro)
        self.assertEqual(macro.rule, "FOR_EACH_CERTIFIED_FAILED_EDGE_REWIRE_TO_UNIQUE_COMPATIBLE_ALTERNATIVE")
        self.assertEqual(len(macro.supporting_contexts), 2)
        self.assertEqual(len(macro.supporting_source_classes), 2)
        self.assertFalse(macro.current_outcomes_required_for_application)

    def test_one_context_is_insufficient_for_macro_authority(self):
        example = compile_training_example("single", 2)
        self.assertIsNone(derive_parametric_rewire_macro((example,)))

    def test_macro_transfers_to_larger_unseen_morphology_without_candidate_evaluation(self):
        macro = derive_parametric_rewire_macro((compile_training_example("train_a", 2), compile_training_example("train_b", 3)))
        genome, _, certificate, alternates = make_world("heldout", 7)
        application = apply_parametric_macro(genome, certificate, macro)
        self.assertIsNotNone(application)
        self.assertEqual(application.candidate_evaluations, 0)
        self.assertEqual(application.structural_lookup_count, 7)
        self.assertEqual(application.mutation_program.depth, 7)
        descendant = apply_mutation_program(genome, application.mutation_program)
        edge_map = {edge.edge_id: edge for edge in descendant.edges}
        for index, target in enumerate(alternates):
            self.assertEqual(edge_map[f"heldout_edge_{index}"].target, target)

    def test_remove_macro_preserves_old_inadequate_body(self):
        genome, _, certificate, alternates = make_world("remove", 4)
        edge_map = {edge.edge_id: edge for edge in genome.edges}
        self.assertTrue(any(edge_map[f"remove_edge_{index}"].target != alternates[index] for index in range(4)))
        self.assertEqual(certificate.lower_bound_program_depth, 4)

    def test_wrong_macro_rule_fails_closed(self):
        macro = ParametricMorphologyMacro(
            macro_id="wrong",
            rule="KEEP_CURRENT_TARGET",
            supporting_contexts=("a", "b"),
            supporting_source_classes=("x", "y"),
            supporting_program_ids=("p1", "p2"),
            inherited_from_external_outcomes=True,
        )
        genome, _, certificate, _ = make_world("wrong", 3)
        self.assertIsNone(apply_parametric_macro(genome, certificate, macro))


if __name__ == "__main__":
    unittest.main()
