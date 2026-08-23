from __future__ import annotations

import unittest

from arte_cognition.comparison_functional_genesis import WorldDerivedComparisonFunctionalInducer
from arte_cognition.equivalence_criterion_genesis import WorldDerivedEquivalenceCriterionInducer
from evaluations.run_comparison_functional_transfer import (
    _predecessor_multiplicity,
    _world,
    main as run_external_comparison_functional_transfer,
)


class ComparisonFunctionalGenesisTests(unittest.TestCase):
    def _training(self):
        return (
            _world("train-a", "alpha", "SOFTWARE", 1.0, 7.0),
            _world("train-b", "beta", "SOFTWARE", 5.0, -11.0),
        )

    def test_order_sign_predecessor_collapses_two_concrete_paths(self):
        worlds=self._training()
        predecessor=WorldDerivedEquivalenceCriterionInducer(min_repeats=2)
        assessment=predecessor.assess_residual(worlds,(0,0),2)
        candidates=predecessor.generate_candidates(assessment,worlds)
        self.assertEqual(len(candidates),1)
        self.assertEqual([_predecessor_multiplicity(w) for w in worlds],[2,2])
        for _ in range(16):
            self.assertEqual(len(predecessor.generate_candidates(assessment,worlds)),1)

    def test_structural_residual_generates_unique_discriminating_functional(self):
        worlds=self._training()
        inducer=WorldDerivedComparisonFunctionalInducer(coefficient_bound=2,min_repeats=2)
        assessment=inducer.assess_residual(worlds,(1,1),(2,2),2)
        schemas=inducer.generate_candidates(assessment,worlds)
        self.assertEqual(len(schemas),2)
        self.assertEqual({schema.coefficients for schema in schemas},{(1,-2,1)})
        self.assertEqual({schema.path_signs for schema in schemas},{(1,1),(-1,-1)})

    def test_generated_functional_transfers_across_scale_offset_measurement_change(self):
        worlds=self._training()
        heldout=_world("heldout","omega","CAUSAL_WORLD",11.0,23.0)
        inducer=WorldDerivedComparisonFunctionalInducer(coefficient_bound=2,min_repeats=2)
        schemas=inducer.generate_candidates(inducer.assess_residual(worlds,(1,1),(2,2),2),worlds)
        self.assertEqual(sum(int(inducer.matches(schema,heldout)) for schema in schemas),2)

    def test_external_comparison_functional_causal_transfer(self):
        report=run_external_comparison_functional_transfer()
        self.assertEqual(report["status"],"PASS_BOUNDED_WORLD_DERIVED_COMPARISON_FUNCTIONAL_AND_PREOUTCOME_TRANSFER")
        self.assertEqual(report["predecessor_order_sign_unique_candidate_count"],1)
        self.assertEqual(report["predecessor_concrete_path_multiplicity"],[2,2])
        self.assertEqual(report["generated_comparison_candidate_count"],2)
        self.assertEqual(report["generated_coefficients"],[1,-2,1])
        self.assertEqual(report["treatment_capability"],1.0)
        self.assertEqual(report["remove_same_checkpoint_capability"],0.0)
        self.assertEqual(report["structurally_valid_wrong_capability"],0.0)


if __name__=="__main__": unittest.main()
