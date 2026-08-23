from __future__ import annotations

import unittest

from arte_cognition.comparison_functional_genesis import WorldDerivedComparisonFunctionalInducer
from arte_cognition.polynomial_relation_genesis import WorldDerivedPolynomialRelationInducer
from evaluations.run_polynomial_relation_transfer import _world, main as run_external_polynomial_relation_transfer


class PolynomialRelationGenesisTests(unittest.TestCase):
    def _training(self):
        return (_world("train-a","alpha","SOFTWARE",1.0,9.0),_world("train-b","beta","SOFTWARE",7.0,-13.0))

    def test_complete_bounded_linear_comparison_grammar_fails(self):
        worlds=self._training()
        predecessor=WorldDerivedComparisonFunctionalInducer(coefficient_bound=2,min_repeats=2)
        assessment=predecessor.assess_residual(worlds,(1,1),(2,2),2)
        self.assertEqual(predecessor.generate_candidates(assessment,worlds),())
        for _ in range(16): self.assertEqual(predecessor.generate_candidates(assessment,worlds),())

    def test_degree_two_residual_generates_unique_relation_partition(self):
        worlds=self._training(); inducer=WorldDerivedPolynomialRelationInducer(degree=2,min_repeats=2)
        schemas=inducer.generate_candidates(inducer.assess_residual(worlds,(0,0),2),worlds)
        self.assertEqual(len(schemas),2)
        self.assertEqual({(s.positive_monomial,s.negative_monomial) for s in schemas},{((0,2,0),(1,0,1))})
        self.assertEqual({s.path_signs for s in schemas},{(-1,-1),(1,1)})

    def test_generated_relation_transfers_across_scale_and_offset(self):
        worlds=self._training(); heldout=_world("heldout","omega","CAUSAL_WORLD",11.0,31.0)
        inducer=WorldDerivedPolynomialRelationInducer(degree=2,min_repeats=2)
        schemas=inducer.generate_candidates(inducer.assess_residual(worlds,(0,0),2),worlds)
        self.assertEqual(sum(int(inducer.matches(s,heldout)) for s in schemas),2)

    def test_external_polynomial_relation_causal_transfer(self):
        report=run_external_polynomial_relation_transfer()
        self.assertEqual(report["status"],"PASS_BOUNDED_WORLD_DERIVED_POLYNOMIAL_RELATION_LIFT_AND_PREOUTCOME_TRANSFER")
        self.assertEqual(report["predecessor_linear_candidate_count"],0)
        self.assertEqual(report["generated_polynomial_candidate_count"],2)
        self.assertEqual(report["generated_positive_monomial"],[0,2,0])
        self.assertEqual(report["generated_negative_monomial"],[1,0,1])
        self.assertEqual(report["treatment_capability"],1.0)
        self.assertEqual(report["remove_same_checkpoint_capability"],0.0)
        self.assertEqual(report["structurally_valid_wrong_capability"],0.0)


if __name__=="__main__": unittest.main()
