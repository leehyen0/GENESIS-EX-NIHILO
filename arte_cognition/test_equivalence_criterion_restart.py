from __future__ import annotations

import unittest

from arte_cognition.equivalence_criterion_genesis import (
    WorldDerivedEquivalenceCriterionInducer,
    derive_equivalence_policy,
    select_authorized_equivalence,
)
from arte_cognition.test_equivalence_criterion_genesis import _pair, _world


class EquivalenceCriterionRestartTests(unittest.TestCase):
    def _training(self):
        return (_world("restart-a", "alpha_restart", 1), _world("restart-b", "beta_restart", 2))

    def test_fresh_inducer_reconstructs_same_criteria_without_object_reuse(self):
        worlds=self._training()
        first=WorldDerivedEquivalenceCriterionInducer(min_repeats=2)
        second=WorldDerivedEquivalenceCriterionInducer(min_repeats=2)
        a=first.generate_candidates(first.assess_residual(worlds,(0,0),2),worlds)
        b=second.generate_candidates(second.assess_residual(worlds,(0,0),2),worlds)
        self.assertIsNot(first,second)
        self.assertEqual(tuple(x.criterion_id for x in a),tuple(x.criterion_id for x in b))
        self.assertEqual(tuple(x.constraints for x in a),tuple(x.constraints for x in b))
        self.assertTrue(all(x is not y for x,y in zip(a,b)))

    def test_fresh_reconstruction_has_no_authority_until_external_reverification(self):
        worlds=self._training()
        inducer=WorldDerivedEquivalenceCriterionInducer(min_repeats=2)
        criteria=inducer.generate_candidates(inducer.assess_residual(worlds,(0,0),2),worlds)
        self.assertEqual(len(criteria),2)
        empty=derive_equivalence_policy(criteria,(),2,2)
        self.assertIsNone(select_authorized_equivalence(criteria,empty))
        selected=criteria[0]
        receipts=tuple(_pair(selected.criterion_id,ctx,cls,1.0) for ctx in ("restart-a","restart-b") for cls in ("A","B"))
        policy=derive_equivalence_policy(criteria,receipts,2,2)
        restored=select_authorized_equivalence(criteria,policy)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.criterion_id,selected.criterion_id)


if __name__=="__main__": unittest.main()
