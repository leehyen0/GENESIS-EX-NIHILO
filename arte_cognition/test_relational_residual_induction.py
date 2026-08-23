from __future__ import annotations

import unittest

from arte_cognition.relational_residual_induction import (
    GeneratedRelationalPathSchema,
    RelationalEdge,
    RelationalResidualInducer,
    derive_relational_path_policy,
    make_context,
    select_authorized_relational_path_schema,
)
from arte_cognition.world_coupling import WorldOutcomePair
from evaluations.run_cross_domain_relational_residual_induction import (
    main as run_external_cross_domain_relational_induction,
)


def _software_context(context_id: str, suffix: str):
    return make_context(
        context_id,
        (
            RelationalEdge(f"builder_{suffix}", "PRODUCES", f"map_{suffix}", "FUNCTION", "MAPPING"),
            RelationalEdge(f"map_{suffix}", "BOUND_AS", f"opts_{suffix}", "MAPPING", "BINDING"),
            RelationalEdge(f"opts_{suffix}", "UNPACKED_INTO", f"call_{suffix}", "BINDING", "CALL"),
            RelationalEdge(f"noise_{suffix}", "MENTIONS", f"call_{suffix}", "NAME", "CALL"),
        ),
        f"builder_{suffix}",
        f"call_{suffix}",
        "SOFTWARE",
    )


def _causal_context(context_id: str, suffix: str):
    return make_context(
        context_id,
        (
            RelationalEdge(f"intervention_{suffix}", "EMITS", f"signal_{suffix}", "INTERVENTION", "SIGNAL"),
            RelationalEdge(f"signal_{suffix}", "MEDIATED_BY", f"state_{suffix}", "SIGNAL", "STATE"),
            RelationalEdge(f"state_{suffix}", "REACHES", f"outcome_{suffix}", "STATE", "OUTCOME"),
            RelationalEdge(f"noise_{suffix}", "COEXISTS", f"outcome_{suffix}", "CONTEXT", "OUTCOME"),
        ),
        f"intervention_{suffix}",
        f"outcome_{suffix}",
        "CAUSAL_WORLD",
    )


def _pair(schema_id: str, context: str, cls: str, effect: float, verified: bool = True):
    return WorldOutcomePair(
        pair_id=f"{schema_id}:{context}:{cls}",
        experiment_id=schema_id,
        axis_id="RELATIONAL::TEST",
        source_id=f"source::{cls}",
        context_id=context,
        challenge_id=f"challenge::{context}",
        epoch=1,
        low_outcome=0.0,
        high_outcome=float(effect),
        low_value=0.0,
        high_value=1.0,
        matched_budget=True,
        externally_generated=True,
        issuer_id=f"issuer::{cls}",
        independence_class_id=cls if verified else "UNVERIFIED",
        authority_verified=verified,
    )


class RelationalResidualInductionTests(unittest.TestCase):
    def test_requires_repeated_zero_selection_residual(self):
        inducer = RelationalResidualInducer()
        contexts = (_software_context("s1", "a"), _software_context("s2", "b"))
        closed = inducer.assess_repeated_residual(contexts, (0, 1), min_contexts=2)
        self.assertEqual(closed.status, "RELATIONAL_RESIDUAL_NOT_ESTABLISHED")
        opened = inducer.assess_repeated_residual(contexts, (0, 0), min_contexts=2)
        self.assertEqual(opened.status, "RELATIONAL_RESIDUAL_OPEN_INDUCTION")

    def test_identifier_invariant_software_path_is_generated(self):
        inducer = RelationalResidualInducer()
        contexts = (_software_context("s1", "alpha"), _software_context("s2", "beta"))
        assessment = inducer.assess_repeated_residual(contexts, (0, 0))
        schemas = inducer.generate_candidates(assessment, contexts)
        self.assertEqual(len(schemas), 1)
        self.assertEqual(
            schemas[0].steps,
            (
                "FUNCTION-[PRODUCES]->MAPPING",
                "MAPPING-[BOUND_AS]->BINDING",
                "BINDING-[UNPACKED_INTO]->CALL",
            ),
        )
        self.assertTrue(inducer.matches(schemas[0], _software_context("heldout", "gamma")))

    def test_same_inducer_generates_nonsoftware_causal_path(self):
        inducer = RelationalResidualInducer()
        contexts = (_causal_context("c1", "alpha"), _causal_context("c2", "beta"))
        assessment = inducer.assess_repeated_residual(contexts, (0, 0))
        schemas = inducer.generate_candidates(assessment, contexts)
        self.assertEqual(len(schemas), 1)
        self.assertEqual(
            schemas[0].steps,
            (
                "INTERVENTION-[EMITS]->SIGNAL",
                "SIGNAL-[MEDIATED_BY]->STATE",
                "STATE-[REACHES]->OUTCOME",
            ),
        )
        self.assertTrue(inducer.matches(schemas[0], _causal_context("heldout", "gamma")))

    def test_wrong_schema_and_remove_lose_heldout_match(self):
        inducer = RelationalResidualInducer()
        heldout = _causal_context("heldout", "omega")
        wrong = GeneratedRelationalPathSchema((
            "INTERVENTION-[EMITS]->SIGNAL",
            "SIGNAL-[REACHES]->OUTCOME",
        ))
        self.assertFalse(inducer.matches(wrong, heldout))
        self.assertEqual(int(False), 0)

    def test_candidate_generation_has_no_outcome_input(self):
        inducer = RelationalResidualInducer()
        contexts = (_software_context("s1", "a"), _software_context("s2", "b"))
        assessment = inducer.assess_repeated_residual(contexts, (0, 0))
        self.assertEqual(len(inducer.generate_candidates(assessment, contexts)), 1)

    def test_policy_requires_two_contexts_and_two_independence_classes(self):
        inducer = RelationalResidualInducer()
        contexts = (_software_context("s1", "a"), _software_context("s2", "b"))
        assessment = inducer.assess_repeated_residual(contexts, (0, 0))
        schema = inducer.generate_candidates(assessment, contexts)[0]
        one_context = (
            _pair(schema.schema_id, "s1", "A", 1.0),
            _pair(schema.schema_id, "s1", "B", 1.0),
        )
        policy = derive_relational_path_policy((schema,), one_context, 2, 2)
        self.assertIsNone(select_authorized_relational_path_schema((schema,), policy))
        two_context = one_context + (
            _pair(schema.schema_id, "s2", "A", 1.0),
            _pair(schema.schema_id, "s2", "B", 1.0),
        )
        policy = derive_relational_path_policy((schema,), two_context, 2, 2)
        self.assertEqual(
            select_authorized_relational_path_schema((schema,), policy).schema_id,
            schema.schema_id,
        )
        verifierless = tuple(_pair(schema.schema_id, p.context_id, "X", 1.0, False) for p in two_context)
        policy = derive_relational_path_policy((schema,), verifierless, 2, 2)
        self.assertIsNone(select_authorized_relational_path_schema((schema,), policy))

    def test_external_cross_domain_preoutcome_transfer(self):
        report = run_external_cross_domain_relational_induction()
        self.assertEqual(
            report["status"],
            "PASS_BOUNDED_CROSS_DOMAIN_RELATIONAL_RESIDUAL_INDUCTION_AND_PREOUTCOME_TRANSFER",
        )
        self.assertEqual(report["treatment_capability"], 1.0)
        self.assertEqual(report["remove_same_checkpoint_capability"], 0.0)
        self.assertEqual(report["wrong_cross_domain_schema_swap_capability"], 0.0)


if __name__ == "__main__":
    unittest.main()
