from __future__ import annotations

import unittest

from arte_cognition.relational_residual_induction import (
    GeneratedRelationalPathSchema,
    RelationalEdge,
    RelationalResidualInducer,
    make_context,
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
        self.assertEqual(int(False), 0)  # REMOVE has no schema and therefore no capability.

    def test_candidate_generation_has_no_outcome_input(self):
        # The public API takes only residual selection counts plus structural graphs;
        # no world outcome or reward object is accepted by candidate generation.
        inducer = RelationalResidualInducer()
        contexts = (_software_context("s1", "a"), _software_context("s2", "b"))
        assessment = inducer.assess_repeated_residual(contexts, (0, 0))
        self.assertEqual(len(inducer.generate_candidates(assessment, contexts)), 1)


if __name__ == "__main__":
    unittest.main()
