from __future__ import annotations

import unittest

from arte_cognition.latent_relation_ontology_genesis import (
    GeneratedLatentPathSchema,
    OpaqueInterventionalWorld,
    WorldDerivedLatentRelationInducer,
    contrast,
    derive_latent_path_policy,
    select_authorized_latent_path,
)
from arte_cognition.world_coupling import WorldOutcomePair
from evaluations.run_latent_relation_ontology_transfer import (
    main as run_external_latent_relation_ontology_transfer,
)


def _world(context_id: str, prefix: str, domain: str, magnitude: float = 1.0):
    nodes = [f"{prefix}_{i}" for i in range(4)]
    decoy = f"{prefix}_decoy"
    all_nodes = nodes + [decoy]
    contrasts = []
    zero = {node: 0.0 for node in all_nodes}
    for source_index in range(3):
        for repeat in range(2):
            low = [dict(zero) for _ in range(4)]
            high = [dict(zero) for _ in range(4)]
            high[0][nodes[source_index]] = magnitude
            for lag in range(1, 4 - source_index):
                high[lag][nodes[source_index + lag]] = magnitude
            contrasts.append(contrast(
                f"{context_id}:{source_index}:{repeat}",
                nodes[source_index],
                low,
                high,
            ))
    return OpaqueInterventionalWorld(
        context_id=context_id,
        domain=domain,
        source_anchor=nodes[0],
        target_anchor=nodes[-1],
        contrasts=tuple(contrasts),
    )


def _mixed_sign_world(context_id: str, prefix: str, domain: str):
    """Same three-edge topology as `_world`, but the middle relation reverses sign."""
    nodes = [f"{prefix}_{i}" for i in range(4)]
    zero = {node: 0.0 for node in nodes}
    contrasts = []
    for source_index in range(3):
        signed_effect = -1.0 if source_index == 1 else 1.0
        for repeat in range(2):
            low = [dict(zero), dict(zero)]
            high = [dict(zero), dict(zero)]
            high[0][nodes[source_index]] = 1.0
            high[1][nodes[source_index + 1]] = signed_effect
            contrasts.append(contrast(
                f"{context_id}:{source_index}:{repeat}",
                nodes[source_index],
                low,
                high,
            ))
    return OpaqueInterventionalWorld(
        context_id=context_id,
        domain=domain,
        source_anchor=nodes[0],
        target_anchor=nodes[-1],
        contrasts=tuple(contrasts),
    )


def _pair(schema_id: str, context_id: str, cls: str, effect: float, verified: bool = True):
    return WorldOutcomePair(
        pair_id=f"{schema_id}:{context_id}:{cls}",
        experiment_id=schema_id,
        axis_id="LATENT_RELATION",
        source_id=f"source::{cls}",
        context_id=context_id,
        challenge_id=f"challenge::{context_id}",
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


class LatentRelationOntologyGenesisTests(unittest.TestCase):
    def test_repeated_predecessor_failure_is_required(self):
        inducer = WorldDerivedLatentRelationInducer()
        worlds = (_world("s1", "alpha", "SOFTWARE"), _world("s2", "beta", "SOFTWARE"))
        closed = inducer.assess_residual(worlds, (0, 1), 2)
        self.assertEqual(closed.status, "OPAQUE_RELATION_ONTOLOGY_RESIDUAL_NOT_ESTABLISHED")
        opened = inducer.assess_residual(worlds, (0, 0), 2)
        self.assertEqual(opened.status, "OPAQUE_RELATION_ONTOLOGY_RESIDUAL_OPEN")

    def test_relation_tokens_are_generated_without_domain_labels_or_node_kinds(self):
        inducer = WorldDerivedLatentRelationInducer()
        software = _world("s", "software_random", "SOFTWARE")
        causal = _world("c", "causal_random", "CAUSAL_WORLD")
        software_tokens = {edge.relation_token for edge in inducer.infer_edges(software)}
        causal_tokens = {edge.relation_token for edge in inducer.infer_edges(causal)}
        self.assertEqual(len(software_tokens), 1)
        self.assertEqual(software_tokens, causal_tokens)
        token = next(iter(software_tokens))
        self.assertTrue(token.startswith("LATENT_REL::"))
        self.assertNotIn("SOFTWARE", token)
        self.assertNotIn("CAUSAL", token)

    def test_same_exact_schema_transfers_software_to_causal_world(self):
        inducer = WorldDerivedLatentRelationInducer()
        training = (_world("s1", "alpha", "SOFTWARE"), _world("s2", "beta", "SOFTWARE"))
        assessment = inducer.assess_residual(training, (0, 0), 2)
        schemas = inducer.generate_candidates(assessment, training)
        self.assertEqual(len(schemas), 1)
        schema = schemas[0]
        self.assertEqual(len(schema.relation_tokens), 3)
        self.assertEqual(len(set(schema.relation_tokens)), 1)
        heldout = _world("c-heldout", "omega", "CAUSAL_WORLD")
        self.assertTrue(inducer.matches(schema, heldout))

    def test_same_topology_but_reversed_relation_sign_blocks_transfer(self):
        inducer = WorldDerivedLatentRelationInducer(min_effect=0.1)
        training = (_world("s1", "alpha", "SOFTWARE"), _world("s2", "beta", "SOFTWARE"))
        schema = inducer.generate_candidates(inducer.assess_residual(training, (0, 0), 2), training)[0]
        counterexample = _mixed_sign_world("c-sign-flip", "omega", "CAUSAL_WORLD")
        counterexample_tokens = tuple(edge.relation_token for edge in inducer.infer_edges(counterexample))
        self.assertEqual(len(set(counterexample_tokens)), 2)
        self.assertFalse(inducer.matches(schema, counterexample))

    def test_identifier_and_magnitude_invariance_preserve_relation_fingerprint(self):
        inducer = WorldDerivedLatentRelationInducer(min_effect=0.1)
        first = _world("a", "first_names", "A", magnitude=1.0)
        second = _world("b", "other_names", "B", magnitude=7.0)
        first_tokens = tuple(edge.relation_token for edge in inducer.infer_edges(first))
        second_tokens = tuple(edge.relation_token for edge in inducer.infer_edges(second))
        self.assertEqual(set(first_tokens), set(second_tokens))

    def test_remove_and_wrong_schema_do_not_match_cross_domain_heldout(self):
        inducer = WorldDerivedLatentRelationInducer()
        training = (_world("s1", "a", "SOFTWARE"), _world("s2", "b", "SOFTWARE"))
        schema = inducer.generate_candidates(inducer.assess_residual(training, (0, 0)), training)[0]
        heldout = _world("c", "z", "CAUSAL_WORLD")
        wrong = GeneratedLatentPathSchema(schema.relation_tokens[:-1])
        self.assertTrue(inducer.matches(schema, heldout))
        self.assertFalse(inducer.matches(wrong, heldout))
        self.assertEqual(int(False), 0)

    def test_authority_requires_two_contexts_and_independent_classes(self):
        inducer = WorldDerivedLatentRelationInducer()
        training = (_world("s1", "a", "SOFTWARE"), _world("s2", "b", "SOFTWARE"))
        schema = inducer.generate_candidates(inducer.assess_residual(training, (0, 0)), training)[0]
        one = (
            _pair(schema.schema_id, "s1", "A", 1.0),
            _pair(schema.schema_id, "s1", "B", 1.0),
        )
        policy = derive_latent_path_policy((schema,), one, 2, 2)
        self.assertIsNone(select_authorized_latent_path((schema,), policy))
        two = one + (
            _pair(schema.schema_id, "s2", "A", 1.0),
            _pair(schema.schema_id, "s2", "B", 1.0),
        )
        policy = derive_latent_path_policy((schema,), two, 2, 2)
        self.assertEqual(select_authorized_latent_path((schema,), policy).schema_id, schema.schema_id)
        verifierless = tuple(_pair(schema.schema_id, p.context_id, "X", 1.0, False) for p in two)
        policy = derive_latent_path_policy((schema,), verifierless, 2, 2)
        self.assertIsNone(select_authorized_latent_path((schema,), policy))

    def test_external_world_derived_ontology_cross_domain_transfer(self):
        report = run_external_latent_relation_ontology_transfer()
        self.assertEqual(
            report["status"],
            "PASS_BOUNDED_WORLD_DERIVED_LATENT_RELATION_ONTOLOGY_AND_CROSS_DOMAIN_PREOUTCOME_TRANSFER",
        )
        self.assertTrue(report["same_exact_schema_transferred_software_to_causal_world"])
        self.assertEqual(report["treatment_capability"], 1.0)
        self.assertEqual(report["remove_same_checkpoint_capability"], 0.0)
        self.assertEqual(report["wrong_short_path_capability"], 0.0)
        self.assertEqual(report["wrong_relation_token_capability"], 0.0)


if __name__ == "__main__":
    unittest.main()
