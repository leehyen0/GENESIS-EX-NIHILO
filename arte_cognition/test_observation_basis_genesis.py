from __future__ import annotations

import unittest

from arte_cognition.latent_relation_ontology_genesis import (
    OpaqueInterventionalWorld,
    WorldDerivedLatentRelationInducer,
    contrast,
)
from arte_cognition.observation_basis_genesis import (
    WorldDerivedObservationBasisInducer,
    derive_observation_basis_policy,
    select_authorized_observation_schema,
)
from arte_cognition.world_coupling import WorldOutcomePair


def _multi_lag_world(
    context_id: str,
    prefix: str,
    domain: str,
    magnitude: float = 1.0,
    lags=(1, 2, 1),
    signs=(1, 1, 1),
):
    nodes = [f"{prefix}_{i}" for i in range(4)]
    decoy = f"{prefix}_decoy"
    all_nodes = nodes + [decoy]
    rows = []
    for source_index in range(3):
        lag = int(lags[source_index])
        sign = int(signs[source_index])
        for repeat in range(2):
            low = [{node: 0.0 for node in all_nodes} for _ in range(3)]
            high = [{node: 0.0 for node in all_nodes} for _ in range(3)]
            high[0][nodes[source_index]] = float(magnitude)
            high[lag][nodes[source_index + 1]] = float(sign) * float(magnitude)
            # Equal low/high decoy activity is observational clutter, not a relation.
            low[min(2, lag)][decoy] = float(repeat + 1)
            high[min(2, lag)][decoy] = float(repeat + 1)
            rows.append(contrast(
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
        contrasts=tuple(rows),
    )


def _pair(schema_id: str, context: str, cls: str, effect: float, verified: bool = True):
    return WorldOutcomePair(
        pair_id=f"{schema_id}:{context}:{cls}",
        experiment_id=schema_id,
        axis_id="OBSERVATION_BASIS",
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


class ObservationBasisGenesisTests(unittest.TestCase):
    def test_fixed_lag_predecessor_is_completely_inexpressive_on_multilag_path(self):
        worlds = (
            _multi_lag_world("train-a", "alpha", "SOFTWARE", 1.0),
            _multi_lag_world("train-b", "beta", "SOFTWARE", 7.0),
        )
        predecessor = WorldDerivedLatentRelationInducer(
            lag=1, min_effect=0.1, min_repeats=2, max_path_depth=8
        )
        assessment = predecessor.assess_residual(worlds, (0, 0), min_contexts=2)
        self.assertEqual(
            assessment.status,
            "OPAQUE_RELATION_ONTOLOGY_RESIDUAL_OPEN",
        )
        self.assertEqual(predecessor.generate_candidates(assessment, worlds), ())
        for _ in range(16):
            self.assertEqual(predecessor.generate_candidates(assessment, worlds), ())

    def test_raw_timelines_generate_the_required_temporal_basis(self):
        inducer = WorldDerivedObservationBasisInducer(min_repeats=2)
        first = _multi_lag_world("a", "first", "SOFTWARE", 1.0)
        second = _multi_lag_world("b", "second", "SOFTWARE", 9.0)
        basis_a = inducer.derive_basis(first)
        basis_b = inducer.derive_basis(second)
        self.assertIsNotNone(basis_a)
        self.assertIsNotNone(basis_b)
        self.assertEqual(basis_a.lag_offsets, (1, 2))
        self.assertEqual(basis_a.lag_offsets, basis_b.lag_offsets)
        self.assertEqual(basis_a.profile_tokens, basis_b.profile_tokens)
        self.assertEqual(len(basis_a.profile_tokens), 2)

    def test_generated_multilag_schema_transfers_software_to_causal_world(self):
        inducer = WorldDerivedObservationBasisInducer(min_repeats=2)
        training = (
            _multi_lag_world("s1", "alpha", "SOFTWARE", 1.0),
            _multi_lag_world("s2", "beta", "SOFTWARE", 5.0),
        )
        assessment = inducer.assess_residual(training, (0, 0), min_contexts=2)
        self.assertEqual(assessment.status, "FIXED_OBSERVATION_BASIS_RESIDUAL_OPEN")
        schemas = inducer.generate_candidates(assessment, training)
        self.assertEqual(len(schemas), 1)
        schema = schemas[0]
        self.assertEqual(len(schema.profile_tokens), 3)
        self.assertEqual(schema.profile_tokens[0], schema.profile_tokens[2])
        self.assertNotEqual(schema.profile_tokens[0], schema.profile_tokens[1])
        heldout = _multi_lag_world("c", "omega", "CAUSAL_WORLD", 3.25)
        self.assertTrue(inducer.matches(schema, heldout))

    def test_wrong_temporal_basis_and_wrong_sign_are_not_equivalent(self):
        inducer = WorldDerivedObservationBasisInducer(min_repeats=2)
        training = (
            _multi_lag_world("s1", "a", "SOFTWARE"),
            _multi_lag_world("s2", "b", "SOFTWARE", 4.0),
        )
        schema = inducer.generate_candidates(
            inducer.assess_residual(training, (0, 0), 2), training
        )[0]
        wrong_lag = _multi_lag_world(
            "wrong-lag", "wl", "CAUSAL_WORLD", lags=(1, 1, 1)
        )
        wrong_sign = _multi_lag_world(
            "wrong-sign", "ws", "CAUSAL_WORLD", lags=(1, 2, 1), signs=(1, -1, 1)
        )
        self.assertFalse(inducer.matches(schema, wrong_lag))
        self.assertFalse(inducer.matches(schema, wrong_sign))

    def test_concrete_names_domains_and_response_magnitudes_do_not_define_basis(self):
        inducer = WorldDerivedObservationBasisInducer(min_repeats=2)
        worlds = (
            _multi_lag_world("x", "totally_different_a", "DOMAIN_A", 0.75),
            _multi_lag_world("y", "totally_different_b", "DOMAIN_B", 11.0),
        )
        bases = tuple(inducer.derive_basis(world) for world in worlds)
        self.assertEqual(bases[0].lag_offsets, bases[1].lag_offsets)
        self.assertEqual(bases[0].profile_tokens, bases[1].profile_tokens)

    def test_authority_requires_repeated_independent_world_support(self):
        inducer = WorldDerivedObservationBasisInducer(min_repeats=2)
        training = (
            _multi_lag_world("s1", "a", "SOFTWARE"),
            _multi_lag_world("s2", "b", "SOFTWARE", 2.0),
        )
        schema = inducer.generate_candidates(
            inducer.assess_residual(training, (0, 0), 2), training
        )[0]
        one_context = (
            _pair(schema.schema_id, "s1", "A", 1.0),
            _pair(schema.schema_id, "s1", "B", 1.0),
        )
        policy = derive_observation_basis_policy((schema,), one_context, 2, 2)
        self.assertIsNone(select_authorized_observation_schema((schema,), policy))
        two_contexts = one_context + (
            _pair(schema.schema_id, "s2", "A", 1.0),
            _pair(schema.schema_id, "s2", "B", 1.0),
        )
        policy = derive_observation_basis_policy((schema,), two_contexts, 2, 2)
        selected = select_authorized_observation_schema((schema,), policy)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.schema_id, schema.schema_id)
        verifierless = tuple(
            _pair(schema.schema_id, pair.context_id, "X", 1.0, False)
            for pair in two_contexts
        )
        policy = derive_observation_basis_policy((schema,), verifierless, 2, 2)
        self.assertIsNone(select_authorized_observation_schema((schema,), policy))


if __name__ == "__main__":
    unittest.main()
