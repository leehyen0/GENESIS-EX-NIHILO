from __future__ import annotations

import unittest

from arte_cognition.latent_relation_ontology_genesis import OpaqueInterventionalWorld, contrast
from arte_cognition.observation_basis_genesis import WorldDerivedObservationBasisInducer
from arte_cognition.response_partition_genesis import (
    WorldDerivedResponsePartitionInducer,
    derive_response_partition_policy,
    select_authorized_response_schema,
)
from arte_cognition.world_coupling import WorldOutcomePair


def _parallel_profile_world(
    context_id: str,
    prefix: str,
    domain: str,
    magnitude: float = 1.0,
    fast_tail: float = 0.25,
    slow_tail: float = 0.75,
):
    root = f"{prefix}_root"
    fast = f"{prefix}_fast"
    slow = f"{prefix}_slow"
    target = f"{prefix}_target"
    decoy = f"{prefix}_decoy"
    nodes = (root, fast, slow, target, decoy)
    rows = []

    def add_source(source: str, effects):
        for repeat in range(2):
            low = [{node: 0.0 for node in nodes} for _ in range(3)]
            high = [{node: 0.0 for node in nodes} for _ in range(3)]
            high[0][source] = float(magnitude)
            for destination, tail in effects:
                high[1][destination] = float(magnitude)
                high[2][destination] = float(magnitude) * float(tail)
            low[2][decoy] = float(repeat + 1)
            high[2][decoy] = float(repeat + 1)
            rows.append(contrast(
                f"{context_id}:{source}:{repeat}", source, low, high
            ))

    # Two concrete root->target routes have identical peak lag/sign but different
    # complete temporal response shapes.
    add_source(root, ((fast, fast_tail), (slow, slow_tail)))
    add_source(fast, ((target, fast_tail),))
    add_source(slow, ((target, slow_tail),))

    return OpaqueInterventionalWorld(
        context_id=context_id,
        domain=domain,
        source_anchor=root,
        target_anchor=target,
        contrasts=tuple(rows),
    )


def _concrete_peak_schema_multiplicity(world: OpaqueInterventionalWorld) -> int:
    predecessor = WorldDerivedObservationBasisInducer(min_repeats=2)
    adjacency = {}
    for edge in predecessor.infer_edges(world):
        adjacency.setdefault(edge.source, []).append(edge)
    paths = []

    def walk(node, visited, tokens):
        for edge in adjacency.get(node, ()):
            if edge.target in visited:
                continue
            nxt = tokens + (edge.profile_token,)
            if edge.target == world.target_anchor:
                paths.append(nxt)
            walk(edge.target, visited + (edge.target,), nxt)

    walk(world.source_anchor, (world.source_anchor,), ())
    if not paths:
        return 0
    first = paths[0]
    return sum(1 for path in paths if path == first)


def _pair(schema_id: str, context: str, cls: str, effect: float, verified: bool = True):
    return WorldOutcomePair(
        pair_id=f"{schema_id}:{context}:{cls}",
        experiment_id=schema_id,
        axis_id="RESPONSE_PARTITION",
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


class ResponsePartitionGenesisTests(unittest.TestCase):
    def test_peak_sign_predecessor_collapses_two_concrete_paths(self):
        worlds = (
            _parallel_profile_world("a", "alpha", "SOFTWARE", 1.0),
            _parallel_profile_world("b", "beta", "SOFTWARE", 8.0),
        )
        multiplicities = tuple(_concrete_peak_schema_multiplicity(world) for world in worlds)
        self.assertEqual(multiplicities, (2, 2))

        predecessor = WorldDerivedObservationBasisInducer(min_repeats=2)
        assessment = predecessor.assess_residual(worlds, (0, 0), 2)
        schemas = predecessor.generate_candidates(assessment, worlds)
        self.assertEqual(len(schemas), 1)
        self.assertEqual(len(schemas[0].profile_tokens), 2)
        self.assertEqual(schemas[0].profile_tokens[0], schemas[0].profile_tokens[1])

    def test_full_profiles_generate_two_distinct_relation_classes(self):
        inducer = WorldDerivedResponsePartitionInducer(min_repeats=2)
        world = _parallel_profile_world("a", "alpha", "SOFTWARE", 3.0)
        partition = inducer.derive_partition(world)
        self.assertIsNotNone(partition)
        self.assertEqual(len(partition.profile_tokens), 2)
        self.assertEqual(
            set(partition.profile_shapes),
            {(1.0, 0.25), (1.0, 0.75)},
        )

    def test_response_partition_generates_two_candidate_paths_before_outcomes(self):
        inducer = WorldDerivedResponsePartitionInducer(min_repeats=2)
        worlds = (
            _parallel_profile_world("a", "alpha", "SOFTWARE", 1.0),
            _parallel_profile_world("b", "beta", "SOFTWARE", 7.0),
        )
        assessment = inducer.assess_residual(worlds, (2, 2), 2)
        self.assertEqual(
            assessment.status,
            "PEAK_SIGN_PARTITION_AMBIGUOUS_OPEN_RESPONSE_PARTITION",
        )
        schemas = inducer.generate_candidates(assessment, worlds)
        self.assertEqual(len(schemas), 2)
        self.assertNotEqual(schemas[0].profile_tokens, schemas[1].profile_tokens)

    def test_exact_profile_paths_transfer_cross_domain_but_wrong_shape_does_not(self):
        inducer = WorldDerivedResponsePartitionInducer(min_repeats=2)
        training = (
            _parallel_profile_world("a", "alpha", "SOFTWARE", 1.0),
            _parallel_profile_world("b", "beta", "SOFTWARE", 9.0),
        )
        schemas = inducer.generate_candidates(inducer.assess_residual(training, (2, 2)), training)
        heldout = _parallel_profile_world("c", "omega", "CAUSAL_WORLD", 4.0)
        self.assertEqual(sum(int(inducer.matches(schema, heldout)) for schema in schemas), 2)

        wrong = _parallel_profile_world(
            "w", "wrong", "CAUSAL_WORLD", 4.0, fast_tail=0.50, slow_tail=0.90
        )
        self.assertEqual(sum(int(inducer.matches(schema, wrong)) for schema in schemas), 0)

    def test_names_domains_and_magnitude_do_not_define_response_partition(self):
        inducer = WorldDerivedResponsePartitionInducer(min_repeats=2)
        first = inducer.derive_partition(
            _parallel_profile_world("a", "first", "DOMAIN_A", 0.5)
        )
        second = inducer.derive_partition(
            _parallel_profile_world("b", "second", "DOMAIN_B", 13.0)
        )
        self.assertEqual(first.profile_shapes, second.profile_shapes)
        self.assertEqual(first.profile_tokens, second.profile_tokens)

    def test_authority_remains_external_and_repeated(self):
        inducer = WorldDerivedResponsePartitionInducer(min_repeats=2)
        training = (
            _parallel_profile_world("a", "alpha", "SOFTWARE"),
            _parallel_profile_world("b", "beta", "SOFTWARE", 5.0),
        )
        schemas = inducer.generate_candidates(inducer.assess_residual(training, (2, 2)), training)
        schema = schemas[0]
        one = (
            _pair(schema.schema_id, "a", "A", 1.0),
            _pair(schema.schema_id, "a", "B", 1.0),
        )
        self.assertIsNone(select_authorized_response_schema(
            schemas, derive_response_partition_policy(schemas, one, 2, 2)
        ))
        two = one + (
            _pair(schema.schema_id, "b", "A", 1.0),
            _pair(schema.schema_id, "b", "B", 1.0),
        )
        selected = select_authorized_response_schema(
            schemas, derive_response_partition_policy(schemas, two, 2, 2)
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.schema_id, schema.schema_id)

        verifierless = tuple(
            _pair(schema.schema_id, pair.context_id, "X", 1.0, False) for pair in two
        )
        self.assertIsNone(select_authorized_response_schema(
            schemas, derive_response_partition_policy(schemas, verifierless, 2, 2)
        ))


if __name__ == "__main__":
    unittest.main()
