import unittest

from arte_cognition.epistemic_memory import EpistemicMemory
from arte_cognition.experiment_genesis import InterventionProposal
from arte_cognition.representation_genesis import RepresentationAxis
from arte_cognition.semantic_genesis import ConceptCandidate, LawCandidate
from arte_cognition.world_coupling import WorldCouplingEngine, WorldOutcomePair
from arte_cognition.world_revision import AuthenticatedWorldCognitionReviser


AXIS_ID = "AXIS::PROJECTION::x|y"


def axis():
    return RepresentationAxis(
        axis_id=AXIS_ID,
        family="PROJECTION",
        inputs=("x", "y"),
        threshold=0.0,
        direction="GT",
        information_gain=1.0,
        train_support=8,
        positive_partition=("b1", "b2", "b3", "b4"),
        formula="x+y",
        coefficients=(("x", 1.0), ("y", 1.0)),
        bias=0.0,
    )


def proposal(name, variable, held):
    return InterventionProposal(
        experiment_id=name,
        axis_id=AXIS_ID,
        manipulated_variable=variable,
        held_fixed=held,
        low_value=-1.0,
        high_value=1.0,
        predicted_low_side="LE_THRESHOLD",
        predicted_high_side="GT_THRESHOLD",
        reason="world revision regression",
    )


def pair(experiment_id, context, class_id, effect, authority=True):
    source = f"source::{context}::{class_id}::{experiment_id}"
    challenge = f"challenge::{context}::{class_id}::{experiment_id}"
    return WorldOutcomePair(
        pair_id=f"PAIR::{experiment_id}::{context}::{class_id}",
        experiment_id=experiment_id,
        axis_id=AXIS_ID,
        source_id=source,
        context_id=context,
        challenge_id=challenge,
        epoch=1,
        low_outcome=0.0,
        high_outcome=float(effect),
        low_value=-1.0,
        high_value=1.0,
        matched_budget=True,
        externally_generated=True,
        issuer_id=f"issuer::{class_id}",
        independence_class_id=class_id if authority else "UNVERIFIED",
        authority_verified=authority,
    )


def trained_memory():
    memory = EpistemicMemory()
    memory.remember_representation(axis())
    memory.remember_experiment(proposal("EXP::1", "x", (("y", 0.0),)))
    memory.remember_experiment(proposal("EXP::2", "y", (("x", 0.0),)))
    concept = ConceptCandidate(
        concept_id=f"CONCEPT::{AXIS_ID}",
        defining_features=(AXIS_ID,),
        support=4,
        information_gain=1.0,
        covered_residuals=("b1", "b2", "b3", "b4"),
    )
    memory.remember_concept(concept)
    memory.ingest_law(LawCandidate(
        law_id=f"LAW::{concept.concept_id}",
        concept_id=concept.concept_id,
        predicted_outcome="B",
        train_support=4,
        train_accuracy=1.0,
        heldout_support=2,
        heldout_accuracy=1.0,
        counterexamples=(),
        status="BOUNDED_LAW",
    ))
    return memory


class WorldRevisionTests(unittest.TestCase):
    def test_two_exact_experiments_with_independent_sign_flip_demote_cognition(self):
        memory = trained_memory()
        world = WorldCouplingEngine(min_independent_classes=2)
        for experiment_id in ("EXP::1", "EXP::2"):
            for class_id in ("class-a", "class-b"):
                world.record_pair(pair(experiment_id, "OLD", class_id, 1.0))
                world.record_pair(pair(experiment_id, "NEW", class_id, -1.0))

        revision = AuthenticatedWorldCognitionReviser().assess_and_apply(
            memory, world, AXIS_ID, "OLD", "NEW"
        )
        self.assertEqual(revision.status, "PASS_BOUNDED_WORLD_CAUSED_COGNITION_DEMOTION")
        self.assertEqual(len(revision.counterevidence), 2)
        self.assertTrue(all(item.contradiction == "SIGN_FLIP" for item in revision.counterevidence))
        self.assertIsNotNone(revision.residual)
        self.assertEqual(memory.representations[AXIS_ID].status, "SHADOW_WORLD_REFUTED")
        self.assertEqual(memory.active_representation_axes(), [])
        self.assertEqual(memory.persisted_intervention_proposals(), [])
        self.assertTrue(all(record.status == "SHADOW_WORLD_REFUTED" for record in memory.experiments.values()))
        self.assertEqual(memory.concepts[f"CONCEPT::{AXIS_ID}"].status, "SHADOW_WORLD_REFUTED")
        self.assertEqual(memory.laws[f"LAW::CONCEPT::{AXIS_ID}"].status, "DEMOTED_WORLD_REFUTED")
        self.assertTrue(any(m.target == AXIS_ID and m.action == "DEMOTE" for m in revision.mutations))

    def test_one_contradicted_experiment_is_not_enough_to_demote_axis(self):
        memory = trained_memory()
        world = WorldCouplingEngine(min_independent_classes=2)
        for class_id in ("class-a", "class-b"):
            world.record_pair(pair("EXP::1", "OLD", class_id, 1.0))
            world.record_pair(pair("EXP::1", "NEW", class_id, -1.0))
        revision = AuthenticatedWorldCognitionReviser().assess_and_apply(
            memory, world, AXIS_ID, "OLD", "NEW"
        )
        self.assertEqual(revision.status, "INSUFFICIENT_ROBUST_WORLD_COUNTEREVIDENCE")
        self.assertEqual(memory.representations[AXIS_ID].status, "ACTIVE_VALIDATED")
        self.assertEqual(len(memory.persisted_intervention_proposals()), 2)
        self.assertEqual(revision.mutations, ())

    def test_unverified_or_one_class_evidence_cannot_rewrite_body(self):
        memory = trained_memory()
        world = WorldCouplingEngine(min_independent_classes=2)
        for experiment_id in ("EXP::1", "EXP::2"):
            world.record_pair(pair(experiment_id, "OLD", "class-a", 1.0))
            world.record_pair(pair(experiment_id, "NEW", "class-a", -1.0))
            world.record_pair(pair(experiment_id, "NEW", "class-b", -1.0, authority=False))
        revision = AuthenticatedWorldCognitionReviser().assess_and_apply(
            memory, world, AXIS_ID, "OLD", "NEW"
        )
        self.assertEqual(revision.status, "INSUFFICIENT_ROBUST_WORLD_COUNTEREVIDENCE")
        self.assertEqual(memory.representations[AXIS_ID].status, "ACTIVE_VALIDATED")
        self.assertFalse(revision.mutations)


if __name__ == "__main__":
    unittest.main()
