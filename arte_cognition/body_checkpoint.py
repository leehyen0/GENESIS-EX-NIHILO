from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict
import json

from .adaptive_cognition import AdaptiveCognitionCompiler
from .cognitive_runtime import PersistentCognitiveRuntime
from .epistemic_memory import ConceptRecord, EpistemicMemory, LawRecord, RepresentationMutation
from .meta_router import CognitionPolicyState, ModuleExperience, OutcomeLearnedCognitionRouter
from .semantic_genesis import ConceptCandidate, LawCandidate
from .topology_learning import CognitionTopologyLearner, EdgeExperience
from .world_coupling import WorldCouplingEngine, WorldOutcomePair


SCHEMA = "arte.cognition_body_checkpoint/v3"
LEGACY_SCHEMAS = {
    "arte.cognition_body_checkpoint/v1",
    "arte.cognition_body_checkpoint/v2",
}


def checkpoint_dict(runtime: PersistentCognitiveRuntime) -> Dict[str, Any]:
    policy = runtime.router.policy
    topology = runtime.topology
    world = runtime.world_coupling
    return {
        "schema": SCHEMA,
        "policy": {
            "learning_rate": policy.learning_rate,
            "min_evidence_before_routing_change": policy.min_evidence_before_routing_change,
            "max_threshold_shift": policy.max_threshold_shift,
            "modules": {
                module: asdict(exp)
                for module, exp in sorted(policy.modules.items())
            },
        },
        "topology": {
            "learning_rate": topology.learning_rate,
            "min_evidence": topology.min_evidence,
            "max_edge_shift": topology.max_edge_shift,
            "macro_min_edge_value": topology.macro_min_edge_value,
            "edges": [
                {"a": a, "b": b, **asdict(exp)}
                for (a, b), exp in sorted(topology.edges.items())
            ],
            "sequence_counts": [
                {"sequence": list(sequence), "count": count}
                for sequence, count in sorted(topology.sequence_counts.items())
            ],
        },
        "world_coupling": {
            "min_independent_classes": world.min_independent_classes,
            "pairs": [asdict(pair) for pair in world.pairs],
        },
        "memory": {
            "concepts": {
                concept_id: {
                    "concept": asdict(record.concept),
                    "status": record.status,
                    "revisions": record.revisions,
                    "last_law_id": record.last_law_id,
                }
                for concept_id, record in sorted(runtime.memory.concepts.items())
            },
            "laws": {
                law_id: {
                    "law": asdict(record.law),
                    "status": record.status,
                    "refutations": list(record.refutations),
                }
                for law_id, record in sorted(runtime.memory.laws.items())
            },
            "mutation_log": [asdict(item) for item in runtime.memory.mutation_log],
        },
    }


def checkpoint_json(runtime: PersistentCognitiveRuntime) -> str:
    return json.dumps(checkpoint_dict(runtime), sort_keys=True, separators=(",", ":"))


def restore_runtime(payload: Dict[str, Any]) -> PersistentCognitiveRuntime:
    schema = payload.get("schema")
    if schema != SCHEMA and schema not in LEGACY_SCHEMAS:
        raise ValueError("unsupported cognition checkpoint schema")

    policy_data = payload.get("policy", {})
    policy = CognitionPolicyState(
        learning_rate=float(policy_data.get("learning_rate", 0.25)),
        min_evidence_before_routing_change=int(policy_data.get("min_evidence_before_routing_change", 3)),
        max_threshold_shift=float(policy_data.get("max_threshold_shift", 0.10)),
    )
    for module, exp in policy_data.get("modules", {}).items():
        policy.modules[module] = ModuleExperience(
            evidence_count=int(exp.get("evidence_count", 0)),
            signed_value=float(exp.get("signed_value", 0.0)),
            positive_count=int(exp.get("positive_count", 0)),
            negative_count=int(exp.get("negative_count", 0)),
        )

    topology_data = payload.get("topology", {})
    topology = CognitionTopologyLearner(
        learning_rate=float(topology_data.get("learning_rate", 0.25)),
        min_evidence=int(topology_data.get("min_evidence", 3)),
        max_edge_shift=float(topology_data.get("max_edge_shift", 0.15)),
        macro_min_edge_value=float(topology_data.get("macro_min_edge_value", 0.05)),
    )
    for edge in topology_data.get("edges", []):
        topology.edges[(edge["a"], edge["b"])] = EdgeExperience(
            evidence_count=int(edge.get("evidence_count", 0)),
            signed_synergy=float(edge.get("signed_synergy", 0.0)),
            positive_count=int(edge.get("positive_count", 0)),
            negative_count=int(edge.get("negative_count", 0)),
        )
    for item in topology_data.get("sequence_counts", []):
        topology.sequence_counts[tuple(item.get("sequence", []))] = int(item.get("count", 0))

    world_data = payload.get("world_coupling", {})
    world = WorldCouplingEngine(
        min_independent_classes=int(world_data.get("min_independent_classes", 2))
    )
    is_authenticated_schema = schema == SCHEMA
    world.restore_pairs([
        WorldOutcomePair(
            pair_id=item["pair_id"],
            experiment_id=item["experiment_id"],
            axis_id=item["axis_id"],
            source_id=item["source_id"],
            context_id=item["context_id"],
            challenge_id=item["challenge_id"],
            epoch=int(item["epoch"]),
            low_outcome=float(item["low_outcome"]),
            high_outcome=float(item["high_outcome"]),
            low_value=float(item["low_value"]),
            high_value=float(item["high_value"]),
            matched_budget=bool(item.get("matched_budget", False)),
            externally_generated=bool(item.get("externally_generated", False)),
            issuer_id=(
                str(item.get("issuer_id", "UNVERIFIED"))
                if is_authenticated_schema
                else "LEGACY_UNVERIFIED"
            ),
            authority_verified=(
                bool(item.get("authority_verified", False))
                if is_authenticated_schema
                else False
            ),
        )
        for item in world_data.get("pairs", [])
    ])

    compiler = AdaptiveCognitionCompiler()
    router = OutcomeLearnedCognitionRouter(compiler=compiler, policy=policy)
    memory = EpistemicMemory()

    memory_data = payload.get("memory", {})
    for concept_id, item in memory_data.get("concepts", {}).items():
        c = item["concept"]
        concept = ConceptCandidate(
            concept_id=c["concept_id"],
            defining_features=tuple(c["defining_features"]),
            support=int(c["support"]),
            information_gain=float(c["information_gain"]),
            covered_residuals=tuple(c["covered_residuals"]),
            status=c.get("status", "PROPOSAL_ONLY"),
        )
        memory.concepts[concept_id] = ConceptRecord(
            concept=concept,
            status=item.get("status", "SHADOW_PROPOSAL"),
            revisions=int(item.get("revisions", 0)),
            last_law_id=item.get("last_law_id"),
        )

    for law_id, item in memory_data.get("laws", {}).items():
        l = item["law"]
        law = LawCandidate(
            law_id=l["law_id"],
            concept_id=l["concept_id"],
            predicted_outcome=l["predicted_outcome"],
            train_support=int(l["train_support"]),
            train_accuracy=float(l["train_accuracy"]),
            heldout_support=int(l["heldout_support"]),
            heldout_accuracy=float(l["heldout_accuracy"]),
            counterexamples=tuple(l["counterexamples"]),
            status=l["status"],
        )
        memory.laws[law_id] = LawRecord(
            law=law,
            status=item.get("status", law.status),
            refutations=list(item.get("refutations", [])),
        )

    memory.mutation_log = [
        RepresentationMutation(
            mutation_id=item["mutation_id"],
            action=item["action"],
            target=item["target"],
            reason=item["reason"],
            reversible=bool(item.get("reversible", True)),
        )
        for item in memory_data.get("mutation_log", [])
    ]

    return PersistentCognitiveRuntime(
        compiler=compiler,
        router=router,
        topology=topology,
        memory=memory,
        world_coupling=world,
    )


def restore_json(text: str) -> PersistentCognitiveRuntime:
    return restore_runtime(json.loads(text))
