from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict
import json

from .adaptive_cognition import AdaptiveCognitionCompiler
from .cognitive_runtime import PersistentCognitiveRuntime
from .epistemic_memory import ConceptRecord, EpistemicMemory, LawRecord, RepresentationMutation
from .meta_router import CognitionPolicyState, ModuleExperience, OutcomeLearnedCognitionRouter
from .semantic_genesis import ConceptCandidate, LawCandidate


SCHEMA = "arte.cognition_body_checkpoint/v1"


def checkpoint_dict(runtime: PersistentCognitiveRuntime) -> Dict[str, Any]:
    policy = runtime.router.policy
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
    if payload.get("schema") != SCHEMA:
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
        memory=memory,
    )


def restore_json(text: str) -> PersistentCognitiveRuntime:
    return restore_runtime(json.loads(text))
