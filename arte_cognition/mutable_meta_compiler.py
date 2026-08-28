from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple
import base64
import hashlib
import json

from .executable_morphology import ExperienceUnit, MorphologyMutation, MorphologyMutator, MutationLevel, OrganKind
from .meta_acceleration import MutationStrategyState
from .self_evolving_body_checkpoint import SelfEvolvingResearchBody


FAMILY_GENERATOR = "AUTONOMOUS_GENERATOR_LANGUAGE"
FAMILY_MUTATOR = "AUTONOMOUS_MUTATOR_POLICY"
FAMILY_TOPOLOGY = "AUTONOMOUS_TOPOLOGY_REWIRE"
FAMILY_ABSTAIN = "AUTONOMOUS_ABSTAIN"
ROUTABLE_FAMILIES = (FAMILY_GENERATOR, FAMILY_MUTATOR, FAMILY_TOPOLOGY, FAMILY_ABSTAIN)


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * ((4 - len(text) % 4) % 4))


@dataclass(frozen=True, order=True)
class MetaCompilerRule:
    signal_slot: str
    family: str
    reward: float
    support: int

    def __post_init__(self) -> None:
        if not self.signal_slot or "/" in self.signal_slot:
            raise ValueError("INVALID_META_COMPILER_SIGNAL_SLOT")
        if self.family not in ROUTABLE_FAMILIES:
            raise ValueError("INVALID_META_COMPILER_FAMILY")
        if int(self.support) < 1:
            raise ValueError("INVALID_META_COMPILER_SUPPORT")


@dataclass(frozen=True)
class MutableMetaCompilerPolicy:
    rules: Tuple[MetaCompilerRule, ...] = ()
    generation: int = 0

    def __post_init__(self) -> None:
        slots = [row.signal_slot for row in self.rules]
        if len(slots) != len(set(slots)):
            raise ValueError("DUPLICATE_META_COMPILER_SIGNAL_RULE")

    def route(self, signal_slot: str) -> Tuple[str, float]:
        rows = [row for row in self.rules if row.signal_slot == str(signal_slot)]
        if not rows:
            return FAMILY_ABSTAIN, 0.0
        if len(rows) != 1:
            raise ValueError("META_COMPILER_SIGNAL_RULE_NOT_UNIQUE")
        return rows[0].family, 1.0

    def fingerprint(self) -> str:
        return _sha(
            {
                "rules": [[row.signal_slot, row.family, row.reward, row.support] for row in self.rules],
                "generation": self.generation,
            }
        )


def meta_compiler_policy_ref(policy: MutableMetaCompilerPolicy) -> str:
    payload = {
        "generation": int(policy.generation),
        "rules": [[row.signal_slot, row.family, float(row.reward), int(row.support)] for row in sorted(policy.rules)],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded = _b64encode(raw)
    suffix = hashlib.sha256(raw).hexdigest()[:16]
    return f"native-meta-compiler://v1/{encoded}/{suffix}"


def compile_meta_compiler_policy(ref: str) -> MutableMetaCompilerPolicy:
    prefix = "native-meta-compiler://v1/"
    if not str(ref).startswith(prefix):
        raise ValueError("NOT_MUTABLE_META_COMPILER_POLICY")
    parts = str(ref)[len(prefix):].split("/")
    if len(parts) != 2:
        raise ValueError("MALFORMED_MUTABLE_META_COMPILER_POLICY")
    encoded, suffix = parts
    raw = _b64decode(encoded)
    if hashlib.sha256(raw).hexdigest()[:16] != suffix:
        raise ValueError("MUTABLE_META_COMPILER_POLICY_HASH_MISMATCH")
    payload = json.loads(raw.decode("utf-8"))
    rules = tuple(
        MetaCompilerRule(str(slot), str(family), float(reward), int(support))
        for slot, family, reward, support in payload.get("rules", ())
    )
    return MutableMetaCompilerPolicy(tuple(sorted(rules)), int(payload.get("generation", 0)))


def meta_compiler_policy_from_body(body: SelfEvolvingResearchBody) -> MutableMetaCompilerPolicy:
    rows = [organ for organ in body.morphology.organs if organ.kind == OrganKind.COMPILER and organ.enabled]
    if len(rows) != 1:
        raise ValueError("MUTABLE_META_COMPILER_REQUIRES_ONE_ACTIVE_COMPILER")
    return compile_meta_compiler_policy(rows[0].implementation_ref)


def initial_meta_compiler_ref() -> str:
    return meta_compiler_policy_ref(MutableMetaCompilerPolicy())


@dataclass(frozen=True)
class MetaCompilerLearningReceipt:
    signal_slot: str
    winning_family: str
    winning_reward: float
    runner_up_reward: float
    proposal_outcomes: Tuple[Tuple[str, float], ...]
    parent_policy_fingerprint: str
    child_policy_fingerprint: str
    mutation_id: str
    generation_uses_future_validation_outcomes: bool = False

    def fingerprint(self) -> str:
        return _sha(self.__dict__)


def learn_meta_compiler_rule(
    body: SelfEvolvingResearchBody,
    *,
    signal_slot: str,
    proposal_outcomes: Mapping[str, float],
    evidence_ref: str,
) -> MetaCompilerLearningReceipt:
    if set(proposal_outcomes) != set(ROUTABLE_FAMILIES):
        raise ValueError("META_COMPILER_LEARNING_REQUIRES_COMPLETE_PROPOSAL_OUTCOMES")
    ranked = sorted(((family, float(value)) for family, value in proposal_outcomes.items()), key=lambda row: (-row[1], row[0]))
    if ranked[0][1] <= 0.0 or ranked[0][1] <= ranked[1][1]:
        raise ValueError("META_COMPILER_LEARNING_WINNER_NOT_IDENTIFIABLE")
    winner, winning_reward = ranked[0]
    runner_up = ranked[1][1]

    parent_policy = meta_compiler_policy_from_body(body)
    existing = [row for row in parent_policy.rules if row.signal_slot == str(signal_slot)]
    if existing:
        if existing[0].family != winner:
            raise ValueError("META_COMPILER_RULE_CONTRADICTS_INHERITED_POLICY")
        raise ValueError("META_COMPILER_RULE_ALREADY_LEARNED")

    child_policy = MutableMetaCompilerPolicy(
        rules=tuple(sorted(parent_policy.rules + (MetaCompilerRule(str(signal_slot), winner, winning_reward, 1),))),
        generation=parent_policy.generation + 1,
    )
    compiler_rows = [organ for organ in body.morphology.organs if organ.kind == OrganKind.COMPILER and organ.enabled]
    if len(compiler_rows) != 1:
        raise ValueError("MUTABLE_META_COMPILER_REQUIRES_ONE_ACTIVE_COMPILER")
    compiler = compiler_rows[0]
    replacement = {
        "organ_id": compiler.organ_id,
        "kind": compiler.kind.value,
        "consumes": list(compiler.consumes),
        "produces": list(compiler.produces),
        "implementation_ref": meta_compiler_policy_ref(child_policy),
        "version": compiler.version + 1,
        "cost_hint": compiler.cost_hint,
        "provenance": list(compiler.provenance)
        + [
            f"meta-compiler-learning::{signal_slot}",
            f"winner::{winner}",
            f"evidence::{evidence_ref}",
            f"parent-policy::{parent_policy.fingerprint()}",
        ],
        "enabled": compiler.enabled,
    }
    mutation_id = "META_COMPILER_POLICY_UPDATE::" + _sha(
        {"parent": body.morphology.fingerprint(), "replacement": replacement, "outcomes": sorted(proposal_outcomes.items())}
    )[:20]
    mutation = MorphologyMutation(
        mutation_id=mutation_id,
        level=MutationLevel.COMPILER_RUNTIME,
        operation="REPLACE_ORGAN",
        payload={"organ": replacement},
        parent_body_hash=body.morphology.fingerprint(),
        rationale=(
            f"learned-signal::{signal_slot}",
            f"causal-winner::{winner}",
            "past-training-outcomes-only",
        ),
        reversible=True,
    )
    pre_hash = body.morphology.fingerprint()
    body.morphology = MorphologyMutator().apply(body.morphology, mutation)

    scores = body.mutation_strategy.score_map()
    support = body.mutation_strategy.support_map()
    key = "META_COMPILER::" + winner
    scores[key] = scores.get(key, 0.0) + winning_reward - runner_up
    support[key] = support.get(key, 0) + 1
    body.mutation_strategy = MutationStrategyState(
        operation_scores=tuple(sorted(scores.items())),
        operation_support=tuple(sorted(support.items())),
        fossilized_operations=body.mutation_strategy.fossilized_operations,
        lineage_hash=_sha(
            {
                "parent": body.mutation_strategy.lineage_hash,
                "mutation": mutation_id,
                "winner": winner,
                "margin": winning_reward - runner_up,
            }
        ),
    )

    episode = ExperienceUnit(
        episode_id="META_COMPILER_LEARNING::" + _sha((signal_slot, winner, evidence_ref, mutation_id))[:20],
        pre_body_hash=pre_hash,
        source_refs=(str(evidence_ref),),
        task_ref=f"opaque-diagnostic::{signal_slot}",
        benchmark_family="NATIVE_META_COMPILER_CHAMBER",
        precommitted_hypotheses=("PAST_OUTCOME_CAN_IMPROVE_FUTURE_META_ROUTING",),
        selected_goal_id="META_MUTATION_COMPILER_SELF_IMPROVEMENT",
        action_trace_hash=_sha(sorted(proposal_outcomes.items())),
        outcome_summary=(
            f"winner={winner};reward={winning_reward:.6f};runner_up={runner_up:.6f};"
            f"future_validation_outcomes_consumed=false"
        ),
        success=True,
        uncertainty_before=1.0,
        uncertainty_after=0.0,
        mutation_ids=(mutation_id,),
        removal_effect=winning_reward - runner_up,
        wrong_swap_effect=winning_reward - runner_up,
        heldout_effect=None,
        delayed_replay_equal=None,
        descendant_body_hash=body.morphology.fingerprint(),
        notes=(
            "L6_COMPILER_RUNTIME_POLICY_UPDATE",
            "past-training-outcomes-only",
            "source-code-unchanged",
        ),
    )
    if not body.experience_archive.append(episode):
        raise ValueError("META_COMPILER_LEARNING_DUPLICATE_EXPERIENCE")

    return MetaCompilerLearningReceipt(
        signal_slot=str(signal_slot),
        winning_family=winner,
        winning_reward=winning_reward,
        runner_up_reward=runner_up,
        proposal_outcomes=tuple(sorted((family, float(value)) for family, value in proposal_outcomes.items())),
        parent_policy_fingerprint=parent_policy.fingerprint(),
        child_policy_fingerprint=child_policy.fingerprint(),
        mutation_id=mutation_id,
        generation_uses_future_validation_outcomes=False,
    )
