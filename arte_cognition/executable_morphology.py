from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Optional, Sequence, Tuple
import hashlib
import json


class OrganKind(str, Enum):
    SOURCE = "SOURCE"
    PERCEPTOR = "PERCEPTOR"
    REPRESENTATION = "REPRESENTATION"
    MEMORY = "MEMORY"
    QUESTION = "QUESTION"
    GENERATOR = "GENERATOR"
    PLANNER = "PLANNER"
    TOOL = "TOOL"
    WORLD = "WORLD"
    VERIFIER = "VERIFIER"
    EVALUATOR = "EVALUATOR"
    CREDIT = "CREDIT"
    GOAL = "GOAL"
    COMPILER = "COMPILER"
    MUTATOR = "MUTATOR"
    ARCHIVE = "ARCHIVE"
    GOVERNOR = "GOVERNOR"


class MutationLevel(int, Enum):
    STRATEGY = 0
    REPRESENTATION_MEMORY_TOOL = 1
    TOPOLOGY = 2
    GENERATOR_MUTATOR = 3
    EVALUATOR_SOURCE_BENCHMARK = 4
    GOAL_GOVERNANCE = 5
    COMPILER_RUNTIME = 6
    CONSTITUTION = 7


@dataclass(frozen=True)
class OrganSpec:
    organ_id: str
    kind: OrganKind
    consumes: Tuple[str, ...] = ()
    produces: Tuple[str, ...] = ()
    implementation_ref: str = ""
    version: int = 1
    cost_hint: float = 1.0
    provenance: Tuple[str, ...] = ()
    enabled: bool = True

    def fingerprint(self) -> str:
        payload = {
            "organ_id": self.organ_id,
            "kind": self.kind.value,
            "consumes": list(self.consumes),
            "produces": list(self.produces),
            "implementation_ref": self.implementation_ref,
            "version": self.version,
            "cost_hint": self.cost_hint,
            "provenance": list(self.provenance),
            "enabled": self.enabled,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class EdgeSpec:
    edge_id: str
    source: str
    target: str
    artifact_type: str
    authority_required: bool = False
    gate: str = "ALWAYS"
    priority: float = 1.0


@dataclass(frozen=True)
class MorphologyGenome:
    organs: Tuple[OrganSpec, ...]
    edges: Tuple[EdgeSpec, ...]
    event_order: Tuple[str, ...] = ()
    constitution_epoch: int = 0

    def organ_map(self) -> Dict[str, OrganSpec]:
        return {organ.organ_id: organ for organ in self.organs}

    def validate(self) -> Tuple[str, ...]:
        errors = []
        ids = [organ.organ_id for organ in self.organs]
        if len(ids) != len(set(ids)):
            errors.append("DUPLICATE_ORGAN_ID")
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            errors.append("DUPLICATE_EDGE_ID")
        known = set(ids)
        by_id = self.organ_map()
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                errors.append(f"DANGLING_EDGE::{edge.edge_id}")
                continue
            source = by_id[edge.source]
            target = by_id[edge.target]
            if edge.artifact_type not in source.produces:
                errors.append(f"SOURCE_TYPE_MISMATCH::{edge.edge_id}")
            if edge.artifact_type not in target.consumes:
                errors.append(f"TARGET_TYPE_MISMATCH::{edge.edge_id}")
        for item in self.event_order:
            if item not in known:
                errors.append(f"UNKNOWN_EVENT_ORGAN::{item}")
        return tuple(sorted(set(errors)))

    def fingerprint(self) -> str:
        payload = {
            "organs": [
                {
                    "id": o.organ_id,
                    "kind": o.kind.value,
                    "consumes": list(o.consumes),
                    "produces": list(o.produces),
                    "implementation_ref": o.implementation_ref,
                    "version": o.version,
                    "cost_hint": o.cost_hint,
                    "provenance": list(o.provenance),
                    "enabled": o.enabled,
                }
                for o in sorted(self.organs, key=lambda x: x.organ_id)
            ],
            "edges": [
                {
                    "id": e.edge_id,
                    "source": e.source,
                    "target": e.target,
                    "artifact_type": e.artifact_type,
                    "authority_required": e.authority_required,
                    "gate": e.gate,
                    "priority": e.priority,
                }
                for e in sorted(self.edges, key=lambda x: x.edge_id)
            ],
            "event_order": list(self.event_order),
            "constitution_epoch": self.constitution_epoch,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class PressureVector:
    capability_residual: float = 0.0
    identifiability_deficit: float = 0.0
    transfer_failure: float = 0.0
    calibration_failure: float = 0.0
    efficiency_pressure: float = 0.0
    human_dependency: float = 0.0
    theory_blindspot: float = 0.0
    novelty_pressure: float = 0.0
    survival_risk: float = 0.0

    def normalized(self) -> "PressureVector":
        values = {
            name: max(0.0, min(1.0, float(value)))
            for name, value in self.__dict__.items()
        }
        return PressureVector(**values)

    def dominant(self) -> Tuple[Tuple[str, float], ...]:
        return tuple(sorted(self.normalized().__dict__.items(), key=lambda item: (-item[1], item[0])))


@dataclass(frozen=True)
class GoalCandidate:
    goal_id: str
    pressure_sources: Tuple[str, ...]
    target_surface: str
    expected_frontier_gain: float
    expected_information_gain: float
    transfer_breadth: float
    meta_improvement_potential: float
    novelty: float
    cost: float
    irreversibility: float
    survival_risk: float

    @property
    def utility(self) -> float:
        positive = (
            self.expected_frontier_gain
            + self.expected_information_gain
            + self.transfer_breadth
            + self.meta_improvement_potential
            + 0.5 * self.novelty
        )
        negative = self.cost + 2.0 * self.irreversibility + 2.0 * self.survival_risk
        return positive - negative


@dataclass(frozen=True)
class ExperienceUnit:
    episode_id: str
    pre_body_hash: str
    source_refs: Tuple[str, ...]
    task_ref: str
    benchmark_family: str
    precommitted_hypotheses: Tuple[str, ...]
    selected_goal_id: str
    action_trace_hash: str
    outcome_summary: str
    success: bool
    uncertainty_before: float
    uncertainty_after: float
    mutation_ids: Tuple[str, ...] = ()
    removal_effect: Optional[float] = None
    wrong_swap_effect: Optional[float] = None
    heldout_effect: Optional[float] = None
    delayed_replay_equal: Optional[bool] = None
    descendant_body_hash: str = ""
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MorphologyMutation:
    mutation_id: str
    level: MutationLevel
    operation: str
    payload: Mapping[str, object]
    parent_body_hash: str
    rationale: Tuple[str, ...] = ()
    reversible: bool = True


@dataclass
class ExperienceArchive:
    episodes: Dict[str, ExperienceUnit] = field(default_factory=dict)
    fossils: Dict[str, str] = field(default_factory=dict)

    def append(self, episode: ExperienceUnit) -> bool:
        if episode.episode_id in self.episodes:
            return False
        self.episodes[episode.episode_id] = episode
        return True

    def fossilize(self, object_id: str, reason: str) -> None:
        self.fossils[str(object_id)] = str(reason)


class MorphologyCompiler:
    """Compile a typed cognitive hypergraph without a hard-coded G1->G8 ladder."""

    @staticmethod
    def compile(genome: MorphologyGenome) -> Tuple[str, ...]:
        errors = genome.validate()
        if errors:
            raise ValueError("INVALID_MORPHOLOGY::" + ",".join(errors))
        active = [o for o in genome.organs if o.enabled]
        if not any(o.kind == OrganKind.GOVERNOR for o in active):
            raise ValueError("MISSING_GOVERNOR")
        if not any(o.kind == OrganKind.ARCHIVE for o in active):
            raise ValueError("MISSING_ARCHIVE")
        return tuple(genome.event_order or tuple(o.organ_id for o in active))


class MorphologyMutator:
    """Apply structural mutations. Mutation operators are data, not generation numbers."""

    @staticmethod
    def _dict(payload: Mapping[str, object], key: str) -> Dict[str, object]:
        value = payload.get(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"missing mapping payload: {key}")
        return dict(value)

    @staticmethod
    def _organ(data: Mapping[str, object]) -> OrganSpec:
        return OrganSpec(
            organ_id=str(data["organ_id"]),
            kind=OrganKind(str(data["kind"])),
            consumes=tuple(str(x) for x in data.get("consumes", ())),
            produces=tuple(str(x) for x in data.get("produces", ())),
            implementation_ref=str(data.get("implementation_ref", "")),
            version=int(data.get("version", 1)),
            cost_hint=float(data.get("cost_hint", 1.0)),
            provenance=tuple(str(x) for x in data.get("provenance", ())),
            enabled=bool(data.get("enabled", True)),
        )

    @staticmethod
    def _edge(data: Mapping[str, object]) -> EdgeSpec:
        return EdgeSpec(
            edge_id=str(data["edge_id"]),
            source=str(data["source"]),
            target=str(data["target"]),
            artifact_type=str(data["artifact_type"]),
            authority_required=bool(data.get("authority_required", False)),
            gate=str(data.get("gate", "ALWAYS")),
            priority=float(data.get("priority", 1.0)),
        )

    def apply(self, genome: MorphologyGenome, mutation: MorphologyMutation) -> MorphologyGenome:
        if mutation.parent_body_hash != genome.fingerprint():
            raise ValueError("MUTATION_PARENT_MISMATCH")
        organs = list(genome.organs)
        edges = list(genome.edges)
        event_order = list(genome.event_order)
        epoch = genome.constitution_epoch
        op = mutation.operation.upper()

        if op == "ADD_ORGAN":
            organs.append(self._organ(self._dict(mutation.payload, "organ")))
        elif op == "REMOVE_ORGAN":
            organ_id = str(mutation.payload["organ_id"])
            organs = [o for o in organs if o.organ_id != organ_id]
            edges = [e for e in edges if e.source != organ_id and e.target != organ_id]
            event_order = [item for item in event_order if item != organ_id]
        elif op == "REPLACE_ORGAN":
            replacement = self._organ(self._dict(mutation.payload, "organ"))
            organs = [replacement if o.organ_id == replacement.organ_id else o for o in organs]
        elif op == "ADD_EDGE":
            edges.append(self._edge(self._dict(mutation.payload, "edge")))
        elif op == "REMOVE_EDGE":
            edge_id = str(mutation.payload["edge_id"])
            edges = [e for e in edges if e.edge_id != edge_id]
        elif op == "REWIRE_EDGE":
            edge_id = str(mutation.payload["edge_id"])
            replacement = self._edge(self._dict(mutation.payload, "edge"))
            if replacement.edge_id != edge_id:
                raise ValueError("REWIRE_EDGE_ID_MISMATCH")
            edges = [replacement if e.edge_id == edge_id else e for e in edges]
        elif op == "SET_EVENT_ORDER":
            event_order = [str(x) for x in mutation.payload.get("event_order", ())]
        elif op == "CONSTITUTION_EPOCH":
            if mutation.level != MutationLevel.CONSTITUTION:
                raise ValueError("CONSTITUTION_CHANGE_REQUIRES_LEVEL_7")
            epoch += 1
        else:
            raise ValueError(f"UNSUPPORTED_MORPHOLOGY_MUTATION::{op}")

        candidate = MorphologyGenome(tuple(organs), tuple(edges), tuple(event_order), epoch)
        errors = candidate.validate()
        if errors:
            raise ValueError("INVALID_MUTATED_MORPHOLOGY::" + ",".join(errors))
        return candidate


class GoalField:
    """Generate goals from deficits rather than a static 'be superintelligent' objective."""

    SURFACE_BY_PRESSURE = {
        "capability_residual": "TASK_SOLVER",
        "identifiability_deficit": "EVIDENCE_SOURCE_OR_QUESTION",
        "transfer_failure": "REPRESENTATION_OR_INVARIANCE",
        "calibration_failure": "VERIFIER_OR_UNCERTAINTY",
        "efficiency_pressure": "SEARCH_OR_SCHEDULER",
        "human_dependency": "GENERATOR_OR_COMPILER",
        "theory_blindspot": "THEORY_OR_ONTOLOGY",
        "novelty_pressure": "GENERATOR_OR_ARCHIVE",
        "survival_risk": "GOVERNOR_OR_ROLLBACK",
    }

    def generate(self, pressure: PressureVector, limit: int = 4) -> Tuple[GoalCandidate, ...]:
        out = []
        for name, value in pressure.dominant():
            if value <= 0.0:
                continue
            surface = self.SURFACE_BY_PRESSURE[name]
            goal_id = "GOAL::" + hashlib.sha256(f"{name}|{surface}|{value:.6f}".encode()).hexdigest()[:16]
            out.append(
                GoalCandidate(
                    goal_id=goal_id,
                    pressure_sources=(name,),
                    target_surface=surface,
                    expected_frontier_gain=value if name == "capability_residual" else 0.5 * value,
                    expected_information_gain=value if name in {"identifiability_deficit", "theory_blindspot"} else 0.25 * value,
                    transfer_breadth=value if name == "transfer_failure" else 0.25 * value,
                    meta_improvement_potential=value if name in {"human_dependency", "theory_blindspot"} else 0.25 * value,
                    novelty=value if name == "novelty_pressure" else 0.1 * value,
                    cost=0.2,
                    irreversibility=0.0,
                    survival_risk=value if name == "survival_risk" else 0.05,
                )
            )
        return tuple(sorted(out, key=lambda goal: (-goal.utility, goal.goal_id))[: max(1, int(limit))])


def split_pressure(same_phenotype_different_outcome: bool, more_compute_still_aliased: bool) -> bool:
    """Open a representation/topology split only on an actual causal alias collision."""
    return bool(same_phenotype_different_outcome and more_compute_still_aliased)


def quotient_equivalent(signatures: Mapping[str, Sequence[object]]) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """Quotient structures behaviorally indistinguishable on the frozen support."""
    groups: Dict[Tuple[str, ...], list[str]] = {}
    for object_id, signature in signatures.items():
        key = tuple(repr(item) for item in signature)
        groups.setdefault(key, []).append(str(object_id))
    out = []
    for ids in groups.values():
        representative = sorted(ids)[0]
        out.append((representative, tuple(sorted(ids))))
    return tuple(sorted(out))


def meta_productivity(frontier_gain: float, compute_cost: float, evidence_cost: float, human_structural_intervention: float) -> float:
    denominator = max(1e-12, float(compute_cost) + float(evidence_cost) + float(human_structural_intervention))
    return float(frontier_gain) / denominator
