from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import hashlib
import json

from .epistemic_memory import EpistemicMemory, RepresentationMutation
from .semantic_genesis import ResidualObservation
from .world_coupling import WorldCouplingEngine


@dataclass(frozen=True)
class ExperimentCounterevidence:
    experiment_id: str
    prior_independent_classes: int
    new_independent_classes: int
    prior_mean_effect: float
    new_mean_effect: float
    contradiction: str


@dataclass(frozen=True)
class WorldCognitionRevision:
    status: str
    axis_id: str
    prior_context_id: str
    new_context_id: str
    counterevidence: Tuple[ExperimentCounterevidence, ...]
    residual: Optional[ResidualObservation]
    mutations: Tuple[RepresentationMutation, ...]
    reasons: Tuple[str, ...]


class AuthenticatedWorldCognitionReviser:
    """Turn robust authenticated world disagreement into reversible BODY revision.

    A single surprising receipt is not enough. The same representation must have
    at least `min_contradicted_experiments` distinct exact experiments that were
    independently supported in the prior context and independently contradicted
    in the new context. Contradiction can be a reproduced sign reversal or a
    collapse of a previously material effect. Only then is the phenotype demoted
    and a fresh residual emitted for the next cognition cycle.
    """

    def __init__(
        self,
        min_contradicted_experiments: int = 2,
        min_abs_effect: float = 0.25,
        collapse_ratio: float = 0.25,
        collapse_absolute: float = 0.05,
    ) -> None:
        self.min_contradicted_experiments = max(1, int(min_contradicted_experiments))
        self.min_abs_effect = max(0.0, float(min_abs_effect))
        self.collapse_ratio = max(0.0, float(collapse_ratio))
        self.collapse_absolute = max(0.0, float(collapse_absolute))

    @staticmethod
    def _experiment_summary(
        world: WorldCouplingEngine,
        experiment_id: str,
        axis_id: str,
        context_id: str,
    ) -> Tuple[int, float]:
        valid = [
            pair for pair in world.pairs
            if pair.experiment_id == experiment_id
            and pair.axis_id == axis_id
            and pair.context_id == context_id
            and pair.matched_budget
            and pair.externally_generated
            and pair.authority_verified
            and pair.independence_class_id != "UNVERIFIED"
        ]
        by_class = {}
        for pair in valid:
            by_class.setdefault(pair.independence_class_id, pair)
        unique = list(by_class.values())
        if not unique:
            return 0, 0.0
        return len(unique), sum(pair.effect for pair in unique) / len(unique)

    def _contradiction(self, prior_effect: float, new_effect: float) -> Optional[str]:
        if abs(prior_effect) < self.min_abs_effect:
            return None
        if abs(new_effect) >= self.min_abs_effect and prior_effect * new_effect < 0.0:
            return "SIGN_FLIP"
        collapse_limit = max(
            self.collapse_absolute,
            abs(prior_effect) * self.collapse_ratio,
        )
        if abs(new_effect) <= collapse_limit:
            return "EFFECT_COLLAPSE"
        return None

    def assess_and_apply(
        self,
        memory: EpistemicMemory,
        world: WorldCouplingEngine,
        axis_id: str,
        prior_context_id: str,
        new_context_id: str,
    ) -> WorldCognitionRevision:
        if prior_context_id == new_context_id:
            return WorldCognitionRevision(
                status="NO_REVISION_SAME_CONTEXT",
                axis_id=axis_id,
                prior_context_id=prior_context_id,
                new_context_id=new_context_id,
                counterevidence=(),
                residual=None,
                mutations=(),
                reasons=("world revision requires a distinct realized context",),
            )

        axis_record = memory.representations.get(axis_id)
        if axis_record is None or axis_record.status != "ACTIVE_VALIDATED":
            return WorldCognitionRevision(
                status="NO_ACTIVE_REPRESENTATION",
                axis_id=axis_id,
                prior_context_id=prior_context_id,
                new_context_id=new_context_id,
                counterevidence=(),
                residual=None,
                mutations=(),
                reasons=("target representation is not currently active",),
            )

        evidence: List[ExperimentCounterevidence] = []
        for experiment_id, record in sorted(memory.experiments.items()):
            if record.proposal.axis_id != axis_id:
                continue
            prior_classes, prior_effect = self._experiment_summary(
                world, experiment_id, axis_id, prior_context_id
            )
            new_classes, new_effect = self._experiment_summary(
                world, experiment_id, axis_id, new_context_id
            )
            if (
                prior_classes < world.min_independent_classes
                or new_classes < world.min_independent_classes
            ):
                continue
            contradiction = self._contradiction(prior_effect, new_effect)
            if contradiction is None:
                continue
            evidence.append(ExperimentCounterevidence(
                experiment_id=experiment_id,
                prior_independent_classes=prior_classes,
                new_independent_classes=new_classes,
                prior_mean_effect=prior_effect,
                new_mean_effect=new_effect,
                contradiction=contradiction,
            ))

        if len(evidence) < self.min_contradicted_experiments:
            return WorldCognitionRevision(
                status="INSUFFICIENT_ROBUST_WORLD_COUNTEREVIDENCE",
                axis_id=axis_id,
                prior_context_id=prior_context_id,
                new_context_id=new_context_id,
                counterevidence=tuple(evidence),
                residual=None,
                mutations=(),
                reasons=(
                    "fewer than the required number of exact experiments were independently contradicted",
                ),
            )

        canonical = {
            "axis_id": axis_id,
            "prior_context_id": prior_context_id,
            "new_context_id": new_context_id,
            "experiments": [
                {
                    "experiment_id": item.experiment_id,
                    "prior_effect": item.prior_mean_effect,
                    "new_effect": item.new_mean_effect,
                    "contradiction": item.contradiction,
                }
                for item in evidence
            ],
        }
        evidence_digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        evidence_tag = f"WORLD_COUNTEREVIDENCE::{evidence_digest}"

        mutations = memory.demote_world_refuted_axis(
            axis_id=axis_id,
            experiment_ids=[item.experiment_id for item in evidence],
            evidence_tag=evidence_tag,
        )
        if not mutations:
            return WorldCognitionRevision(
                status="NO_BODY_STATE_CHANGE",
                axis_id=axis_id,
                prior_context_id=prior_context_id,
                new_context_id=new_context_id,
                counterevidence=tuple(evidence),
                residual=None,
                mutations=(),
                reasons=("counterevidence existed but the target phenotype was already non-active",),
            )

        residual = ResidualObservation(
            residual_id=f"WORLD_REVISION::{evidence_digest}",
            features=("AUTHENTICATED_WORLD_COUNTEREXAMPLE", axis_id),
            outcome="WORLD_MODEL_MISMATCH",
            source_class="AUTHENTICATED_WORLD_REVISION",
            heldout=False,
        )
        return WorldCognitionRevision(
            status="PASS_BOUNDED_WORLD_CAUSED_COGNITION_DEMOTION",
            axis_id=axis_id,
            prior_context_id=prior_context_id,
            new_context_id=new_context_id,
            counterevidence=tuple(evidence),
            residual=residual,
            mutations=tuple(mutations),
            reasons=(
                "multiple exact experiments changed under independently authenticated world evidence",
                "representation, generated experiments, and directly dependent semantic state were demoted without lineage deletion",
                "fresh residual emitted for regeneration rather than self-certifying a replacement",
            ),
        )
