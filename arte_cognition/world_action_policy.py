from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .experiment_genesis import InterventionProposal
from .world_coupling import WorldCouplingEngine


@dataclass(frozen=True)
class WorldActionDecision:
    status: str
    proposal: Optional[InterventionProposal]
    context_id: Optional[str]
    independent_evidence_classes: int
    routing_score: float
    reasons: Tuple[str, ...]


class EvidenceBoundWorldActionPolicy:
    """Separate experimental generation from evidence-supported action choice.

    Authority is bound to the exact generated `experiment_id`, not merely the
    representation axis. Evidence that one manipulation of an axis works cannot
    promote another manipulation on the same axis. This closes an action-authority
    aliasing path while keeping proposal generation cheap and exploratory.
    """

    @staticmethod
    def _experiment_evidence(
        proposal: InterventionProposal,
        world: WorldCouplingEngine,
        context_id: Optional[str],
    ) -> tuple[int, float]:
        valid = [
            pair for pair in world.pairs
            if pair.experiment_id == proposal.experiment_id
            and pair.axis_id == proposal.axis_id
            and pair.matched_budget
            and pair.externally_generated
            and pair.authority_verified
            and pair.independence_class_id != "UNVERIFIED"
            and (context_id is None or pair.context_id == context_id)
        ]
        by_class = {}
        for pair in valid:
            by_class.setdefault(pair.independence_class_id, pair)
        unique = list(by_class.values())
        if not unique:
            return 0, 0.0
        mean_abs_effect = sum(abs(pair.effect) for pair in unique) / len(unique)
        independence_factor = min(1.0, len(unique) / world.min_independent_classes)
        return len(unique), mean_abs_effect * independence_factor

    def select(
        self,
        proposals: Sequence[InterventionProposal],
        world: WorldCouplingEngine,
        context_id: Optional[str] = None,
    ) -> WorldActionDecision:
        if not proposals:
            return WorldActionDecision(
                status="NO_INTERVENTION_CANDIDATES",
                proposal=None,
                context_id=context_id,
                independent_evidence_classes=0,
                routing_score=0.0,
                reasons=("no generated intervention proposals are available",),
            )

        if context_id is None:
            transport = world.assess_transport(proposals)
            if not transport.safe_for_global_transport:
                return WorldActionDecision(
                    status="ABSTAIN_REGIME_CONFLICT",
                    proposal=None,
                    context_id=None,
                    independent_evidence_classes=0,
                    routing_score=0.0,
                    reasons=transport.reasons,
                )

        supported = []
        for index, proposal in enumerate(proposals):
            evidence_classes, score = self._experiment_evidence(proposal, world, context_id)
            if evidence_classes >= world.min_independent_classes and score > 0.0:
                supported.append((proposal, evidence_classes, score, index))

        if not supported:
            return WorldActionDecision(
                status="EXPLORE_ONLY_NO_WORLD_SUPPORTED_ACTION",
                proposal=None,
                context_id=context_id,
                independent_evidence_classes=0,
                routing_score=0.0,
                reasons=(
                    "generated experiments may be explored, but no exact experiment has enough authenticated independent world evidence",
                ),
            )

        chosen, evidence_classes, score, _ = sorted(
            supported,
            key=lambda item: (-item[2], item[3], item[0].experiment_id),
        )[0]
        return WorldActionDecision(
            status="WORLD_SUPPORTED_ACTION",
            proposal=chosen,
            context_id=context_id,
            independent_evidence_classes=evidence_classes,
            routing_score=score,
            reasons=("authenticated independent world outcomes support this exact generated experiment",),
        )
