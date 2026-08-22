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

    Generated interventions remain exploration candidates. They do not become a
    learned action preference merely because the BODY can propose them. A proposal
    is selected for exploitation only after its axis has enough authenticated,
    matched, independent world-outcome evidence. Contextless action also obeys the
    transportability guard: conflicting supported regimes force abstention.
    """

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
        for proposal in proposals:
            summary = world.summary(proposal.axis_id, context_id=context_id)
            if (
                summary.independent_evidence_classes >= world.min_independent_classes
                and summary.routing_score > 0.0
            ):
                supported.append((proposal, summary))

        if not supported:
            return WorldActionDecision(
                status="EXPLORE_ONLY_NO_WORLD_SUPPORTED_ACTION",
                proposal=None,
                context_id=context_id,
                independent_evidence_classes=0,
                routing_score=0.0,
                reasons=(
                    "generated experiments may be explored, but no action has enough authenticated independent world evidence",
                ),
            )

        supported_ids = {proposal.experiment_id for proposal, _ in supported}
        ranked = world.rank_proposals(proposals, context_id=context_id)
        chosen = next(
            proposal for proposal in ranked
            if proposal.experiment_id in supported_ids
        )
        summary = world.summary(chosen.axis_id, context_id=context_id)
        return WorldActionDecision(
            status="WORLD_SUPPORTED_ACTION",
            proposal=chosen,
            context_id=context_id,
            independent_evidence_classes=summary.independent_evidence_classes,
            routing_score=summary.routing_score,
            reasons=("authenticated independent world outcomes support this action preference",),
        )
