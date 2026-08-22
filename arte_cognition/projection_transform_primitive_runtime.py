from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Optional, Tuple

from .experiment_genesis import ExperimentGenesisEngine, InterventionProposal
from .projection_transform_primitive_genesis import (
    TRANSFORM_PRIMITIVE_MARKER,
    ProjectionPrimitiveFrontier,
    ProjectionPrimitivePolicy,
    derive_projection_primitive_frontier,
    derive_projection_primitive_policy,
    generate_projection_power_primitive_programs,
)
from .representation_genesis import RepresentationAxis


class TransformPrimitiveAlphabetOrgan:
    """Stateless organ that derives alphabet expansion only from canonical BODY state.

    No mutable policy or authority is stored here. The organ consumes the BODY's
    persisted proposals, reverified world pairs, and transform-depth falsification
    assessment on every call. Checkpoint/restore therefore inherits evidence but not
    primitive authority unless the external verifier reauthorizes the world pairs.
    """

    def __init__(self, body) -> None:
        self.body = body

    def _proposals(self):
        return tuple(record.proposal for record in self.body.memory.experiments.values())

    def programs(self):
        return generate_projection_power_primitive_programs()

    def policy(self) -> ProjectionPrimitivePolicy:
        return derive_projection_primitive_policy(
            proposals=self._proposals(),
            world_pairs=self.body.world_coupling.pairs,
            min_independent_classes=self.body.world_coupling.min_independent_classes,
            programs=self.programs(),
            strong_effect_threshold=0.9,
            min_contexts=2,
        )

    def frontier(
        self,
        context_id: str,
        left: float,
        right: float,
        alphabet_failure_brackets: Mapping[str, Tuple[float, float]],
        current_depth: int = 3,
        max_candidates: int = 64,
        apply_learned_primitive: bool = True,
    ) -> ProjectionPrimitiveFrontier:
        assessment = self.body.projection_transform_depth_assessment(
            context_brackets=alphabet_failure_brackets,
            current_depth=current_depth,
            next_depth=current_depth + 1,
        )
        policy: Optional[ProjectionPrimitivePolicy] = (
            self.policy() if apply_learned_primitive else None
        )
        return derive_projection_primitive_frontier(
            depth_assessment=assessment,
            left=left,
            right=right,
            policy=policy,
            programs=self.programs(),
            max_candidates=max_candidates,
        )

    def generate_interventions(
        self,
        axis: RepresentationAxis,
        reference_values: Mapping[str, float],
        context_id: str,
        left: float,
        right: float,
        alphabet_failure_brackets: Mapping[str, Tuple[float, float]],
        current_depth: int = 3,
        max_candidates: int = 64,
        apply_learned_primitive: bool = True,
    ) -> list[InterventionProposal]:
        if axis.family != "PROJECTION":
            return []
        frontier = self.frontier(
            context_id=context_id,
            left=left,
            right=right,
            alphabet_failure_brackets=alphabet_failure_brackets,
            current_depth=current_depth,
            max_candidates=max_candidates,
            apply_learned_primitive=apply_learned_primitive,
        )
        generated: list[InterventionProposal] = []
        for candidate in frontier.candidates:
            engine = ExperimentGenesisEngine(
                relative_margin=self.body.experiment.relative_margin,
                max_proposals=max(self.body.experiment.max_proposals, len(axis.coefficients)),
                projection_margin_multipliers=(candidate.scale,),
            )
            for proposal in engine.propose(axis, reference_values):
                reason = (
                    f"{proposal.reason} {TRANSFORM_PRIMITIVE_MARKER}"
                    f"{'|'.join(candidate.program_ids)}"
                )
                bound = replace(proposal, reason=reason)
                self.body.memory.remember_experiment(bound)
                generated.append(bound)
        return generated
