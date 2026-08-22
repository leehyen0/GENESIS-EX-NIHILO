from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import hashlib
import itertools

from .causal_model_genesis import InterventionDescriptor


@dataclass(frozen=True)
class InterventionSurfaceSpec:
    max_targets: int = 2
    max_blocked: int = 1
    delay_steps: Tuple[int, ...] = (0, 1)
    context_modes: Tuple[bool, ...] = (False, True)
    include_context_only: bool = True


class InterventionSurfaceGenesisEngine:
    """Generate intervention semantics instead of receiving a hand-authored probe list.

    The capability schema remains bounded and human-authored, but concrete target,
    blocking, delay and context combinations are synthesized from observable
    variables. Generated ids are semantic hashes, so evaluator naming cannot steer
    selection. Cost grows with intervention complexity rather than being a hidden
    answer key.
    """

    def __init__(self, spec: InterventionSurfaceSpec | None = None, budget: int = 256) -> None:
        self.spec = spec or InterventionSurfaceSpec()
        self.budget = max(1, int(budget))
        self.last_truncated = False
        self.last_full_count = 0

    @staticmethod
    def _id(
        targets: Sequence[str],
        blocked: Sequence[str],
        delay_steps: int,
        context_shift: bool,
    ) -> str:
        raw = (
            "targets=" + ",".join(targets)
            + "|blocked=" + ",".join(blocked)
            + f"|delay={int(delay_steps)}|context={int(bool(context_shift))}"
        ).encode()
        return "GENINT::" + hashlib.sha256(raw).hexdigest()[:16]

    @staticmethod
    def _cost(targets: Sequence[str], blocked: Sequence[str], delay_steps: int, context_shift: bool) -> float:
        return float(
            1.0
            + 1.5 * max(0, len(targets) - 1)
            + 3.0 * len(blocked)
            + 3.0 * max(0, int(delay_steps))
            + (5.0 if context_shift else 0.0)
        )

    def generate(self, variables: Sequence[str]) -> List[InterventionDescriptor]:
        variables = tuple(sorted({str(v) for v in variables if str(v)}))
        rows = {}
        max_targets = min(max(1, int(self.spec.max_targets)), len(variables))
        max_blocked = min(max(0, int(self.spec.max_blocked)), len(variables))

        target_sets: List[Tuple[str, ...]] = []
        for width in range(1, max_targets + 1):
            target_sets.extend(itertools.combinations(variables, width))
        if self.spec.include_context_only:
            target_sets.append(())

        blocked_sets: List[Tuple[str, ...]] = [()]
        for width in range(1, max_blocked + 1):
            blocked_sets.extend(itertools.combinations(variables, width))

        for targets in target_sets:
            for blocked in blocked_sets:
                # A completely empty non-context intervention has no semantics.
                for delay_steps in self.spec.delay_steps:
                    for context_shift in self.spec.context_modes:
                        if not targets and not context_shift:
                            continue
                        if not targets and int(delay_steps) > 0:
                            continue
                        intervention_id = self._id(targets, blocked, int(delay_steps), bool(context_shift))
                        rows[intervention_id] = InterventionDescriptor(
                            intervention_id=intervention_id,
                            targets=tuple(targets),
                            blocked=tuple(blocked),
                            delay_steps=int(delay_steps),
                            context_shift=bool(context_shift),
                            cost=self._cost(targets, blocked, int(delay_steps), bool(context_shift)),
                        )

        ordered = sorted(
            rows.values(),
            key=lambda d: (
                d.cost,
                len(d.targets),
                len(d.blocked),
                d.delay_steps,
                d.context_shift,
                d.intervention_id,
            ),
        )
        self.last_full_count = len(ordered)
        self.last_truncated = len(ordered) > self.budget
        return ordered[: self.budget]

    def novel(
        self,
        variables: Sequence[str],
        observed_intervention_ids: Iterable[str] = (),
    ) -> List[InterventionDescriptor]:
        observed = {str(value) for value in observed_intervention_ids}
        return [row for row in self.generate(variables) if row.intervention_id not in observed]
