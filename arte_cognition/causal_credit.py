from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from .adaptive_cognition import ModuleCredit


@dataclass(frozen=True)
class OutcomeAblationCredit:
    module: str
    full_outcome: float
    ablated_outcome: float
    marginal_contribution: float
    causal_credit: float
    causal_harm: float
    matched_compute: bool


@dataclass(frozen=True)
class PairSynergyCredit:
    module_i: str
    module_j: str
    full_outcome: float
    without_i: float
    without_j: float
    without_both: float
    synergy: float


class OutcomeAblationCreditEngine:
    """Assign module credit from realized outcome ablations, not decision proxies.

    For module i, marginal contribution is Y(full) - Y(full without i).
    Positive contribution becomes credit; negative contribution becomes harm.
    Missing ablations receive no credit. Callers may require matched-compute
    controls so more computation is not silently credited as cognition quality.
    """

    def assign(
        self,
        full_outcome: float,
        ablation_outcomes: Mapping[str, float],
        active_modules: Iterable[str],
        matched_compute: Optional[Mapping[str, bool]] = None,
    ) -> List[OutcomeAblationCredit]:
        matched_compute = dict(matched_compute or {})
        result: List[OutcomeAblationCredit] = []
        seen = set()
        for module in active_modules:
            if module in seen or module not in ablation_outcomes:
                continue
            seen.add(module)
            ablated = float(ablation_outcomes[module])
            delta = float(full_outcome) - ablated
            matched = bool(matched_compute.get(module, True))
            # Unmatched-compute comparisons remain evidence records but cannot
            # update policy credit.
            credit = max(0.0, delta) if matched else 0.0
            harm = max(0.0, -delta) if matched else 0.0
            result.append(OutcomeAblationCredit(
                module=module,
                full_outcome=float(full_outcome),
                ablated_outcome=ablated,
                marginal_contribution=delta,
                causal_credit=credit,
                causal_harm=harm,
                matched_compute=matched,
            ))
        return result

    def pair_synergy(
        self,
        full_outcome: float,
        without_i: float,
        without_j: float,
        without_both: float,
        module_i: str,
        module_j: str,
    ) -> PairSynergyCredit:
        # Interaction term relative to removing either module from the full system.
        synergy = (
            float(full_outcome)
            - float(without_i)
            - float(without_j)
            + float(without_both)
        )
        return PairSynergyCredit(
            module_i=module_i,
            module_j=module_j,
            full_outcome=float(full_outcome),
            without_i=float(without_i),
            without_j=float(without_j),
            without_both=float(without_both),
            synergy=synergy,
        )

    @staticmethod
    def to_router_credits(
        credits: Iterable[OutcomeAblationCredit],
        epsilon: float = 1e-12,
    ) -> List[ModuleCredit]:
        """Adapt outcome-ablation evidence to the bounded meta-router learner."""
        result: List[ModuleCredit] = []
        for item in credits:
            delta = item.marginal_contribution if item.matched_compute else 0.0
            result.append(ModuleCredit(
                module=item.module,
                used=True,
                decision_changed=bool(item.matched_compute and abs(delta) > epsilon),
                outcome_delta=delta,
                causal_credit=max(0.0, delta) if item.matched_compute else 0.0,
            ))
        return result
