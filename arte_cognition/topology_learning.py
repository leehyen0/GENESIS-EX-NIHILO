from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import math


@dataclass
class EdgeExperience:
    evidence_count: int = 0
    signed_synergy: float = 0.0
    positive_count: int = 0
    negative_count: int = 0


@dataclass(frozen=True)
class MacroCognitionCandidate:
    macro_id: str
    sequence: Tuple[str, ...]
    evidence_count: int
    mean_edge_value: float
    status: str = "PROPOSAL_ONLY"


class CognitionTopologyLearner:
    """Learn bounded routing edges from pair-ablation synergy evidence.

    Topology updates require explicit synergy evidence; mere co-occurrence does not
    change edge weights. Learned weights only reorder already-selected modules and
    cannot add/remove hard-required cognition. Repeated positive chains can become
    macro proposals, never automatically authoritative macros.
    """

    def __init__(
        self,
        learning_rate: float = 0.25,
        min_evidence: int = 3,
        max_edge_shift: float = 0.15,
        macro_min_edge_value: float = 0.05,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.min_evidence = max(1, int(min_evidence))
        self.max_edge_shift = max(0.0, float(max_edge_shift))
        self.macro_min_edge_value = float(macro_min_edge_value)
        self.edges: Dict[Tuple[str, str], EdgeExperience] = {}
        self.sequence_counts: Dict[Tuple[str, ...], int] = {}

    def observe_sequence(
        self,
        sequence: Sequence[str],
        edge_synergy: Mapping[Tuple[str, str], float],
    ) -> None:
        seq = tuple(sequence)
        if len(seq) >= 2:
            self.sequence_counts[seq] = self.sequence_counts.get(seq, 0) + 1
        for a, b in zip(seq, seq[1:]):
            edge = (a, b)
            if edge not in edge_synergy:
                continue
            signal = max(-1.0, min(1.0, float(edge_synergy[edge])))
            exp = self.edges.setdefault(edge, EdgeExperience())
            exp.evidence_count += 1
            exp.signed_synergy = max(
                -1.0,
                min(1.0, (1.0 - self.learning_rate) * exp.signed_synergy + self.learning_rate * signal),
            )
            if signal > 0:
                exp.positive_count += 1
            elif signal < 0:
                exp.negative_count += 1

    def edge_shift(self, a: str, b: str) -> float:
        exp = self.edges.get((a, b))
        if exp is None or exp.evidence_count < self.min_evidence:
            return 0.0
        return self.max_edge_shift * math.tanh(2.0 * exp.signed_synergy)

    def reorder(self, modules: Iterable[str]) -> List[str]:
        modules = list(dict.fromkeys(modules))
        if len(modules) <= 1:
            return modules
        # Stable score from learned incoming/outgoing preferences. Only ordering
        # changes; membership remains fixed by the sparse cognition compiler.
        score = {module: 0.0 for module in modules}
        for a in modules:
            for b in modules:
                if a == b:
                    continue
                shift = self.edge_shift(a, b)
                score[a] -= shift / 2.0
                score[b] += shift / 2.0
        original_index = {module: i for i, module in enumerate(modules)}
        return sorted(modules, key=lambda m: (score[m], original_index[m]))

    def propose_macros(self, budget: int = 8) -> List[MacroCognitionCandidate]:
        out: List[MacroCognitionCandidate] = []
        for sequence, count in self.sequence_counts.items():
            if count < self.min_evidence or len(sequence) < 2:
                continue
            values = [self.edge_shift(a, b) for a, b in zip(sequence, sequence[1:])]
            if not values or min(values) <= self.macro_min_edge_value:
                continue
            out.append(MacroCognitionCandidate(
                macro_id="MACRO::" + "->".join(sequence),
                sequence=sequence,
                evidence_count=count,
                mean_edge_value=sum(values) / len(values),
            ))
        return sorted(out, key=lambda m: (-m.mean_edge_value, -m.evidence_count, m.macro_id))[:max(1, int(budget))]
