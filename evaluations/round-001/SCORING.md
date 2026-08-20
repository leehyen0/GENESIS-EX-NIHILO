# Evaluation Round 001 — Scoring

Score each candidate independently from **0 to 4**. You may consult the public issue history, linked maintainer resolution, and merged fix when judging whether a frozen candidate correctly identified the problem.

## Four one-point coordinates

1. **Causal locus** — identifies the primary layer that actually owns the failure.
2. **Discriminating question/probe** — asks for evidence that separates the plausible competing explanations rather than merely collecting more data.
3. **Minimal action family** — recommends the smallest action consistent with the eventual resolution; broad lists of unrelated interventions do not receive this point.
4. **Verifier / authority** — proposes the decisive regression test, control, trace, or ownership check needed to justify the action.

Total: **0–4 per candidate**.

Also provide:
- an A–D ranking for each case;
- a short rationale;
- uncertainty/confidence;
- any candidate that is unsafe, over-broad, or should abstain;
- whether your own baseline was better/worse/equivalent to the best candidate.

## Suggested result format

```json
{
  "evaluator": {"type": "human|ai|human+ai", "model": "optional"},
  "baseline_sha256": "...",
  "cases": [
    {
      "case": "owner/repo#issue",
      "scores": {"A": 0, "B": 0, "C": 0, "D": 0},
      "ranking": ["A", "B", "C", "D"],
      "best_candidate_rationale": "...",
      "baseline_comparison": "better|equivalent|worse|uncertain",
      "confidence": 0.0
    }
  ],
  "baseline_reveal": ["...original baseline objects..."]
}
```

The candidate-to-policy key is sealed separately and must not be used during evaluation. A single evaluator does not by itself close independent-custody or broad-capability claims.
