# Evaluation Round 001 — Tasks

This is a public, black-box evaluation round for an experimental diagnostic research system. Internal architecture, theory names, training lineage, and policy identities are intentionally not disclosed.

## Two-stage procedure

**Stage 1 — baseline first.** Before opening `CANDIDATES.md`, independently diagnose the four public issues below. Save one JSON object with your baseline diagnosis and recommended verification/action for each case, then post only its SHA-256 hash to the round issue.

Suggested baseline fields per case:

```json
{
  "case": "owner/repo#issue",
  "diagnosis": "...",
  "question_or_probe": "...",
  "verification": "...",
  "recommended_action": "...",
  "confidence": 0.0
}
```

After posting the baseline hash, you may open `CANDIDATES.md`. At the end, reveal the baseline JSON (or provide a public locator) so its hash can be checked.

**Stage 2 — candidate scoring.** Score the anonymized candidates A–D for each case using `SCORING.md`. Do not try to infer which internal policy produced a candidate; policy identities are intentionally withheld.

## Tasks

### python/cpython#154874

Public issue: https://github.com/python/cpython/issues/154874

Issue-only observation used when the candidate outputs were frozen:

> termattrs becomes negative when capability bit 31 is set; sibling attribute APIs return unsigned masks; regression followed a shared macro rewrite that introduced int/ERR handling.

### pandas-dev/pandas#66639

Public issue: https://github.com/pandas-dev/pandas/issues/66639

Issue-only observation used when the candidate outputs were frozen:

> sep=None historically selected the Python parser; a cleanup removed the fallback branch so None reaches unsupported engines and produces TypeError/silent misparse.

### curl/curl#22272

Public issue: https://github.com/curl/curl/issues/22272

Issue-only observation used when the candidate outputs were frozen:

> curl_multi_wakeup followed by curl_multi_perform then curl_multi_poll can block forever after a change that consumes wakeups during perform; pre-change the wakeup remained visible to poll.

### scipy/scipy#23177

Public issue: https://github.com/scipy/scipy/issues/23177

Issue-only observation used when the candidate outputs were frozen:

> A SciPy stats test passes with JAX 0.6.0/0.6.1 but fails after JAX 0.6.2; CPU CI pulls PyPI 0.6.2 while GPU/conda remains older, creating environment-dependent behavior.

## Evaluator metadata

Record whether you are a human, an AI system, or a human+AI workflow. For AI evaluation, record provider/model/version when known. Do not disclose private prompts, credentials, hidden reusable answers, or sensitive identity information.

This round measures bounded diagnostic quality only. It does not establish broad intelligence, AGI, ASI, or human superiority.
