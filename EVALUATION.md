# Independent Evaluation Guide

The goal of this public surface is to make external criticism and evaluation possible without disclosing unpublished internal research.

## Preferred evaluation types

### 1. Hidden-task evaluation
The evaluator chooses or holds the task before seeing the system output. The tested system commits to an answer before the hidden reference is revealed.

### 2. Counterexample evaluation
Provide a case that should break an apparent generalization. A useful result explains whether the failure comes from missing information, representation, reasoning, measurement, or execution.

### 3. Model-disjoint evaluation
Use another AI system to independently analyze the same frozen input or output. Record the model/provider, visible input, evaluation rubric, and result.

### 4. Source-disjoint transfer
Test the same claimed capability on a substantially different source or problem family rather than a renamed copy of the original task.

### 5. Reproduction
Attempt to reproduce a public result from the information made available in this repository. Non-reproduction is useful evidence and should be reported.

## Minimal report format

```text
Task:
What was hidden before commitment:
What the tested system could see:
What the evaluator could see:
Frozen system output:
Evaluation method:
Result:
Failure or uncertainty:
Reproduction information:
```

## Evidence rules

- Separate generation from evaluation.
- Do not count self-description as verification.
- Preserve negative results.
- Do not change the rubric after seeing the result.
- State when information, tools, or compute differ between comparisons.
- Do not infer broad capability from one narrow success.

## Privacy

Do not publish private prompts, credentials, identity documents, unreleased theory, hidden answers that must remain reusable, or material obtained from private repository access.
