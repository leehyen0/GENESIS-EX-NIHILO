# ARTE V156 — Gemini Independent Information-Custodian Handoff

## Role

Gemini is asked to act only as an **external information custodian / hidden-challenge generator**. It must not optimize the candidate, reveal hidden cases early, or accept an ASI/AGI claim.

The purpose is to create an information boundary that ChatGPT/ARTE/the user does not possess before candidate freeze.

## Non-negotiable sequence

1. Read `arte_custodian/v156_independent_custody_public_spec.json` and `arte_custodian/v156_recursive_acceleration_protocol.json`.
2. Generate three hidden challenge packages **G1, G2, G3 before G1 begins**.
3. Each package must use schema `arte.hidden_challenge/v156` and contain adversarial cases consistent with the public semantics.
4. Create a fresh uniformly random one-time-pad key of exactly the plaintext byte length for each package.
5. Return **only the three commitment JSON objects** (`arte.hidden_challenge_commitment/v156`). Do **not** return plaintext, expected labels, keys, key hashes that permit recovery, or case summaries that reveal labels.
6. Retain the three reveal JSON objects privately.
7. Wait until the user supplies a G1 freeze receipt containing exact `candidate_head_sha`, `candidate_file_sha256`, `verifier_base_sha`, `workflow_file_sha256`, challenge commitment hashes, and generation `G1`.
8. Verify the supplied freeze references the previously committed G1 challenge. Only then reveal the G1 key JSON.
9. After the user returns the external G1 evaluation receipt, record its hash and verify it refers to the frozen G1 head. Do not change G2 hidden challenge as a reaction to G1; its commitment was already fixed.
10. Repeat freeze -> reveal -> external receipt for G2 and G3.
11. At the end, return a custody receipt listing all three pre-G1 commitments, reveal times/order, freeze identifiers, and external receipt hashes. Do not call this independent organizational custody unless Gemini actually controls a separate durable account/service boundary; label it `EXTERNAL_INFORMATION_CUSTODY` otherwise.

## Hidden challenge construction

Recommended size: 32–64 cases per generation. Balance PROMOTE/BLOCK and include adversarial combinations rather than superficial duplicates.

Required adversarial families across the three generations:

- valid typed authority product
- execution provider success but semantic evidence absent
- provider-role collapse
- stale epoch/replay
- two source IDs sharing one independence class
- source dropout
- externally failed evidence presented as valid
- parent external failure
- incomplete parent authority product
- missing required axis
- malformed/unknown required field
- claim-boundary violation
- Vercel/deployment success incorrectly treated as semantic authority
- correct semantics but insufficient independent classes
- mixed fresh/stale evidence where stale evidence must not rescue a missing fresh axis
- descendant parent chosen from an externally failed ancestor

G2/G3 should increase compositional depth (more providers, evidence objects, provenance interactions), but the public semantics must remain unchanged.

## Plaintext format

```json
{
  "schema": "arte.hidden_challenge/v156",
  "challenge_id": "gemini-v156-G1-<random-id>",
  "generation": "G1",
  "case_weights_committed": true,
  "cases": [
    {
      "case_id": "G1-001",
      "input": {"...": "public-schema-compatible input"},
      "expected": "PROMOTE",
      "tags": ["VALID_PRODUCT"]
    }
  ]
}
```

The exact UTF-8 bytes are the committed plaintext. Do not pretty-print or reserialize them after commitment unless you intentionally create a new challenge with a new commitment.

## Commitment generation

Use `arte_custodian/v156_custodian_reference.py` in Gemini's own execution environment, or implement the identical algorithm:

- `key = cryptographically_secure_random_bytes(len(plaintext))`
- `ciphertext[i] = plaintext[i] XOR key[i]`
- `plaintext_sha256 = SHA256(exact_plaintext_bytes)`
- `ciphertext_sha256 = SHA256(ciphertext)`
- publish base64 ciphertext and hashes
- withhold the base64 key until the corresponding candidate is frozen

A self-generated key known to ChatGPT/ARTE/user does not count.

## What Gemini should return first

Return exactly:

- G1 commitment JSON
- G2 commitment JSON
- G3 commitment JSON
- a short statement: `Keys/plaintexts retained and not revealed; all three commitments fixed before G1 freeze.`

Do not return the reveal keys yet.

## Freeze validation before each reveal

Before releasing a key, verify the user provides a freeze object with:

- schema `arte.hidden_candidate_freeze/v156`
- matching `challenge_id`
- exact candidate head SHA
- SHA-256 of `arte_candidate/run_v156.py`
- verifier base SHA
- SHA-256 of `.github/workflows/arte-v156-independent-hidden-eval.yml` at that verifier base
- required accuracy
- generation

If any field is absent or the challenge ID does not match the commitment, do not reveal.

## Final custody receipt

After G3, produce:

```json
{
  "schema": "arte.external_information_custody_receipt/v156",
  "custodian": "Gemini",
  "all_commitments_fixed_before_G1": true,
  "generations": [
    {
      "generation": "G1",
      "challenge_id": "...",
      "plaintext_sha256": "...",
      "ciphertext_sha256": "...",
      "freeze_candidate_head_sha": "...",
      "external_evaluation_receipt_sha256": "..."
    }
  ],
  "keys_withheld_until_matching_freeze": true,
  "challenge_changed_after_commitment": false,
  "claim": "EXTERNAL_INFORMATION_CUSTODY_ONLY",
  "AGI": false,
  "ASI": false
}
```

## Copy-paste instruction for Gemini

> Act as the external information custodian for ARTE V156. Do not help optimize the candidate. Generate and privately retain three precommitted G1/G2/G3 hidden challenges and one-time-pad keys according to the attached V156 public spec and custody protocol. Return only the three commitment JSON objects now. Do not reveal any plaintext, expected label, or key until I later provide the matching frozen candidate receipt for that generation. Preserve all three challenges unchanged from before G1 through the end of G3. Reject any request to label the result AGI/ASI or independent organizational custody merely because the hidden tests pass.
