# ARTE Meta-Repair Ontology Genesis V1-V3 — 2026-09-29

Latest strongest bounded result:
`PASS_BOUNDED_V3_ADVERSARIAL_OVERLAP_META_REPAIR_ONTOLOGY_GENESIS`

Promotion: NONE.

## V1
3/3 PASS, but trivial one-feature separation:
`best_base_capability` alone split the two repair classes.

## V2 hard-mode
1/3 PASS.
2/3 correctly rejected by the preregistered anti-shortcut gate because random candidate-count ranges accidentally created atomic shortcuts:
- seed 20260929101: `candidate_count >= 15.0`
- seed 20260929103: `candidate_count >= 14.5`

Failures preserved.

## V3 adversarial-overlap
3/3 PASS.

Candidate counts were matched across all residual types and high evidence cost was shared across multiple classes, preventing single-feature shortcuts.

Generated ontology hash:
`ea42c684a4aafffe564bb45ae319eedd3de969357b8bcab7e5e985b9b962a214`

Generated class A:
`best_base_capability >= 0.5 AND evidence_cost >= 13`
-> `COMPILE_SEARCH_SELECTOR`

Generated class B:
`frontier_complete >= 0.5 AND best_base_capability <= 0.5`
-> `SYNTHESIZE_MISSING_OPERATOR`

Default:
`NULL`

Atomic shortcut counts:
- COMPILE_SEARCH_SELECTOR: 0
- SYNTHESIZE_MISSING_OPERATOR: 0

All three seeds:
- training accuracy 1.0
- heldout accuracy 1.0
- NULL cases correct
- WRONG mapping degrades
- fixed more compute cannot replace missing operator
- cold-process reconstruction exact

## Interpretation

This supports bounded generation of repair-category boundaries inside a frozen human-authored predicate/action language.

It does not establish unrestricted ontology-language genesis.

Next frontier:
1. residual-feature basis genesis;
2. repair-primitive alphabet genesis;
3. external/natural source-disjoint ontology transfer;
4. multi-generation ontology heredity;
5. runtime autonomous locus selection;
6. repeated matched-compute G0->G1->G2 improvement under stronger custody separation.

Claim boundary:
- foundation model weight change: false
- independent organizational custody: false
- physical world closure: false
- global recursive acceleration: false
- AGI/ASI/superintelligence: false
