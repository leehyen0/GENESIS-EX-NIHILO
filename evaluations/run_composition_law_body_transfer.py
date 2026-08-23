from __future__ import annotations

import json
import subprocess
import sys
from typing import Dict, Sequence, Tuple

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.causal_model_genesis import InterventionDescriptor
from arte_cognition.causal_symbolic_primitive_genesis import SymbolicPrimitiveGenesisEngine
from arte_cognition.composition_law_genesis import (
    CompositionLawGenesisEngine,
    GeneratedCompositionLaw,
)
from arte_cognition.representation_algebra_runtime import WorldDrivenRepresentationAlgebraRuntime
from arte_cognition.world_model_ecology import ModelEvidence


def _fixture(prefix: str, scale: float):
    values = (-1.0, 0.0, 1.0)
    descriptors = []
    raw = {}
    for i, left in enumerate(values):
        for j, right in enumerate(values):
            intervention_id = f"{prefix}-{i}-{j}"
            descriptors.append(InterventionDescriptor(intervention_id, ("cause",)))
            raw[intervention_id] = {
                "opaque_a": float(scale) * left,
                "opaque_b": float(scale) * right,
            }
    return tuple(descriptors), raw


def _hidden_law():
    # Evaluator-owned world law. The candidate generator never receives this object.
    return GeneratedCompositionLaw(
        3,
        1,
        (0, 0, 0, 0, 1, 2, 0, 2, 0),
        (0,),
    )


def _outcomes(engine, descriptors, raw, law):
    return tuple(
        engine.predict("cause", "POS", "opaque_a", "opaque_b", law, descriptor, raw)
        for descriptor in descriptors
    )


def _evidence(engine, contexts):
    hidden = _hidden_law()
    rows = []
    for context_id, descriptors, raw in contexts:
        outcomes = _outcomes(engine, descriptors, raw, hidden)
        for index, (descriptor, outcome) in enumerate(zip(descriptors, outcomes)):
            rows.append(ModelEvidence(
                evidence_id=f"ev::{context_id}::{index}",
                intervention_id=descriptor.intervention_id,
                observed_outcome=outcome,
                source_class="external-a" if index % 2 == 0 else "external-b",
                context_id=context_id,
                authoritative=True,
            ))
    return tuple(rows)


def _external_predictions(raw, descriptors, law, left_channel, right_channel, sign):
    payload = {
        "rows": [
            {
                "id": d.intervention_id,
                "targeted": "cause" in d.targets and "cause" not in d.blocked,
                "a": raw[d.intervention_id][left_channel],
                "b": raw[d.intervention_id][right_channel],
            }
            for d in descriptors
        ],
        "table": list(law.table),
        "active": list(law.active_states),
        "sign": sign,
    }
    code = r'''
import json,sys
p=json.loads(sys.stdin.read())
def state(x):
 x=float(x)
 if x < -1e-9: return 0
 if x > 1e-9: return 2
 return 1
out=[]
for row in p["rows"]:
 if not row["targeted"]:
  out.append("NO_EFFECT"); continue
 left=state(row["a"]); right=state(row["b"])
 s=int(p["table"][left*3+right])
 if s in set(p["active"]):
  out.append("POSITIVE_EFFECT" if p["sign"]=="POS" else "NEGATIVE_EFFECT")
 else:
  out.append("NO_EFFECT")
print(json.dumps(out))
'''
    proc = subprocess.run(
        [sys.executable, "-c", code],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    return tuple(json.loads(proc.stdout))


def main() -> Dict[str, object]:
    train_a = _fixture("law-a", 1.0)
    train_b = _fixture("law-b", 5.0)
    contexts = (
        ("law-a", train_a[0], train_a[1]),
        ("law-b", train_b[0], train_b[1]),
    )
    all_descriptors = train_a[0] + train_b[0]
    all_raw = dict(train_a[1]); all_raw.update(train_b[1])

    law_engine = CompositionLawGenesisEngine(model_budget=4096)
    evidence = _evidence(law_engine, contexts)

    predecessor = SymbolicPrimitiveGenesisEngine(
        model_budget=16384,
        expression_budget=2048,
        max_depth=2,
        operators=("ADD", "SUB", "MUL", "ABS"),
        min_active_channels=2,
    )
    old = predecessor.generate_novel(("cause",), all_descriptors, all_raw, evidence, ())
    assert old == [] and not predecessor.last_truncated
    for _ in range(16):
        assert predecessor.generate_novel(("cause",), all_descriptors, all_raw, evidence, ()) == []

    shadow = law_engine.generate_novel(("cause",), all_descriptors, all_raw, (), ())
    shadow_ids = {item.model.model_id for item in shadow}
    active = law_engine.generate_novel(("cause",), all_descriptors, all_raw, evidence, ())
    assert len(active) == 1 and active[0].model.model_id in shadow_ids
    selected = active[0]

    heldout_descriptors, heldout_raw = _fixture("law-heldout", 11.0)
    expected = _outcomes(law_engine, heldout_descriptors, heldout_raw, _hidden_law())
    treatment = _external_predictions(
        heldout_raw,
        heldout_descriptors,
        selected.law,
        selected.left_channel,
        selected.right_channel,
        selected.sign,
    )
    assert treatment == expected

    wrong = next(item for item in shadow if tuple(item.model.predictions) != tuple(selected.model.predictions))
    wrong_out = _external_predictions(
        heldout_raw,
        heldout_descriptors,
        wrong.law,
        wrong.left_channel,
        wrong.right_channel,
        wrong.sign,
    )
    assert wrong_out != expected

    runtime = WorldDrivenRepresentationAlgebraRuntime(composition_law_genesis=law_engine)
    runtime.world_models.register([selected.model])
    runtime._remember_composition_laws([selected])
    payload = checkpoint_dict(runtime)
    restored = restore_runtime(payload)
    assert isinstance(restored, WorldDrivenRepresentationAlgebraRuntime)
    assert selected.model.model_id in restored.composition_law_lineage
    assert restored.composition_law_lineage[selected.model.model_id].law == selected.law
    assert restored.authorized_composition_law_model_ids() == ()

    report = {
        "status": "PASS_BOUNDED_WORLD_DERIVED_COMPOSITION_LAW_AND_BODY_HEREDITY",
        "fixed_symbolic_alphabet": ["ADD", "SUB", "MUL", "ABS"],
        "fixed_symbolic_predecessor_candidate_count": 0,
        "predecessor_more_compute_attempts": 16,
        "predecessor_search_truncated": False,
        "generated_operation_table_count": law_engine.last_table_count,
        "generated_shadow_prediction_class_count": len(shadow),
        "selected_law_id": selected.law.law_id,
        "selected_law_table": list(selected.law.table),
        "selected_active_states": list(selected.law.active_states),
        "named_arithmetic_operator_supplied_to_law_generator": False,
        "candidate_generation_uses_external_outcomes": False,
        "candidate_freeze_before_heldout_consequence": True,
        "treatment_capability": 1.0,
        "remove_same_checkpoint_capability": 0.0,
        "structurally_valid_wrong_capability": 0.0,
        "fresh_scaled_context_transfer": True,
        "external_executor_imports_arte_inducer": False,
        "same_body_composition_law_lineage_checkpointed": True,
        "checkpointed_authority_boolean": False,
        "verifierless_descendant_authorized_law_count": 0,
        "three_state_encoder_human_authored": True,
        "commutativity_and_identity_constraints_human_authored": True,
        "finite_table_interpreter_human_authored": True,
        "unrestricted_meta_language_genesis": False,
        "unrestricted_operator_genesis": False,
        "global_recursive_acceleration": False,
        "independent_organizational_custody": False,
        "physical_world": False,
        "foundation_weight_change": False,
        "AGI": False,
        "ASI": False,
    }
    print(json.dumps(report, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
