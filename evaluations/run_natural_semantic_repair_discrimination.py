from __future__ import annotations

import dataclasses
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.primitive_genesis_runtime import WorldDrivenPrimitiveRuntime
from arte_cognition.software_failure_extractor_program_genesis import (
    ExtractorPatchCandidate,
    generate_failure_extractor_programs,
    interpret_with_extractor_program,
)
from arte_cognition.software_repair_semantic_discrimination import (
    RepairSemanticDiscriminatorOrgan,
    generate_repair_semantic_discriminators,
    propose_repair_semantic_discriminator,
    required_binding_preserved,
    select_authorized_repair_semantic_discriminator,
    select_patch_candidate,
)
from evaluations.run_natural_repair_constructor_genesis import (
    FreshRepositoryEnvironment,
    _authority,
    _execute_candidate,
    _git_blob_sha,
)


G5_PATH = "arte_cognition/test_causal_primitive_genesis.py"
G5_FIXTURE = ROOT / "evaluations/fixtures/historical_194dda71/test_causal_primitive_genesis.py"
G5_BLOB = "c40f511b8b27c94028fa553c3821db4597af4ce1"
G5_SELECTOR = (
    "arte_cognition.test_causal_primitive_genesis."
    "RawThresholdPrimitiveGenesisTests."
    "test_runtime_cannot_open_g5_before_g4_is_falsified"
)
G5_METHOD = "generate_world_driven_primitive_models"

G6_PATH = "arte_cognition/test_causal_linear_primitive_genesis.py"
G6_FIXTURE = ROOT / "evaluations/fixtures/historical_0610bbaa/test_causal_linear_primitive_genesis.py"
G6_BLOB = "d5a2ff9319eb7fbf03586ac517b07400d993d747"
G6_SELECTOR = (
    "arte_cognition.test_causal_linear_primitive_genesis."
    "LinearFormPrimitiveGenesisTests."
    "test_runtime_cannot_open_g6_before_g5_is_falsified"
)
G6_METHOD = "generate_world_driven_linear_primitive_models"

G7_PATH = "arte_cognition/test_causal_symbolic_primitive_genesis.py"
G7_FIXTURE = ROOT / "evaluations/fixtures/historical_c540a487/test_causal_symbolic_primitive_genesis.py"
G7_BLOB = "2722ce1e5e5d2705481c76324facae75c8f655f5"
G7_SELECTOR = (
    "arte_cognition.test_causal_symbolic_primitive_genesis."
    "SymbolicPrimitiveGenesisTests."
    "test_runtime_cannot_open_symbolic_search_before_g6_falsification"
)
G7_METHOD = "generate_world_driven_symbolic_primitive_models"

_CAPTURE_PREFIX = "ARTE_CALL_CAPTURE="


def _norm(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if dataclasses.is_dataclass(value):
        return {"__type__": type(value).__name__, "value": _norm(dataclasses.asdict(value))}
    if isinstance(value, Mapping):
        return {str(key): _norm(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_norm(item) for item in value]
    return {"__type__": type(value).__name__, "repr": repr(value)}


def _fingerprint(value) -> str:
    return json.dumps(_norm(value), sort_keys=True, separators=(",", ":"))


class SemanticRepositoryEnvironment(FreshRepositoryEnvironment):
    def __init__(self, source: str, path: str, selector: str, method_name: str) -> None:
        super().__init__(source, path, "UNITTEST", selector=selector)
        self.method_name = str(method_name)
        signature = inspect.signature(getattr(WorldDrivenPrimitiveRuntime, self.method_name))
        parameters = [
            parameter
            for name, parameter in signature.parameters.items()
            if name != "self" and parameter.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        self.parameter_names = tuple(parameter.name for parameter in parameters)
        self.required_parameter_names = tuple(
            parameter.name for parameter in parameters
            if parameter.default is inspect.Parameter.empty
        )
        self.parameter_optional_by_position = tuple(
            parameter.default is not inspect.Parameter.empty for parameter in parameters
        )
        self._baseline_capture: Optional[Dict[str, str]] = None

    @staticmethod
    def _selector_parts(selector: str) -> Tuple[str, str, str]:
        module_name, class_name, method_name = selector.rsplit(".", 2)
        return module_name, class_name, method_name

    def _capture(self, source: str, timeout: float = 25.0) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]]:
        module_name, class_name, test_method = self._selector_parts(str(self.selector))
        script = f'''\
import dataclasses, importlib, json
from unittest.mock import patch
from arte_cognition.primitive_genesis_runtime import WorldDrivenPrimitiveRuntime

def norm(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if dataclasses.is_dataclass(value):
        return {{"__type__": type(value).__name__, "value": norm(dataclasses.asdict(value))}}
    if isinstance(value, dict):
        return {{str(key): norm(value[key]) for key in sorted(value, key=lambda item: str(item))}}
    if isinstance(value, (list, tuple)):
        return [norm(item) for item in value]
    return {{"__type__": type(value).__name__, "repr": repr(value)}}

module = importlib.import_module({module_name!r})
case = getattr(module, {class_name!r})({test_method!r})
captures = []
def recorder(self, *args, **kwargs):
    captures.append({{"args": [norm(item) for item in args], "kwargs": {{str(k): norm(v) for k, v in sorted(kwargs.items())}}}})
    return []
case.setUp()
try:
    with patch.object(WorldDrivenPrimitiveRuntime, {self.method_name!r}, recorder):
        getattr(case, {test_method!r})()
finally:
    tear = getattr(case, "tearDown", None)
    if callable(tear):
        tear()
print({_CAPTURE_PREFIX!r} + json.dumps(captures, sort_keys=True, separators=(",", ":")))
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "arte_cognition", root / "arte_cognition")
            target = root / self.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(source), encoding="utf-8")
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["PYTHONNOUSERSITE"] = "1"
            completed = subprocess.run(
                [sys.executable, "-c", script], cwd=root, env=env,
                capture_output=True, text=True, timeout=max(2.0, float(timeout)), check=False,
            )
        if completed.returncode != 0:
            raise AssertionError(f"call-capture subprocess failed: {completed.stderr[-1200:]}")
        line = next((item for item in completed.stdout.splitlines() if item.startswith(_CAPTURE_PREFIX)), None)
        if line is None:
            raise AssertionError("call-capture subprocess emitted no capture payload")
        payload = json.loads(line[len(_CAPTURE_PREFIX):])
        if len(payload) != 1:
            raise AssertionError(f"expected exactly one target call capture, got {len(payload)}")
        args = tuple(json.dumps(item, sort_keys=True, separators=(",", ":")) for item in payload[0]["args"])
        kwargs = tuple(
            sorted((str(key), json.dumps(value, sort_keys=True, separators=(",", ":"))) for key, value in payload[0]["kwargs"].items())
        )
        return args, kwargs

    def binding_map(self, source: str) -> Dict[str, str]:
        args, kwargs = self._capture(source)
        mapping: Dict[str, str] = {}
        for index, value in enumerate(args):
            if index < len(self.parameter_names):
                mapping[self.parameter_names[index]] = value
        for name, value in kwargs:
            mapping[name] = value
        return mapping

    def baseline_binding_map(self) -> Dict[str, str]:
        if self._baseline_capture is None:
            self._baseline_capture = self.binding_map(self.source)
        return dict(self._baseline_capture)

    def semantic_run(self, source: str, timeout: float = 25.0) -> Tuple[float, str, str]:
        executable, stdout, stderr = super().run(source, timeout=timeout)
        if executable != 1.0:
            return 0.0, stdout, stderr
        try:
            candidate_bindings = self.binding_map(source)
        except Exception as exc:
            return 0.0, stdout, f"semantic capture failed: {exc!r}"
        semantic = required_binding_preserved(
            self.baseline_binding_map(),
            candidate_bindings,
            self.required_parameter_names,
        )
        return (1.0 if semantic else 0.0), stdout, stderr

    def run(self, source: Optional[str] = None, timeout: float = 25.0):
        candidate = self.source if source is None else str(source)
        return self.semantic_run(candidate, timeout=timeout)


def _extractor_interpretation(environment: SemanticRepositoryEnvironment):
    # Baseline capability is intentionally zero under the real production method.
    executable_environment = FreshRepositoryEnvironment(
        environment.source, environment.path, "UNITTEST", selector=environment.selector
    )
    baseline, _, stderr = executable_environment.run()
    if baseline != 0.0 or "ValueError" not in stderr:
        raise AssertionError(f"natural pre-fix failure did not reproduce: {stderr[-1000:]}")
    interpretations = []
    for program in generate_failure_extractor_programs():
        item = interpret_with_extractor_program(program, stderr, environment.source, environment.path)
        if item is not None and item.patch_candidates:
            interpretations.append(item)
    if len(interpretations) != 1:
        raise AssertionError(f"expected one inherited #82 extractor program family, got {len(interpretations)}")
    return interpretations[0]


def _train_context(body, organ, proposals, environment, context, signers, verifier, epoch_base):
    interpretation = _extractor_interpretation(environment)
    if len(interpretation.patch_candidates) != 3:
        raise AssertionError("natural ambiguity benchmark no longer produces exactly three extractor edits")
    baseline_bindings = environment.baseline_binding_map()
    rows = []
    for proposal in proposals:
        chosen = select_patch_candidate(
            proposal.discriminator,
            interpretation.patch_candidates,
            environment.parameter_optional_by_position,
        )
        if chosen is None:
            rows.append((proposal.discriminator, None, (0.0, 0.0)))
            continue
        effects = _execute_candidate(
            body, proposal, environment, chosen.patched_source,
            context, signers, verifier, epoch_base,
        )
        rows.append((proposal.discriminator, chosen, effects))
    strong = [row for row in rows if row[1] is not None and min(row[2]) >= 0.9]
    if len(strong) != 1:
        raise AssertionError(f"semantic software world must isolate one discriminator; got {len(strong)}")
    chosen = strong[0][1]
    chosen_map = environment.binding_map(chosen.patched_source)
    if not required_binding_preserved(
        baseline_bindings, chosen_map, environment.required_parameter_names
    ):
        raise AssertionError("world-supported semantic discriminator did not preserve required bindings")
    return interpretation, rows, strong[0][0]


def main() -> None:
    sources = [
        G5_FIXTURE.read_text(encoding="utf-8"),
        G6_FIXTURE.read_text(encoding="utf-8"),
        G7_FIXTURE.read_text(encoding="utf-8"),
    ]
    if tuple(_git_blob_sha(source) for source in sources) != (G5_BLOB, G6_BLOB, G7_BLOB):
        raise AssertionError("natural semantic-discrimination fixtures diverged from exact historical blobs")

    discriminators = generate_repair_semantic_discriminators()
    proposals = tuple(propose_repair_semantic_discriminator(item) for item in discriminators)
    if len(discriminators) != 2:
        raise AssertionError("semantic discriminator shadow language changed unexpectedly")
    signers, verifier = _authority()
    parent = PersistentCognitiveRuntime()
    organ = RepairSemanticDiscriminatorOrgan(parent)
    organ.remember(proposals)

    g5_env = SemanticRepositoryEnvironment(sources[0], G5_PATH, G5_SELECTOR, G5_METHOD)
    g5_interpretation, g5_rows, g5_winner = _train_context(
        parent, organ, proposals, g5_env, "semantic-g5", signers, verifier, 51000
    )
    if organ.policy().status == "REPRODUCED_REPAIR_SEMANTIC_DISCRIMINATOR":
        raise AssertionError("one semantic context incorrectly authorized a discriminator")

    g6_env = SemanticRepositoryEnvironment(sources[1], G6_PATH, G6_SELECTOR, G6_METHOD)
    g6_interpretation, g6_rows, g6_winner = _train_context(
        parent, organ, proposals, g6_env, "semantic-g6", signers, verifier, 52000
    )
    if g5_winner.discriminator_id != g6_winner.discriminator_id:
        raise AssertionError("natural G5/G6 semantic worlds did not reproduce one discriminator")
    parent_policy = organ.policy()
    if (
        parent_policy.status != "REPRODUCED_REPAIR_SEMANTIC_DISCRIMINATOR"
        or parent_policy.discriminator_id != g5_winner.discriminator_id
        or set(parent_policy.supporting_contexts) != {"semantic-g5", "semantic-g6"}
    ):
        raise AssertionError(f"semantic discriminator failed BODY authority gate: {parent_policy}")

    checkpoint = checkpoint_dict(parent)
    verifierless = restore_runtime(checkpoint)
    verifierless_policy = RepairSemanticDiscriminatorOrgan(verifierless).policy()
    if verifierless_policy.status == "REPRODUCED_REPAIR_SEMANTIC_DISCRIMINATOR":
        raise AssertionError("semantic discriminator authority leaked through checkpoint")
    reverified = restore_runtime(checkpoint, world_verifier=verifier)
    reverified_policy = RepairSemanticDiscriminatorOrgan(reverified).policy()
    if reverified_policy != parent_policy:
        raise AssertionError("external reverification did not reconstruct semantic discriminator policy")

    g7_env = SemanticRepositoryEnvironment(sources[2], G7_PATH, G7_SELECTOR, G7_METHOD)
    g7_interpretation = _extractor_interpretation(g7_env)
    candidates = g7_interpretation.patch_candidates
    if len(candidates) != 3:
        raise AssertionError("heldout G7 no longer presents the three-edit semantic ambiguity")
    full_outcomes = tuple(g7_env.run(candidate.patched_source)[0] for candidate in candidates)
    if sum(1 for value in full_outcomes if value == 1.0) != 1:
        raise AssertionError(f"independent semantic world did not uniquely identify a repair: {full_outcomes}")

    if select_authorized_repair_semantic_discriminator(discriminators, verifierless_policy) is not None:
        raise AssertionError("verifierless descendant selected a semantic discriminator")
    learned = select_authorized_repair_semantic_discriminator(discriminators, reverified_policy)
    if learned is None:
        raise AssertionError("reverified descendant failed to recover semantic discriminator")
    treatment_candidate = select_patch_candidate(
        learned, candidates, g7_env.parameter_optional_by_position
    )
    if treatment_candidate is None:
        raise AssertionError("learned discriminator selected no heldout repair")
    treatment_capability = g7_env.run(treatment_candidate.patched_source)[0]

    # REMOVE: same inherited extractor program and same one-candidate execution budget,
    # but no semantic discriminator. Deterministic fallback is the first extractor edit.
    remove_candidate = sorted(candidates, key=lambda item: item.edit_id)[0]
    remove_capability = g7_env.run(remove_candidate.patched_source)[0]
    wrong = next(item for item in discriminators if item.discriminator_id != learned.discriminator_id)
    wrong_candidate = select_patch_candidate(wrong, candidates, g7_env.parameter_optional_by_position)
    wrong_capability = 0.0 if wrong_candidate is None else g7_env.run(wrong_candidate.patched_source)[0]

    if treatment_capability != 1.0 or remove_capability != 0.0 or wrong_capability != 0.0:
        raise AssertionError("heldout Treatment/REMOVE/WRONG semantic causal isolation failed")

    selected_index = next(
        index for index, candidate in enumerate(candidates)
        if candidate.edit_id == treatment_candidate.edit_id
    )
    full_success_index = next(index for index, value in enumerate(full_outcomes) if value == 1.0)
    if selected_index != full_success_index:
        raise AssertionError("learned semantic discriminator missed the unique heldout semantic repair")

    result = {
        "status": "PASS_BOUNDED_WORLD_AUTHORIZED_SEMANTIC_REPAIR_DISCRIMINATION_AND_NATURAL_G7_TRANSFER",
        "repository": "leehyen0/GENESIS-EX-NIHILO",
        "parent_main": "e944ca0578f7cde989d454a38b457bddd5029f4f",
        "natural_training_files": [G5_PATH, G6_PATH],
        "natural_heldout_file": G7_PATH,
        "historical_fixtures_exact_git_blobs": True,
        "inherited_extractor_program_family": g7_interpretation.program_id,
        "inherited_extractor_candidate_count_per_context": [
            len(g5_interpretation.patch_candidates),
            len(g6_interpretation.patch_candidates),
            len(g7_interpretation.patch_candidates),
        ],
        "narrow_executable_patch_ambiguity_preexisting": True,
        "semantic_world_definition": "actual_historical_unit_test_AND_required_callable_bindings_preserved_from_pre_fix_call_capture",
        "semantic_baseline_generated_from_pre_fix_execution": True,
        "later_human_fix_exposed_to_body": False,
        "candidate_generation_uses_semantic_outcomes": False,
        "generated_semantic_discriminator_count": len(discriminators),
        "learned_discriminator_id": learned.discriminator_id,
        "learned_selection_rule": learned.selection_rule,
        "supporting_contexts": list(parent_policy.supporting_contexts),
        "one_context_insufficient_for_authority": True,
        "verifierless_semantic_discriminator_authority": False,
        "semantic_policy_rederived_after_external_reverification": True,
        "heldout_full_candidate_count": len(candidates),
        "heldout_full_semantic_outcomes": list(full_outcomes),
        "heldout_unique_semantically_valid_candidate_count": 1,
        "treatment_candidate_count": 1,
        "treatment_capability": treatment_capability,
        "remove_same_checkpoint_candidate_count": 1,
        "remove_same_checkpoint_capability": remove_capability,
        "wrong_candidate_count": 1 if wrong_candidate is not None else 0,
        "wrong_capability": wrong_capability,
        "candidate_execution_reduction_vs_full": 1.0 - (1.0 / len(candidates)),
        "selected_edit_index": selected_index,
        "selected_edit_matches_unique_semantic_world_success": True,
        "required_binding_contract_derived_from_callable_signature": True,
        "semantic_discriminator_grammar_human_authored": True,
        "call_capture_normalizer_human_authored": True,
        "unrestricted_semantic_operator_genesis": False,
        "unrestricted_failure_extractor_operator_genesis": False,
        "global_recursive_acceleration": False,
        "independent_organizational_custody": False,
        "physical_world": False,
        "foundation_weight_change": False,
        "AGI": False,
        "ASI": False,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
