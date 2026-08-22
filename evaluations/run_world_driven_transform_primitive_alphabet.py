from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arte_cognition.canonical_body_checkpoint import checkpoint_dict, restore_runtime
from arte_cognition.cognitive_runtime import PersistentCognitiveRuntime
from arte_cognition.experiment_genesis import ExperimentGenesisEngine
from arte_cognition.projection_generator_transform_grammar import (
    DEEP_TRANSFORM_SIGNATURE_ANCHORS,
    generate_projection_transform_programs,
)
from arte_cognition.projection_scale_genesis import projection_scale_scores
from arte_cognition.projection_transform_primitive_genesis import generate_projection_power_primitive_programs
from arte_cognition.projection_transform_primitive_runtime import TransformPrimitiveAlphabetOrgan
from arte_cognition.representation_genesis import RepresentationAxis
from arte_cognition.world_coupling import HMACWorldReceiptSigner, HMACWorldReceiptVerifier, WorldOutcomeReceipt


def probe_scale(proposal):
    marker = "probe_scale="
    return float(str(proposal.reason).split(marker, 1)[1].split()[0].rstrip(",;)") )


def axis(label, rng):
    x = f"sensor_{rng.randrange(10000000,99999999)}"
    z = f"sensor_{rng.randrange(10000000,99999999)}"
    return RepresentationAxis(
        axis_id=f"AXIS::PROJECTION::{x}|{z}", family="PROJECTION", inputs=(x,z),
        threshold=0.0, direction="GT", information_gain=1.0, train_support=8,
        positive_partition=(f"{label}-positive",), formula=f"(1)*{x} + (1)*{z}",
        coefficients=((x,1.0),(z,1.0)), bias=0.0, status="PROPOSAL_ONLY",
    )


class HiddenScaleWorld:
    def __init__(self, target, signer, source_id, challenge_id, context_id, epoch):
        self.target=float(target); self.signer=signer; self.source_id=str(source_id)
        self.challenge_id=str(challenge_id); self.context_id=str(context_id); self.epoch=int(epoch)
    def execute(self, proposal, arm, value):
        effect = 1.0 if abs(probe_scale(proposal)-self.target)<=1e-9 else 0.25
        outcome = 0.0 if str(arm).upper()=="LOW" else effect
        return self.signer.sign(WorldOutcomeReceipt(
            receipt_id=f"{self.challenge_id}::{proposal.experiment_id}::{arm}",
            experiment_id=proposal.experiment_id, axis_id=proposal.axis_id, arm=arm,
            intervention_value=float(value), outcome=float(outcome), source_id=self.source_id,
            context_id=self.context_id, challenge_id=self.challenge_id, epoch=self.epoch,
            budget_token=f"budget::{self.challenge_id}", externally_generated=True,
        ))


def endpoints(ax,left,right):
    return ExperimentGenesisEngine(
        projection_margin_multipliers=(float(left),float(right)), max_proposals=64
    ).propose(ax,{ax.inputs[0]:0.0,ax.inputs[1]:0.0})


def execute(body, proposals, target, context, epoch_base, signers, verifier):
    for pi, proposal in enumerate(proposals):
        body.memory.remember_experiment(proposal)
        for ii,(issuer,signer) in enumerate(signers.items()):
            pair=body.execute_world_intervention(
                proposal,
                HiddenScaleWorld(target,signer,f"{context}-{epoch_base}-s-{pi}-{issuer}",
                                 f"{context}-{epoch_base}-c-{pi}-{issuer}",context,
                                 epoch_base+pi*10+ii),
                verifier=verifier,
            )
            if not pair.authority_verified:
                raise AssertionError("hidden primitive world pair lost authority")


def capability(body,context,target):
    scores=projection_scale_scores(
        (r.proposal for r in body.memory.experiments.values()),body.world_coupling.pairs,
        body.world_coupling.min_independent_classes,probe_scale,context_id=context,
    )
    return float(scores.get(round(float(target),12),0.0)>=0.9)


def old_values(programs,left,right):
    return {p.apply(left,right) for p in programs if p.apply(left,right) is not None}


def choose_bracket(rng, primitive, old_depth4, all_primitives, require_not_first=False):
    for _ in range(5000):
        left=round(rng.uniform(1.5,12.0),6)
        right=round(left*rng.uniform(2.2,6.5),6)
        target=primitive.apply(left,right)
        if target is None: continue
        if any(abs(float(v)-target)<=1e-9 for v in old_values(old_depth4,left,right)): continue
        primitive_values=sorted({p.apply(left,right) for p in all_primitives if p.apply(left,right) is not None})
        if require_not_first and primitive_values and abs(primitive_values[0]-target)<=1e-9: continue
        colliders=[p.program_id for p in all_primitives if p.apply(left,right) is not None and abs(p.apply(left,right)-target)<=1e-9]
        if len(colliders)!=1: continue
        return (left,right,target)
    raise AssertionError("failed to sample primitive-novel bracket")


def falsify_old(body, brackets, primitive, rng, signers, verifier, epoch_base):
    for index,(context,(left,right)) in enumerate(brackets.items()):
        target=primitive.apply(left,right); ax=axis(context,rng); body.memory.remember_representation(ax)
        execute(body,endpoints(ax,left,right),target,context,epoch_base+index*10000,signers,verifier)
        generated=body.generate_projection_transform_adaptive_interventions(
            ax,{ax.inputs[0]:0.0,ax.inputs[1]:0.0},context,left,right,brackets,
            current_depth=3,next_depth=4,max_candidates=128,
            allow_depth_expansion=False,apply_learned_program=False,
        )
        execute(body,generated,target,context,epoch_base+1000+index*10000,signers,verifier)
    assessment=body.projection_transform_depth_assessment(brackets,current_depth=3,next_depth=4)
    if assessment.status!="TRANSFORM_GRAMMAR_DEPTH_FALSIFIED_OPEN_NEXT":
        raise AssertionError(f"old alphabet not completely falsified: {assessment}")
    if any(item.missing_program_ids for item in assessment.context_assessments):
        raise AssertionError("absence was used as old-alphabet refutation")
    return assessment


def train_primitive(body, brackets, primitive, training, rng, signers, verifier, epoch_base):
    organ=TransformPrimitiveAlphabetOrgan(body)
    for index,(left,right,target) in enumerate(training):
        context=f"primitive-train-{rng.randrange(10000000,99999999)}"
        ax=axis(context,rng); body.memory.remember_representation(ax)
        frontier=organ.frontier(context,left,right,brackets,current_depth=3,max_candidates=64,
                                apply_learned_primitive=False)
        if round(float(target),12) not in {c.scale for c in frontier.candidates}:
            raise AssertionError("hidden generated primitive absent from outcome-independent shadow")
        generated=organ.generate_interventions(
            ax,{ax.inputs[0]:0.0,ax.inputs[1]:0.0},context,left,right,brackets,
            current_depth=3,max_candidates=64,apply_learned_primitive=False,
        )
        execute(body,generated,target,context,epoch_base+index*10000,signers,verifier)
    policy=organ.policy()
    if policy.program_id!=primitive.program_id:
        raise AssertionError(f"wrong primitive selected: {policy}")
    return policy


def main(seed_path):
    seed=int(Path(seed_path).read_text().strip()); rng=random.Random(seed)
    primitives=generate_projection_power_primitive_programs()
    old_depth4=generate_projection_transform_programs(
        max_transform_depth=4,signature_anchors=DEEP_TRANSFORM_SIGNATURE_ANCHORS
    )
    hidden=rng.choice(primitives)

    samples=[choose_bracket(rng,hidden,old_depth4,primitives,require_not_first=(i==4)) for i in range(5)]
    wrong_candidates=[]
    for candidate in primitives:
        if candidate.program_id==hidden.program_id: continue
        if all(candidate.apply(l,r) is not None and
               all(abs(float(v)-candidate.apply(l,r))>1e-9 for v in old_values(old_depth4,l,r))
               for l,r,_ in samples):
            wrong_candidates.append(candidate)
    if not wrong_candidates: raise AssertionError("no independent wrong primitive control available")
    wrong_hidden=rng.choice(wrong_candidates)

    issuer_a=f"issuer-{rng.randrange(10**7,10**8)}"; issuer_b=f"issuer-{rng.randrange(10**7,10**8)}"
    key_a=hashlib.sha256(f"{seed}:primitive:a".encode()).digest(); key_b=hashlib.sha256(f"{seed}:primitive:b".encode()).digest()
    signers={issuer_a:HMACWorldReceiptSigner(issuer_a,key_a),issuer_b:HMACWorldReceiptSigner(issuer_b,key_b)}
    verifier=HMACWorldReceiptVerifier({issuer_a:key_a,issuer_b:key_b},independence_classes={issuer_a:"LAB_A",issuer_b:"LAB_B"})

    body=PersistentCognitiveRuntime()
    failure_contexts=[f"alphabet-f-{rng.randrange(10000000,99999999)}" for _ in range(2)]
    failure_brackets={c:(samples[i][0],samples[i][1]) for i,c in enumerate(failure_contexts)}
    assessment=falsify_old(body,failure_brackets,hidden,rng,signers,verifier,10000)
    learned=train_primitive(body,failure_brackets,hidden,samples[2:4],rng,signers,verifier,40000)

    checkpoint=checkpoint_dict(body)
    verifierless=restore_runtime(checkpoint)
    verifierless_policy=TransformPrimitiveAlphabetOrgan(verifierless).policy()
    if verifierless_policy.primitive_id is not None:
        raise AssertionError("primitive authority restored without verifier")

    heldout_left,heldout_right,heldout_target=samples[4]
    heldout_context=f"heldout-{rng.randrange(10000000,99999999)}"; heldout_axis=axis(heldout_context,rng)
    treatment=restore_runtime(checkpoint,world_verifier=verifier); remove=restore_runtime(checkpoint,world_verifier=verifier)
    for b in (treatment,remove):
        b.memory.remember_representation(heldout_axis)
        execute(b,endpoints(heldout_axis,heldout_left,heldout_right),heldout_target,heldout_context,80000,signers,verifier)
    torgan=TransformPrimitiveAlphabetOrgan(treatment); rorgan=TransformPrimitiveAlphabetOrgan(remove)
    tf=torgan.frontier(heldout_context,heldout_left,heldout_right,failure_brackets,current_depth=3,max_candidates=1,apply_learned_primitive=True)
    tg=torgan.generate_interventions(heldout_axis,{heldout_axis.inputs[0]:0.0,heldout_axis.inputs[1]:0.0},heldout_context,heldout_left,heldout_right,failure_brackets,current_depth=3,max_candidates=1,apply_learned_primitive=True)
    execute(treatment,tg,heldout_target,heldout_context,90000,signers,verifier)
    rf=rorgan.frontier(heldout_context,heldout_left,heldout_right,failure_brackets,current_depth=3,max_candidates=1,apply_learned_primitive=False)
    rg=rorgan.generate_interventions(heldout_axis,{heldout_axis.inputs[0]:0.0,heldout_axis.inputs[1]:0.0},heldout_context,heldout_left,heldout_right,failure_brackets,current_depth=3,max_candidates=1,apply_learned_primitive=False)
    execute(remove,rg,heldout_target,heldout_context,90000,signers,verifier)

    wrong=PersistentCognitiveRuntime(); wrong_contexts=[f"wrong-f-{rng.randrange(10000000,99999999)}" for _ in range(2)]
    wrong_failure={c:(samples[i][0],samples[i][1]) for i,c in enumerate(wrong_contexts)}
    falsify_old(wrong,wrong_failure,wrong_hidden,rng,signers,verifier,120000)
    wrong_training=[(samples[i][0],samples[i][1],wrong_hidden.apply(samples[i][0],samples[i][1])) for i in (2,3)]
    wrong_policy=train_primitive(wrong,wrong_failure,wrong_hidden,wrong_training,rng,signers,verifier,150000)
    wrong=restore_runtime(checkpoint_dict(wrong),world_verifier=verifier); wc=f"wrong-h-{rng.randrange(10000000,99999999)}"; wa=axis(wc,rng); wrong.memory.remember_representation(wa)
    execute(wrong,endpoints(wa,heldout_left,heldout_right),heldout_target,wc,180000,signers,verifier)
    worgan=TransformPrimitiveAlphabetOrgan(wrong); wg=worgan.generate_interventions(wa,{wa.inputs[0]:0.0,wa.inputs[1]:0.0},wc,heldout_left,heldout_right,wrong_failure,current_depth=3,max_candidates=1,apply_learned_primitive=True)
    execute(wrong,wg,heldout_target,wc,190000,signers,verifier)

    treatment_cap=capability(treatment,heldout_context,heldout_target); remove_cap=capability(remove,heldout_context,heldout_target); wrong_cap=capability(wrong,wc,heldout_target)
    if treatment_cap!=1.0 or remove_cap!=0.0 or wrong_cap!=0.0:
        raise AssertionError("primitive causal transfer controls failed")

    result={
        "status":"PASS_BOUNDED_WORLD_FALSIFICATION_DRIVEN_TRANSFORM_PRIMITIVE_ALPHABET_AND_DESCENDANT_CAUSAL_TRANSFER",
        "old_alphabet":["LOG","INV"],"old_depth":3,"old_depth_program_count":assessment.current_program_count,
        "old_alphabet_falsified_contexts":len(assessment.falsified_contexts),
        "old_alphabet_missing_programs":sum(len(x.missing_program_ids) for x in assessment.context_assessments),
        "primitive_shadow_program_count":len(primitives),"learned_primitive_id":learned.primitive_id,
        "learned_exponent":learned.exponent,"learned_alpha":learned.alpha,
        "heldout_bracket":[heldout_left,heldout_right],"heldout_target":heldout_target,
        "target_absent_from_old_alphabet_depth4":True,"treatment_candidate_count":len(tf.candidates),
        "remove_candidate_count":len(rf.candidates),"treatment_capability":treatment_cap,
        "remove_same_checkpoint_capability":remove_cap,"wrong_learned_primitive_id":wrong_policy.primitive_id,
        "wrong_capability":wrong_cap,"verifierless_primitive_authority":False,
        "candidate_primitive_generation_uses_world_outcomes":False,"power_schema_human_authored":True,
        "unrestricted_operator_invention":False,"foundation_weight_change":False,"physical_world":False,
        "independent_organizational_custody":False,"global_recursive_acceleration":False,"AGI":False,"ASI":False,
    }
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    if len(sys.argv)!=2: raise SystemExit("usage: run_world_driven_transform_primitive_alphabet.py <seed_path>")
    main(sys.argv[1])
