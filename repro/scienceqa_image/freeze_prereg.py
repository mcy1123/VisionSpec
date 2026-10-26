#!/usr/bin/env python3
"""Freeze the experiment protocol and pre-registration BEFORE any model output.

Everything that could be used to choose a favourable analysis is written here,
once, and hashed. The dataset selection was already frozen by
``build_dataset.py`` (``dataset/readiness_gate.json``). The literal prompt is
recorded verbatim from ``build_dataset.PROMPT_TEMPLATE`` so a later edit of the
code cannot retroactively change what was run.

Run this before any training or evaluation. Re-running it must reproduce the same
bytes; it is not a place to record results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import PROMPT_TEMPLATE  # noqa: E402

SPLIT_SEED = 20261015
TRAIN_SEEDS = [20261013, 20261014]
TARGET_MODEL = "/mnt/local/mcy/GLANCE/models/Qwen3-VL-8B-Instruct"
WARM_HEAD = "/mnt/local/mcy/GLANCE/checkpoints/head"

PROTOCOL = {
    "task": "ScienceQA image-reasoning: can a correctly implemented online "
            "Full-KL OPD beat an equal-budget continue-SFT at raising b63k8 "
            "acceptance length and real decode time, under one frozen Qwen3-VL-8B "
            "target, one image draft start point, one prompt and one verifier?",
    "source_of_record": "docs/tasks/10.15-1948.md",
    "arms": {
        "I0": "the verified GLANCE image warm draft `${WARM_HEAD}`; zero updates",
        "I-SFT": "continue image QA variable-reveal CE SFT from exactly I0",
        "I-OPD": "correctly implemented online Full-KL OPD from exactly I0, reusing "
                 "the frozen QA position contract (repro/qa_opd_fair)",
    },
    "frozen": {
        "target": TARGET_MODEL,
        "target_mode": "greedy, temperature=0, the independent AR is the lossless oracle",
        "head_start": WARM_HEAD,
        "block_size": 16,
        "budget": 63,
        "budget_includes_root": False,
        "branch_k": 8,
        "eos_stop": "immediate",
        "precision": {"target": "float32", "head": "bfloat16",
                      "master_weights": "cpu fp32", "attn": "sdpa"},
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "weight_decay": 0.0,
                      "grad_clip": 1.0, "lr": 3e-5, "schedule": "constant"},
        "objective": "accepted one-hot CE + rejected full-vocabulary reverse KL "
                     "with gamma**(k-1), gamma=0.8, update-global two-stream "
                     "denominators, one stream dropped and renormalised when empty",
        "forbidden_this_round": ["k3 rejected KL", "Bernoulli forward KL",
                                 "TV loss", "P2/P3", "new tree or budget",
                                 "new visual module", "hyper-parameter search"],
        "time_axis": "not applicable (single images; there is no video timeline)",
    },
    "budget_plan": {
        "train_pool_docs": 1200,
        "qa_per_update": 4,
        "updates_target": 300,
        "qa_exposures_per_arm_per_seed": 1200,
        "anchors_per_qa_cap": 6,
        "anchors_per_update_cap": 24,
        "seeds": TRAIN_SEEDS,
        "split_seed": SPLIT_SEED,
        "checkpoint_grid": [30, 100, 200, 300],
        "note": "a uniform, pre-registered amendment may reduce the formal update "
                "count and/or the on-disk checkpoint form if measured disk or GPU "
                "limits make the above unsafe; any amendment must apply to BOTH "
                "arms of BOTH seeds and is recorded in amendments.jsonl BEFORE "
                "the formal runs start",
    },
    "dev_test": {
        "dev": "official ScienceQA-IMG validation, 240 frozen records",
        "test": "official ScienceQA-IMG test, 240 frozen records, opened once "
                "after all method/analysis decisions are frozen",
        "primary_estimate": "OPD(step_final) - SFT(step_final) on mean "
                            "per-image-group online tau",
        "bootstrap": {"n": 2000, "seed": 20261015,
                      "unit": "image group (one picture may carry several questions)"},
        "secondary": ["pooled tau", "median tau", "rounds", "tokens per round",
                      "top-8 coverage on the shared correct AR prefix",
                      "first blocker reason", "decode-only ms/token",
                      "prefill+decode ms/request", "end-to-end ms/request",
                      "peak GiB"],
    },
    "gates": {
        "readiness": "train>=1200 and dev>=240 and test>=240 real decodable "
                     "images with official splits and no cross-set image leakage",
        "prompt_length": "set max_new_tokens=512 if the train-only 32-record "
                         "pilot shows >10% of natural runs above 256 tokens; "
                         "else 256; one pre-registered revision to 1024 is allowed "
                         "only if >10% are hard-capped at 512 and before any dev "
                         "or head result is looked at",
        "image_binding": "the pixels must demonstrably change the target's fused "
                         "hidden state and its greedy continuation; otherwise stop",
        "lossless": "SPEC output token ids must equal independent greedy AR "
                    "token ids exactly, including EOS, bonus EOS, root EOS and the "
                    "max_new_tokens cap; comparing visible text is not enough",
        "position_contract": "n_acc from the current proposal against the current "
                             "target at anchor-relative positions; draft and "
                             "teacher logits at the same next-token position; the "
                             "whole legal rejected suffix supervised",
        "optimizer": "fp32 master + Adam moments + RNG + data cursor must restore "
                     "bit-exactly before any formal run",
        "disaster_stop": "if at the step-30 dev check the OPD arm's mean tau falls "
                         "more than 0.50 token below I0 AND its top-8 coverage falls "
                         "more than 10 percentage points below I0, and the cause is "
                         "independently confirmed not to be an EOS/truncation/path "
                         "bug, stop both arms at that point and report the stop "
                         "point as such (do not claim 300 updates were run)",
    },
    "reporting_rules": {
        "stop_rules": [
            "ordinary negative result: OPD does not reliably beat equal-budget SFT "
            "-> report it and close this Full-KL ScienceQA recipe; do not switch to "
            "k3/TV/P2/P3 and do not add training rounds",
            "mechanism-only: offline top-8 improves but online tau/time does not -> "
            "report the mechanism evidence, claim no speedup",
            "short answers: if most ScienceQA outputs are under 32 tokens, the "
            "conclusion is restricted to short/medium length VQA reasoning",
            "blocking inconsistency: any arm's SPEC != independent AR, or image "
            "path / official split recheck fails -> no speedup conclusion",
        ],
        "claim_limits": [
            "the test split is a 240-record subset of official ScienceQA test, not "
            "the full official test set and not an external unseen dataset",
            "the target is frozen and lossless, so no arm can be claimed to answer "
            "ScienceQA better; gold accuracy is an appendix on the target only",
            "single-seed deltas are never called a stable advantage; the two seeds "
            "are paired replicates, not independent datasets",
            "n=29-style dev reuse: this dev split is used for development and is "
            "not an untouched test set until the final frozen test is opened",
        ],
    },
    "artifacts": {
        "run_root": "results/scienceqa_image_reasoning_20261015T1955",
        "dataset_root": "datasets/scienceqa_glance_20261015/lmms-lab-encoder_ScienceQA@69dd4d6b",
        "all_new_data_on": ["/mnt/local", "/mnt/local2", "/mnt/local3"],
    },
}

PROMPT = {
    "template": PROMPT_TEMPLATE,
    "template_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest(),
    "image_binding": "the real picture is passed as a Qwen3-VL image object "
                     "(common.build_processor_messages / common.encode), never as "
                     "an <image> placeholder string",
    "leakage_rule": "gold answer index, lecture, solution and hint are never "
                    "placed in the prompt",
    "option_letters": "ABCDE, in the original official option order",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(json.dumps(PROTOCOL))
    protocol["warm_head"] = WARM_HEAD
    protocol["prompt"] = PROMPT
    protocol["frozen"]["head_start"] = WARM_HEAD

    # the dataset selection is already frozen; carry its identity forward
    rg = out / "dataset" / "readiness_gate.json"
    if not rg.exists():
        raise SystemExit("dataset/readiness_gate.json missing: run build_dataset.py first")
    readiness = json.load(open(rg))
    protocol["dataset"] = {
        "source": readiness["source"],
        "achieved": readiness["achieved"],
        "distinct_images": readiness["distinct_images"],
        "gate_pass": readiness["gate_pass"],
        "readiness_gate_sha256": hashlib.sha256(rg.read_bytes()).hexdigest(),
    }
    blob = json.dumps(protocol, indent=2, ensure_ascii=False, sort_keys=True)
    (out / "experiment_protocol.json").write_text(blob)
    (out / "prereg.json").write_text(json.dumps({
        "created_before_any_model_output": True,
        "protocol_sha256": hashlib.sha256(blob.encode()).hexdigest(),
        "prompt_template_sha256": PROMPT["template_sha256"],
        "protocol": protocol,
    }, indent=2, ensure_ascii=False, sort_keys=True))
    print("prompt sha256:", PROMPT["template_sha256"])
    print("protocol sha256:", hashlib.sha256(blob.encode()).hexdigest())
    print("wrote", out / "experiment_protocol.json")
    print("wrote", out / "prereg.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
