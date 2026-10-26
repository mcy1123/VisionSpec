#!/usr/bin/env python3
"""Write `prereg.json` for task B.  Must run before any model forward.

Everything here is stated before the first O/T/K logit exists: the units, the
three conditions, the response metric, the NA thresholds, the aggregation, the
gates and the stop rules.  The script refuses to overwrite an existing prereg
so a rerun cannot quietly edit the frozen protocol.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_common as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run directory")
    a = ap.parse_args()
    run = os.path.abspath(a.out)
    target = os.path.join(run, "prereg.json")
    if os.path.exists(target):
        raise SystemExit(f"refusing to overwrite frozen prereg {target}")

    sel = B.read_json(os.path.join(run, "stageA", "selection.json"))
    audit = B.read_json(os.path.join(run, "stageA", "audit_c6.json"))

    prereg = {
        "created": datetime.now().astimezone().isoformat(),
        "direction": "B",
        "task_file": "docs/tasks/parallel_20261012/02_task_B_temporal_response.md",
        "question": (
            "When a change in video event order / entity state moves the frozen "
            "target's token preference at a shared text prefix, does the C0 "
            "draft's distribution respond systematically less than the "
            "target's, and is any such gap specific to temporal change rather "
            "than to a general visual perturbation?"),
        "not_claimed": [
            "nothing here is an online tau or a speed number",
            "no training, no loss change, no head/verifier modification",
            "the frozen new test sets stay closed",
            "family C units are not causal evidence (their prefix asserts "
            "timing and a full reversal makes it false)",
        ],
        "models": {
            "target": B.C0_PATH.replace("head_video439_aligned/step1000",
                                        "target is frozen Qwen3-VL-8B-Instruct"),
            "target_path": "/mnt/local/mcy/GLANCE/models/Qwen3-VL-8B-Instruct",
            "draft": B.C0_PATH,
            "draft_model_sha256": B.C0_MODEL_SHA,
            "dtype": "float32 target and float32 draft (C6 convention)",
            "device": "one idle A6000, under the shared flock",
        },
        "units": {
            "source": "C6 frozen 6 videos / 48 rounds, plus the same 6 videos' "
                      "round-0 block offsets 2..8",
            "selection_rule": audit["selection_rule"],
            "entropy_min_nats": audit["entropy_min_nats"],
            "max_units_per_video": audit["max_units_per_video"],
            "order_markers": audit["order_markers"],
            "analysis_position": audit["analysis_position"],
            "prefix_construction": audit["prefix_construction"],
            "family_N": {"role": "main causal comparison",
                         "count": sel["family_N_count"],
                         "units": [u["unit_id"] for u in sel["family_N_units"]]},
            "family_C": {"role": "secondary, prefix falsified by T, not causal",
                         "count": sel["family_C_count"],
                         "units": [u["unit_id"] for u in sel["family_C_units"]]},
        },
        "conditions": {
            "O": "the original cached clip, unmodified (the file C6 read)",
            "T": "temporal-order intervention: the 16 sampled frames in exact "
                 "reverse order, written through the same cv2 mp4v writer and "
                 "the same 384-px geometry, 16 frames at 30 fps",
            "K": "order-preserving control: the same 16 frames in the same "
                 "order, mirrored horizontally, same writer and geometry",
            "O_rec": "gate only, not a condition: the same 16 frames in the "
                     "original order written through the same writer, to bound "
                     "the codec/pipeline noise floor of the response metric",
        },
        "frame_timeline_disclosure": (
            "each cached clip is 16 frames resampled by np.linspace over the "
            "ORIGINAL source video, which is 18-225 s long; the cached file's "
            "0.53 s packaging time is not the original timeline. The "
            "intervention is therefore defined on the sampled 16-frame "
            "sequence, whose original frame indices and times are recorded in "
            "frames/frame_manifest.json; T reverses that observed sequence."),
        "response_metric": {
            "candidate_set_raw": "union of the target top-8 sets of O, T and K "
                                 "at the analysis position (fixed k=8)",
            "candidate_set_primary":
                "the raw union minus every token whose probability stays below "
                "0.005 under O, T and K (probability floor)",
            "lt_c": "log_softmax of the target logits at the analysis position "
                    "under condition c",
            "ld_c": "log_softmax of the draft's offset-1 logits at the same "
                    "position under condition c",
            "delta_X": "vector (lt_X - lt_O) or (ld_X - ld_O) for X in {T, K}",
            "L1_target_X": "sum |lt_X - lt_O| over the primary candidate set",
            "L1_draft_X": "sum |ld_X - ld_O| over the primary candidate set",
            "R_X": "L1_draft_X / L1_target_X; NA when L1_target_X < 0.05 nats",
            "cos_X": "cosine(draft delta, target delta); NA when either norm "
                     "< 1e-8",
            "coverage_change_X": "membership change of the reference token in "
                                 "the draft top-8 between O and X",
            "secondary": "the same log-probability quantities on the raw union, "
                         "and the probability-space L1 (= 2 x total variation) "
                         "and cosine on the raw union",
            "noise_floor": "the same quantities computed for O vs O_rec; any "
                           "unit whose O vs O_rec L1_target is not clearly "
                           "below its O vs T L1_target is flagged",
            "metric_revision": (
                "prereg v1 defined the primary set as the raw top-8 union. The "
                "2-video gate pilot then showed that log-probability L1 over "
                "that set is dominated by tokens sitting at log p ~ -25 whose "
                "magnitude swamps real mass movement, so the O vs O_rec "
                "pipeline noise became comparable to O vs T. v2 adds the "
                "probability floor to the primary set and keeps the v1 numbers "
                "as an explicitly reported secondary. The revision was made "
                "after a gate pass on 2 of 6 videos, before any 6-video result "
                "and before any O vs T vs K comparison was read; the "
                "intervention, units, thresholds, aggregation and stop rules "
                "are unchanged, and prereg v1 is preserved as "
                "prereg_v1_superseded.json."),
        },
        "pilot_disclosure": {
            "v1_pilot": "v_-D1gdv_gQyw and v_1kwIYXB4b8s, all four conditions, "
                        "G1/G2/G2b/G3 passed; only the metric definition defect "
                        "above was found",
            "superseded_artifacts": "pilot_v1_metric/ (kept, not used as "
                                    "evidence)",
        },
        "thresholds": {
            "target_l1_na": B.TARGET_L1_NA,
            "prob_floor": B.PROB_FLOOR,
            "vec_norm_eps": B.VEC_NORM_EPS,
            "k_top": B.K_TOP,
            "entropy_min_nats": audit["entropy_min_nats"],
            "bootstrap_resamples": B.BOOT_N,
            "bootstrap_seed": B.BOOT_SEED,
        },
        "gates": {
            "G1_input_O": "re-running the processor on the original cache "
                          "reproduces C6's saved input_ids and pixel hashes",
            "G2_reproduce_C6": "at every C6-anchor unit, the recomputed draft "
                               "top-8 ids equal C6's captured trace top-8 ids, "
                               "and the recomputed target argmax equals C6's "
                               "recorded target_argmax (this target-side check "
                               "also covers the round0_offset units, whose "
                               "target position is the same)",
            "G2b_round0_implementation": (
                "in C6's own round-0 block (fused = prompt only, noise block "
                "root = a_1) all 15 draft offsets' top-8 must equal C6's "
                "captured GPU candidates. This is the implementation gate for "
                "the head call. It is needed because the round0_offset units "
                "sit at anchor_pos = prompt_len + j - 1, a boundary C6 never "
                "probed: there the header sees a_1..a_j in the fused context "
                "instead of only a_1 inside the noise block, so their draft row "
                "is deliberately not equal to C6's round-0 offset j."),
            "G3_text_identity": "input_ids are identical across O, T, K and "
                                "O_rec for the same video (only pixels move)",
            "G4_frame_permutation": "decoding each written file reproduces the "
                                    "intended frame permutation (mean abs pixel "
                                    "difference below 8/255)",
            "G5_pipeline_noise": "O vs O_rec response L1 is reported; units "
                                 "whose noise floor is not below their O vs T "
                                 "target L1 are flagged",
            "gate_failure_policy": "a failed model gate is localised and "
                                   "reported; the affected unit is excluded and "
                                   "listed, and no positive claim is made from "
                                   "a run that did not pass G1/G2/G2b/G3.",
        },
        "aggregation": (
            "per unit compute R_T, R_K, cos_T, cos_K and the raw L1s; average "
            "the units of a video with equal weight; report the 6-video mean "
            "with a video-level bootstrap CI (resample the 6 videos, seed "
            "20261012, 2000 draws). The paired contrast is R_T - R_K per "
            "video. n=6 videos is a probe, so an interval that touches 0 is "
            "'not distinguishable', never 'equivalent'."),
        "decision_reads": {
            "supports_temporal_specific_draft_weakness": (
                "video-level mean R_T < 1 and mean R_T < mean R_K, with the "
                "bootstrap CI of (R_T - R_K) entirely below 0"),
            "general_visual_weakness_not_temporal": (
                "R_T and R_K move together; the (R_T - R_K) interval spans 0"),
            "draft_over_responds": "mean R_T > 1 with a CI above 1",
            "undecidable": "too few non-NA units, or the target does not "
                           "respond at all (all L1_target below threshold)",
        },
        "stop_rules": [
            "if fewer than 6 non-NA family-N units survive the gates, report "
            "the coverage failure and do not claim a temporal mechanism",
            "if the target does not respond to T (all family-N L1_target < "
            "0.05), report 'the frozen target itself is insensitive to this "
            "intervention at these prefixes' and stop",
            "no threshold, unit, condition or marker list is changed after the "
            "first forward; a changed rule means a new run directory and a new "
            "prereg",
        ],
        "exploratory_analyses_declared_now": [
            "offsets 2..8 of every selected anchor, teacher-forced along the "
            "C0 AR continuation (exploratory; the suffix is teacher-forced, not "
            "an online round)",
            "family C units under O/T/K, reported separately",
            "per-condition target entropy, target greedy change, draft rank of "
            "the original reference token",
        ],
        "forbidden": [
            "using target logits, the correct token, an oracle path or a unit's "
            "measured benefit to choose the intervention, the units or the "
            "split",
            "calling a local logit change a tau or a speedup",
        ],
        "provenance": {
            "c6_manifest_sha256": audit["c6_manifest_sha256"],
            "c6_trace_sha256": audit["c6_trace_sha256"],
            "c6_token_metrics_sha256": audit["c6_token_metrics_sha256"],
            "baseline_eval_sha256": audit["baseline_eval_sha256"],
            "c0_model_sha256": audit["c0_model_sha256_actual"],
            "selection_json_sha256": B.sha256_file(
                os.path.join(run, "stageA", "selection.json")),
            "code": {p: B.sha256_file(os.path.join(
                "/home/mcy/Projects/GLANCE/repro/experiments/temporal_d4_b", p))
                for p in sorted(os.listdir(
                    "/home/mcy/Projects/GLANCE/repro/experiments/temporal_d4_b"))
                if p.endswith(".py")},
        },
    }
    B.write_json(target, prereg)
    print("wrote", target)


if __name__ == "__main__":
    main()
