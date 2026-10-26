#!/usr/bin/env python3
"""Stage D (CPU): aggregate the frozen response metric and bootstrap by video.

Reads `forward_records.jsonl` and writes per-unit, per-video and summary tables
plus a video-level bootstrap.  Unit of resampling is the video: the 29 units
are nested in 6 videos and are never treated as independent draws.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/repro")

import b_common as B  # noqa: E402


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def boot_ci(per_video, stat, n=None, seed=B.BOOT_SEED):
    vids = sorted(per_video)
    vals = [stat(per_video[v]) for v in vids]
    if len(vids) < 2:
        return {"point": vals[0] if vals else None, "lo": None, "hi": None,
                "n_video": len(vids)}
    rng = random.Random(seed)
    draws = []
    for _ in range(n or B.BOOT_N):
        sample = [rng.choice(vals) for _ in vids]
        draws.append(mean(sample))
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(int(0.975 * len(draws)), len(draws) - 1)]
    return {"point": mean(vals), "lo": lo, "hi": hi, "n_video": len(vids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    out = os.path.abspath(a.out or os.path.join(run, "analysis"))
    os.makedirs(out, exist_ok=True)

    src = os.path.join(run, "forward_records.jsonl")
    if not os.path.exists(src):
        alt = os.path.join(run, "main", "forward_records.jsonl")
        if os.path.exists(alt):
            src = alt
        else:
            raise SystemExit(f"no forward records under {run}")
    records = B.read_jsonl(src)
    gates_path = os.path.join(os.path.dirname(src), "gates.json")
    gates = B.read_json(gates_path) if os.path.exists(gates_path) else None

    # ---------------- per-unit table ----------------
    per_unit = []
    for r in records:
        e = {
            "unit_id": r["unit_id"], "family": r["family"],
            "video": r["video"], "family_source": r["family_source"],
            "round_index": r["round_index"], "anchor_rel": r["anchor_rel"],
            "prefix_len": r["prefix_len"],
            "order_marker": int(r["order_marker"]),
            "order_marker_hits": r["order_marker_hits"],
            "gate_G2_pass": int(bool(r.get("gate_G2", {}).get("pass", False))),
            "target_entropy_O": r["per_condition"]["O"]["target_entropy"],
            "reference_in_draft_top8_O":
                int(r["per_condition"]["O"]["reference_in_draft_top8"]),
            "reference_draft_rank_O":
                r["per_condition"]["O"]["reference_draft_rank"],
        }
        for key, cond in [("T", "T"), ("K", "K"), ("Orec", "O_rec")]:
            resp = r["response_" + key]
            raw = r["response_raw_" + key]
            e["L1_target_" + key] = resp["L1_target"]
            e["L1_draft_" + key] = resp["L1_draft"]
            e["R_" + key] = resp["R"]
            e["cos_" + key] = resp["cos"]
            e["prob_tv_target_" + key] = resp["prob_tv_target"]
            e["prob_tv_draft_" + key] = resp["prob_tv_draft"]
            e["prob_R_" + key] = resp["prob_R"]
            e["prob_cos_" + key] = resp["prob_cos"]
            e["raw_L1_target_" + key] = raw["L1_target"]
            e["raw_L1_draft_" + key] = raw["L1_draft"]
            e["raw_R_" + key] = raw["R"]
            e["n_cand_raw"] = len(raw["candidate_tokens"])
            e["n_cand_eff"] = len(resp["candidate_tokens"])
            e["target_greedy_changed_" + key] = int(
                r["per_condition"][cond]["target_argmax"]
                != r["per_condition"]["O"]["target_argmax"])
            e["coverage_change_" + key] = int(
                r["per_condition"][cond]["reference_in_draft_top8"]
                != r["per_condition"]["O"]["reference_in_draft_top8"])
            e["target_entropy_" + key] = r["per_condition"][cond]["target_entropy"]
            e["draft_entropy_" + key] = r["per_condition"][cond]["draft_entropy"]
        e["noise_below_T"] = int(bool(r["gate_G5"]["noise_below_T"]))
        per_unit.append(e)

    cols = list(per_unit[0].keys())
    with open(os.path.join(out, "per_unit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(per_unit)

    # ---------------- per-video table ----------------
    per_video = {}
    for fam in ("N", "C", "all"):
        for vid in B.VIDEOS:
            units = [u for u in per_unit
                     if u["video"] == vid and (fam == "all" or u["family"] == fam)]
            if not units:
                continue
            key = f"{fam}|{vid}"
            per_video[key] = {
                "n_units": len(units),
                "n_with_R_T": sum(1 for u in units if u["R_T"] is not None),
                "n_with_R_K": sum(1 for u in units if u["R_K"] is not None),
                "L1_target_T": mean([u["L1_target_T"] for u in units]),
                "L1_draft_T": mean([u["L1_draft_T"] for u in units]),
                "L1_target_K": mean([u["L1_target_K"] for u in units]),
                "L1_draft_K": mean([u["L1_draft_K"] for u in units]),
                "L1_target_Orec": mean([u["L1_target_Orec"] for u in units]),
                "L1_draft_Orec": mean([u["L1_draft_Orec"] for u in units]),
                "R_T": mean([u["R_T"] for u in units]),
                "R_K": mean([u["R_K"] for u in units]),
                "R_T_minus_K": mean([u["R_T"] - u["R_K"] for u in units
                                     if u["R_T"] is not None
                                     and u["R_K"] is not None]),
                "cos_T": mean([u["cos_T"] for u in units]),
                "cos_K": mean([u["cos_K"] for u in units]),
                "prob_R_T": mean([u["prob_R_T"] for u in units]),
                "prob_R_K": mean([u["prob_R_K"] for u in units]),
                "prob_R_T_minus_K": mean([u["prob_R_T"] - u["prob_R_K"]
                                          for u in units
                                          if u["prob_R_T"] is not None
                                          and u["prob_R_K"] is not None]),
                "prob_tv_target_T": mean([u["prob_tv_target_T"] for u in units]),
                "prob_tv_draft_T": mean([u["prob_tv_draft_T"] for u in units]),
                "prob_tv_target_K": mean([u["prob_tv_target_K"] for u in units]),
                "prob_tv_draft_K": mean([u["prob_tv_draft_K"] for u in units]),
                "prob_tv_target_Orec": mean([u["prob_tv_target_Orec"] for u in units]),
                "raw_R_T": mean([u["raw_R_T"] for u in units]),
                "raw_R_K": mean([u["raw_R_K"] for u in units]),
                "target_greedy_changed_T":
                    mean([u["target_greedy_changed_T"] for u in units]),
                "target_greedy_changed_K":
                    mean([u["target_greedy_changed_K"] for u in units]),
                "coverage_change_T": mean([u["coverage_change_T"] for u in units]),
                "coverage_change_K": mean([u["coverage_change_K"] for u in units]),
                "noise_below_T_frac": mean([u["noise_below_T"] for u in units]),
                "target_entropy_O": mean([u["target_entropy_O"] for u in units]),
            }
    with open(os.path.join(out, "per_video.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group"] + list(
            next(iter(per_video.values())).keys()))
        w.writeheader()
        for k, v in per_video.items():
            w.writerow({"group": k, **v})

    # ---------------- bootstrap / summary ----------------
    summary = {
        "source_records": src,
        "n_records": len(records),
        "n_family_N": sum(1 for u in per_unit if u["family"] == "N"),
        "n_family_C": sum(1 for u in per_unit if u["family"] == "C"),
        "bootstrap": {"n_resamples": B.BOOT_N, "seed": B.BOOT_SEED,
                      "unit": "video"},
        "families": {},
    }
    for fam in ("N", "C"):
        pv = {v: per_video[f"{fam}|{v}"] for v in B.VIDEOS
              if f"{fam}|{v}" in per_video}
        if not pv:
            continue
        block = {
            "n_units": sum(d["n_units"] for d in pv.values()),
            "n_units_with_R_T": sum(d["n_with_R_T"] for d in pv.values()),
            "n_units_with_R_K": sum(d["n_with_R_K"] for d in pv.values()),
            "per_video": pv,
            "mean_R_T": boot_ci(pv, lambda d: d["R_T"]),
            "mean_R_K": boot_ci(pv, lambda d: d["R_K"]),
            "mean_R_T_minus_K": boot_ci(pv, lambda d: d["R_T_minus_K"]),
            "mean_cos_T": boot_ci(pv, lambda d: d["cos_T"]),
            "mean_cos_K": boot_ci(pv, lambda d: d["cos_K"]),
            "mean_L1_target_T": boot_ci(pv, lambda d: d["L1_target_T"]),
            "mean_L1_draft_T": boot_ci(pv, lambda d: d["L1_draft_T"]),
            "mean_L1_target_K": boot_ci(pv, lambda d: d["L1_target_K"]),
            "mean_L1_draft_K": boot_ci(pv, lambda d: d["L1_draft_K"]),
            "mean_L1_target_Orec": boot_ci(pv, lambda d: d["L1_target_Orec"]),
            "mean_L1_draft_Orec": boot_ci(pv, lambda d: d["L1_draft_Orec"]),
            "mean_prob_R_T": boot_ci(pv, lambda d: d["prob_R_T"]),
            "mean_prob_R_K": boot_ci(pv, lambda d: d["prob_R_K"]),
            "mean_prob_R_T_minus_K": boot_ci(pv, lambda d: d["prob_R_T_minus_K"]),
            "mean_prob_tv_target_T": boot_ci(pv, lambda d: d["prob_tv_target_T"]),
            "mean_prob_tv_draft_T": boot_ci(pv, lambda d: d["prob_tv_draft_T"]),
            "mean_prob_tv_target_K": boot_ci(pv, lambda d: d["prob_tv_target_K"]),
            "mean_prob_tv_draft_K": boot_ci(pv, lambda d: d["prob_tv_draft_K"]),
            "mean_prob_tv_target_Orec":
                boot_ci(pv, lambda d: d["prob_tv_target_Orec"]),
            "mean_raw_R_T": boot_ci(pv, lambda d: d["raw_R_T"]),
            "mean_raw_R_K": boot_ci(pv, lambda d: d["raw_R_K"]),
            "mean_target_greedy_changed_T":
                boot_ci(pv, lambda d: d["target_greedy_changed_T"]),
            "mean_target_greedy_changed_K":
                boot_ci(pv, lambda d: d["target_greedy_changed_K"]),
            "mean_coverage_change_T":
                boot_ci(pv, lambda d: d["coverage_change_T"]),
            "mean_coverage_change_K":
                boot_ci(pv, lambda d: d["coverage_change_K"]),
            "videos_with_R_T_below_1": [
                v for v, d in pv.items() if (d["R_T"] or 1) < 1],
            "videos_with_R_T_below_R_K": [
                v for v, d in pv.items()
                if d["R_T"] is not None and d["R_K"] is not None
                and d["R_T"] < d["R_K"]],
        }
        summary["families"][fam] = block

    # ---------------- exploratory offsets ----------------
    expl_rows = []
    for r in records:
        for e in r.get("exploratory_offsets", []):
            row = {"unit_id": r["unit_id"], "family": r["family"],
                   "video": r["video"], "offset": e["offset"],
                   "reference_token": e["reference_token"],
                   "target_entropy_O": e["O"]["target_entropy"],
                   "target_argmax_changed_T": int(
                       e["T"]["target_argmax"] != e["O"]["target_argmax"]),
                   }
            for key in ("T", "K", "Orec"):
                row[f"L1_target_{key}"] = e["response_" + key]["L1_target"]
                row[f"L1_draft_{key}"] = e["response_" + key]["L1_draft"]
                row[f"R_{key}"] = e["response_" + key]["R"]
                row[f"cos_{key}"] = e["response_" + key]["cos"]
            expl_rows.append(row)
    if expl_rows:
        with open(os.path.join(out, "per_offset_exploratory.csv"), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(expl_rows[0].keys()))
            w.writeheader()
            w.writerows(expl_rows)
        # aggregate offsets 2..8 per family/video
        agg = {}
        for row in expl_rows:
            key = (row["family"], row["video"])
            agg.setdefault(key, []).append(row)
        offset_summary = {}
        for (fam, vid), rows in agg.items():
            offset_summary[f"{fam}|{vid}"] = {
                "n_offset_units": len(rows),
                "n_with_R_T": sum(1 for r in rows if r["R_T"] is not None),
                "mean_L1_target_T": mean([r["L1_target_T"] for r in rows]),
                "mean_L1_draft_T": mean([r["L1_draft_T"] for r in rows]),
                "mean_R_T": mean([r["R_T"] for r in rows]),
                "mean_R_K": mean([r["R_K"] for r in rows]),
                "mean_cos_T": mean([r["cos_T"] for r in rows]),
                "frac_target_argmax_changed_T":
                    mean([r["target_argmax_changed_T"] for r in rows]),
            }
        summary["exploratory_offsets"] = offset_summary

    summary["gates"] = {
        "G1": gates.get("G1_summary") if gates else None,
        "G2": gates.get("G2_summary") if gates else None,
        "G3": gates.get("G3_summary") if gates else None,
        "pilot": gates.get("pilot_gate") if gates else None,
    }
    B.write_json(os.path.join(out, "summary.json"), summary)

    # ---------------- prefix / position provenance ----------------
    pre_path = os.path.join(run, "stageA", "prefixes.json")
    if os.path.exists(pre_path):
        pref = B.read_json(pre_path)
        prov = []
        for r in records:
            p = pref.get(r["unit_id"], {})
            pid = p.get("prefix_ids") or []
            prov.append({
                "unit_id": r["unit_id"], "family": r["family"],
                "video": r["video"],
                "prompt_len": len(p.get("prompt_ids") or []),
                "prefix_len": r["prefix_len"],
                "anchor_pos": r["anchor_pos"],
                "root": r["per_condition"]["O"]["root"],
                "reference_token_O": r["per_condition"]["O"]["reference_token"],
                "prefix_ids_sha256": B.sha256_bytes(
                    json.dumps(pid).encode()) if pid else None,
                "prefix_text_sha256": B.sha256_bytes(
                    r["prefix_text"].encode()),
                "mrope": ("explicit position_ids from "
                          "target.model.get_rope_index(input_ids, None, "
                          "video_grid_thw, None), identical to "
                          "glance.decode.generate's prefill path"),
                "attention": "use_cache=True + DynamicCache, no 2-D mask "
                             "(is_causal), the decode branch",
                "vision_binding": "pixel_values_videos + video_grid_thw from "
                                  "the real processor per condition; each "
                                  "condition rebuilds its own vision KV",
            })
        import csv as _csv
        with open(os.path.join(out, "prefix_provenance.csv"), "w",
                  newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(prov[0].keys()))
            w.writeheader()
            w.writerows(prov)

    print("family N:")
    for k, v in summary.get("families", {}).get("N", {}).items():
        if isinstance(v, dict) and "point" in v:
            print(f"  {k:32s} {v['point']}  CI[{v['lo']},{v['hi']}]")
    print("per-video R_T / R_K (family N):")
    for v in B.VIDEOS:
        d = summary.get("families", {}).get("N", {}).get("per_video", {}).get(v)
        if d:
            print(f"  {v:18s} n={d['n_units']} R_T={d['R_T']} R_K={d['R_K']} "
                  f"L1t_T={d['L1_target_T']} L1t_K={d['L1_target_K']} "
                  f"L1t_noise={d['L1_target_Orec']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
