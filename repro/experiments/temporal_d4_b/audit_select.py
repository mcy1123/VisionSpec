#!/usr/bin/env python3
"""Stage A (pure CPU): audit C6 and freeze the task-B comparison units.

Reads only pre-existing C6 artifacts plus the frozen C0 AR trajectory.  No
model is loaded and no GPU is touched, so this can run while direction A holds
the GPU lock.

Two unit families are frozen, both defined before any O/T/K forward exists:

  family N  "order-neutral" prefixes -- text whose prefix makes no claim about
            *when* things happen, so a full temporal reversal leaves it true.
            Main causal comparison.
  family C  "order-committal" prefixes -- the prefix contains ordering
            vocabulary, so a full reversal falsifies it.  Listed separately for
            completeness, never used as causal evidence.

A prefix is eligible only when C6's own original-condition record puts the
target entropy at the analysis position at >= ENTROPY_MIN nats, because the
pre-registered response ratio divides by the target's L1 change and a
near-deterministic target makes that NA by construction.

No O/T/K model output and no target top-1 identity participates in selection.

Writes into `--out`:
  audit_c6.json     provenance/hashes of every input this stage read
  candidates.csv    every candidate with its pre-registered decision
  selection.json    the frozen units and the rule text
  prefixes.json     per unit: prompt ids, prefix ids, decoded text, verdict
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/repro")

import b_common as B

ENTROPY_MIN = 0.30
MAX_PER_VIDEO = 4
R0_OFFSETS = (2, 3, 4, 5, 6, 7, 8)


def build_candidates():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(
        "/mnt/local/mcy/GLANCE/models/Qwen3-VL-8B-Instruct")

    manifest = B.read_json(os.path.join(B.C6_RUN, "manifest.json"))
    baseline = B.read_json(B.BASELINE_EVAL)
    base_rows = {r["video"]: r for r in baseline["rows"]}
    traces = B.read_jsonl(os.path.join(B.C6_RUN, "trace_current_ar.jsonl"))
    token_metrics = B.read_jsonl(
        os.path.join(B.C6_RUN, "token_metrics_current_ar.jsonl"))
    trace_index = {(t["video"], t["round_index"]): t for t in traces
                   if t["arm"] == "C0"}
    tm_index = {(m["video"], m["round_index"], m["offset"]): m
                for m in token_metrics if m["arm"] == "C0"}
    inputs = {v: B.read_json(os.path.join(B.C6_RUN, f"input_{v}.json"))
              for v in B.VIDEOS}
    return manifest, base_rows, trace_index, tm_index, inputs, tk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    (manifest, base_rows, trace_index, tm_index, inputs,
     tk) = build_candidates()

    audit = {
        "created": datetime.now().astimezone().isoformat(),
        "c6_run": B.C6_RUN,
        "c6_manifest_sha256": B.sha256_file(os.path.join(B.C6_RUN, "manifest.json")),
        "c6_trace_sha256": B.sha256_file(
            os.path.join(B.C6_RUN, "trace_current_ar.jsonl")),
        "c6_token_metrics_sha256": B.sha256_file(
            os.path.join(B.C6_RUN, "token_metrics_current_ar.jsonl")),
        "baseline_eval": B.BASELINE_EVAL,
        "baseline_eval_sha256": B.sha256_file(B.BASELINE_EVAL),
        "c0_path": B.C0_PATH,
        "c0_model_sha256_expected": B.C0_MODEL_SHA,
        "c0_model_sha256_actual": B.sha256_file(
            os.path.join(B.C0_PATH, "model.safetensors")),
        "c6_input_files_sha256": {
            v: B.sha256_file(os.path.join(B.C6_RUN, f"input_{v}.json"))
            for v in B.VIDEOS},
        "entropy_min_nats": ENTROPY_MIN,
        "max_units_per_video": MAX_PER_VIDEO,
        "round0_offsets_considered": list(R0_OFFSETS),
        "order_markers": B.ORDER_MARKERS,
        "selection_rule": (
            "unit pool = {C6 anchors with H_valid>=1, offset 1} U "
            "{round-0 block offsets 2..8}; keep those whose shared text prefix "
            "has no ordering marker (family N) and whose C6 original-condition "
            "target entropy at the analysis position is >= 0.30 nats; rank by "
            "(entropy desc, family, anchor index asc); take up to 4 per video. "
            "Family C = the same pool with an ordering marker, ranked and "
            "capped identically, reported separately."),
        "analysis_position": (
            "the first offset the draft predicts from the fixed shared prefix "
            "(the token right after the prefix; the root token at anchor_pos is "
            "part of the prefix)"),
        "prefix_construction": (
            "prefix = question tokens + the C0 AR tokens before the predicted "
            "position, i.e. ar[:anchor_rel+1] for a C6 anchor and ar[:j] for "
            "round-0 offset j"),
    }
    assert audit["c0_model_sha256_actual"] == B.C0_MODEL_SHA, "C0 hash moved"
    B.write_json(os.path.join(out, "audit_c6.json"), audit)

    rows = []
    prefixes = {}
    prompt_ids_by_video = {}
    for vinfo in manifest["videos"]:
        vid = vinfo["video"]
        assert vid in B.VIDEOS, vid
        row = base_rows[vid]
        ar = row["ar_ids"]
        gen = row["gen_tokens"]
        prompt_ids = inputs[vid]["input_ids"]
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        assert len(prompt_ids) == row["prompt_len"] == vinfo["prompt_len"]
        prompt_ids_by_video[vid] = prompt_ids
        cands = []
        for rnd, rel in zip(vinfo["round_indices"], vinfo["anchor_rel"]):
            tr = trace_index[(vid, rnd)]
            assert ar[rel] == tr["root"], (vid, rnd)
            m1 = tm_index.get((vid, rnd, 1))
            if m1 is None:
                continue
            pref_ids = ar[: rel + 1]
            cands.append({
                "family_source": "C6_anchor", "video": vid,
                "round_index": rnd, "anchor_rel": rel,
                "pos": tr["pos"], "root": ar[rel],
                "H_valid": tr["H_valid"],
                "target_entropy": m1["target_entropy"],
                "draft_entropy": m1["draft_entropy"],
                "reference_token": m1["reference_token"],
                "rank": m1["rank"], "top1": m1["top1"], "top8": m1["top8"],
                "first_block_reason": tr["oracle"]["reason"],
                "prefix_ids": pref_ids, "prefix_len": len(pref_ids),
            })
        for j in R0_OFFSETS:
            mj = tm_index.get((vid, 0, j))
            if mj is None:
                continue
            tr = trace_index[(vid, 0)]
            pref_ids = ar[:j]
            cands.append({
                "family_source": "round0_offset", "video": vid,
                "round_index": 0, "anchor_rel": j - 1,
                "pos": row["prompt_len"] + j - 1, "root": ar[j - 1],
                "H_valid": None,
                "target_entropy": mj["target_entropy"],
                "draft_entropy": mj["draft_entropy"],
                "reference_token": mj["reference_token"],
                "rank": mj["rank"], "top1": mj["top1"], "top8": mj["top8"],
                "first_block_reason": tr["oracle"]["reason"],
                "prefix_ids": pref_ids, "prefix_len": len(pref_ids),
            })
        for c in cands:
            text = tk.decode(c["prefix_ids"])
            flag, hits = B.has_order_marker(text)
            c["prefix_text"] = text
            c["order_marker"] = flag
            c["order_marker_hits"] = ";".join(hits)
            c["family"] = "C" if flag else "N"
            c["eligible"] = c["target_entropy"] >= ENTROPY_MIN
            c["unit_id"] = (f"{vid}::{c['family_source']}"
                            f"::{c['round_index']}::{c['anchor_rel']}")
        rows.extend(cands)

    # ---- ranking + cap -------------------------------------------------
    units, secondary = [], []
    for vid in B.VIDEOS:
        for fam in ("N", "C"):
            pool = [r for r in rows if r["video"] == vid and r["family"] == fam
                    and r["eligible"]]
            pool.sort(key=lambda r: (-float(r["target_entropy"]),
                                     r["family_source"], r["round_index"],
                                     r["anchor_rel"]))
            for i, r in enumerate(pool):
                r["selection_rank"] = i + 1
                r["selected"] = i < MAX_PER_VIDEO
                r["exclusion_reason"] = (
                    None if i < MAX_PER_VIDEO else
                    f"rank>{MAX_PER_VIDEO} under (entropy desc) in family {fam}")
                if not r["selected"]:
                    continue
                u = {
                    "unit_id": r["unit_id"], "family": fam,
                    "video": vid, "family_source": r["family_source"],
                    "round_index": r["round_index"],
                    "anchor_rel": r["anchor_rel"], "anchor_pos": r["pos"],
                    "root": r["root"], "prefix_len": r["prefix_len"],
                    "prefix_text": r["prefix_text"],
                    "order_marker": r["order_marker"],
                    "order_marker_hits": r["order_marker_hits"],
                    "target_entropy_offset1_C6": r["target_entropy"],
                    "reference_token_offset1_C6": r["reference_token"],
                    "rank_offset1_C6": r["rank"],
                    "top8_offset1_C6": r["top8"],
                    "first_block_reason_C6": r["first_block_reason"],
                }
                (units if fam == "N" else secondary).append(u)
                prefixes[u["unit_id"]] = {
                    "video": vid, "family_source": r["family_source"],
                    "round_index": r["round_index"],
                    "anchor_pos": r["pos"], "anchor_rel": r["anchor_rel"],
                    "root": r["root"],
                    "prompt_ids": prompt_ids_by_video[vid],
                    "prefix_ids": r["prefix_ids"], "prefix_len": len(r["prefix_ids"]),
                    "prefix_text": r["prefix_text"],
                    "reference_token": r["reference_token"],
                }
        # non-eligible candidates get an explicit reason
        for r in rows:
            if r["video"] == vid and not r["eligible"]:
                r["selected"] = False
                r["selection_rank"] = None
                r["exclusion_reason"] = (
                    "prefix asserts temporal order (family C not eligible for "
                    "the main comparison)" if r["order_marker"]
                    else f"target entropy {r['target_entropy']:.3f} < "
                         f"{ENTROPY_MIN} nats")
                if r["family"] == "C" and r["family_source"] == "C6_anchor" \
                        and r["target_entropy"] >= ENTROPY_MIN:
                    pass

    B.write_json(os.path.join(out, "prefixes.json"), prefixes)

    cols = ["video", "family_source", "round_index", "anchor_rel", "pos",
            "root", "H_valid", "target_entropy", "draft_entropy",
            "reference_token", "rank", "top1", "top8", "first_block_reason",
            "order_marker", "order_marker_hits", "family", "eligible",
            "selected", "selection_rank", "exclusion_reason", "prefix_len",
            "unit_id"]
    with open(os.path.join(out, "candidates.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["video"], r["family_source"],
                                             r["round_index"], r["anchor_rel"])):
            w.writerow(r)

    sel = {
        "created": datetime.now().astimezone().isoformat(),
        "entropy_min_nats": ENTROPY_MIN,
        "max_units_per_video": MAX_PER_VIDEO,
        "rule": audit["selection_rule"],
        "family_N_units": units,
        "family_N_count": len(units),
        "family_N_per_video": {v: sum(1 for u in units if u["video"] == v)
                               for v in B.VIDEOS},
        "family_C_units": secondary,
        "family_C_count": len(secondary),
        "family_C_note": (
            "the family C prefixes assert timing, so a full time reversal makes "
            "them false. They are measured and reported separately; their O->T "
            "numbers are not causal evidence about temporal response."),
        "pool_stats": {},
    }
    for vid in B.VIDEOS:
        pool = [r for r in rows if r["video"] == vid
                and r["family_source"] == "C6_anchor"]
        sel["pool_stats"][vid] = {
            "c6_anchors": len(pool),
            "c6_anchors_with_valid_offset1": sum(
                1 for r in pool if (r["H_valid"] or 0) >= 1),
            "c6_anchors_order_neutral": sum(1 for r in pool
                                           if not r["order_marker"]),
            "c6_anchors_H_ge_min": sum(1 for r in pool if r["eligible"]),
        }
    B.write_json(os.path.join(out, "selection.json"), sel)

    print(f"family N units {len(units)}  family C units {len(secondary)}")
    for v in B.VIDEOS:
        us = [u for u in units if u["video"] == v]
        print(f"  N {v:18s} {len(us)}: " + ", ".join(
            f"{u['family_source']}:{u['round_index']}(H={u['target_entropy_offset1_C6']:.2f})"
            for u in us))
    print("pool stats:")
    for v, s in sel["pool_stats"].items():
        print("  ", v, s)
    print("wrote", out)


if __name__ == "__main__":
    main()
