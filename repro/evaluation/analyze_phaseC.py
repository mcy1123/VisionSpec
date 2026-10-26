#!/usr/bin/env python3
"""Phase C seed-0 analysis: paired per-video Delta-tau with video-level bootstrap CI.

Plan section 10.6 asks for:
  * per-video paired Delta-tau, mean, median, positive-video fraction, and a
    paired bootstrap 95% CI **resampling videos** (never tokens -- tokens inside a
    video are not independent);
  * the main comparisons C2-C0 and C2-C1;
  * the lossless gate and timing alongside tau, because a tau gain alone does not
    establish an end-to-end speedup;
  * the pre-registered stop/extend rule: if on the monitoring subset C2 falls
    more than 0.05 mean tau below C0 at two consecutive nodes, pause the extension
    and re-check on the full validation set.

C0 needs no evaluation of its own: it is `head_video439_aligned/step1000`, whose
fp32/b63k8/AR eval already exists from phase A4 under the identical protocol.

No try/except: missing inputs surface.
"""
from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import argparse
import json
import os
import statistics as st

import numpy as np

RES = "/mnt/local/mcy/GLANCE/results"
A4 = f"{RES}/phaseA4/sft439_step1000_fresh30_tree_float32.json"
PHASEC = f"{RES}/phaseC"
MONITOR = 8
FULL = 29


def load(path):
    with open(path) as fh:
        return json.load(fh)


def kept(path):
    doc = load(path)
    return {r["video"]: r for r in doc["rows"] if not r["warmup"]}, doc["rows"]


def bootstrap_ci(deltas, n_boot=10000, seed=0, lo=2.5, hi=97.5):
    """Percentile CI over VIDEO-level resamples of the paired differences."""
    d = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def compare(name, a, b, common=None):
    """a - b on the videos both scored."""
    keys = sorted(set(a) & set(b)) if common is None else sorted(common)
    d = [a[k]["tau"] - b[k]["tau"] for k in keys]
    ci = bootstrap_ci(d)
    t = st.fmean(d) / (st.stdev(d) / len(d) ** 0.5) if len(d) > 1 and st.stdev(d) > 0 else float("nan")
    return {"name": name, "n": len(keys), "mean": st.fmean(d), "median": st.median(d),
            "positive": sum(1 for x in d if x > 0), "ci95": ci, "t": t,
            "per_video": {k: a[k]["tau"] - b[k]["tau"] for k in keys},
            "mean_tau_a": st.fmean(a[k]["tau"] for k in keys),
            "mean_tau_b": st.fmean(b[k]["tau"] for k in keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{PHASEC}/summary_phaseC.json")
    a = ap.parse_args()

    c0_full, c0_all = kept(A4)
    arms = {}
    for tag, path in (("C1@10", f"{PHASEC}/C1_update10_monitor8.json"),
                      ("C2@10", f"{PHASEC}/C2_update10_monitor8.json"),
                      ("C1@25", f"{PHASEC}/C1_update25_monitor8.json"),
                      ("C2@25", f"{PHASEC}/C2_update25_monitor8.json"),
                      ("C1@50", f"{PHASEC}/C1_update50_fresh30.json"),
                      ("C2@50", f"{PHASEC}/C2_update50_fresh30.json")):
        if os.path.exists(path):
            arms[tag] = kept(path)[0]
        else:
            print(f"!! missing {path}")

    # C0 restricted to each protocol's video set
    mon_keys = sorted(arms["C1@10"]) if "C1@10" in arms else []
    c0_mon = {k: c0_full[k] for k in mon_keys if k in c0_full}

    print("=" * 84)
    print("Phase C seed-0 -- video_fresh30, fp32, GLANCE tree b63/k8, AR lossless gate")
    print("=" * 84)

    print("\n--- monitoring subset (8 videos, score-free selection) ---")
    print(f"{'node':<8} {'arm':<5} {'n':>3} {'mean tau':>9} {'median':>8} {'lossless':>10}")
    for tag in ("C1@10", "C2@10", "C1@25", "C2@25"):
        if tag not in arms:
            continue
        d = arms[tag]
        taus = [d[k]["tau"] for k in sorted(d)]
        ll = sum(1 for k in d if d[k].get("lossless"))
        print(f"{tag.split('@')[1]:<8} {tag.split('@')[0]:<5} {len(taus):>3} "
              f"{st.fmean(taus):>9.4f} {st.median(taus):>8.4f} {ll:>4}/{len(taus):<5}")
    if c0_mon:
        taus = [c0_mon[k]["tau"] for k in sorted(c0_mon)]
        print(f"{'0':<8} {'C0':<5} {len(taus):>3} {st.fmean(taus):>9.4f} "
              f"{st.median(taus):>8.4f} {'(from A4)':>10}")

    print("\n--- monitoring-subset paired comparisons (bootstrap unit = video) ---")
    for tag in ("C1@10", "C2@10", "C1@25", "C2@25"):
        if tag in arms and c0_mon:
            r = compare(f"C0 - {tag}", c0_mon, arms[tag])
            print(f"  {tag:<7} vs C0: mean {r['mean']:+.4f}  median {r['median']:+.4f}  "
                  f"95% CI [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]  "
                  f"positive {r['positive']}/{r['n']}")
    for tag in ("C2@10", "C2@25"):
        c1 = tag.replace("C2", "C1")
        if tag in arms and c1 in arms:
            r = compare(f"{tag} - {c1}", arms[tag], arms[c1])
            print(f"  {tag:<7} vs {c1:<7}: mean {r['mean']:+.4f}  median {r['median']:+.4f}  "
                  f"95% CI [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]  "
                  f"positive {r['positive']}/{r['n']}")

    print("\n--- full validation, update 50 (30 rows, row 0 warmup, n=29) ---")
    full_keys = sorted(set(c0_full) & set(arms.get("C1@50", {})) & set(arms.get("C2@50", {})))
    print(f"common videos across C0/C1/C2: {len(full_keys)}")
    rows = {"C0": c0_full}
    for tag in ("C1@50", "C2@50"):
        if tag in arms:
            rows[tag.split("@")[0]] = arms[tag]
    print(f"{'arm':<6} {'n':>3} {'mean tau':>9} {'median':>8} {'EOS':>7} {'lossless':>10}")
    for name, d in rows.items():
        sub = {k: d[k] for k in full_keys}
        taus = [sub[k]["tau"] for k in full_keys]
        ll = sum(1 for k in full_keys if sub[k].get("lossless"))
        eos = sum(1 for k in full_keys if sub[k].get("hit_eos"))
        print(f"{name:<6} {len(taus):>3} {st.fmean(taus):>9.4f} {st.median(taus):>8.4f} "
              f"{eos:>4}/{len(taus):<2} {ll:>4}/{len(taus):<5}")

    print("\n--- full-validation paired comparisons (bootstrap unit = video) ---")
    main_cmp = {}
    for name, x, y in (("C2-C0", "C2", "C0"), ("C2-C1", "C2", "C1"), ("C1-C0", "C1", "C0")):
        if x in rows and y in rows:
            r = compare(name, rows[x], rows[y], common=full_keys)
            main_cmp[name] = r
            print(f"  {name}: mean {r['mean']:+.4f}  median {r['median']:+.4f}  "
                  f"95% CI [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]  t {r['t']:+.2f}  "
                  f"positive {r['positive']}/{r['n']}")

    # --- timing (decode-only, from the eval JSONs) ---
    print("\n--- decode-only timing (median over the common videos) ---")
    for name, d in rows.items():
        sub = [d[k] for k in full_keys]
        sm = st.median(r["spec_decode_ms_per_tok"] for r in sub)
        ar = st.median(r["ar_decode_ms_per_tok"] for r in sub)
        print(f"  {name}: spec {sm:.2f} ms/tok  AR {ar:.2f} ms/tok  "
              f"ratio {ar/sm:.3f}x (decode-only, NOT end-to-end)")

    # --- pre-registered stop / extend rule ---
    print("\n--- pre-registered stop/extend rule (plan 10.6) ---")
    nodes = []
    for tag in ("C2@10", "C2@25"):
        if tag in arms and c0_mon:
            nodes.append(compare(tag, arms[tag], c0_mon)["mean"])
    pause = len(nodes) >= 2 and all(x < -0.05 for x in nodes[:2])
    print(f"  C2-C0 on monitor at nodes 10/25: {[round(x,4) for x in nodes]}")
    print(f"  pause condition (>=2 consecutive nodes worse by >0.05): {pause}")
    if "C2-C0" in main_cmp and "C2-C1" in main_cmp:
        c2_best = main_cmp["C2-C0"]["mean"] > 0 and main_cmp["C2-C1"]["mean"] > 0
        print(f"  extend condition (C2 > C0 AND C2 > C1 at update50): {c2_best}")
        if pause:
            print("  DECISION: PAUSE extension; re-check the latest node on full validation")
        elif c2_best:
            print("  DECISION: EXTEND C1/C2 to update 100 (exploratory, per rule 6)")
        else:
            print("  DECISION: CLOSE OUT -- do not auto-extend or add seeds")

    json.dump({"monitor_n": MONITOR, "full_n": len(full_keys),
               "main_comparisons": main_cmp,
               "monitor_c2_minus_c0_nodes": nodes,
               "pause": bool(pause)},
              open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
