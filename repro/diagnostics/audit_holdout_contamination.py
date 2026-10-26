#!/usr/bin/env python3
"""Audit which recorded results were scored on rows their head had trained on.

Background. `pool_499.parquet` splits 499 videos into 439 train + 60 heldout, and
`data/video_heldout_30.parquet` is a 30-video subset of the **train** side (it was
the OPD pipeline's validation split: 409 train + 30 val = 439). Every video head
in this project except the image-side zero-shot head descends, by training or by
warm start, from a run over those 439 videos — so for those heads the 30 videos in
`video_heldout_30` are TRAINING data, and any number scored on them is a
memorisation estimate, not a generalisation estimate.

This script resolves, for every result JSON under the results tree:
  * which head it evaluated,
  * which video set it scored,
  * and whether that set intersects the training videos of that head or of any
    head in its warm-start ancestry.

The ancestry matters: `head_opd_video409` was trained only on 409 videos (with the
30 genuinely excluded), but it was warm-started from `head_video439/step1000`,
which saw all 439 — so information about those 30 videos is in its initial
weights and the contamination is inherited, not absent.

No try/except swallowing: an unreadable file is reported, not skipped.
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
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common

DATA = common.DATA


def vids(path):
    df = pd.read_parquet(path)
    return set(df["video_name"].astype(str))


def split_of(path):
    df = pd.read_parquet(path)
    return {s: set(g["video_name"].astype(str)) for s, g in df.groupby("split")}


# ---------------------------------------------------------------------------
# Provenance: each head -> (videos it trained on, warm-start ancestor or None)
# Determined from the training logs (`train rows: N (of M records)`) and from the
# recipe docstrings; see the audit table in ViSpec判定_日期.md section 1.
# ---------------------------------------------------------------------------
def build_provenance():
    p150 = split_of(os.path.join(DATA, "pool_150.parquet"))
    p499 = split_of(os.path.join(DATA, "pool_499.parquet"))
    h30 = vids(os.path.join(DATA, "video_heldout_30.parquet"))
    f30 = vids(os.path.join(DATA, "video_fresh30.parquet"))
    t409 = vids(os.path.join(DATA, "video_opd_train_409.parquet"))
    prov = {
        # image-side DFlash head: no video training at all
        "head": (set(), None),
        "gate": (set(), None),
        # log 72 had the split-filter bug (trained on all 150 of a 120/30 pool)
        "head_video_sft": (p150["train"] | p150["heldout"], None),
        "head_video120": (p150["train"], None),
        # 439 = all of pool_499's train side, which CONTAINS video_heldout_30
        "head_video439": (p499["train"], None),
        "head_video439_h8": (p499["train"], "head_video439"),
        "head_video439_h12": (p499["train"], "head_video439"),
        "head_h8_init": (set(), "head_video439"),
        "head_h12_init": (set(), "head_video439"),
        # multiprompt: 439 train videos x 4 prompts
        "head_video_mp": (p499["train"], None),
        "head_opd_video409": (t409, "head_video439"),
        "head_opd_video409_lr1e4": (t409, "head_video439"),
        "head_opd_fix_aligned": (t409, "head_video439"),
        "head_opd_fix_anchors16": (t409, "head_video439"),
        "head_opd_fix_revkl": (t409, "head_video439"),
        "head_opd_cap": (t409, "head_video439"),
        "head_opd_mp": (t409, "head_video_mp"),
    }
    return prov, {"heldout30": h30, "fresh30": f30, "pool499_train": p499["train"],
                  "pool499_heldout": p499["heldout"], "pool150_train": p150["train"],
                  "opd409": t409}


def head_of(cfg):
    h = cfg.get("head") or cfg.get("init_from") or ""
    return os.path.basename(os.path.dirname(str(h))) if str(h).endswith(tuple(
        f"step{i}" for i in range(0, 20001, 250))) else os.path.basename(str(h))


def eval_set_of(cfg, facts):
    """Video set a result was scored on, or None if it cannot be resolved."""
    rows = cfg.get("rows")
    if rows:
        p = str(rows)
        if not os.path.exists(p):
            return None, f"missing parquet {p}"
        return vids(p), os.path.basename(p)
    pool = cfg.get("pool")
    if pool and os.path.exists(str(pool)):
        s = split_of(str(pool))
        want = s.get(cfg.get("split", ""), set())
        rt = cfg.get("restrict_to")
        if rt and os.path.exists(str(rt)):
            want = want & vids(str(rt))
        return want, f"{os.path.basename(str(pool))}:{cfg.get('split')}" + (
            f"∩{os.path.basename(str(rt))}" if rt else "")
    return None, "unresolved"


def contaminated(head, ev, prov):
    """Videos in `ev` that `head` or any warm-start ancestor trained on."""
    seen, bad, chain = set(), set(), []
    h = head
    while h and h not in seen:
        seen.add(h)
        if h not in prov:
            chain.append(f"{h}(unknown)")
            break
        trained, parent = prov[h]
        bad |= (ev & trained)
        chain.append(h)
        h = parent
    return bad, chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(common.ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(common.ROOT, "results", "diag",
                                                  "holdout_audit.json"))
    a = ap.parse_args()

    prov, facts = build_provenance()
    print("=== known video sets ===")
    for k, v in facts.items():
        print(f"  {k:16s} n={len(v)}")
    print(f"\n  heldout30 in pool499 train? {len(facts['heldout30'] & facts['pool499_train'])}/30")
    print(f"  heldout30 in pool499 heldout? {len(facts['heldout30'] & facts['pool499_heldout'])}/30")
    print(f"  fresh30  in pool499 train?   {len(facts['fresh30'] & facts['pool499_train'])}/30")
    print(f"  fresh30  in pool499 heldout? {len(facts['fresh30'] & facts['pool499_heldout'])}/30")

    files = sorted(glob.glob(os.path.join(a.results, "**", "*.json"), recursive=True))
    rows, unresolved = [], []
    for f in files:
        with open(f) as fh:
            try:
                d = json.load(fh)
            except json.JSONDecodeError:
                unresolved.append((os.path.relpath(f, a.results), "not JSON"))
                continue
        cfg = d.get("config") if isinstance(d, dict) else None
        if not isinstance(cfg, dict) or "head" not in cfg and "pool" not in cfg and "rows" not in cfg:
            continue
        head = head_of(cfg)
        ev, label = eval_set_of(cfg, facts)
        if ev is None:
            unresolved.append((os.path.relpath(f, a.results), label))
            continue
        nres = len(d.get("rows", []) or [])
        if nres == 0:
            continue
        bad, chain = contaminated(head, ev, prov)
        rows.append({
            "file": os.path.relpath(f, a.results), "head": head, "eval_set": label,
            "n_rows": nres, "n_eval_videos": len(ev),
            "n_leaked": len(bad), "leaked_frac": len(bad) / max(len(ev), 1),
            "ancestry": " <- ".join(chain),
            "verdict": ("CLEAN" if not bad else
                        ("CONTAMINATED (trained)" if head in prov and bad & prov[head][0]
                         else "CONTAMINATED (inherited via warm start)")),
        })

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"facts": {k: sorted(v) for k, v in facts.items()},
               "provenance": {k: [sorted(v[0]), v[1]] for k, v in prov.items()},
               "results": rows, "unresolved": unresolved}, open(a.out, "w"), indent=1)

    bad = [r for r in rows if r["verdict"] != "CLEAN"]
    print(f"\n=== {len(rows)} scored results, {len(bad)} with leakage ===")
    print(f"{'file':44s} {'head':24s} {'eval_set':30s} {'n':>3s} {'leak':>5s}  verdict")
    for r in sorted(rows, key=lambda r: (r["verdict"] == "CLEAN", r["file"])):
        print(f"{r['file'][:44]:44s} {r['head'][:24]:24s} {r['eval_set'][:30]:30s} "
              f"{r['n_rows']:3d} {r['n_leaked']:5d}  {r['verdict']}")
    if unresolved:
        print(f"\n=== {len(unresolved)} unresolved (not silently dropped) ===")
        for f, why in unresolved[:25]:
            print(f"  {f}: {why}")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
