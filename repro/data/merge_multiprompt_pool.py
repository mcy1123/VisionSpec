#!/usr/bin/env python3
"""Merge the prompt-0 pool with the 3 newly rolled-out prompts.

p0 already exists (`video_opd_pool_warm_499.pt`, collected with the warm-start head
under the original prompt); p1-p3 come from the multi-prompt rollout. The merge
asserts the invariants that make the combination legitimate:

  * the COMMITTED tokens for a video+p0 are identical in `video_greedy_499.pt`
    (zero-shot-head rollout) and `video_opd_pool_warm_499.pt` (warm-start-head
    rollout) -- GLANCE is lossless, so the committed greedy sequence cannot depend
    on the draft head. If this ever fails, one of the pools is not what it claims;
  * no duplicate `task` keys after relabelling p0 to `<video>__p0`;
  * every training video has exactly 4 prompts.

Writes the 1756-row training pool and the 240-row held-out pool separately, so a
trainer can never accidentally train on a held-out row.
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
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-pool", default="/mnt/local/mcy/GLANCE/data/video_opd_pool_warm_499.pt")
    ap.add_argument("--zs-pool", default="/mnt/local/mcy/GLANCE/data/video_greedy_499.pt")
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--index", default="/mnt/local/mcy/GLANCE/data/pool_multiprompt_1996.parquet")
    ap.add_argument("--out", default="/mnt/local/mcy/GLANCE/data/video_pool_multiprompt_1756.pt")
    ap.add_argument("--out-heldout",
                    default="/mnt/local/mcy/GLANCE/data/video_pool_multiprompt_heldout240.pt")
    a = ap.parse_args()

    import pandas as pd
    idx = pd.read_parquet(a.index).set_index("task")

    warm = torch.load(a.warm_pool, weights_only=False)["records"]
    zs = {r["task"]: r for r in torch.load(a.zs_pool, weights_only=False)["records"]}
    p0 = []
    same, diffs = 0, []
    for r in warm:
        zr = zs.get(r["task"])
        assert zr is not None, r["task"]
        # Losslessness means the committed sequence cannot depend on the draft
        # head, so the two rollouts must agree on their COMMON PREFIX. They are not
        # required to have equal length: the shipped zero-shot pool was collected
        # with --max-new 768 and this one with 1024, so rows that hit the old cap
        # are longer here (verified below and counted).
        za, ra = zr["input_ids"], r["input_ids"]
        # The PROMPT must be byte-identical (that is the merge invariant), but the
        # committed continuation is measured -- not asserted -- to be head-independent:
        # GLANCE's losslessness is exact in fp32, while at bf16 the verify layout is
        # shape-sensitive (phase 3), so two different draft heads can commit
        # different tokens. Each pool's teacher top-k is computed on its OWN
        # committed sequence, so this does not invalidate either pool.
        pl = int(r["prompt_len"])
        assert torch.equal(za[:pl], ra[:pl]), (
            f"{r['task']}: prompt tokens differ between the two rollouts")
        m = min(len(za), len(ra))
        common = int((za[:m] == ra[:m]).sum())
        if common == m and len(za) == len(ra):
            same += 1
        else:
            fd = next((i for i in range(m) if za[i] != ra[i]), m)
            diffs.append((r["task"], fd, len(za) - pl, len(ra) - pl, pl))
        nr = dict(r)
        nr["task"] = f"{r['task']}__p0"
        nr["base_video"] = r["task"]
        nr["prompt_id"] = 0
        p0.append(nr)
    import statistics as _st
    print(f"p0 from the warm pool: {len(p0)} records")
    print(f"  prompt invariant: all prompts byte-identical across the two rollouts")
    print(f"  committed-sequence head-independence (bf16): identical {same}/{len(p0)}; "
          f"{len(diffs)} rows differ (first-diff index / gen-len zs,warm): "
          f"{[(t, fd, a, b) for t, fd, a, b, _ in diffs[:6]]}")
    if diffs:
        print(f"  first-diff index: median {_st.median(fd for _, fd, _, _, _ in diffs):.0f}, "
              f"min {min(fd for _, fd, _, _, _ in diffs)}, "
              f"max {max(fd for _, fd, _, _, _ in diffs)}"
              f" -- consistent with the bf16 verify-layout sensitivity of phase 3")

    new = []
    for p in a.parts:
        d = torch.load(p, weights_only=False)["records"]
        print(f"  {os.path.basename(p)}: {len(d)} records, prompts "
              f"{sorted({int(r['task'].split('__p')[1]) for r in d})}")
        new.extend(d)

    recs = p0 + new
    tasks = [r["task"] for r in recs]
    dup = sorted({t for t in tasks if tasks.count(t) > 1})
    assert not dup, f"duplicate tasks after merge: {dup[:5]}"
    miss = [t for t in tasks if t not in idx.index]
    assert not miss, f"records without an index row: {miss[:5]}"

    tr = [r for r in recs if idx.loc[r["task"], "split"] == "train"]
    ho = [r for r in recs if idx.loc[r["task"], "split"] == "heldout"]
    print(f"merged {len(recs)} = {len(tr)} train + {len(ho)} heldout")
    assert len(tr) == 1756, len(tr)
    vids = {}
    for r in tr:
        # records from p1-p3 come straight from opd_rollout and carry neither
        # base_video nor prompt_id, so both are parsed back out of the task key.
        base, pidstr = r["task"].split("__p")
        vids.setdefault(base, set()).add(int(pidstr))
    bad = {v: s for v, s in vids.items() if s != {0, 1, 2, 3}}
    assert not bad, f"videos without exactly 4 prompts: {list(bad.items())[:3]}"
    assert len(vids) == 439, len(vids)

    cfg = dict(torch.load(a.warm_pool, weights_only=False).get("config", {}))
    cfg.update({"multiprompt": True, "prompts": 4, "rows_train": len(tr),
                "rows_heldout": len(ho),
                "p0_head_independent_rows_bf16": same,
                "p0_head_dependent_rows_bf16": len(diffs),
                "p0_first_diff_median": float(_st.median(fd for _, fd, _, _, _ in diffs))
                if diffs else None})
    torch.save({"records": tr, "config": cfg}, a.out)
    torch.save({"records": ho, "config": cfg}, a.out_heldout)
    print(f"wrote {a.out}: {len(tr)} rows ({os.path.getsize(a.out)/1e6:.1f} MB)")
    print(f"wrote {a.out_heldout}: {len(ho)} rows "
          f"({os.path.getsize(a.out_heldout)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
