#!/usr/bin/env python3
"""Fixed-anchor candidate and tree coverage probe (the "why" behind online tau).

For every dev record, on the frozen correct AR prefix, at a fixed set of anchors
chosen by the same pure `qa_anchors.select_anchors` rule the trainers use, this
measures what the DEPLOYMENT draft actually proposed and what the DEPLOYMENT tree
did with it:

  top1 / top8 membership of the correct token at the anchor's first offset;
  ``L_raw``   consecutive offsets whose correct next token is inside top-8;
  ``L_tree``  consecutive offsets whose correct next token the *budget-63 tree*
              actually contains, walked from the root as a real path;
  ``L_actual`` the accepted length of the deployment tree under the frozen
              target's own greedy walk (so a tree bug and a target-disagreement
              are distinguishable);
  ``first_blocker`` one of ``not_in_topk`` (the candidate never existed),
              ``in_topk_not_in_tree`` (it existed but the budget did not buy it),
              ``in_tree`` (the tree had it and the target walked elsewhere),
              ``past_block`` (the block ran out);
  ``phi``     sum over nodes of the path probability of the tree, the objective
              the budget-63 best-first selection actually optimises.

It is a FIXED-ANCHOR probe, not a replay of an online round: the deployment feeds
the head only the newly consumed context rows of a round, while this builds the
fresh-cache call at the same anchor. The project measured that those can differ in
the last bf16 bits, so these numbers must never be pooled with an arm's own online
tau. Every row records the anchor and the correct-path length so the two stay
separable.

The tree is rebuilt a second time from the captured candidate ids/scores and the
two node lists are asserted identical, so a silent tree-construction change in
this probe cannot pass as a measurement.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path[:0] = [str(HERE), str(REPRO), str(REPRO / "qa_opd_fair"),
                str(ROOT / "GLANCE_repo"),
                str(ROOT / "scripts" / "dflash_pkg" / "dflash-0.1.0")]

from shared import common  # noqa: E402
from historical import opd_fullkl  # noqa: E402
import qa_anchors  # noqa: E402
from metrics_contract import classify_lengths
from imageencode import prefix_image_vision_positions  # noqa: E402
from train_sqa_sft import draft_block_logits_mixed  # noqa: E402

import glance  # noqa: E402
from transformers import DynamicCache  # noqa: E402


def load_split(run, split, limit=None):
    rows = []
    with open(pathlib.Path(run) / "dataset" / "records_manifest.csv") as fh:
        for r in csv.DictReader(fh):
            if r["canonical_split"] == split:
                rows.append(r)
    rows.sort(key=lambda r: r["official_row_index"])
    return rows[:limit] if limit else rows


def cache_prefix(prefix_kv, n):
    return DynamicCache(ddp_cache_data=[
        (l.keys[..., :n, :], l.values[..., :n, :]) for l in prefix_kv.layers])


def tree_nodes_key(nodes):
    return [(n.token, n.depth, n.parent, n.rank) for n in nodes]


def phi_of(nodes, top_ids, top_logp):
    """Sum over tree nodes of the product of the path's offset log-probs."""
    total = 0.0
    for i, n in enumerate(nodes):
        lp = 0.0
        cur, d = n, n.depth
        while cur is not None and d >= 1:
            lp += float(top_logp[d - 1, cur.rank])
            cur = nodes[cur.parent] if cur.parent >= 0 else None
            d -= 1
        total += math.exp(lp)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--budget", type=int, default=63)
    ap.add_argument("--branch-k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-pixels", type=int, default=768)
    ap.add_argument("--anchors", type=int, default=3,
                    help="anchors per record, chosen by the frozen "
                         "qa_anchors.select_anchors rule over the correct AR "
                         "answer positions")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--head-dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--ar-cache", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    torch.cuda.set_device(a.gpu)
    torch.set_grad_enabled(False)

    head = common.load_head(a.head, dtype=getattr(torch, a.head_dtype), device=dev).eval()
    target, proc = common.load_target(device=dev, dtype=torch.float32)
    head_dtype = next(head.parameters()).dtype
    block = int(head.block_size)
    eos = int(proc.tokenizer.eos_token_id)
    rows = load_split(a.run, a.split, a.limit or None)
    ar_path = a.ar_cache or (pathlib.Path(a.run) / "eval" / f"ar_{a.split}.pt")
    blob = torch.load(ar_path, weights_only=False, map_location="cpu")
    ar = {r["pid"]: r for r in blob["records"]}
    print(f"[cov] {a.tag} {a.split}: {len(rows)} records, {a.anchors} anchors each, "
          f"budget {a.budget} branch {a.branch_k}, head {a.head_dtype}", flush=True)

    out, t_all, n_tree_ok = [], time.time(), 0
    for i, r in enumerate(rows):
        rec = ar.get(r["pid"])
        if rec is None:
            continue
        img = Image.open(r["image_path"]).convert("RGB")
        if a.max_pixels and max(img.size) > a.max_pixels:
            w, h = img.size
            s = a.max_pixels / max(w, h)
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
        ids, pixels, grid = common.encode(proc, r["prompt"], img)
        ids = ids.to(dev)
        pix = pixels.to(dev) if pixels is not None else None
        g = grid.to(dev) if grid is not None else None
        prompt_len = int(ids.shape[1])
        full = torch.cat([ids[0], torch.tensor(rec["tokens"], dtype=torch.long,
                                              device=dev)])
        n_ans = len(rec["tokens"])
        toks = rec["tokens"]
        eos_pos = (toks.index(eos) if eos in toks else None)
        vispos = prefix_image_vision_positions(target, full[None], pix, g)
        fused, prefix_kv = opd_fullkl.target_prefill_shared(
            target, full[None], max_seq_len=common.MAX_SEQ_LEN, vispos=vispos)
        if fused.dtype != head_dtype:
            fused = fused.to(head_dtype)

        cand = qa_anchors.sft_candidate_positions(prompt_len, eos_pos, n_ans)
        anchors = qa_anchors.select_anchors(cand, a.anchors)
        for p in anchors:
            if p + 1 >= int(full.shape[0]):
                continue
            q_logits = draft_block_logits_mixed(
                head, target, fused[:, :p], anchor_pos=p,
                root_token=int(full[p]), head_dtype=head_dtype)
            top = torch.topk(F.log_softmax(q_logits.float(), dim=-1),
                             a.branch_k, dim=-1)
            top_ids, top_logp = top.indices.cpu(), top.values.cpu()
            nodes = glance.build_budget_tree(top_ids, top_logp, a.budget, a.branch_k)
            # independent rebuild: the probe's own tree must be reproducible
            nodes2 = glance.build_budget_tree(top_ids, top_logp, a.budget, a.branch_k)
            if tree_nodes_key(nodes) != tree_nodes_key(nodes2):
                raise RuntimeError(f"{r['pid']}@{p}: tree rebuild differs")
            n_tree_ok += 1

            horizon = min(block - 1, int(full.shape[0]) - 1 - p)
            l_raw, l_tree, blocker, blocker_depth = 0, 0, None, None
            for k in range(1, horizon + 1):
                want = int(full[p + k])
                row_ids = top_ids[k - 1].tolist()
                if want in row_ids and k == l_raw + 1:
                    l_raw += 1
                elif blocker is None and want not in row_ids:
                    blocker, blocker_depth = "not_in_topk", k
            children = glance.children_of(nodes)
            cursor = -1
            for k in range(1, horizon + 1):
                want = int(full[p + k])
                hit = next((c for c in children.get(cursor, [])
                            if nodes[c].token == want), None)
                if hit is None:
                    if blocker is None:
                        in_topk = want in top_ids[k - 1].tolist()
                        blocker = "in_topk_not_in_tree" if in_topk else "not_in_topk"
                        blocker_depth = k
                    break
                l_tree += 1
                cursor = hit
            blocker, blocker_depth = classify_lengths(l_raw, l_tree, horizon)

            # the deployment verification of THIS tree under the frozen target
            n_prefix = p
            packed = torch.tensor([[int(full[p])] + [n.token for n in nodes]],
                                  device=dev)
            packed_pos = torch.tensor([[p] + [p + n.depth for n in nodes]],
                                      device=dev)
            mask = glance.tree_mask(n_prefix, nodes, dev,
                                    next(target.parameters()).dtype)
            outv = target(packed, position_ids=vispos(packed_pos),
                          past_key_values=cache_prefix(prefix_kv, n_prefix),
                          use_cache=True, attention_mask=mask)
            packed_logits = outv.logits[0]
            ch = glance.children_of(nodes)
            path, cur = [], -1
            while len(path) < horizon:
                token = int(packed_logits[0 if cur == -1 else cur + 1].argmax())
                hit = next((c for c in ch.get(cur, []) if nodes[c].token == token), None)
                if hit is None:
                    break
                path.append(hit)
                cur = hit
                if token == eos:
                    break
            l_packed_bounded = len(path)
            l_correct_actual = 0
            cur = -1
            for k in range(1, horizon + 1):
                want = int(full[p + k])
                hit = next((c for c in ch.get(cur, []) if nodes[c].token == want), None)
                if hit is None or hit not in path:
                    break
                l_correct_actual += 1
                cur = hit

            if l_packed_bounded != l_correct_actual:
                raise RuntimeError("packed teacher path differs from frozen AR within valid horizon")
            l_actual = l_correct_actual
            if not 0 <= l_actual <= l_tree <= l_raw <= horizon:
                raise RuntimeError("local coverage bounds violated")

            out.append({
                "tag": a.tag, "split": a.split, "pid": r["pid"],
                "image_group_id": r["image_group_id"], "subject": r["subject"],
                "anchor": p, "prompt_len": prompt_len, "n_ans": n_ans,
                "horizon": horizon, "boundary_anchor": horizon < block - 1,
                "correct_top1_first": int(top_ids[0, 0]) == int(full[p + 1]),
                "correct_top8_first": int(full[p + 1]) in top_ids[0].tolist(),
                "correct_rank_first": (int((top_ids[0] == int(full[p + 1]))
                                           .nonzero()[0]) + 1
                                       if int(full[p + 1]) in top_ids[0].tolist()
                                       else None),
                "top1_top8_logp_margin": float(top_logp[0, 0] - top_logp[0, -1]),
                "L_raw": l_raw, "L_tree": l_tree, "L_actual": l_actual,
                "L_correct_in_actual_path": l_correct_actual,
                "L_packed_verified_path_bounded": l_packed_bounded,
                "tree_gap": l_raw - l_tree,
                "first_blocker": blocker, "first_blocker_depth": blocker_depth,
                "tree_nodes": len(nodes), "phi": phi_of(nodes, top_ids, top_logp),
                "n_correct_in_tree": l_tree, "horizon_correct": horizon,
                "answer_in_block": bool(n_ans and p + horizon >= prompt_len + n_ans),
            })
            del q_logits, outv, packed_logits
            torch.cuda.empty_cache()
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[cov] {i+1}/{len(rows)} {r['pid']} anchors {len(anchors)} "
                  f"elapsed {(time.time()-t_all)/60:.1f}min", flush=True)
        del ids, pix, g, full, vispos, fused, prefix_kv
        torch.cuda.empty_cache()

    def agg(sel):
        if not sel:
            return {}
        return {
            "n": len(sel),
            "top1_first": float(np.mean([x["correct_top1_first"] for x in sel])),
            "top8_first": float(np.mean([x["correct_top8_first"] for x in sel])),
            "L_raw": float(np.mean([x["L_raw"] for x in sel])),
            "L_tree": float(np.mean([x["L_tree"] for x in sel])),
            "L_actual": float(np.mean([x["L_actual"] for x in sel])),
            "L_correct_in_actual_path": float(np.mean(
                [x["L_correct_in_actual_path"] for x in sel])),
            "tree_gap": float(np.mean([x["tree_gap"] for x in sel])),
            "phi": float(np.exp(np.mean([math.log(max(x["phi"], 1e-12)) for x in sel]))),
            "first_blocker": dict((k, sum(1 for x in sel if x["first_blocker"] == k))
                                  for k in sorted({x["first_blocker"] for x in sel})),
        }

    groups = {}
    for x in out:
        groups.setdefault(x["image_group_id"], []).append(x)
    per_group = {g: {"L_tree": float(np.mean([x["L_tree"] for x in v])),
                     "L_raw": float(np.mean([x["L_raw"] for x in v])),
                     "L_actual": float(np.mean([x["L_actual"] for x in v])),
                     "top8_first": float(np.mean([x["correct_top8_first"] for x in v])),
                     "n_anchors": len(v), "subject": v[0]["subject"]}
                 for g, v in groups.items()}
    summary = {
        "tag": a.tag, "split": a.split, "head": a.head, "head_dtype": a.head_dtype,
        "target_dtype": "float32", "budget": a.budget, "branch_k": a.branch_k,
        "anchors_per_record": a.anchors, "max_pixels": a.max_pixels,
        "n_records": len(rows), "n_anchors": len(out), "n_image_groups": len(groups),
        "tree_rebuild_checks_passed": n_tree_ok,
        "overall": agg(out),
        "by_subject": {s: agg([x for x in out if x["subject"] == s])
                       for s in sorted({x["subject"] for x in out})},
        "per_image_group": per_group,
        "anchors": out,
        "note": ("FIXED-ANCHOR probe on the frozen correct AR prefix. Not a replay "
                 "of an online round: the deployment feeds the head only a round's "
                 "newly consumed rows while this uses the fresh-cache call at the "
                 "same anchor, and those can differ in the last bf16 bits. These "
                 "numbers must not be pooled with any arm's own online tau."),
        "wall_minutes": (time.time() - t_all) / 60,
    }
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=2)
    csvp = str(a.out).rsplit(".", 1)[0] + "_anchors.csv"
    with open(csvp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("anchors", "per_image_group", "by_subject")},
                     indent=2), flush=True)
    print(json.dumps(summary["overall"], indent=2), flush=True)
    print("wrote", a.out, csvp, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
