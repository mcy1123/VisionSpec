#!/usr/bin/env python3
"""Stage C (GPU): real O / O_rec / T / K target and C0 draft forward.

For every frozen unit (family N and family C) and every condition this script

  1. re-runs the real Qwen3-VL processor on the condition's clip,
  2. runs the frozen target once over `question + C0 AR prefix` on the decode
     cache path (use_cache=True, explicit M-RoPE position_ids) and keeps the
     kept-layer fused hidden states plus the logits,
  3. runs the frozen C0 head at the unit's anchor with the same fused context
     and the same root token the deployment decoder had,
  4. stores the full-vocabulary fp32 log-softmax rows so the pre-registered
     candidate union can be formed after all conditions exist.

Nothing is trained, no parameter changes, no file outside this run directory is
written.  The target and the head are loaded in float32 exactly as C6 did.

Gates implemented here: G1 (O input hash), G2 (reproduce C6's captured draft
top-8 and target argmax), G3 (identical text ids across conditions),
G5 (O vs O_rec pipeline noise).  G4 lives in `build_conditions.py`.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/repro")
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

import b_common as B  # noqa: E402
from shared import common  # noqa: E402
from historical import opd_fullkl  # noqa: E402

CONDITIONS = ["O", "O_rec", "T", "K"]


def sha_tensor(t: torch.Tensor) -> str:
    return B.sha256_bytes(t.detach().cpu().contiguous().numpy().tobytes())


def topk_rows(logp: torch.Tensor, k: int):
    v, i = torch.topk(logp, k, dim=-1)
    return i.tolist(), v.tolist()


def entropy(logp: torch.Tensor) -> float:
    return float(-(logp.exp() * logp).sum())


def cos(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < B.VEC_NORM_EPS or nb < B.VEC_NORM_EPS:
        return None
    return dot / (na * nb)


def l1(a, b) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def response(cand_tokens, lt_o, ld_o, lt_x, ld_x,
             pt_o=None, pd_o=None, pt_x=None, pd_x=None):
    """Pre-registered response metric on a fixed candidate token set.

    Primary numbers are log-probability deltas (as the task asks).  When the
    probability vectors are supplied the same function also returns the
    probability-space L1 (= 2 x total variation) as a bounded secondary.
    """
    dT = [lt_x[t] - lt_o[t] for t in cand_tokens]
    dD = [ld_x[t] - ld_o[t] for t in cand_tokens]
    L1t = sum(abs(x) for x in dT)
    L1d = sum(abs(x) for x in dD)
    out = {
        "candidate_tokens": list(cand_tokens),
        "L1_target": L1t,
        "L1_draft": L1d,
        "R": None if L1t < B.TARGET_L1_NA else L1d / L1t,
        "R_na_reason": ("target L1 %.6f < %.3f" % (L1t, B.TARGET_L1_NA)
                        if L1t < B.TARGET_L1_NA else None),
        "cos": cos(dD, dT),
        "target_delta": dT,
        "draft_delta": dD,
    }
    if pt_o is not None:
        eT = [pt_x[t] - pt_o[t] for t in cand_tokens]
        eD = [pd_x[t] - pd_o[t] for t in cand_tokens]
        Tt = sum(abs(x) for x in eT)
        Td = sum(abs(x) for x in eD)
        out.update({
            "prob_L1_target": Tt, "prob_L1_draft": Td,
            "prob_tv_target": Tt / 2.0, "prob_tv_draft": Td / 2.0,
            "prob_R": None if Tt < B.TARGET_L1_NA else Td / Tt,
            "prob_cos": cos(eD, eT),
            "prob_target_delta": eT, "prob_draft_delta": eD,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="frozen run dir (stageA/prereg)")
    ap.add_argument("--out", required=True, help="output dir for this pass")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--videos", default=None,
                    help="comma list; default all six frozen videos")
    ap.add_argument("--extra-offsets", type=int, default=8,
                    help="teacher-forced offsets stored beyond offset 1")
    ap.add_argument("--pilot-videos", type=int, default=2,
                    help="abort unless the first N videos pass the model gates")
    a = ap.parse_args()

    run = os.path.abspath(a.run)
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        raise SystemExit(f"output dir exists, refusing to overwrite: {out}")
    os.makedirs(out)

    sel = B.read_json(os.path.join(run, "stageA", "selection.json"))
    conds = B.read_json(os.path.join(run, "conditions", "conditions.json"))

    units = sel["family_N_units"] + sel["family_C_units"]
    videos = B.VIDEOS if a.videos is None else a.videos.split(",")
    for v in videos:
        assert v in B.VIDEOS, v

    device = f"cuda:{a.gpu}"
    calls = {"target_forwards": 0, "head_forwards": 0, "processor_calls": 0}
    t_start = time.perf_counter()

    import pandas as pd
    from transformers import AutoProcessor
    from glance.head import context_feature

    pool = pd.read_parquet(B.FRESH30).set_index("video_name")
    target, _ = common.load_target(device=device, dtype=torch.float32)
    proc = AutoProcessor.from_pretrained(
        "/mnt/local/mcy/GLANCE/models/Qwen3-VL-8B-Instruct")
    head = common.load_head(B.C0_PATH, dtype=torch.float32, device=device).eval()
    head.requires_grad_(False)

    c6_inputs = {v: B.read_json(os.path.join(B.C6_RUN, f"input_{v}.json"))
                 for v in videos}
    baseline = {r["video"]: r for r in B.read_json(B.BASELINE_EVAL)["rows"]}
    tm_index = {}
    for m in B.read_jsonl(os.path.join(B.C6_RUN, "token_metrics_current_ar.jsonl")):
        if m["arm"] == "C0":
            tm_index[(m["video"], m["round_index"], m["offset"])] = m
    trace_index = {}
    for t in B.read_jsonl(os.path.join(B.C6_RUN, "trace_current_ar.jsonl")):
        if t["arm"] == "C0":
            trace_index[(t["video"], t["round_index"])] = t

    gates = {"G1_input_O": {}, "G2_reproduce_C6": [],
             "G2b_round0_implementation": {},
             "G3_text_identity": {},
             "G4_frame_permutation": {}, "G5_pipeline_noise": {},
             "pilot_gate": {"n_pilot_videos": a.pilot_videos, "checked": False}}
    for vid, entry in conds["videos"].items():
        gates["G4_frame_permutation"][vid] = {
            k: {"sha256": v["sha256"],
                "mean_abs_pixel_diff_max": v["mean_abs_pixel_diff_max"],
                "frame_order_argmin_identity": v["frame_order_argmin_identity"]}
            for k, v in entry["conditions"].items()}
    B.write_json(os.path.join(out, "gates_partial.json"), gates)

    records = []
    for vi, vid in enumerate(videos):
        row = pool.loc[vid]
        ar = baseline[vid]["ar_ids"]
        prompt_ids = c6_inputs[vid]["input_ids"]
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        vunits = [u for u in units if u["video"] == vid]
        assert vunits, vid
        cont_len = max(u["anchor_rel"] for u in vunits) + 1 + a.extra_offsets
        cont_len = min(cont_len, len(ar))

        text = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"},
                                          {"type": "text", "text": str(row.question)}]}],
            add_generation_prompt=True, tokenize=False)

        per_cond = {}
        ref_ids = ref_pv_grid = None
        for cond in CONDITIONS:
            path = (str(row.cache_path) if cond == "O"
                    else conds["videos"][vid]["conditions"][cond]["path"])
            ins = proc(text=[text], videos=[path], fps=float(row.fps),
                       return_tensors="pt")
            calls["processor_calls"] += 1
            ids = ins["input_ids"].to(device)
            pv = ins["pixel_values_videos"].to(device)
            vg = ins["video_grid_thw"].to(device)
            grid = vg.cpu().tolist()
            if cond == "O":
                known = c6_inputs[vid]
                g1 = {
                    "input_ids_sha256": sha_tensor(ids[0]),
                    "expected_input_ids_sha256": known["input_ids_sha256"],
                    "input_ids_match": sha_tensor(ids[0]) == known["input_ids_sha256"],
                    "pixel_sha256": sha_tensor(pv),
                    "expected_pixel_sha256": known["pixel_sha256"],
                    "pixel_match": sha_tensor(pv) == known["pixel_sha256"],
                    "grid": grid, "expected_grid": known["grid"],
                    "prompt_len": int(ids.shape[1]),
                }
                gates["G1_input_O"][vid] = g1
                assert g1["input_ids_match"], vid
                assert g1["pixel_match"], vid
                assert grid == known["grid"], vid
                ref_ids, ref_pv_grid = ids, grid
            else:
                same_ids = torch.equal(ids, ref_ids)
                gates["G3_text_identity"].setdefault(vid, {})[cond] = {
                    "input_ids_identical_to_O": same_ids,
                    "grid_identical_to_O": grid == ref_pv_grid,
                }
                assert same_ids, (vid, cond)
                assert grid == ref_pv_grid, (vid, cond)

            full = torch.cat([ids, torch.tensor([ar[:cont_len]], device=device)],
                             dim=1)
            pos_ids = target.model.get_rope_index(full, None, vg, None)[0]
            with torch.inference_mode():
                o = target(full, use_cache=True, past_key_values=DynamicCache(),
                           output_hidden_states=True, logits_to_keep=0,
                           pixel_values_videos=pv, video_grid_thw=vg,
                           position_ids=pos_ids)
                calls["target_forwards"] += 1
                fused = context_feature(o.hidden_states,
                                        common.KEPT_LAYERS).detach()
                logits = o.logits[0]
                del o

                n_need = 1 + a.extra_offsets
                store = {}
                for u in vunits:
                    p = u["anchor_pos"]
                    root = int(full[0, p])
                    tl = logits[p:p + n_need].float()
                    tlp = torch.log_softmax(tl, dim=-1).cpu()
                    q = opd_fullkl.draft_block_logits(
                        head, target, fused[:, :p], anchor_pos=p,
                        root_token=root)
                    calls["head_forwards"] += 1
                    qlp = torch.log_softmax(q[:n_need].float(), dim=-1).cpu()
                    store[u["unit_id"]] = {
                        "tgt_lp": tlp, "drf_lp": qlp, "root": root,
                        "ref_tokens": ar[u["anchor_rel"] + 1:
                                         u["anchor_rel"] + 1 + n_need]}
                    del tl, q

                # --- G2b: reproduce C6's round-0 block, all 15 offsets ------
                # The frozen C6 anchors all sit at a round boundary the real
                # decoder actually reached.  The `round0_offset` units
                # deliberately sit at an earlier boundary (anchor_pos =
                # prompt_len + j - 1), so their draft row cannot equal C6's
                # round-0 offset j: in C6's block only a_1 is inside the noise
                # block, while here a_1..a_j sit in the fused context.  That
                # difference is the point of those units.  This gate therefore
                # verifies the head call itself against C6 at the one anchor
                # both share.
                if cond == "O" and (vid, 0) in trace_index:
                    exp = trace_index[(vid, 0)]["top8_ids"]
                    qr = opd_fullkl.draft_block_logits(
                        head, target, fused[:, :int(ids.shape[1])],
                        anchor_pos=int(ids.shape[1]), root_token=ar[0])
                    calls["head_forwards"] += 1
                    got = [torch.topk(qr[k].float(), B.K_TOP).indices.tolist()
                           for k in range(len(exp))]
                    gates["G2b_round0_implementation"][vid] = {
                        "n_offsets": len(exp),
                        "all_top8_match": bool(got == [list(x) for x in exp]),
                        "n_offsets_matching": sum(
                            1 for k in range(len(exp))
                            if got[k] == list(exp[k])),
                        "first_mismatch_offset": next(
                            (k + 1 for k in range(len(exp))
                             if got[k] != list(exp[k])), None),
                    }
                    del qr, got
            per_cond[cond] = {"store": store,
                              "pixel_sha256": sha_tensor(pv),
                              "input_ids_sha256": sha_tensor(ids[0])}
            del ins, ids, pv, vg, full, pos_ids, fused, logits, store
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  {vid} {cond} done", flush=True)

        # ---------------- metrics ----------------
        for u in vunits:
            uid = u["unit_id"]
            rec = {
                "unit_id": uid, "family": u["family"], "video": vid,
                "family_source": u["family_source"],
                "round_index": u["round_index"], "anchor_rel": u["anchor_rel"],
                "anchor_pos": u["anchor_pos"], "prefix_len": u["prefix_len"],
                "prefix_text": u["prefix_text"],
                "order_marker": u["order_marker"],
                "order_marker_hits": u["order_marker_hits"],
                "target_entropy_offset1_C6": u["target_entropy_offset1_C6"],
                "reference_token_offset1_C6": u["reference_token_offset1_C6"],
                "first_block_reason_C6": u["first_block_reason_C6"],
                "per_condition": {},
            }
            for cond in CONDITIONS:
                st = per_cond[cond]["store"][uid]
                t0v = st["tgt_lp"][0]
                d0v = st["drf_lp"][0]
                ids8_t, val8_t = topk_rows(t0v, B.K_TOP)
                ids8_d, val8_d = topk_rows(d0v, B.K_TOP)
                ref = int(st["ref_tokens"][0])
                rank_d = int((d0v > d0v[ref]).sum().item()) + 1
                rec["per_condition"][cond] = {
                    "target_top8_ids": ids8_t, "target_top8_logp": val8_t,
                    "draft_top8_ids": ids8_d, "draft_top8_logp": val8_d,
                    "target_argmax": int(t0v.argmax()),
                    "target_entropy": entropy(t0v),
                    "draft_entropy": entropy(d0v),
                    "reference_token": ref,
                    "root": st["root"],
                    "reference_in_draft_top8": bool(ref in ids8_d),
                    "reference_draft_rank": rank_d,
                    "reference_target_logp": float(t0v[ref]),
                    "reference_draft_logp": float(d0v[ref]),
                }
            # ---- gate G2 on O ----
            g2 = {"unit_id": uid, "video": vid, "pass": True}
            if u["family_source"] == "C6_anchor":
                tr = trace_index[(vid, u["round_index"])]
                exp8 = tr["top8_ids"][0]
                got8 = rec["per_condition"]["O"]["draft_top8_ids"]
                g2["draft_top8_match"] = bool(exp8 == got8)
                g2["draft_gate"] = "direct_C6_anchor_top8"
                g2["expected_draft_top8"] = exp8
                g2["got_draft_top8"] = got8
                tm1 = tm_index.get((vid, u["round_index"], 1))
            else:
                # round0_offset units sit at anchor_pos = prompt_len + j - 1,
                # a boundary C6 never probed; the draft row is not comparable to
                # C6's round-0 offset j (different noise block). The head call
                # itself is covered by G2b, and the target side is comparable
                # and gated below.
                g2["draft_gate"] = "not_directly_C6_comparable_round_boundary"
                g2["draft_covered_by_G2b"] = True
                tm1 = tm_index.get((vid, 0, u["anchor_rel"] + 1))
            if tm1 is not None:
                g2["target_argmax_match"] = bool(
                    rec["per_condition"]["O"]["target_argmax"]
                    == tm1["target_argmax"])
                g2["expected_target_argmax"] = tm1["target_argmax"]
                g2["target_entropy_C6"] = tm1["target_entropy"]
                g2["target_entropy_recomputed"] = \
                    rec["per_condition"]["O"]["target_entropy"]
                g2["target_entropy_abs_diff"] = abs(
                    tm1["target_entropy"]
                    - rec["per_condition"]["O"]["target_entropy"])
                g2["reference_token"] = tm1["reference_token"]
                g2["reference_draft_rank_C6"] = tm1["rank"]
                g2["reference_draft_rank_recomputed"] = \
                    rec["per_condition"]["O"]["reference_draft_rank"]
            g2["pass"] = bool(g2.get("draft_top8_match", True)
                              and g2.get("target_argmax_match", True))
            gates["G2_reproduce_C6"].append(g2)
            rec["gate_G2"] = g2

            # ---- pre-registered response metric on offset 1 ----
            C_raw = set(rec["per_condition"]["O"]["target_top8_ids"])
            C_raw |= set(rec["per_condition"]["T"]["target_top8_ids"])
            C_raw |= set(rec["per_condition"]["K"]["target_top8_ids"])
            C_raw = sorted(C_raw)
            lps = {cond: (per_cond[cond]["store"][uid]["tgt_lp"][0].numpy(),
                          per_cond[cond]["store"][uid]["drf_lp"][0].numpy())
                   for cond in CONDITIONS}

            def lvec(cond, which, tokens):
                arr = lps[cond][0] if which == "t" else lps[cond][1]
                return {int(c): float(arr[int(c)]) for c in tokens}

            def pvec(cond, which, tokens):
                arr = lps[cond][0] if which == "t" else lps[cond][1]
                return {int(c): float(math.exp(float(arr[int(c)])))
                        for c in tokens}

            # primary candidate set: union of target top-8, minus tokens that
            # stay below the pre-registered probability floor under all three
            # conditions (their log-prob tail otherwise dominates the L1).
            C_eff = [t for t in C_raw
                     if max(pvec(c, "t", [t])[t] for c in ("O", "T", "K"))
                     >= B.PROB_FLOOR]
            rec["candidate_union_raw"] = C_raw
            rec["candidate_eff"] = C_eff
            for cond, key in [("T", "T"), ("K", "K"), ("O_rec", "Orec")]:
                rec["response_" + key] = response(
                    C_eff, lvec("O", "t", C_eff), lvec("O", "d", C_eff),
                    lvec(cond, "t", C_eff), lvec(cond, "d", C_eff),
                    pvec("O", "t", C_eff), pvec("O", "d", C_eff),
                    pvec(cond, "t", C_eff), pvec(cond, "d", C_eff))
                rec["response_raw_" + key] = response(
                    C_raw, lvec("O", "t", C_raw), lvec("O", "d", C_raw),
                    lvec(cond, "t", C_raw), lvec(cond, "d", C_raw),
                    pvec("O", "t", C_raw), pvec("O", "d", C_raw),
                    pvec(cond, "t", C_raw), pvec(cond, "d", C_raw))
            g5 = {
                "unit_id": uid, "family": u["family"],
                "L1_target_O_vs_Orec": rec["response_Orec"]["L1_target"],
                "L1_draft_O_vs_Orec": rec["response_Orec"]["L1_draft"],
                "L1_target_O_vs_T": rec["response_T"]["L1_target"],
                "L1_target_O_vs_K": rec["response_K"]["L1_target"],
                "prob_tv_target_O_vs_Orec":
                    rec["response_Orec"]["prob_tv_target"],
                "prob_tv_target_O_vs_T": rec["response_T"]["prob_tv_target"],
                "prob_tv_target_O_vs_K": rec["response_K"]["prob_tv_target"],
                "noise_below_T": (rec["response_Orec"]["L1_target"]
                                  < rec["response_T"]["L1_target"]),
            }
            gates["G5_pipeline_noise"].setdefault(vid, []).append(g5)
            rec["gate_G5"] = g5

            # ---- exploratory offsets 2..8 ----
            expl = []
            n_avail = min(per_cond[c]["store"][uid]["tgt_lp"].shape[0]
                          for c in CONDITIONS)
            for m in range(2, a.extra_offsets + 1):
                if m > n_avail:
                    break
                Cm = set()
                for cond in ["O", "T", "K"]:
                    Cm |= set(topk_rows(
                        per_cond[cond]["store"][uid]["tgt_lp"][m - 1],
                        B.K_TOP)[0])
                Cm = sorted(Cm)
                e = {"offset": m,
                     "reference_token": int(per_cond["O"]["store"][uid]["ref_tokens"][m - 1])}
                for cond in CONDITIONS:
                    st = per_cond[cond]["store"][uid]
                    e[cond] = {
                        "tgt": lvec(cond, "t", Cm), "drf": lvec(cond, "d", Cm),
                        "tgt_p": pvec(cond, "t", Cm), "drf_p": pvec(cond, "d", Cm),
                        "target_argmax": int(st["tgt_lp"][m - 1].argmax()),
                        "target_entropy": entropy(st["tgt_lp"][m - 1]),
                        "draft_entropy": entropy(st["drf_lp"][m - 1]),
                        "reference_target_logp": float(
                            st["tgt_lp"][m - 1][e["reference_token"]]),
                        "reference_draft_logp": float(
                            st["drf_lp"][m - 1][e["reference_token"]]),
                    }
                for cond, key in [("T", "T"), ("K", "K"), ("O_rec", "Orec")]:
                    e["response_" + key] = response(
                        Cm, e["O"]["tgt"], e["O"]["drf"], e[cond]["tgt"],
                        e[cond]["drf"],
                        e["O"]["tgt_p"], e["O"]["drf_p"], e[cond]["tgt_p"],
                        e[cond]["drf_p"])
                e["candidate_union"] = Cm
                expl.append(e)
            rec["exploratory_offsets"] = expl
            records.append(rec)

        B.write_json(os.path.join(out, "gates_partial.json"), gates)
        print(f"{vid}: {len(vunits)} units done", flush=True)
        del per_cond, lps
        gc.collect()
        torch.cuda.empty_cache()

        if vi + 1 == a.pilot_videos:
            ok = (all(g["pass"] for g in gates["G2_reproduce_C6"])
                  and all(v["input_ids_identical_to_O"]
                          for d in gates["G3_text_identity"].values()
                          for v in d.values())
                  and all(g["all_top8_match"]
                          for g in gates["G2b_round0_implementation"].values())
                  and all(g["input_ids_match"] and g["pixel_match"]
                          for g in gates["G1_input_O"].values()))
            gates["pilot_gate"] = {
                "checked": True, "n_pilot_videos": a.pilot_videos,
                "pass": bool(ok),
                "G2_failures": [g["unit_id"] for g in gates["G2_reproduce_C6"]
                                if not g["pass"]],
                "G2b_failures": [v for v, g in
                                 gates["G2b_round0_implementation"].items()
                                 if not g["all_top8_match"]],
                "videos_done": videos[:vi + 1],
            }
            print("PILOT GATE", gates["pilot_gate"], flush=True)
            if not ok:
                B.write_json(os.path.join(out, "gates.json"), gates)
                raise SystemExit("pilot gate failed; stopping before extension")

    with open(os.path.join(out, "forward_records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")

    gates["G2_summary"] = {
        "n": len(gates["G2_reproduce_C6"]),
        "n_pass": sum(1 for g in gates["G2_reproduce_C6"] if g["pass"]),
        "failed_units": [g["unit_id"] for g in gates["G2_reproduce_C6"]
                         if not g["pass"]],
        "max_target_entropy_abs_diff": max(
            (g.get("target_entropy_abs_diff", 0.0)
             for g in gates["G2_reproduce_C6"]), default=None),
    }
    gates["G1_summary"] = {
        "n": len(gates["G1_input_O"]),
        "all_input_ids_match": all(g["input_ids_match"]
                                   for g in gates["G1_input_O"].values()),
        "all_pixel_match": all(g["pixel_match"]
                               for g in gates["G1_input_O"].values()),
    }
    gates["G3_summary"] = {
        "all_identical": all(v["input_ids_identical_to_O"]
                             for d in gates["G3_text_identity"].values()
                             for v in d.values()),
    }
    gates["G2b_summary"] = {
        "n_videos": len(gates["G2b_round0_implementation"]),
        "n_videos_all_15_offsets_match": sum(
            1 for g in gates["G2b_round0_implementation"].values()
            if g["all_top8_match"]),
        "detail": gates["G2b_round0_implementation"],
    }
    B.write_json(os.path.join(out, "gates.json"), gates)
    calls["elapsed_s"] = time.perf_counter() - t_start
    calls["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    calls["units"] = len(records)
    calls["videos"] = videos
    calls["device"] = device
    calls["created"] = datetime.now().astimezone().isoformat()
    calls["checkpoint_sha256_after"] = B.sha256_file(
        os.path.join(B.C0_PATH, "model.safetensors"))
    calls["checkpoint_unchanged"] = (calls["checkpoint_sha256_after"]
                                     == B.C0_MODEL_SHA)
    assert calls["checkpoint_unchanged"]
    B.write_json(os.path.join(out, "calls.json"), calls)
    print(json.dumps({k: calls[k] for k in
                      ["units", "target_forwards", "head_forwards",
                       "processor_calls", "elapsed_s", "peak_gpu_gib"]}, indent=1))
    print("gates:", gates["G1_summary"], gates["G2_summary"], gates["G3_summary"])
    print("wrote", out)


if __name__ == "__main__":
    main()
