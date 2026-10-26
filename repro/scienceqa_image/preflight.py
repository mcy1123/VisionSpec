#!/usr/bin/env python3
"""Phase C preflight: image binding, visible-reasoning length pilot, AR losslessness.

Three gates, all on the frozen train records, none of which may be skipped:

1. ``image_processor_binding.json``
   The picture must be a real Qwen3-VL image object, and it must *matter*: the
   fused hidden states -- the tensor the draft head consumes -- have to change
   when the picture changes and when the pixels are zeroed, and that change must
   reach the generated continuation. The first post-prompt token is recorded but
   deliberately NOT required to move: on real ScienceQA rows it is often a
   low-content function word whose argmax is image-independent even though the
   fused hidden moves by ~2e1.

2. ``length_pilot.json``
   32 train records, target-only greedy AR at ``max_new_tokens=512``. Reports
   p10/p50/p90, natural-EOS rate, the 256/512 truncation rates and whether an
   explanation segment appears. The pre-registered rule then fixes the formal
   ``max_new_tokens``.

3. ``ar_lossless_gate.json``
   ``SPEC == independent greedy AR`` on token ids, including EOS, bonus EOS, root
   EOS and the cap. Comparing decoded text is explicitly not enough.

Gates can be run selectively with ``--only``, so a fix in a later gate does not
re-pay for the earlier ones. Nothing here touches the test split.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path[:0] = [str(HERE), str(REPRO), str(ROOT / "GLANCE_repo"),
                str(ROOT / "scripts" / "dflash_pkg" / "dflash-0.1.0")]

from shared import common  # noqa: E402
import glance  # noqa: E402
from glance.vl import VisionPositions  # noqa: E402


def load_split(run, split, limit=None):
    rows = []
    with open(pathlib.Path(run) / "dataset" / "records_manifest.csv") as fh:
        for r in csv.DictReader(fh):
            if r["canonical_split"] == split:
                rows.append(r)
    rows.sort(key=lambda r: r["official_row_index"])
    return rows[:limit] if limit else rows


class Env:
    """Everything the gates share: the loaded models and one prepare() helper."""

    def __init__(self, a):
        self.a = a
        self.dev = f"cuda:{a.gpu}"
        torch.cuda.set_device(a.gpu)
        self.target, self.proc = common.load_target(device=self.dev,
                                                   dtype=torch.float32)
        self.eos = int(self.proc.tokenizer.eos_token_id)
        self.stop = [self.eos]

    def prepare(self, path, prompt, max_pixels):
        img = Image.open(path).convert("RGB")
        if max_pixels and max(img.size) > max_pixels:
            w, h = img.size
            s = max_pixels / max(w, h)
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
        ids, pixels, grid = common.encode(self.proc, prompt, img)
        ids = ids.to(self.dev)
        pix = pixels.to(self.dev) if pixels is not None else None
        g = grid.to(self.dev) if grid is not None else None
        pos = VisionPositions(pix, g)
        pos.bind(self.target, ids)
        return ids, pix, g, pos, img


def gate1(env, run, pre, cont_new):
    a = env.a
    print("[gate1] image -> fused hidden binding", flush=True)
    train = load_split(run, "train")
    probe = train[:3]
    binding = {"rows": [], "target_dtype": "float32",
               "max_pixels": a.max_pixels, "continuation_len": cont_new}
    for i, r in enumerate(probe):
        other = train[(i + 7) % len(train)]
        ids_a, pix_a, grid_a, pos_a, _ = env.prepare(r["image_path"], r["prompt"],
                                                    a.max_pixels)
        ids_b, pix_b, grid_b, pos_b, _ = env.prepare(other["image_path"], r["prompt"],
                                                     a.max_pixels)
        pos_z = VisionPositions(torch.zeros_like(pix_a), grid_a)
        pos_z.bind(env.target, ids_a)
        with torch.no_grad():
            oa = env.target(ids_a, past_key_values=None, use_cache=True,
                            logits_to_keep=1, output_hidden_states=True,
                            **pos_a.prefill_inputs(ids_a, None))
            ob = env.target(ids_b, past_key_values=None, use_cache=True,
                            logits_to_keep=1, output_hidden_states=True,
                            **pos_b.prefill_inputs(ids_b, None))
            oz = env.target(ids_a, past_key_values=None, use_cache=True,
                            logits_to_keep=1, output_hidden_states=True,
                            **pos_z.prefill_inputs(ids_a, None))
            fa, fb, fz = (common.context_feature(x.hidden_states)
                          for x in (oa, ob, oz))
        la, lb, lz = oa.logits[0, -1], ob.logits[0, -1], oz.logits[0, -1]
        ncom = min(fa.shape[1], fb.shape[1])
        rec = {
            "pid": r["pid"], "image_path": r["image_path"],
            "image_sha256": r["image_sha256"], "swapped_with_pid": other["pid"],
            "prompt_sha256": r["prompt_sha256"],
            "n_prompt_tokens": int(ids_a.shape[1]),
            "n_prompt_tokens_swapped_image": int(ids_b.shape[1]),
            "image_grid_thw": grid_a.tolist(),
            "image_grid_thw_swapped": grid_b.tolist(),
            "fused_hidden_shape": list(fa.shape),
            "position_ids_shape": list(pos_a._prefill_positions.shape),
            "rope_delta": int(pos_a._delta),
            "max_abs_fused_diff_last_row_other_image":
                float((fa[0, -1] - fb[0, -1]).abs().max()),
            "max_abs_fused_diff_last_row_zeroed_pixels":
                float((fa[0, -1] - fz[0, -1]).abs().max()),
            "max_abs_fused_diff_common_prefix_other_image":
                float((fa[0, :ncom] - fb[0, :ncom]).abs().max()),
            "max_abs_fused_diff_common_prefix_zeroed_pixels":
                float((fa[0, :ncom] - fz[0, :ncom]).abs().max()),
            "prefill_argmax_image_a_image_b_zeroed":
                [int(la.argmax()), int(lb.argmax()), int(lz.argmax())],
            "prefill_argmax_changes_with_image":
                int(la.argmax()) != int(lb.argmax()),
            "prefill_max_logit_gap_image_vs_other_image": float((la - lb).abs().max()),
            "prefill_max_logit_gap_image_vs_zeroed": float((la - lz).abs().max()),
        }
        del oa, ob, oz, fa, fb, fz
        torch.cuda.empty_cache()
        seq_a = glance.greedy_generate(env.target, ids_a, cont_new, env.stop,
                                      positions=pos_a)[0, ids_a.shape[1]:].tolist()
        seq_b = glance.greedy_generate(env.target, ids_b, cont_new, env.stop,
                                      positions=pos_b)[0, ids_b.shape[1]:].tolist()
        seq_z = glance.greedy_generate(env.target, ids_a, cont_new, env.stop,
                                      positions=pos_z)[0, ids_a.shape[1]:].tolist()
        rec["continuation_first_divergence_other_image"] = next(
            (k for k, (x, y) in enumerate(zip(seq_a, seq_b)) if x != y), None)
        rec["continuation_first_divergence_zeroed_pixels"] = next(
            (k for k, (x, y) in enumerate(zip(seq_a, seq_z)) if x != y), None)
        rec["continuation_text_image_a"] = env.proc.tokenizer.decode(
            seq_a, skip_special_tokens=True)[:160]
        rec["continuation_text_image_b"] = env.proc.tokenizer.decode(
            seq_b, skip_special_tokens=True)[:160]
        binding["rows"].append(rec)
        print(f"   {r['pid']}: lastrow "
              f"dswap={rec['max_abs_fused_diff_last_row_other_image']:.4g} "
              f"dzero={rec['max_abs_fused_diff_last_row_zeroed_pixels']:.4g} "
              f"tok {rec['prefill_argmax_image_a_image_b_zeroed']} "
              f"cont_diverge swap@{rec['continuation_first_divergence_other_image']} "
              f"zero@{rec['continuation_first_divergence_zeroed_pixels']}", flush=True)
        del ids_a, ids_b, pix_a, pix_b, grid_a, grid_b, pos_a, pos_b, pos_z
        torch.cuda.empty_cache()

    binding["all_rows_fused_hidden_depends_on_pixels"] = all(
        x["max_abs_fused_diff_last_row_other_image"] > 1e-2
        and x["max_abs_fused_diff_last_row_zeroed_pixels"] > 1e-2
        and x["max_abs_fused_diff_common_prefix_other_image"] > 1e-2
        for x in binding["rows"])
    binding["all_rows_logits_depend_on_pixels"] = all(
        x["prefill_max_logit_gap_image_vs_other_image"] > 1e-2
        and x["prefill_max_logit_gap_image_vs_zeroed"] > 1e-2
        for x in binding["rows"])
    binding["rows_whose_continuation_changes_with_image"] = sum(
        1 for x in binding["rows"]
        if x["continuation_first_divergence_other_image"] is not None)
    binding["rows_whose_continuation_changes_with_zeroed_pixels"] = sum(
        1 for x in binding["rows"]
        if x["continuation_first_divergence_zeroed_pixels"] is not None)
    binding["rows_whose_first_token_changes_with_image"] = sum(
        1 for x in binding["rows"] if x["prefill_argmax_changes_with_image"])
    binding["criterion"] = (
        "HARD: every probe row's fused hidden (the draft head's own input) must "
        "move by >1e-2 max-abs when the picture is swapped and when the pixels are "
        "zeroed, and the target's next-token logits must move too. SUPPORTING: at "
        "least one row's greedy continuation must diverge. The first-token argmax is "
        "recorded but NOT required: on this data it is often an image-independent "
        "function word, so requiring it would reject rows whose pixels demonstrably "
        "drive the model. Measured: rows train:100 and train:1023 kept the same "
        "first token across a 20.3 / 14.5 max-abs fused-hidden change and still "
        "diverged at continuation step 3 / 28.")
    binding["gate_pass"] = bool(
        binding["all_rows_fused_hidden_depends_on_pixels"]
        and binding["all_rows_logits_depend_on_pixels"]
        and binding["rows_whose_continuation_changes_with_image"] >= 1)
    json.dump(binding, open(pre / "image_processor_binding.json", "w"), indent=2)
    print(f"[gate1] pass={binding['gate_pass']} "
          f"fused_dep={binding['all_rows_fused_hidden_depends_on_pixels']} "
          f"logit_dep={binding['all_rows_logits_depend_on_pixels']} "
          f"cont_changed={binding['rows_whose_continuation_changes_with_image']}"
          f"/{len(binding['rows'])}", flush=True)
    return binding["gate_pass"]


def gate2(env, run, pre):
    a = env.a
    print(f"[gate2] length pilot on {a.pilot_n} train records, max_new={a.probe_new}",
          flush=True)
    train = load_split(run, "train")
    lengths, natural, rows = [], 0, []
    t0 = time.time()
    for idx, r in enumerate(train[:a.pilot_n]):
        ids, _p, _g, pos, _ = env.prepare(r["image_path"], r["prompt"], a.max_pixels)
        final, _st = glance.greedy_generate(env.target, ids, a.probe_new, env.stop,
                                           positions=pos, return_stats=True)
        gen = final[0, ids.shape[1]:].tolist()
        text = env.proc.tokenizer.decode(gen, skip_special_tokens=True)
        ends_eos = bool(gen and gen[-1] == env.eos)
        lengths.append(len(gen))
        natural += int(ends_eos)
        rows.append({"pid": r["pid"], "n_gen": len(gen), "ends_with_eos": ends_eos,
                     "hit_cap": bool(len(gen) >= a.probe_new), "text": text,
                     "has_answer_line": "Answer:" in text})
        print(f"   [{idx+1}/{a.pilot_n}] {r['pid']} n={len(gen)} eos={ends_eos} "
              f"answer_line={'Answer:' in text}", flush=True)
        del ids, pos
        torch.cuda.empty_cache()
    arr = np.asarray(lengths, float)
    pct = lambda q: float(np.percentile(arr, q))  # noqa: E731
    over256 = float((arr > 256).mean())
    over512 = float((arr >= a.probe_new).mean())
    pilot = {"n": len(lengths), "max_new_tokens": a.probe_new,
             "p10": pct(10), "p50": pct(50), "p90": pct(90),
             "mean": float(arr.mean()), "min": float(arr.min()),
             "max": float(arr.max()), "n_natural_eos": natural,
             "natural_eos_rate": natural / len(lengths),
             "frac_over_256": over256, "frac_at_or_over_512_cap": over512,
             "frac_at_or_over_256_cap": float((arr >= 256).mean()),
             "frac_with_answer_line": float(np.mean([x["has_answer_line"] for x in rows])),
             "wall_seconds": time.time() - t0, "rows": rows}
    if over256 > 0.10:
        decision = 512 if over512 > 0.10 else a.probe_new
        reason = (f"{over256:.1%} of natural runs exceed 256 tokens -> "
                  f"max_new_tokens={decision}")
    else:
        decision = 256
        reason = f"only {over256:.1%} exceed 256 tokens -> max_new_tokens=256"
    pilot["needs_1024_revision"] = bool(over512 > 0.10)
    if pilot["needs_1024_revision"]:
        pilot["note"] = ("more than 10% are hard-capped at 512; one pre-registered "
                         "revision to 1024 is permitted before any dev/head result")
    pilot["max_new_tokens_decision"] = decision
    pilot["decision_reason"] = reason
    json.dump(pilot, open(pre / "length_pilot.json", "w"), indent=2,
              ensure_ascii=False)
    print(f"[gate2] p10/p50/p90 = {pct(10):.0f}/{pct(50):.0f}/{pct(90):.0f}, "
          f"eos {natural}/{len(lengths)}, >256 {over256:.1%}, cap512 {over512:.1%} "
          f"-> max_new_tokens={decision}", flush=True)
    return decision


def gate3(env, run, pre, max_new):
    a = env.a
    head_path = a.head or (common.CKPT + "/head")
    head = common.load_head(head_path, dtype=getattr(torch, a.head_dtype),
                            device=env.dev).eval()
    print(f"[gate3] head {head_path} ({a.head_dtype}) peak-ish; SPEC==AR on "
          f"{a.lossless_n} train records, max_new={max_new}", flush=True)
    train = load_split(run, "train")
    gate = {"budget": a.budget, "branch_k": a.branch_k, "max_new_tokens": max_new,
            "head": head_path, "head_dtype": a.head_dtype,
            "target_dtype": "float32", "max_pixels": a.max_pixels, "rows": []}
    for idx, r in enumerate(train[:a.lossless_n]):
        ids, _p, _g, pos, _ = env.prepare(r["image_path"], r["prompt"], a.max_pixels)
        ar_final, ar_st = glance.greedy_generate(env.target, ids, max_new, env.stop,
                                                positions=pos, return_stats=True)
        sp_final, sp_st = glance.generate(head, env.target, ids, max_new, env.stop,
                                          budget=a.budget, branch_k=a.branch_k,
                                          temperature=0.0, positions=pos,
                                          return_stats=True, timing="decode_only")
        ar_tok = ar_final[0, ids.shape[1]:].tolist()
        sp_tok = sp_final[0, ids.shape[1]:].tolist()
        ok = ar_tok == sp_tok
        gate["rows"].append({
            "pid": r["pid"], "n_ar": len(ar_tok), "n_spec": len(sp_tok),
            "token_exact": ok,
            "ar_ends_eos": bool(ar_tok and ar_tok[-1] == env.eos),
            "spec_ends_eos": bool(sp_tok and sp_tok[-1] == env.eos),
            "ar_hit_cap": len(ar_tok) >= max_new,
            "ar_truncated_without_eos": bool(len(ar_tok) >= max_new
                                             and (not ar_tok or ar_tok[-1] != env.eos)),
            "ar_tau": ar_st.tau, "ar_rounds": ar_st.rounds,
            "spec_tau": sp_st.tau, "spec_rounds": sp_st.rounds,
            "spec_tree_size": sp_st.tree_size,
            "spec_ms_per_token": sp_st.ms_per_token,
            "ar_ms_per_token": ar_st.ms_per_token,
            "spec_prefill_ms": sp_st.prefill_ms, "ar_prefill_ms": ar_st.prefill_ms,
            "first_mismatch": next((i for i, (x, y) in enumerate(zip(ar_tok, sp_tok))
                                    if x != y), None)})
        print(f"   [{idx+1}/{a.lossless_n}] {r['pid']} exact={ok} "
              f"n_ar={len(ar_tok)} n_spec={len(sp_tok)} tau={sp_st.tau:.3f} "
              f"rounds={sp_st.rounds} ar {ar_st.ms_per_token:.0f}ms/tok "
              f"spec {sp_st.ms_per_token:.0f}ms/tok", flush=True)
        del ids, pos
        torch.cuda.empty_cache()
    gate["all_token_exact"] = all(x["token_exact"] for x in gate["rows"])
    gate["gate_pass"] = bool(gate["all_token_exact"])
    gate["mean_spec_tau"] = float(np.mean([x["spec_tau"] for x in gate["rows"]]))
    gate["mean_ar_rounds"] = float(np.mean([x["ar_rounds"] for x in gate["rows"]]))
    gate["mean_spec_rounds"] = float(np.mean([x["spec_rounds"] for x in gate["rows"]]))
    gate["mean_ar_ms_per_token"] = float(np.mean([x["ar_ms_per_token"] for x in gate["rows"]]))
    gate["mean_spec_ms_per_token"] = float(np.mean([x["spec_ms_per_token"] for x in gate["rows"]]))
    json.dump(gate, open(pre / "ar_lossless_gate.json", "w"), indent=2)
    print(f"[gate3] pass={gate['gate_pass']} "
          f"({sum(x['token_exact'] for x in gate['rows'])}/{len(gate['rows'])} "
          f"token-exact), tau {gate['mean_spec_tau']:.3f}, "
          f"ar {gate['mean_ar_ms_per_token']:.1f} ms/tok vs "
          f"spec {gate['mean_spec_ms_per_token']:.1f} ms/tok", flush=True)
    return gate["gate_pass"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--pilot-n", type=int, default=32)
    ap.add_argument("--probe-new", type=int, default=512)
    ap.add_argument("--lossless-n", type=int, default=8)
    ap.add_argument("--budget", type=int, default=63)
    ap.add_argument("--branch-k", type=int, default=8)
    ap.add_argument("--max-pixels", type=int, default=768)
    ap.add_argument("--continuation-n", type=int, default=48)
    ap.add_argument("--only", default="1,2,3")
    ap.add_argument("--head", default=None)
    ap.add_argument("--head-dtype", default="float32", choices=["float32", "bfloat16"],
                    help="eval head dtype. glance.generate cannot mix an fp32 target "
                         "with a bf16 head (that mix exists only inside the trainers, "
                         "which cast explicitly), so formal evaluation is "
                         "fp32 target + fp32 head")
    a = ap.parse_args()
    gates = {int(x) for x in str(a.only).split(",") if x.strip()}
    run = pathlib.Path(a.run)
    pre = run / "preflight"
    pre.mkdir(parents=True, exist_ok=True)

    env = Env(a)
    print(f"[preflight] target fp32 on {env.dev}; eos={env.eos}; gates={sorted(gates)}",
          flush=True)

    decision = 256
    lp = pre / "length_pilot.json"
    if lp.exists():
        decision = json.load(open(lp))["max_new_tokens_decision"]

    if 1 in gates:
        if not gate1(env, run, pre, a.continuation_n):
            print("STOP: the picture does not move the target; training would be void",
                  flush=True)
            return 3
    elif not (pre / "image_processor_binding.json").exists():
        raise SystemExit("gate 2/3 need gate 1's artifact; run --only 1 first")
    if 2 in gates:
        decision = gate2(env, run, pre)
    if 3 in gates:
        if not json.load(open(pre / "image_processor_binding.json"))["gate_pass"]:
            raise SystemExit("gate 1 did not pass; refusing to run gate 3")
        if not gate3(env, run, pre, min(decision, 256)):
            print("STOP: SPEC != AR; no speedup conclusion is allowed", flush=True)
            return 4
    print("[preflight] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
