#!/usr/bin/env python3
"""One video head, one explicit row list: tau + per-round anchors + AR/SPEC timing.

This is the eval vehicle for the video-OPD long task. It is additive: it does not
touch `eval_video_head.py`, whose numbers the earlier video stages were reported
with. What it adds is what this task needs and that script does not produce:

  * the rows come from an EXPLICIT parquet (the frozen 30-row held-out), not from
    a split filter -- so a split re-shuffle cannot silently change the denominator;
  * `on_round` is captured, so every arm's anchor positions, accepted lengths and
    rounds-to-60%-coverage are recorded (task sections 6.4/6.5);
  * `--ar` additionally runs the repository's own autoregressive greedy oracle
    (`glance.decode.greedy_generate`) on the SAME encoded inputs, interleaved with
    the speculative run, and checks the two token streams for exact equality --
    which is both the lossless gate and the speedup baseline (task sections 4/5).

Row timing is wall-clock around the whole row (video decode + processor + prefill
+ decode) and, separately, the decode-only window the repository's DecodeStats
reports. The first `--warmup` rows are excluded from aggregates but never from the
lossless check. No try/except: a failure surfaces with its traceback.
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
import hashlib
import json
from pathlib import Path
import statistics as st
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, str(_REPRO_ROOT))
from shared import common
from qa_opd_fair.videoencode import (build_video_text, encode_inputs,
                                    validate_real_metadata)
from evaluation.visual_probe import VisualZeroProbe
from shared.model_contract import validate_model_contract


def validate_rows(df, time_mode, *, require_exact_time=False):
    """CPU-only manifest gate, run before allocating any model weights."""
    identities = []
    for _, series in df.iterrows():
        row = series.to_dict()
        name = row.get("video_name", row.get("video_id"))
        if not isinstance(name, str) or not name:
            raise ValueError("each row requires a nonempty video_name or video_id")
        question = row.get("question", row.get("prompt"))
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{name}: missing question/prompt")
        cache = Path(str(row.get("cache_path", "")))
        if not cache.is_file():
            raise ValueError(f"{name}: cache_path is not a file: {cache}")
        info = {"video": name, "cache_path": str(cache.resolve())}
        if time_mode == "real":
            info.update(validate_real_metadata(row, require_cache=True,
                                               require_exact=True))
            expected = row.get("cache_hash")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"{name}: real input requires cache_hash (SHA256)")
            digest = hashlib.sha256()
            with cache.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"{name}: cache SHA256 differs from manifest")
            info["cache_sha256"] = expected
        else:
            if cache.suffix.lower() == ".npz":
                raise ValueError(f"{name}: NPZ frame caches require --time-mode real")
            fps = float(row.get("fps", 0))
            if not 0 < fps < float("inf"):
                raise ValueError(f"{name}: legacy mode needs a finite positive fps")
        identities.append(info)
    if not identities:
        raise ValueError("no evaluation rows remain after selection/sharding")
    return identities


def write_result(path, payload):
    """Never overwrite an existing evaluation or diagnostic report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=1, allow_nan=False)


def encoded_identity(inputs):
    """Hash the actual processor output, independent of source-path aliases."""
    hashes = {}
    for key in ("input_ids", "pixel_values_videos", "video_grid_thw"):
        tensor = inputs[key].detach().cpu().contiguous()
        hashes[key] = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    return hashes


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluation_artifacts(target_path, head_path):
    """Hash the head being evaluated, plus configs and executed source files.

    Target weight shards are not re-hashed here; their provenance must come
    from the target's verified download manifest. Paths/dimensions alone cannot
    identify a head because a checkpoint may have been replaced in place.
    """
    head = Path(head_path)
    weights = sorted(head.glob("*.safetensors")) + sorted(head.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError("evaluation head has no weight files")
    root = _REPRO_ROOT.parent
    sources = [Path(__file__), _REPRO_ROOT / "evaluation/visual_probe.py",
               _REPRO_ROOT / "qa_opd_fair/videoencode.py",
               root / "GLANCE_repo/glance/decode.py",
               root / "scripts/dflash_pkg/dflash-0.1.0/dflash/model.py"]
    return {"head_config_sha256": file_sha256(head / "config.json"),
            "head_weights_sha256": {p.name: file_sha256(p) for p in weights},
            "target_config_sha256": file_sha256(Path(target_path) / "config.json"),
            "target_weight_verification": "external_verified_manifest_required",
            "source_sha256": {str(p.relative_to(root)): file_sha256(p) for p in sources}}


def coverage_at(rounds, k, prompt_len, gen_tokens):
    """Fraction of the generated response reached after `k` draft rounds.

    Same definition the image-side diagnosis used (`多模态OPD诊断2_2026-09-24.md`
    section 3.1): a rollout truncated to k rounds covers this share of the
    response. `rounds[i]["pos"]` is the absolute block start, and a round commits
    `accepted + 1` tokens, so the end of round k is round k's pos + committed.
    """
    if k <= 0:
        return 0.0
    if len(rounds) <= k:
        return 1.0
    end = rounds[k - 1]["pos"] + rounds[k - 1]["accepted"] + 1
    return max(0.0, (end - prompt_len) / max(gen_tokens, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True,
                    help="parquet whose rows are exactly the rows to score")
    ap.add_argument("--head", required=True)
    ap.add_argument("--target", default=common.TARGET_PATH,
                    help="local target path; must match the head config")
    ap.add_argument("--visual-memory", default=None,
                    help="optional .pt mapping video_name -> raw visual tokens")
    ap.add_argument("--n", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--budget", type=int, default=63)
    ap.add_argument("--branch", type=int, default=8)
    ap.add_argument("--ar", action="store_true",
                    help="also run the AR oracle, check losslessness, time it")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--shards", type=int, default=1,
                    help="split the row list into N strided shards (one process "
                         "per GPU); merge the JSONs with merge_eval_jsons.py")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="fp32 is the losslessness-gate dtype: the repository's own "
                         "audit runs fp32 because a bf16 tree pass and a bf16 "
                         "single-token pass reduce in different orders and near-ties "
                         "land on either side (audit_lossless.py docstring).")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--device-map", default=None,
                    help="opt-in multi-GPU target placement (e.g. 'auto'); only for high-token diagnostics")
    ap.add_argument("--max-memory", default=None,
                    help="accelerate max_memory, e.g. '0:44GiB,1:44GiB'")
    ap.add_argument("--tag", default="")
    ap.add_argument("--zero-visual", action="store_true",
                    help="diagnostic: zero visual-token rows of target_hidden before the head")
    ap.add_argument("--time-mode", choices=("legacy", "real"), default="legacy",
                    help="real uses unique source-frame indices and explicit timeline metadata")
    ap.add_argument("--require-exact-time", action="store_true",
                    help="explicit strict-timeline declaration (real mode always enforces this)")
    ap.add_argument("--eos-stop", choices=("legacy", "immediate"), default="legacy",
                    help="explicit decoder EOS protocol; use immediate for new experiments")
    ap.add_argument("--preflight-only", action="store_true",
                    help="validate manifests/configs and write JSON without loading model weights")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if Path(a.out).exists():
        ap.error(f"output already exists; choose a new --out: {a.out}")
    if a.n < 0 or a.warmup < 0 or a.max_new <= 0 or a.budget <= 0 or a.branch <= 0:
        ap.error("n/warmup must be nonnegative; max-new/budget/branch must be positive")
    if a.shards < 1 or not 0 <= a.shard < a.shards:
        ap.error("require shards >= 1 and 0 <= shard < shards")
    if a.require_exact_time and a.time_mode != "real":
        ap.error("--require-exact-time requires --time-mode real")
    if a.time_mode == "real" and a.visual_memory:
        ap.error("real mode rejects legacy video_name-only visual memories; rebuild an "
                 "input-identity-bound memory and validate its integration first")

    dev = f"cuda:{a.gpu}"
    dt = torch.float32 if a.dtype == "float32" else None
    df = pd.read_parquet(a.rows)
    if a.n:
        df = df.head(a.n)
    if a.shards > 1:
        df = df.iloc[a.shard::a.shards].reset_index(drop=True)
    if "video_name" not in df and "video_id" in df:
        df["video_name"] = df["video_id"]
    if "question" not in df and "prompt" in df:
        df["question"] = df["prompt"]
    row_identities = validate_rows(df, a.time_mode,
                                  require_exact_time=a.require_exact_time)
    model_contract = validate_model_contract(a.target, a.head)
    protocol = {"version": "video_eval_real_timeline_v1",
                "time_mode": a.time_mode, "eos_stop": a.eos_stop,
                "visual_mask": "absolute_stream_positions_v1" if a.zero_visual else None,
                "rows_sha256": hashlib.sha256(Path(a.rows).read_bytes()).hexdigest(),
                "target": str(Path(a.target).resolve()),
                "model_contract": model_contract,
                "artifacts": evaluation_artifacts(a.target, a.head),
                "lossless_checked": bool(a.ar and not a.preflight_only),
                "timing_scope": "encoded_input_to_output; CPU preprocessing separately recorded",
                "spec_prefill_definition": "DecodeStats startup window includes target prefill "
                                           "and the first draft; not pure target prefill",
                "tau_definition": "mean(raw accepted + 1) over verification rounds",
                "effective_tokens_per_round_definition": "final output tokens after EOS/cap "
                                                          "divided by verification rounds",
                "diagnostic_only": True}
    if a.preflight_only:
        write_result(a.out, {"config": vars(a), "protocol": protocol,
                             "preflight": row_identities,
                             "status": "CPU_PREFLIGHT_ONLY_NO_MODEL_EXECUTION"})
        print(f"CPU preflight passed for {len(df)} rows -> {a.out}")
        return

    import glance
    from glance.decode import greedy_generate
    from glance.vl import VisionPositions
    from transformers import AutoProcessor

    # DecodeStats synchronizes the current CUDA device internally. Match that
    # device to the explicit head/input placement, including --gpu != 0.
    torch.cuda.set_device(a.gpu)
    common.TARGET_PATH = a.target
    print(f"rows {a.rows}: scoring {len(df)} rows (shard {a.shard}/{a.shards}), "
          f"head {a.head}, dtype {a.dtype}, max_new {a.max_new} "
          f"b{a.budget}k{a.branch} ar={a.ar}", flush=True)

    max_memory = None
    if a.max_memory:
        max_memory = {}
        for part in a.max_memory.split(","):
            k, v = part.split(":", 1)
            max_memory[int(k) if k.strip().isdigit() else k.strip()] = v.strip()
    target, _ = common.load_target(path=a.target, device=dev, dtype=dt,
                                   device_map=a.device_map, max_memory=max_memory)
    placement = getattr(target, "hf_device_map", None)
    protocol["target_device_map"] = ({str(k): str(v) for k, v in placement.items()}
                                     if placement else None)
    protocol["uses_cpu_or_disk_offload"] = bool(
        placement and any(str(v) in ("cpu", "disk") for v in placement.values()))
    if a.device_map:
        # accelerate places different layers on different GPUs. The decode code
        # below owns the head and the KV/out tensors on cuda:0, so bring target
        # outputs back to the input device and route lm_head through its owner.
        import torch as _t
        _dev = torch.device(dev)
        _lm = target.lm_head
        class _LM(_t.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, x):
                return self.inner(x.to(self.inner.weight.device)).to(x.device)
            @property
            def weight(self):
                return self.inner.weight
        target.lm_head = _LM(_lm)
        _orig_forward = target.forward
        def _forward(*args, **kwargs):
            out = _orig_forward(*args, **kwargs)
            if getattr(out, "logits", None) is not None:
                out.logits = out.logits.to(_dev)
            hs = getattr(out, "hidden_states", None)
            if hs is not None:
                out.hidden_states = tuple(h.to(_dev) for h in hs)
            return out
        target.forward = _forward
        print(f"[eval] target device_map={a.device_map} max_memory={max_memory} outputs->{_dev} map={getattr(target, 'hf_device_map', None)}", flush=True)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    head = common.load_head(a.head, device=dev, dtype=dt).eval()
    zero_probe = None
    if a.zero_visual:
        zero_probe = VisualZeroProbe(head)
        head.forward = zero_probe
        print("[eval] zero-visual diagnostic enabled", flush=True)
    visual_memory = None
    if a.visual_memory:
        visual_memory = torch.load(a.visual_memory, map_location="cpu",
                                   weights_only=False)["features"]
        print(f"[eval] visual memories: {len(visual_memory)} videos", flush=True)
        missing = [name for name in df["video_name"] if name not in visual_memory]
        if missing:
            raise ValueError(f"visual memory missing {len(missing)} evaluation videos: {missing[:5]}")
    eos = proc.tokenizer.eos_token_id

    res = []
    for i, row in df.reset_index(drop=True).iterrows():
        t_row0 = time.perf_counter()
        text = build_video_text(proc, row["question"])
        ins = encode_inputs(proc, row, a.time_mode, text=text)
        ids = ins["input_ids"].to(dev)
        video_token_id = proc.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        visual_mask = ids[0] == video_token_id
        if not bool(visual_mask.any()):
            raise ValueError(f"{row['video_name']}: processor produced no video tokens")
        pv = ins["pixel_values_videos"].to(dev)
        vg = ins["video_grid_thw"].to(dev)
        n_in = int(ids.shape[1])
        torch.cuda.synchronize()
        preprocessing_s = time.perf_counter() - t_row0
        t_identity = time.perf_counter()
        input_identity = encoded_identity(ins)
        identity_audit_s = time.perf_counter() - t_identity

        def run_spec():
            pos = VisionPositions(pixel_values_videos=pv, video_grid_thw=vg)
            rounds = []
            if zero_probe is not None:
                zero_probe.reset(visual_mask)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            head.visual_memory = (visual_memory[row["video_name"]].to(dev)
                                  if visual_memory and row["video_name"] in visual_memory else None)
            out, stats = glance.generate(head, target, ids, a.max_new, [eos],
                                         budget=a.budget, branch_k=a.branch,
                                         temperature=0.0, positions=pos,
                                         return_stats=True, timing="decode_only",
                                         eos_stop=a.eos_stop,
                                         on_round=rounds.append)
            torch.cuda.synchronize()
            return out, stats, time.perf_counter() - t0, rounds

        def run_ar():
            pos = VisionPositions(pixel_values_videos=pv, video_grid_thw=vg)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out, stats = greedy_generate(target, ids, a.max_new, [eos],
                                         positions=pos, return_stats=True)
            torch.cuda.synchronize()
            return out, stats, time.perf_counter() - t0

        # Interleave the order so a drifting clock cannot favour one arm.
        if i % 2 == 0:
            spec_out, spec_st, spec_s, spec_rounds = run_spec()
            ar_out, ar_st, ar_s = (run_ar() if a.ar else (None, None, None))
        else:
            ar_out, ar_st, ar_s = (run_ar() if a.ar else (None, None, None))
            spec_out, spec_st, spec_s, spec_rounds = run_spec()
        t_row1 = time.perf_counter()

        n_spec = int(spec_out.shape[1]) - n_in
        spec_ids = spec_out[0, n_in:].tolist()
        anchor_rel = [int(r["pos"]) - n_in for r in spec_rounds]
        accepted = [int(r["accepted"]) for r in spec_rounds]
        n_rounds = len(anchor_rel)
        rec = {"video": str(row["video_name"]), "L": n_in,
               "prompt_len": n_in, "gen_tokens": n_spec,
               "rounds": int(spec_st.rounds), "tau": float(spec_st.tau),
               "effective_tokens_per_round": n_spec / max(int(spec_st.rounds), 1),
               "Ea": float(spec_st.tau) - 1.0,
               "hit_eos": bool(spec_ids and spec_ids[-1] == eos),
               "truncated": bool(n_spec >= a.max_new and
                                 not (spec_ids and spec_ids[-1] == eos)),
               "spec_text": proc.tokenizer.decode(spec_ids, skip_special_tokens=True),
               "tree_size": float(spec_st.tree_size),
               "spec_wall_s": spec_s, "spec_decode_ms_per_tok": float(spec_st.ms_per_token),
               "spec_prefill_ms": float(spec_st.prefill_ms),
               "preprocessing_s": preprocessing_s,
               "input_identity_audit_s": identity_audit_s,
               "spec_end_to_end_s": preprocessing_s + spec_s,
               "input_identity": input_identity,
               "time_mode": a.time_mode,
               "n_visual_tokens": int(visual_mask.sum()),
               "video_grid_thw": vg.detach().cpu().tolist(),
               "timeline": row_identities[i],
               "zero_visual_calls": zero_probe.records if zero_probe is not None else None,
               "row_wall_s": t_row1 - t_row0, "warmup": i < a.warmup,
               "width": int(row.get("width", 0)), "height": int(row.get("height", 0)),
               "n_frames": int(row.get("n_frames", 0)),
               "anchor_rel": anchor_rel, "accepted": accepted,
               "cov60": coverage_at(spec_rounds, 60, n_in, n_spec),
               "front32_frac": (sum(1 for p in anchor_rel if p < 32) / max(n_rounds, 1)),
               "third_frac": [
                   sum(1 for p in anchor_rel if p < n_spec / 3) / max(n_rounds, 1),
                   sum(1 for p in anchor_rel if n_spec / 3 <= p < 2 * n_spec / 3) / max(n_rounds, 1),
                   sum(1 for p in anchor_rel if p >= 2 * n_spec / 3) / max(n_rounds, 1)]}
        if a.ar:
            ar_ids = ar_out[0, n_in:].tolist()
            m = min(len(ar_ids), len(spec_ids))
            first_bad = next((j for j in range(m) if ar_ids[j] != spec_ids[j]), None)
            if first_bad is None and len(ar_ids) != len(spec_ids):
                first_bad = m
            rec.update({
                "ar_tokens": len(ar_ids),
                "ar_text": proc.tokenizer.decode(ar_ids, skip_special_tokens=True),
                "ar_hit_eos": bool(ar_ids and ar_ids[-1] == eos),
                "ar_wall_s": ar_s,
                "ar_end_to_end_s": preprocessing_s + ar_s,
                "ar_decode_ms_per_tok": float(ar_st.ms_per_token),
                "ar_ms_per_tok_wall": float(ar_st.ms_per_token_wall),
                "ar_prefill_ms": float(ar_st.prefill_ms),
                "lossless": bool(first_bad is None and len(ar_ids) == len(spec_ids)),
                "first_mismatch": first_bad,
                "spec_ids": spec_ids, "ar_ids": ar_ids,
            })
        res.append(rec)
        print(f"  [{i+1}/{len(df)}] {rec['video'][:16]:16s} L={n_in:4d} "
              f"gen={n_spec:4d} rounds={rec['rounds']:3d} tau={rec['tau']:.3f} "
              f"spec={spec_s:6.2f}s"
              + (f" ar={ar_s:6.2f}s lossless={rec.get('lossless')} "
                 f"mm={rec.get('first_mismatch')}" if a.ar else ""), flush=True)

    bad = [r["video"] for r in res if a.ar and not r.get("lossless")]
    write_result(a.out, {"config": vars(a), "protocol": protocol, "rows": res,
                         "status": "LOSSLESS_FAILED" if bad else
                         "LOSSLESS_PASSED" if a.ar else "LOSSLESS_NOT_CHECKED"})

    keep = [r for r in res if not r["warmup"]]
    if not keep:
        print(f"No non-warmup rows; raw results saved -> {a.out}")
        if bad:
            raise SystemExit(2)
        return
    taus = [r["tau"] for r in keep]
    sd = f"{st.stdev(taus):.4f}" if len(taus) > 1 else "n/a"
    print(f"\nn={len(keep)} (warmup dropped {len(res)-len(keep)})  "
          f"tau median {st.median(taus):.4f} mean {st.fmean(taus):.4f} "
          f"min {min(taus):.4f} max {max(taus):.4f} sd {sd}")
    print(f"Ea median {st.median(r['Ea'] for r in keep):.4f}  "
          f"gen median {st.median(r['gen_tokens'] for r in keep):.0f}")
    print(f"no-EOS {sum(1 for r in keep if not r['hit_eos'])}/{len(keep)}  "
          f"spec wall median {st.median(r['spec_wall_s'] for r in keep):.1f}s")
    if a.ar:
        bad = [r["video"] for r in res if not r.get("lossless")]
        print(f"SPEC wall total {sum(r['spec_wall_s'] for r in res):.0f}s  "
              f"AR wall total {sum(r['ar_wall_s'] for r in res):.0f}s  "
              f"row-wall total {sum(r['row_wall_s'] for r in res):.0f}s")
        print(f"LOSSLESS {len(res)-len(bad)}/{len(res)} exact; mismatched: {bad}")
    print(f"-> {a.out}")
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
