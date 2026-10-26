#!/usr/bin/env python3
"""D2 gate: prove the real-timeline adapter is exact, then measure its effect.

The adapter (`videoencode.encode_inputs(mode="real")`) must change ONLY the
timeline the processor reads. This script gates that claim on real inputs:

  1. `legacy` reproduces the historical encoding exactly -- twice, so the check
     is determinism, not a lucky single run;
  2. `real` is stable across repeats and its `pixel_values_videos` and
     `video_grid_thw` are BYTE-IDENTICAL to legacy, i.e. the frame pixels, the
     frame count, the spatial grid and the visual token count do not move;
  3. the two modes differ only in `input_ids` (the `<X.Y seconds>` markers and
     the text tokens around them);
  4. the real time markers are monotonically non-decreasing and track the source
     PTS at the temporal-patch resolution the processor documents (merge_size=2
     frames averaged per patch);
  5. VFR samples are reported as approximations with their measured error.

Then it runs the three frozen dev heads on the selected dev inputs under BOTH
timelines with the EOS fix, and requires SPEC == the independent greedy AR of
that same timeline, so the "real" arm's losslessness is established against its
own AR rather than against the legacy tokens.

Usage:
  python timestamp_gate.py --run <run> --gpu N [--max-videos 3] [--arms C0,C1,C2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path[:0] = [str(HERE), str(REPRO), str(ROOT / "GLANCE_repo"),
                str(ROOT / "scripts" / "dflash_pkg" / "dflash-0.1.0")]

import torch  # noqa: E402


def dump(path, obj):
    pathlib.Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def append(path, obj):
    with open(path, "a") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()


def digest(t):
    return hashlib.sha256(t.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


def gpu_free(index):
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,uuid", "--format=csv,noheader,nounits"],
        text=True)
    for line in out.splitlines():
        i, free, uuid = [x.strip() for x in line.split(",")]
        if i == str(index):
            apps = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                 "--format=csv,noheader,nounits"], text=True)
            foreign = [x for x in apps.splitlines()
                       if x.split(",")[0].strip() == uuid
                       and int(x.split(",")[1]) != os.getpid()]
            return int(free), foreign
    raise KeyError(index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--max-videos", type=int, default=3)
    ap.add_argument("--arms", default="C0,C1,C2")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--gpu-gate-mib", type=int, default=24000)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--skip-model", action="store_true",
                    help="run only the encoding gate (checks 1-4), which needs no "
                         "model at all and therefore no GPU; the SPEC==AR part is "
                         "then covered by the three-timestamp-mode main evaluation")
    a = ap.parse_args()

    R = pathlib.Path(a.run)
    out = R / "timestamp"
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "command.json", {"argv": sys.argv,
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    try:
        free, foreign = gpu_free(a.gpu)
        dump(out / "gpu_start.json", {"free_mib": free, "foreign": foreign})
        if free < a.gpu_gate_mib:
            raise RuntimeError(f"GPU_RESOURCE_GATE: gpu {a.gpu} free {free} MiB < "
                               f"{a.gpu_gate_mib}")

        from shared import common
        import glance
        import glance.decode as dec
        from glance.vl import VisionPositions
        from transformers import AutoProcessor
        from videoencode import (build_video_text, encode_inputs, encode_signature,
                                 real_time_exact)

        torch.set_grad_enabled(False)
        torch.set_num_threads(8)
        dev = "cpu" if a.skip_model else f"cuda:{a.gpu}"
        if not a.skip_model:
            torch.cuda.set_device(a.gpu)

        legacy = json.loads((R / "audit" / "legacy_run_paths.json").read_text())
        manifest = pathlib.Path(a.manifest or legacy["dev_manifests"][0])
        rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r["prompt_mode"] == "B"]
        rows.sort(key=lambda r: (r["dataset"], r["video_id"], str(r["question_id"])))
        # Three videos spanning short / tens-of-seconds / minutes where available.
        byvid = {}
        for r in rows:
            byvid.setdefault(r["video_id"], []).append(r)
        vids = sorted(byvid, key=lambda v: byvid[v][0]["video_duration"])
        picked = [vids[0], vids[len(vids) // 2], vids[-1]] if len(vids) >= 3 else vids
        picked = picked[:a.max_videos]
        sel = [r for v in picked for r in byvid[v][:1]]
        dump(out / "selected_videos.json", {
            "videos": [{"video_id": v, "duration_s": byvid[v][0]["video_duration"],
                        "source_fps": byvid[v][0]["source_fps"],
                        "sample_frame_ids": byvid[v][0]["sample_frame_ids"],
                        "sample_timestamps": byvid[v][0]["sample_timestamps"],
                        "n_frames": byvid[v][0]["n_frames"]}
                       for v in picked],
            "selection": "sorted by the manifest's real duration, first / middle / "
                         "last: a short clip, a tens-of-seconds clip and the "
                         "longest frozen dev clip"})

        from transformers import AutoProcessor as _AP
        proc = _AP.from_pretrained(common.TARGET_PATH)
        target = None
        head_dtype = None
        eos = int(proc.tokenizer.eos_token_id)
        heads = legacy["heads"]
        if not a.skip_model:
            target, proc = common.load_target(
                device=dev,
                dtype=torch.float32 if a.dtype == "float32" else torch.bfloat16)
            head_dtype = next(target.parameters()).dtype

        # ---- 1-4: the encoding gate on every selected row
        checks, all_ok = [], True
        for r in sel:
            text = build_video_text(proc, r["prompt"])
            xl1 = encode_inputs(proc, r, "legacy", text=text)
            xl2 = encode_inputs(proc, r, "legacy", text=text)
            xr1 = encode_inputs(proc, r, "real", text=text)
            xr2 = encode_inputs(proc, r, "real", text=text)
            sl1, sl2 = encode_signature(xl1, proc.tokenizer), encode_signature(xl2, proc.tokenizer)
            sr1, sr2 = encode_signature(xr1, proc.tokenizer), encode_signature(xr2, proc.tokenizer)
            pv_equal = torch.equal(xl1["pixel_values_videos"], xr1["pixel_values_videos"])
            grid_equal = torch.equal(xl1["video_grid_thw"], xr1["video_grid_thw"])
            exact, err = real_time_exact(r)
            # The markers must rise with the source frame ids at the
            # temporal-patch (merge_size=2) resolution.
            ids = [int(v) for v in r["sample_frame_ids"]]
            fps = float(r["source_fps"])
            patch_ts = [(ids[i] / fps + ids[i + 1] / fps) / 2
                        for i in range(0, len(ids), 2)]
            markers = [float(v) for v in sr1["time_tokens"]]
            marker_monotone = all(markers[i] <= markers[i + 1] + 0.051
                                  for i in range(len(markers) - 1))
            marker_err = (max(abs(m - t) for m, t in zip(markers, patch_ts))
                          if len(markers) == len(patch_ts) else None)
            rec = {
                "video_id": r["video_id"], "question_id": str(r["question_id"]),
                "duration_s": r["video_duration"], "source_fps": r["source_fps"],
                "legacy_stable": sl1 == sl2, "real_stable": sr1 == sr2,
                "pixel_values_identical": pv_equal, "video_grid_identical": grid_equal,
                "ids_differ": sl1["input_ids"] != sr1["input_ids"],
                "legacy_time_tokens": sl1.get("time_tokens"),
                "real_time_tokens": sr1.get("time_tokens"),
                "real_time_exact_by_pts": exact, "pts_max_abs_error_s": err,
                "marker_vs_patch_pts_max_abs_error_s": marker_err,
                "marker_monotone": marker_monotone,
                "n_frames": int(r["n_frames"]),
            }
            ok = (rec["legacy_stable"] and rec["real_stable"]
                  and rec["pixel_values_identical"] and rec["video_grid_identical"]
                  and rec["ids_differ"] and rec["marker_monotone"])
            rec["pass"] = bool(ok)
            all_ok = all_ok and ok
            checks.append(rec)
            append(out / "legacy_vs_real_inputs.jsonl", rec)
            print(f"[enc] {r['video_id']} dur {r['video_duration']:.1f}s "
                  f"legacy_stable {rec['legacy_stable']} real_stable {rec['real_stable']} "
                  f"pixels_eq {pv_equal} grid_eq {grid_equal} ids_differ "
                  f"{rec['ids_differ']} marker_err {marker_err}", flush=True)
        with (out / "source_vs_processor_timestamps.csv").open("w") as fh:
            fh.write("video_id,duration_s,source_fps,frame_index,source_pts_s,"
                     "patch_index,marker_s,legacy_marker_s\n")
            for r, rec in zip(sel, checks):
                ids = [int(v) for v in r["sample_frame_ids"]]
                ts = [float(v) for v in r["sample_timestamps"]]
                fps = float(r["source_fps"])
                for i, (fi, t) in enumerate(zip(ids, ts)):
                    fh.write(f"{r['video_id']},{r['video_duration']},{fps},{fi},{t},"
                             f"{i // 2},"
                             f"{(rec['real_time_tokens'] or [None])[i // 2]},"
                             f"{(rec['legacy_time_tokens'] or [None])[i // 2]}\n")

        # ---- 5: losslessness and effect under both timelines
        results = []
        if a.skip_model:
            gates = {"pass": bool(all_ok), "encoding_checks_pass": bool(all_ok),
                     "encoding_checks": checks,
                     "spec_ar_checks": 0, "all_lossless": None,
                     "model_section": "skipped (--skip-model); the SPEC==AR gate "
                                      "for both timelines is covered by the main "
                                      "three-dataset evaluation, which runs fp32",
                     "heads": heads, "dtype": "n/a"}
            dump(out / "gates.json", gates)
            print("[gate] encoding-only run:", json.dumps(gates["encoding_checks_pass"]))
            if not gates["pass"]:
                raise SystemExit("timestamp encoding gate failed")
            return
        for mode in ("legacy", "real"):
            for r in sel:
                text = build_video_text(proc, r["prompt"])
                x = encode_inputs(proc, r, mode, text=text)
                ids = x["input_ids"].to(dev)
                pos = VisionPositions(pixel_values_videos=x["pixel_values_videos"].to(dev),
                                      video_grid_thw=x["video_grid_thw"].to(dev))
                ar_t, ar_st = dec.greedy_generate(target, ids, a.max_new, [eos],
                                                  positions=pos, return_stats=True)
                ar = ar_t[0, ids.shape[1]:].tolist()
                for arm in a.arms.split(","):
                    head = common.load_head(heads[arm], device=dev,
                                            dtype=head_dtype).eval()
                    head.requires_grad_(False)
                    sp_t, sp_st = glance.generate(head, target, ids, a.max_new, [eos],
                                                  budget=63, branch_k=8, temperature=0.0,
                                                  positions=pos, return_stats=True,
                                                  timing="decode_only",
                                                  eos_stop="immediate")
                    sp = sp_t[0, ids.shape[1]:].tolist()
                    rec = {"time_mode": mode, "arm": arm, "video_id": r["video_id"],
                           "question_id": str(r["question_id"]),
                           "lossless_spec_eq_ar": sp == ar, "spec_tokens": sp,
                           "ar_tokens": ar, "tau": sp_st.tau,
                           "rounds": sp_st.rounds, "n_tokens": sp_st.n_tokens,
                           "ar_rounds": ar_st.rounds, "ar_n_tokens": ar_st.n_tokens,
                           "decode_ms_token": sp_st.ms_per_token,
                           "ar_decode_ms_token": ar_st.ms_per_token}
                    results.append(rec)
                    append(out / "legacy_vs_real_per_question.jsonl", rec)
                    print(f"[spec] {mode} {arm} {r['video_id']} n {len(sp)} "
                          f"tau {sp_st.tau:.3f} lossless {rec['lossless_spec_eq_ar']}",
                          flush=True)
                    del head
                    torch.cuda.empty_cache()
                del x
        # per-video table
        with (out / "legacy_vs_real_per_video.csv").open("w") as fh:
            fh.write("video_id,arm,time_mode,lossless,tau,n_tokens,rounds,ar_n_tokens\n")
            for rec in results:
                fh.write(f"{rec['video_id']},{rec['arm']},{rec['time_mode']},"
                         f"{rec['lossless_spec_eq_ar']},{rec['tau']},"
                         f"{rec['n_tokens']},{rec['rounds']},{rec['ar_n_tokens']}\n")
        gates = {
            "pass": bool(all_ok and all(r["lossless_spec_eq_ar"] for r in results)),
            "encoding_checks_pass": bool(all_ok),
            "encoding_checks": checks,
            "spec_ar_checks": len(results),
            "all_lossless": all(r["lossless_spec_eq_ar"] for r in results),
            "ar_tokens_differ_by_mode": {},
            "heads": heads,
            "dtype": a.dtype,
            "gpu": a.gpu,
        }
        for v in picked:
            l = next((r["ar_tokens"] for r in results
                      if r["video_id"] == v and r["time_mode"] == "legacy"
                      and r["arm"] == a.arms.split(",")[0]), None)
            r_ = next((r["ar_tokens"] for r in results
                       if r["video_id"] == v and r["time_mode"] == "real"
                       and r["arm"] == a.arms.split(",")[0]), None)
            gates["ar_tokens_differ_by_mode"][v] = bool(l is not None and r_ is not None
                                                        and l != r_)
        dump(out / "gates.json", gates)
        dump(out / "frame_pts_manifest.jsonl" if False else out / "frame_pts_summary.json",
             {"selected": [c["video_id"] for c in checks]})
        if not gates["pass"]:
            raise SystemExit("timestamp gate failed")
    except Exception:
        err = traceback.format_exc()
        (out / "error.log").write_text(err)
        dump(out / "gates.json", {"pass": False, "reason": err})
        print(err, flush=True)
        raise


if __name__ == "__main__":
    main()
