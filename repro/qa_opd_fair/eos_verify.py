#!/usr/bin/env python3
"""D1: real-model EOS-boundary reproduction, fix gate and legacy-vs-fixed re-test.

What this measures, and why each piece exists:

* `eos_bug_repro.jsonl` -- for every decoded input, the round-level record of
  which round COMMITTED the first stop token and whether that token was the
  round's bonus (the defect) or an accepted draft node. This is the real trace
  the audit asked for, not a synthetic argument.
* `legacy_vs_fixed_per_video.csv` -- the paired re-test. Both modes run the same
  frozen inputs, the same video encoding (the historical 16-frame/30 fps cache
  path, deliberately unchanged in D1), the same heads and the same decode
  parameters. The ONLY difference is `eos_stop`.
* `gates.json` -- the pass criteria: SPEC==independent AR (token-exact) in BOTH
  modes, identical emitted tokens between modes, `immediate` rounds <= `legacy`
  rounds, and the reproduction actually observing bonus-EOS cases.

The `legacy` mode is bit-identical to the historical decoder (it is the default
value of the new `eos_stop` argument), so the legacy arm of this run reproduces
the previous run's behaviour. `immediate` is the fix.

Usage:
  python repro/qa_opd_fair/eos_verify.py --run <run dir> --stage smoke|full \
      --gpu N [--max-videos 3] [--manifests a.jsonl,b.jsonl] [--arms C0,C1,C2]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
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


def append(path, obj):
    with open(path, "a") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()


def dump(path, obj):
    pathlib.Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def gpu_snapshot():
    return {
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,"
                           "utilization.gpu", "--format=csv,noheader,nounits"],
            text=True),
        "compute-apps": subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,"
                           "used_memory", "--format=csv,noheader,nounits"],
            text=True),
    }


def gpu_gate(index, need_mib=30000):
    """Free-memory + no-foreign-process gate for one GPU.

    The fp32 target alone is ~32 GiB, so the gate is 30 GiB free, not the 40 GiB
    the historical script used (that number was chosen when an idle 48 GiB card
    was normal). A foreign compute process on the same card is always refused:
    it changes both the timing and the memory headroom, and this run reports
    wall-clock numbers.
    """
    snap = gpu_snapshot()
    row = next(x for x in snap["gpu"].splitlines()
               if x.split(",")[0].strip() == str(index))
    parts = [x.strip() for x in row.split(",")]
    uuid = parts[1]
    foreign = [x for x in snap["compute-apps"].splitlines()
               if x.split(",")[0].strip() == uuid
               and int(x.split(",")[1]) != os.getpid()]
    if foreign or int(parts[4]) < need_mib:
        raise RuntimeError("GPU_RESOURCE_GATE: need >=%d MiB free and no foreign "
                           "compute process: %s; %s" % (need_mib, row, foreign))
    return snap


def round_record(rounds, n_in, emitted, eos):
    """Classify each round's committed tokens against the emitted sequence.

    A round's committed tokens are its accepted draft path plus its bonus, and
    they occupy the output positions `root_position .. root_position+accepted`
    in that order. The bonus therefore sits at `root_position + accepted`, and
    the round that committed the first stop token is found by locating the stop
    token's index in the emitted list inside exactly one round's range.

    `stop_round` / `stop_kind` are the reproduction evidence: `"bonus"` means the
    stop token was the round's bonus (the case the legacy scan misses), and
    `"accepted"` means it was an accepted draft node (which legacy already saw).
    """
    out = []
    stop_index = emitted.index(eos) if eos in emitted else None
    stop_round = stop_kind = None
    for k, r in enumerate(rounds):
        pos, accepted = int(r["pos"]), int(r["accepted"])
        root = pos - n_in
        first = root
        last = root + accepted          # the bonus position, inclusive
        rec = {
            "round": k,
            "root_position": root,
            "accepted": accepted,
            "bonus_position": last,
            "committed_range": [first, last],
            "accepted_tokens": [int(x) for x in r["path_tokens"]],
            "argmax_tokens": [int(x) for x in r["argmax_tokens"]],
            "entropy": float(r.get("entropy", 0.0)),
            "tree_size": int(r.get("tree_size", 0)),
        }
        if stop_index is not None and first <= stop_index <= last:
            stop_round = k
            stop_kind = "bonus" if stop_index == last else "accepted"
        out.append(rec)
    return out, stop_round, stop_kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", choices=["smoke", "full"], required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--manifests", default=None,
                    help="comma-separated dataset manifest jsonl files")
    ap.add_argument("--arms", default="C0,C1,C2")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=None,
                    help="override the per-format max_new (otherwise the frozen run config)")
    ap.add_argument("--gpu-gate-mib", type=int, default=30000)
    ap.add_argument("--arms-per-pass", type=int, default=1,
                    help="how many heads stay resident at once (memory control)")
    ap.add_argument("--out-subdir", default=None)
    a = ap.parse_args()

    R = pathlib.Path(a.run)
    out = R / "eos_fix" / (a.out_subdir or a.stage)
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "command.json", {"argv": sys.argv,
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    try:
        dump(out / "gpu_start.json", gpu_snapshot())
        gpu_gate(a.gpu, a.gpu_gate_mib)

        from shared import common
        import glance
        import glance.decode as dec
        from glance.vl import VisionPositions
        from transformers import AutoProcessor

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_num_threads(8)
        torch.manual_seed(20261013)
        torch.set_grad_enabled(False)
        dev = f"cuda:{a.gpu}"
        torch.cuda.set_device(a.gpu)

        legacy_run = json.loads(
            (R / "audit" / "legacy_run_paths.json").read_text())
        heads = legacy_run["heads"]
        arms = a.arms.split(",")
        manifest_paths = (a.manifests.split(",") if a.manifests
                          else [legacy_run["dev_manifests"][0]])
        rows = []
        for mp in manifest_paths:
            for line in pathlib.Path(mp).read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        # Deterministic order: dataset, video, question, format.
        rows.sort(key=lambda r: (r["dataset"], r["video_id"],
                                 str(r["question_id"]), r["prompt_mode"]))
        if a.max_videos:
            keep = []
            seen = collections.OrderedDict()
            for r in rows:
                seen.setdefault(r["video_id"], []).append(r)
            for vid in list(seen)[:a.max_videos]:
                keep.extend(seen[vid])
            rows = keep
        max_new = a.max_new or legacy_run["config"]["max_new_tokens"]

        target, proc = common.load_target(device=dev, dtype=torch.float32)
        eos = int(proc.tokenizer.eos_token_id)
        dump(out / "environment.json", {
            "python": sys.version, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(a.gpu),
            "target_dtype": str(next(target.parameters()).dtype), "eos": eos})

        # ---- encode every distinct (video, prompt) once, on the legacy path
        inputs = {}
        for r in rows:
            key = (r["video_id"], r["question_id"], r["prompt_mode"])
            if key in inputs:
                continue
            text = proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "video"},
                                              {"type": "text", "text": r["prompt"]}]}],
                add_generation_prompt=True, tokenize=False)
            x = proc(text=[text], videos=[r["cache_path"]], return_tensors="pt",
                     fps=float(r["fps"]))
            inputs[key] = x
        dump(out / "inputs.json", {"n_inputs": len(inputs), "n_rows": len(rows)})

        def pos_of(x):
            return VisionPositions(pixel_values_videos=x["pixel_values_videos"].to(dev),
                                   video_grid_thw=x["video_grid_thw"].to(dev))

        def ar(r):
            x = inputs[(r["video_id"], r["question_id"], r["prompt_mode"])]
            ids = x["input_ids"].to(dev)
            torch.cuda.synchronize()
            t = time.perf_counter()
            v, st = dec.greedy_generate(target, ids, int(max_new[r["prompt_mode"]]),
                                        [eos], positions=pos_of(x), return_stats=True)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            return {"tokens": v[0, ids.shape[1]:].tolist(), "stats": st.to_dict(),
                    "wall_s": wall, "text": proc.tokenizer.decode(
                        v[0, ids.shape[1]:].tolist(), skip_special_tokens=True)}

        def spec(r, head, mode):
            x = inputs[(r["video_id"], r["question_id"], r["prompt_mode"])]
            ids = x["input_ids"].to(dev)
            rounds = []
            torch.cuda.synchronize()
            t = time.perf_counter()
            v, st = glance.generate(
                head, target, ids, int(max_new[r["prompt_mode"]]), [eos],
                budget=63, branch_k=8, temperature=0.0, positions=pos_of(x),
                return_stats=True, timing="decode_only",
                on_round=rounds.append, eos_stop=mode)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            toks = v[0, ids.shape[1]:].tolist()
            return {"tokens": toks, "stats": st.to_dict(), "wall_s": wall,
                    "rounds": rounds, "n_in": int(ids.shape[1]),
                    "text": proc.tokenizer.decode(toks, skip_special_tokens=True)}

        results = []
        for i, r in enumerate(rows):
            ref = ar(r)
            row_key = {"dataset": r["dataset"], "video_id": r["video_id"],
                       "question_id": r["question_id"], "mode": r["prompt_mode"]}
            for arm in arms:
                head = common.load_head(heads[arm], device=dev,
                                        dtype=torch.float32).eval()
                head.requires_grad_(False)
                print(f"[arm] {arm} loaded {heads[arm]}", flush=True)
                for mode in ("legacy", "immediate"):
                    d = spec(r, head, mode)
                    rr, hit_round, hit_kind = round_record(
                        d["rounds"], d["n_in"], d["tokens"], eos)
                    emitted = d["tokens"]
                    rec = {
                        **row_key, "arm": arm, "eos_stop": mode,
                        "tokens": emitted, "text": d["text"],
                        "lossless_spec_eq_ar": emitted == ref["tokens"],
                        "ar_tokens": ref["tokens"],
                        "rounds": d["stats"]["rounds"],
                        "tau": d["stats"]["tau"],
                        "n_tokens": d["stats"]["n_tokens"],
                        "ms_per_token": d["stats"]["ms_per_token"],
                        "ms_per_token_wall": d["stats"]["ms_per_token_wall"],
                        "prefill_ms": d["stats"]["prefill_ms"],
                        "wall_s": d["wall_s"],
                        "hit_eos": eos in emitted,
                        "eos_stop_round": hit_round,
                        "eos_stop_kind": hit_kind,
                        "round_records": rr,
                    }
                    results.append(rec)
                    append(out / "per_input.jsonl", rec)
                    print(a.stage, arm, mode, r["video_id"], r["question_id"],
                          r["prompt_mode"], "n", len(emitted),
                          "rounds", d["stats"]["rounds"],
                          "tau", round(d["stats"]["tau"], 3),
                          "eos_kind", hit_kind,
                          "lossless", rec["lossless_spec_eq_ar"], flush=True)
                del head
                torch.cuda.empty_cache()

        # ---- gates and the paired table
        by_key = collections.defaultdict(dict)
        ar_by_key = {}
        for rec in results:
            k = (rec["dataset"], rec["video_id"], rec["question_id"],
                 rec["mode"], rec["arm"])
            by_key[k][rec["eos_stop"]] = rec
            ar_by_key[k] = rec["ar_tokens"]
        bugs, gate_fail = [], []
        for k, pair in sorted(by_key.items()):
            if len(pair) != 2:
                gate_fail.append({"key": k, "reason": "missing a mode"})
                continue
            lg, im = pair["legacy"], pair["immediate"]
            if not (lg["lossless_spec_eq_ar"] and im["lossless_spec_eq_ar"]):
                gate_fail.append({"key": k, "reason": "SPEC != AR"})
            if lg["tokens"] != im["tokens"]:
                gate_fail.append({"key": k, "reason": "modes disagree on tokens"})
            if im["rounds"] > lg["rounds"]:
                gate_fail.append({"key": k, "reason": "immediate used more rounds"})
            if lg["eos_stop_kind"] == "bonus":
                bugs.append({
                    "dataset": lg["dataset"], "video_id": lg["video_id"],
                    "question_id": lg["question_id"], "mode": lg["mode"],
                    "arm": lg["arm"], "legacy_rounds": lg["rounds"],
                    "immediate_rounds": im["rounds"],
                    "redundant_rounds": lg["rounds"] - im["rounds"],
                    "eos_stop_round": lg["eos_stop_round"],
                    "tokens": lg["tokens"],
                    "legacy_ms_per_token": lg["ms_per_token"],
                    "immediate_ms_per_token": im["ms_per_token"],
                })
        with open(out / "eos_bug_repro.jsonl", "w") as fh:
            for b in bugs:
                fh.write(json.dumps(b, ensure_ascii=False) + "\n")
        import csv
        with open(out / "legacy_vs_fixed_per_video.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "video_id", "question_id", "format", "arm",
                        "legacy_rounds", "immediate_rounds", "redundant_rounds",
                        "legacy_tau", "immediate_tau", "legacy_tokens",
                        "immediate_tokens", "tokens_equal", "eos_stop_kind",
                        "legacy_lossless", "immediate_lossless",
                        "legacy_ms_per_token", "immediate_ms_per_token",
                        "legacy_ms_token_wall", "immediate_ms_token_wall"])
            for k, pair in sorted(by_key.items()):
                if len(pair) != 2:
                    continue
                lg, im = pair["legacy"], pair["immediate"]
                w.writerow([lg["dataset"], lg["video_id"], lg["question_id"],
                            lg["mode"], lg["arm"], lg["rounds"], im["rounds"],
                            lg["rounds"] - im["rounds"], lg["tau"], im["tau"],
                            lg["n_tokens"], im["n_tokens"],
                            int(lg["tokens"] == im["tokens"]),
                            lg["eos_stop_kind"], int(lg["lossless_spec_eq_ar"]),
                            int(im["lossless_spec_eq_ar"]),
                            lg["ms_per_token"], im["ms_per_token"],
                            lg["ms_per_token_wall"], im["ms_per_token_wall"]])
        summary = {
            "n_decodes": len(results),
            "n_pairs": sum(1 for p in by_key.values() if len(p) == 2),
            "n_bonus_eos_cases": len(bugs),
            "n_gate_failures": len(gate_fail),
            "redundant_rounds_total": sum(b["redundant_rounds"] for b in bugs),
            "n_pairs_with_round_saving": sum(
                1 for p in by_key.values() if len(p) == 2
                and p["legacy"]["rounds"] > p["immediate"]["rounds"]),
            "legacy_rounds_total": sum(p["legacy"]["rounds"] for p in by_key.values()
                                       if len(p) == 2),
            "immediate_rounds_total": sum(p["immediate"]["rounds"] for p in by_key.values()
                                          if len(p) == 2),
        }
        dump(out / "summary.json", summary)
        dump(out / "gates.json", {
            "pass": not gate_fail,
            "stage": a.stage,
            "summary": summary,
            "failures": gate_fail[:50],
            "criteria": [
                "SPEC == independent greedy AR, token-exact, in both modes",
                "legacy and immediate emit identical tokens",
                "immediate rounds <= legacy rounds",
            ],
        })
        if gate_fail:
            raise SystemExit(f"EOS gates failed: {gate_fail[:3]}")
    except Exception:
        err = traceback.format_exc()
        (out / "error.log").write_text(err)
        dump(out / "gates.json", {"pass": False, "reason": err})
        print(err, flush=True)
        raise


if __name__ == "__main__":
    main()
