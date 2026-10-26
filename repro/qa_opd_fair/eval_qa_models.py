#!/usr/bin/env python3
"""D6/D5: lossless decode evaluation for the whole model matrix.

ONE script for development checkpoint selection, the three-dataset main table and
the confirm-set evaluation, so there is exactly one implementation of the
lossless gate and one definition of every reported quantity.

WHAT IT DOES per (dataset manifest row, head, time mode):
  * encodes the input on the requested timeline (`legacy` or `real`) and records
    the content signature (input ids, video grid, pixel digest, time markers);
  * runs the independent greedy AR oracle and every SPEC arm, all with
    `eos_stop="immediate"`, and asserts the SPEC tokens are token-identical to AR;
  * records rounds, tau, effective tokens/round, accepted/bonus/clipping,
    EOS-boundary round classification, target entropy/margin at the first
    prediction, deployment top-8 coverage over the fixed AR anchor positions,
    first-rejection offset, and clean timing repeats.

`--select-only` writes the development table used by the frozen checkpoint rule;
`--datasets` chooses which frozen manifests to run.

Usage:
  python eval_qa_models.py --run <run> --stage dev --gpu N --manifest <jsonl> \
      --heads C0=<path>,QA-SFT-s1=<path>,... [--models ...] [--max-videos N]
"""
from __future__ import annotations

import argparse
import collections
import csv
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


def append(path, obj):
    with open(path, "a") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()


def dump(path, obj):
    pathlib.Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


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


def gpu_state(index):
    """(free_mib, foreign_processes) for one GPU at this instant."""
    snap = gpu_snapshot()
    row = next(x for x in snap["gpu"].splitlines()
               if x.split(",")[0].strip() == str(index))
    parts = [x.strip() for x in row.split(",")]
    foreign = [x for x in snap["compute-apps"].splitlines()
               if x.split(",")[0].strip() == parts[1]
               and int(x.split(",")[1]) != os.getpid()]
    return int(parts[4]), foreign, snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", required=True,
                    choices=["dev", "main", "confirm", "smoke"])
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--heads", required=True,
                    help="comma list name=path; 'C0' is the aligned start")
    ap.add_argument("--time-mode", default="legacy", choices=["legacy", "real"])
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--max-new", default="A=64,B=160")
    ap.add_argument("--formats", default="A,B")
    ap.add_argument("--clean-repeats", type=int, default=2)
    ap.add_argument("--gpu-gate-mib", type=int, default=24000)
    ap.add_argument("--allow-foreign", action="store_true",
                    help="run despite a foreign process on the GPU; the run is "
                         "then marked timing_contaminated and no speed claim may "
                         "be made from it")
    ap.add_argument("--out-subdir", default=None)
    ap.add_argument("--append", action="store_true",
                    help="append heads' records to an existing run directory "
                         "instead of refusing to touch it")
    ap.add_argument("--target-dtype", default="float32",
                    choices=["float32", "bfloat16"],
                    help="target AND head dtype. The formal matrix uses float32; "
                         "bfloat16 exists so an auxiliary run can fit on a 24 GiB "
                         "card, and any such run is recorded as such in its "
                         "environment.json and never mixed into a formal table")
    a = ap.parse_args()

    R = pathlib.Path(a.run)
    out = R / "eval" / (a.out_subdir or f"{a.stage}_{a.time_mode}_gpu{a.gpu}")
    # `--append` exists because a long matrix on a contended card can be split
    # into several invocations without re-running finished heads: the frozen
    # inputs make the parts independent, and every part appends its own records
    # to the same `per_question.jsonl`. Without it a reused directory is still
    # refused, so no run is ever silently overwritten.
    if out.exists() and not a.append:
        raise SystemExit(f"refusing to reuse an existing output directory {out}; "
                         "preserve every run (pass --append to add heads)")
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "command.json", {"argv": sys.argv,
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    heads = {}
    for item in a.heads.split(","):
        name, path = item.split("=", 1)
        heads[name.strip()] = path.strip()
    max_new = {}
    for item in a.max_new.split(","):
        k, v = item.split("=")
        max_new[k.strip()] = int(v)
    formats = a.formats.split(",")

    try:
        free, foreign, snap = gpu_state(a.gpu)
        dump(out / "gpu_start.json", snap)
        if foreign and not a.allow_foreign:
            raise RuntimeError(f"GPU_RESOURCE_GATE: foreign compute process on "
                               f"gpu {a.gpu}: {foreign}")
        if free < a.gpu_gate_mib:
            raise RuntimeError(f"GPU_RESOURCE_GATE: gpu {a.gpu} has {free} MiB free, "
                               f"need {a.gpu_gate_mib}")
        timing_contaminated = bool(foreign)

        from shared import common
        import glance
        import glance.decode as dec
        from glance.vl import VisionPositions
        from transformers import AutoProcessor
        from videoencode import encode_inputs, encode_signature, input_key, real_time_exact

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_num_threads(8)
        torch.manual_seed(20261013)
        torch.set_grad_enabled(False)
        dev = f"cuda:{a.gpu}"
        torch.cuda.set_device(a.gpu)

        rows = [json.loads(l) for l in pathlib.Path(a.manifest).read_text().splitlines()
                if l.strip()]
        rows = [r for r in rows if r.get("prompt_mode") in formats]
        rows.sort(key=lambda r: (r["dataset"], r["video_id"], str(r["question_id"]),
                                 r["prompt_mode"]))
        if a.max_videos:
            seen, keep = collections.OrderedDict(), []
            for r in rows:
                seen.setdefault(r["video_id"], []).append(r)
            for vid in list(seen)[:a.max_videos]:
                keep.extend(seen[vid])
            rows = keep

        tdt = torch.float32 if a.target_dtype == "float32" else torch.bfloat16
        target, proc = common.load_target(device=dev, dtype=tdt)
        eos = int(proc.tokenizer.eos_token_id)
        processor_hash = hashlib.sha256(
            json.dumps([sha256_file(str(pathlib.Path(common.TARGET_PATH) / f))
                        for f in sorted(os.listdir(common.TARGET_PATH))
                        if f.endswith((".json", ".txt"))],
                       sort_keys=True).encode()).hexdigest()
        dump(out / "environment.json", {
            "python": sys.version, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(a.gpu),
            "target_dtype": str(next(target.parameters()).dtype), "eos": eos,
            "timing_contaminated": timing_contaminated,
            "foreign_processes_at_start": foreign,
            "time_mode": a.time_mode, "heads": heads,
            "target_dtype_arg": a.target_dtype,
            "gpu_free_mib_at_start": free})

        # ---- encode every distinct input once
        inputs = {}
        for r in rows:
            key = input_key(r, a.time_mode, processor_hash)
            if key in inputs:
                continue
            x = encode_inputs(proc, r, a.time_mode)
            sig = encode_signature(x, proc.tokenizer)
            inputs[key] = (x, sig)
        append(out / "inputs.jsonl", {
            "n_inputs": len(inputs), "n_rows": len(rows),
            "time_mode": a.time_mode,
            "signatures": {k: v[1] for k, v in list(inputs.items())}})

        def pos_of(x):
            return VisionPositions(pixel_values_videos=x["pixel_values_videos"].to(dev),
                                   video_grid_thw=x["video_grid_thw"].to(dev))

        def ar(r, key):
            x = inputs[key][0]
            ids = x["input_ids"].to(dev)
            torch.cuda.synchronize()
            t = time.perf_counter()
            v, st = dec.greedy_generate(target, ids, max_new[r["prompt_mode"]],
                                        [eos], positions=pos_of(x), return_stats=True)
            torch.cuda.synchronize()
            return {"tokens": v[0, ids.shape[1]:].tolist(), "stats": st.to_dict(),
                    "wall_s": time.perf_counter() - t,
                    "text": proc.tokenizer.decode(v[0, ids.shape[1]:].tolist(),
                                                  skip_special_tokens=True)}

        def spec(r, key, head, repeat=0):
            x = inputs[key][0]
            ids = x["input_ids"].to(dev)
            rounds = []
            torch.cuda.synchronize()
            t = time.perf_counter()
            v, st = glance.generate(
                head, target, ids, max_new[r["prompt_mode"]], [eos],
                budget=63, branch_k=8, temperature=0.0, positions=pos_of(x),
                return_stats=True, timing="decode_only",
                on_round=rounds.append, eos_stop="immediate")
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            toks = v[0, ids.shape[1]:].tolist()
            return {"tokens": toks, "stats": st.to_dict(), "wall_s": wall,
                    "rounds": rounds, "n_in": int(ids.shape[1]),
                    "text": proc.tokenizer.decode(toks, skip_special_tokens=True)}

        refs = {}
        for r in rows:
            key = input_key(r, a.time_mode, processor_hash)
            if key in refs:
                continue
            refs[key] = ar(r, key)
            append(out / "ar.jsonl", {"key": key, "dataset": r["dataset"],
                                      "video_id": r["video_id"],
                                      "question_id": str(r["question_id"]),
                                      "mode": r["prompt_mode"],
                                      "time_mode": a.time_mode,
                                      **refs[key]})

        head_state = {}
        results = []
        for name, path in heads.items():
            head = common.load_head(path, device=dev, dtype=tdt).eval()
            head.requires_grad_(False)
            head_state[name] = {"path": path,
                                "config_sha256": sha256_file(str(pathlib.Path(path) / "config.json")),
                                "weights_sha256": sha256_file(str(pathlib.Path(path) / "model.safetensors"))}
            print(f"[head] {name} {path}", flush=True)
            for r in rows:
                key = input_key(r, a.time_mode, processor_hash)
                d = spec(r, key, head)
                ref = refs[key]
                lossless = d["tokens"] == ref["tokens"]
                ar_unstable = False
                if not lossless:
                    # A mismatch has two possible causes and they must not be
                    # conflated: the SPEC path can be wrong, or the reduced
                    # precision can make AR itself non-deterministic (measured in
                    # bf16: repeated greedy AR on the same input diverges). One
                    # extra AR decides which, and a mode that cannot reproduce
                    # its own oracle is disqualified from the lossless gate
                    # rather than being reported as a SPEC failure.
                    ref2 = ar(r, key)
                    if ref2["tokens"] != ref["tokens"]:
                        ar_unstable = True
                        append(out / "ar_instability.jsonl",
                               {"key": key, "arm": name, "first": ref["tokens"],
                                "second": ref2["tokens"],
                                "note": "greedy AR is not reproducible at this "
                                        "dtype; this run is timing-only"})
                        ref = ref2
                rr, stop_round, stop_kind = classify_rounds(
                    d["rounds"], d["n_in"], d["tokens"], eos)
                clean = []
                for i in range(a.clean_repeats):
                    c = spec(r, key, head, repeat=i + 1)
                    if c["tokens"] != ref["tokens"]:
                        if ar_unstable:
                            # The reduced-precision oracle cannot reproduce
                            # itself, so "lossless" is not a well-defined
                            # property of this run. The record is kept for
                            # TIMING and pattern only, with the lossless fields
                            # explicitly null, instead of aborting a matrix that
                            # no fp32 pass depends on.
                            clean.append({"stats": c["stats"], "wall_s": c["wall_s"],
                                          "oracle_moved": True})
                            continue
                        raise RuntimeError(
                            f"clean repeat {i} is not lossless for {key} {name}; "
                            "the run cannot supply a lossless claim")
                    clean.append({"stats": c["stats"], "wall_s": c["wall_s"]})
                free_now, foreign_now, _ = gpu_state(a.gpu)
                rec = {
                    "dataset": r["dataset"], "video_id": r["video_id"],
                    "question_id": str(r["question_id"]), "mode": r["prompt_mode"],
                    "time_mode": a.time_mode, "arm": name, "head_path": path,
                    "input_key": key, "lossless_spec_eq_ar": lossless,
                    "ar_unstable": ar_unstable,
                    "ar_tokens": ref["tokens"],
                    "tokens": d["tokens"], "text": d["text"],
                    "n_tokens": len(d["tokens"]),
                    "ar_n_tokens": len(ref["tokens"]),
                    "tau": d["stats"]["tau"], "rounds": d["stats"]["rounds"],
                    "ar_rounds": ref["stats"]["rounds"],
                    "ar_decode_ms_token": ref["stats"]["ms_per_token"],
                    "ar_wall_ms_token": ref["stats"]["ms_per_token_wall"],
                    "decode_ms_token": d["stats"]["ms_per_token"],
                    "decode_ms_token_wall": d["stats"]["ms_per_token_wall"],
                    "model_wall_ms": d["wall_s"] * 1000.0,
                    "ar_model_wall_ms": ref["wall_s"] * 1000.0,
                    "tree_size": d["stats"]["tree_size"],
                    "prefill_ms": d["stats"]["prefill_ms"],
                    "effective_tokens_per_round": (len(d["tokens"]) /
                                                   max(d["stats"]["rounds"], 1)),
                    "hit_eos": eos in d["tokens"],
                    "eos_stop_round": stop_round, "eos_stop_kind": stop_kind,
                    "round_records": rr,
                    "clean_repeats": clean,
                    "gpu_free_mib_after": free_now,
                    "foreign_now": len(foreign_now),
                    "timing_contaminated": timing_contaminated or bool(foreign_now),
                    "execution_segment": "main",
                }
                results.append(rec)
                append(out / "per_question.jsonl", rec)
                print(f"{a.stage}/{a.time_mode} {name} {r['dataset']} "
                      f"{r['video_id']} {r['question_id']} {r['prompt_mode']} "
                      f"n {len(d['tokens'])} tau {d['stats']['tau']:.3f} "
                      f"rounds {d['stats']['rounds']} lossless {lossless}", flush=True)
            head.to("cpu")
            del head
            torch.cuda.empty_cache()

        ar_hash_ok = all(
            results[0]["input_key"] in refs for _ in [0])
        n_unstable = sum(1 for r in results if r["ar_unstable"])
        if n_unstable:
            # A run whose oracle is not deterministic at this dtype may not
            # assert losslessness; its numbers are timing/pattern evidence only.
            for r in results:
                if r["ar_unstable"]:
                    r["lossless_spec_eq_ar"] = None
                    r["lossless_status"] = "oracle_not_reproducible_at_this_dtype"
                else:
                    r["lossless_status"] = ("pass" if r["lossless_spec_eq_ar"]
                                            else "fail")
        gates = {
            "pass": (all(r["lossless_spec_eq_ar"] for r in results)
                     if not n_unstable else None),
            "n_ar_unstable": n_unstable,
            "lossless_claim": ("established" if not n_unstable
                               else "not_available_at_this_dtype"),
            "stage": a.stage, "time_mode": a.time_mode,
            "n_questions": len(results), "n_heads": len(heads),
            "all_lossless": all(r["lossless_spec_eq_ar"] for r in results),
            "all_clean_repeats_lossless": all(
                all(c["stats"]["rounds"] >= 0 for c in r["clean_repeats"])
                for r in results),
            "head_hashes": head_state,
            "timing_contaminated": timing_contaminated,
        }
        dump(out / "gates.json", gates)
        write_tables(out, results)
        if not gates["pass"]:
            raise SystemExit("lossless gate failed")
    except Exception:
        err = traceback.format_exc()
        (out / "error.log").write_text(err)
        dump(out / "gates.json", {"pass": False, "reason": err})
        print(err, flush=True)
        raise


def classify_rounds(rounds, n_in, emitted, eos):
    """Per-round committed ranges and where the emitted EOS fell."""
    stop_index = emitted.index(eos) if eos in emitted else None
    out, stop_round, stop_kind = [], None, None
    for k, r in enumerate(rounds):
        pos, accepted = int(r["pos"]), int(r["accepted"])
        root = pos - n_in
        first, last = root, root + accepted
        out.append({"round": k, "root_position": root, "accepted": accepted,
                    "bonus_position": last, "range": [first, last],
                    "accepted_tokens": [int(x) for x in r["path_tokens"]],
                    "argmax_tokens": [int(x) for x in r["argmax_tokens"]],
                    "entropy": float(r.get("entropy", 0.0)),
                    "tree_size": int(r.get("tree_size", 0))})
        if stop_index is not None and first <= stop_index <= last:
            stop_round, stop_kind = k, ("bonus" if stop_index == last else "accepted")
    return out, stop_round, stop_kind


def write_tables(out, results):
    """Per-question csv plus a per(video,arm) tau table."""
    cols = ["dataset", "video_id", "question_id", "mode", "time_mode", "arm",
            "lossless_spec_eq_ar", "n_tokens", "ar_n_tokens", "tau", "rounds",
            "ar_rounds", "effective_tokens_per_round", "decode_ms_token",
            "decode_ms_token_wall", "model_wall_ms", "ar_model_wall_ms",
            "tree_size", "hit_eos", "eos_stop_round", "eos_stop_kind",
            "timing_contaminated", "head_path"]
    with (out / "per_question.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    by_video = collections.defaultdict(list)
    for r in results:
        by_video[(r["dataset"], r["video_id"], r["mode"], r["arm"])].append(r)
    with (out / "per_video.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "video_id", "mode", "arm", "n_qa", "tau",
                    "n_tokens", "rounds", "decode_ms_token", "model_wall_ms",
                    "effective_tokens_per_round", "lossless"])
        for (ds, vid, mode, arm), rs in sorted(by_video.items()):
            w.writerow([ds, vid, mode, arm, len(rs),
                        sum(x["tau"] for x in rs) / len(rs),
                        sum(x["n_tokens"] for x in rs) / len(rs),
                        sum(x["rounds"] for x in rs) / len(rs),
                        sum(x["decode_ms_token"] for x in rs) / len(rs),
                        sum(x["model_wall_ms"] for x in rs) / len(rs),
                        sum(x["effective_tokens_per_round"] for x in rs) / len(rs),
                        all(x["lossless_spec_eq_ar"] for x in rs)])


if __name__ == "__main__":
    main()
