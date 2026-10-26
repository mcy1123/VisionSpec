#!/usr/bin/env python3
"""Run the remaining stages whenever an accelerator is actually free enough.

This host is multi-tenant and the fp32 Qwen3-VL-8B target needs about 33 GiB for
its weights alone, so a stage cannot start unless some card has the headroom. A
previous attempt died with `torch.OutOfMemoryError` because another user's job
took GPU 7 while the test-AR build was starting. Rather than guess, this
supervisor polls every card, starts the next pending stage on the emptiest card
that satisfies that stage's requirement, and records every wait and every launch.

It never kills another user's process and never touches a shared dependency.

    python supervise_remaining.py --run <run> --stages test-ar,train-all,eval
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

REPRO = pathlib.Path("/home/mcy/Projects/GLANCE/repro/scienceqa_image")
PY = "/home/mcy/.conda/envs/fin_research/bin/python"
# GiB free required by each stage. Measured, not guessed: the AR build's process
# held ~32.7 GiB on the card (fp32 Qwen3-VL-8B weights plus a short cache);
# training peaked at 38.9 GiB of torch allocations; the evaluation holds fp32
# target AND fp32 head, so it needs slightly more than the AR build. The
# supervisor also makes ONE attempt on the best card within SLACK of the
# requirement, because the instantaneous free figure moves.
GPU_REQ = {"test-ar": 33.0, "train": 38.0, "eval": 38.0, "coverage": 38.0}
SLACK = 2.0
BLOCK_MINUTES = 25.0


def gpu_state():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().splitlines()
    st = []
    for line in out:
        i, used, total = [x.strip() for x in line.split(",")]
        st.append({"index": int(i),
                   "used_mib": int(used.split()[0]),
                   "total_mib": int(total.split()[0]),
                   "free_gib": (int(total.split()[0]) - int(used.split()[0])) / 1024.0})
    return st


BLOCKED = {}


def pick(need, slack=0.0):
    now = time.time()
    st = [g for g in gpu_state()
          if g["free_gib"] >= need - slack
          and now - BLOCKED.get(g["index"], 0) > BLOCK_MINUTES * 60]
    return max(st, key=lambda g: g["free_gib"]) if st else None


def _oom(path):
    try:
        return "OutOfMemoryError" in pathlib.Path(path).read_text()[-200000:]
    except Exception:
        return False


def say(msg, log):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {msg}"
    print(line, flush=True)
    with open(log, "a") as fh:
        fh.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stages", default="test-ar,train-all,eval")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--max-wait-hours", type=float, default=14.0)
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    log = run / "logs" / "supervisor.log"
    env = dict(os.environ)
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    stages = [s for s in a.stages.split(",") if s.strip()]
    t0 = time.time()
    say(f"supervisor start, stages={stages}", log)

    for stage in stages:
        if stage == "test-ar":
            target = run / "eval" / "ar_test.pt"
            if target.exists():
                say("test-ar already done", log)
                continue
            need = GPU_REQ["test-ar"]
            while True:
                g = pick(need, SLACK)
                if g:
                    break
                if (time.time() - t0) / 3600 > a.max_wait_hours:
                    say("GIVE UP waiting for test-ar: no card with enough free memory",
                        log)
                    return 3
                say(f"waiting for >= {need} GiB free; now "
                    f"{[(x['index'], round(x['free_gib'],1)) for x in gpu_state()]}", log)
                time.sleep(a.poll)
            say(f"test-ar on GPU {g['index']} ({g['free_gib']:.1f} GiB free)", log)
            with open(run / "logs" / "build_eval_ar_test.log", "a") as lf:
                rc = subprocess.call(
                    [PY, "-u", str(REPRO / "build_eval_ar.py"), "--run", str(run),
                     "--split", "test", "--gpu", str(g["index"]), "--max-new", "256",
                     "--max-pixels", "768", "--out", str(target)],
                    stdout=lf, stderr=subprocess.STDOUT, env=env)
            say(f"test-ar rc={rc}", log)
            if rc != 0 or not target.exists():
                say("test-ar failed; stopping so it can be inspected", log)
                return 4

        elif stage == "train-all":
            # checkpoints are spread over the three permitted volumes so no single
            # 98-99% full filesystem is filled: seed1 on local3, seed2's SFT on
            # local2 and seed2's OPD on local.
            ROOTS = {("sft", "20261013"):
                     "/mnt/local3/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955",
                     ("opd", "20261013"):
                     "/mnt/local3/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955",
                     ("sft", "20261014"):
                     "/mnt/local2/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955",
                     ("opd", "20261014"):
                     "/mnt/local/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955"}
            arms = [("sft", "20261013"), ("opd", "20261013"),
                    ("sft", "20261014"), ("opd", "20261014")]
            for arm, seed in arms:
                root = pathlib.Path(ROOTS[(arm, seed)])
                out = root / "train" / f"seed{seed}" / arm
                if (out / "step120" / "model.safetensors").exists():
                    say(f"train {arm}/{seed} already done", log)
                    continue
                need = GPU_REQ["train"]
                while True:
                    g = pick(need)
                    if g:
                        break
                    if (time.time() - t0) / 3600 > a.max_wait_hours:
                        say(f"GIVE UP waiting for train {arm}/{seed}", log)
                        return 3
                    say(f"waiting for >= {need} GiB free for train {arm}/{seed}; now "
                        f"{[(x['index'], round(x['free_gib'],1)) for x in gpu_state()]}", log)
                    time.sleep(a.poll)
                say(f"train {arm}/{seed} on GPU {g['index']} "
                    f"({g['free_gib']:.1f} GiB free) -> {out}", log)
                cmd = [PY, "-u", str(REPRO / ("train_sqa_sft.py" if arm == "sft"
                                              else "train_sqa_opd.py")),
                       "--run", str(run), "--seed", seed, "--gpu", str(g["index"]),
                       "--pool", str(run / "dataset" / f"sqa_pool_df{seed}.parquet"),
                       "--records", str(run / "dataset" / f"sqa_target_ar_df{seed}.pt"),
                       "--init-from", "/mnt/local/mcy/GLANCE/checkpoints/head",
                       "--out", str(out), "--updates", "120", "--qa-per-update", "4",
                       "--anchors-per-qa", "6", "--lr", "3e-5",
                       "--master-weights", "cpu", "--target-dtype", "float32",
                       "--max-pixels", "768", "--save-at", "30,60,120"]
                if arm == "opd":
                    cmd += ["--gamma", "0.8"]
                out.mkdir(parents=True, exist_ok=True)
                with open(run / "logs" / f"train_{arm}_seed{seed}.log", "a") as lf:
                    rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
                tries = 0
                while rc != 0 and _oom(run / "logs" / f"train_{arm}_seed{seed}.log") \
                        and tries < 6:
                    tries += 1
                    BLOCKED[g["index"]] = time.time()
                    say(f"train {arm}/{seed} OOM on GPU {g['index']}; blocking that card "
                        f"for {BLOCK_MINUTES:.0f} min (try {tries})", log)
                    while True:
                        g2 = pick(GPU_REQ["train"])
                        if g2:
                            break
                        say("waiting for a card for the retry", log)
                        time.sleep(a.poll)
                    say(f"train {arm}/{seed} retry on GPU {g2['index']}", log)
                    cmd[cmd.index("--gpu") + 1] = str(g2["index"])
                    g = g2
                    with open(run / "logs" / f"train_{arm}_seed{seed}.log", "a") as lf:
                        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                             env=env)
                say(f"train {arm}/{seed} rc={rc}", log)
                if rc != 0:
                    say(f"train {arm}/{seed} failed; stopping for inspection", log)
                    return 5

        elif stage == "eval":
            script = REPRO / "run_eval_matrix.sh"
            while True:
                need = 38.0
                g = pick(need)
                if g:
                    break
                if (time.time() - t0) / 3600 > a.max_wait_hours:
                    say("GIVE UP waiting for the eval matrix", log)
                    return 3
                say(f"waiting for >= {need} GiB free for eval; now "
                    f"{[(x['index'], round(x['free_gib'],1)) for x in gpu_state()]}", log)
                time.sleep(a.poll)
            say(f"eval matrix on GPU {g['index']} ({g['free_gib']:.1f} GiB free)", log)
            e = dict(env)
            e["GPU"] = str(g["index"])
            e["RUN"] = str(run)
            e["PYTHON"] = PY
            # The matrix is resumable and internally re-picks nothing; if the card
            # is taken mid-way the failing pass is logged and the loop below
            # restarts it, which the per-artifact skip makes safe.
            for attempt in range(6):
                rc = subprocess.call(["bash", str(script)], env=e,
                                     stdout=open(log, "a"), stderr=subprocess.STDOUT)
                say(f"eval matrix attempt {attempt+1} rc={rc}", log)
                pending = subprocess.run(
                    ["bash", "-c", f"ls {run}/eval/*.json 2>/dev/null | wc -l"],
                    capture_output=True, text=True).stdout.strip()
                if rc == 0:
                    break
                g = pick(38.0)
                if not g:
                    say("eval stalled waiting for a free card; will retry", log)
                    time.sleep(a.poll * 5)
                    continue
                e["GPU"] = str(g["index"])
                say(f"eval retry on GPU {g['index']}", log)

        else:
            say(f"unknown stage {stage}", log)
            return 2

    say("supervisor finished all requested stages", log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
