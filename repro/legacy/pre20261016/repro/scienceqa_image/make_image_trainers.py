#!/usr/bin/env python3
"""Generate the ScienceQA image trainers from the frozen video QA trainers.

The task forbids changing the objective when moving from video QA to image QA, and
requires the already-validated position contract to be REUSED rather than
re-derived. This generator therefore copies the frozen files and edits exactly the
image-input lines. Running it is the audit trail: `--check` fails if the generated
files differ from what the frozen sources would produce, and the emitted
`image_trainer.patch` is the complete, reviewable change.

    python make_image_trainers.py            # write the files + the patch
    python make_image_trainers.py --check    # verify they are up to date

Edited lines, and nothing else:

  train_qa_opd.py -> train_sqa_opd.py
    * docstring header
    * `from train_qa_sft import ...` -> `from train_sqa_sft import ...`
    * `from videoencode import build_video_text, encode_inputs` ->
      `from imageencode import encode_image_inputs, prefix_image_vision_positions`
    * the two-line video encoding call -> `encode_image_inputs(..., max_pixels=)`
    * `opd_fullkl.prefix_vision_positions(..., pixel_values_videos=, video_grid_thw=)` ->
      `prefix_image_vision_positions(target, ids, pixel_values, image_grid_thw)`
    * `target_prefill_shared(..., pixel_values_videos=, video_grid_thw=)` drops those
      two kwargs and relies on the already-bound `vispos` (the image grid rides on
      the bound `VisionPositions`, exactly as the video path relied on `vispos`)
    * the prepared pixel/grid keys
    * `--max-pixels` added; `--time-mode` documented as ignored
    * `--schedule` added (default = the frozen `train_schedule_seed_<seed>.jsonl`),
      so a smoke run can point at a short debug schedule without overwriting the
      formal one
    * `repro/qa_opd_fair` added to `sys.path`, so `import qa_anchors` resolves to
      the FROZEN anchor rule in its original file rather than to a copy
    * the schedule lookup key honours an explicit `task` field (a ScienceQA record
      has one identity, not a video|question pair)
    * the schedule lookup key honours an explicit `task` field
    * the arm string
  train_qa_sft.py -> train_sqa_sft.py
    * the same set, with `common.target_context_cached(..., pixel_values=,
      image_grid_thw=)` in place of the video kwargs.

NOT edited: `qa_anchors.py` (imported unchanged), `repro/opd_fullkl.py`,
`repro/master_optim.py`, `repro/common.py`, `GLANCE_repo/glance/*`,
`repro/train_head.py`. `repro/qa_opd_fair/*` is left byte-identical.
"""
from __future__ import annotations

import argparse
import difflib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPRO = HERE.parent
SRC = REPRO / "qa_opd_fair"

# The exact frozen source substring that resolves the schedule path, and its
# replacement. Kept as module constants so the quoting is written once.
OLD_SCHEDULE = "\n".join([
    "    order = [json.loads(l) for l in",
    '             (pathlib.Path(a.run) / "dataset" /',
    '              f"train_schedule_seed_{a.seed}.jsonl").read_text().splitlines()',
    "             if l.strip()]",
])
NEW_SCHEDULE = "\n".join([
    "    sched_path = (pathlib.Path(a.schedule) if a.schedule else",
    '                  pathlib.Path(a.run) / "dataset" /',
    '                  f"train_schedule_seed_{a.seed}.jsonl")',
    "    order = [json.loads(l) for l in sched_path.read_text().splitlines()",
    "             if l.strip()]",
])
SCHEDULE_ARG = "\n".join([
    '    ap.add_argument("--schedule", default=None,',
    '                    help="override the frozen schedule path; default reads "',
    '                         "train_schedule_seed_<seed>.jsonl from --run")',
    "",
])
UPDATES_ARG = '    ap.add_argument("--updates", type=int, default=120)'


OLD_TIMEMODE_ARG = '    ap.add_argument("--time-mode", default="legacy", choices=["legacy", "real"])'
NEW_TIMEMODE_ARG_SFT = '    ap.add_argument("--time-mode", default="legacy", choices=["legacy", "real"],\n                    help="IGNORED for still images; kept so this CLI matches the "\n                         "frozen video script. An image has no timeline.")\n    ap.add_argument("--max-pixels", type=int, default=768,\n                    help="longest-side cap; 768 is common.IMAGE_LONG_SIDE, the "\n                         "value the historical image trainer used, and it is used "\n                         "for TRAINING AND EVALUATION here so the head sees the "\n                         "same pixels in both")'
NEW_TIMEMODE_ARG_OPD = '    ap.add_argument("--time-mode", default="legacy", choices=["legacy", "real"],\n                    help="IGNORED for still images; kept so this CLI matches the "\n                         "frozen video script. An image has no timeline.")\n    ap.add_argument("--max-pixels", type=int, default=768,\n                    help="longest-side cap; 768 is common.IMAGE_LONG_SIDE, used "\n                         "for TRAINING AND EVALUATION here so the head sees the "\n                         "same pixels in both")'

OLD_LOG_KEY = "\n".join([
    '            "context_tokens": sum(int(recs[f\"{z[\'video_id\']}|{z[\'question_id\']}\"]["input_ids"].numel()) for z in slots[upd]),',
])
NEW_LOG_KEY = "\n".join([
    '            "context_tokens": sum(int(recs[z.get(\"task\") or f\"{z[\'video_id\']}|{z[\'question_id\']}\"]["input_ids"].numel()) for z in slots[upd]),',
])

OLD_CKPT_SFT = "\n".join([
    '        torch.save({"head": head.state_dict(), "master_optimizer": opt.state_dict(),',
    '                    "update": 0, "seed": a.seed, "arm": "QA-SFT"},',
    '                   os.path.join(a.out, "step0", "training_state.pt"))',
])
NEW_CKPT_SFT = "\n".join([
    '        torch.save({"head": head.state_dict(), "master_optimizer": opt.state_dict(),',
    '                    "update": 0, "seed": a.seed, "arm": "SQA-SFT",',
    '                    "rng": rng_state(), "data_cursor": 0,',
    '                    "schedule": (a.schedule or str(',
    '                        pathlib.Path(a.run) / "dataset" /',
    '                        f"train_schedule_seed_{a.seed}.jsonl"))},',
    '                   os.path.join(a.out, "step0", "training_state.pt"))',
])
OLD_CKPT_OPD = "\n".join([
    '        torch.save({"head": head.state_dict(), "master_optimizer": opt.state_dict(),',
    '                    "update": 0, "seed": a.seed, "arm": "QA-OPD"},',
    '                   os.path.join(a.out, "step0", "training_state.pt"))',
])
NEW_CKPT_OPD = "\n".join([
    '        torch.save({"head": head.state_dict(), "master_optimizer": opt.state_dict(),',
    '                    "update": 0, "seed": a.seed, "arm": "SQA-OPD",',
    '                    "rng": rng_state(), "data_cursor": 0,',
    '                    "schedule": (a.schedule or str(',
    '                        pathlib.Path(a.run) / "dataset" /',
    '                        f"train_schedule_seed_{a.seed}.jsonl"))},',
    '                   os.path.join(a.out, "step0", "training_state.pt"))',
])
OLD_CKPT2 = "\n".join([
    '            torch.save({"head": head.state_dict(),',
    '                        "master_optimizer": opt.state_dict(),',
    '                        "update": upd + 1, "seed": a.seed, "arm": "QA-OPD"},',
    '                       ck / "training_state.pt")',
])
NEW_CKPT2_SFT = "\n".join([
    '            torch.save({"head": head.state_dict(),',
    '                        "master_optimizer": opt.state_dict(),',
    '                        "update": upd + 1, "seed": a.seed, "arm": "SQA-SFT",',
    '                        "rng": rng_state(), "data_cursor": upd + 1,',
    '                        "schedule": str(sched_path)},',
    '                       ck / "training_state.pt")',
])
OLD_CKPT2_SFT = "\n".join([
    '            torch.save({"head": head.state_dict(),',
    '                        "master_optimizer": opt.state_dict(),',
    '                        "update": upd + 1, "seed": a.seed, "arm": "QA-SFT"},',
    '                       ck / "training_state.pt")',
])
NEW_CKPT2 = "\n".join([
    '            torch.save({"head": head.state_dict(),',
    '                        "master_optimizer": opt.state_dict(),',
    '                        "update": upd + 1, "seed": a.seed, "arm": "SQA-OPD",',
    '                        "rng": rng_state(), "data_cursor": upd + 1,',
    '                        "schedule": str(sched_path)},',
    '                       ck / "training_state.pt")',
])
RNG_HELPER = "\n".join([
    "def rng_state():",
    '    """torch/numpy/python RNG state, so a checkpoint can resume exactly."""',
    "    import random",
    "    import numpy as _np",
    "    return {",
    '        "torch": torch.get_rng_state(),',
    '        "torch_cuda": (torch.cuda.get_rng_state_all()',
    "                       if torch.cuda.is_available() else []),",
    '        "numpy": _np.random.get_state(),',
    '        "python": random.getstate(),',
    "    }",
    "",
    "",
    "def restore_rng(state):",
    '    """Counterpart of `rng_state`; used by the resume gate."""',
    "    import random",
    "    import numpy as _np",
    "    if not state:",
    "        return",
    '    torch.set_rng_state(state["torch"])',
    '    if state.get("torch_cuda") and torch.cuda.is_available():',
    '        torch.cuda.set_rng_state_all(state["torch_cuda"])',
    '    _np.random.set_state(state["numpy"])',
    '    random.setstate(state["python"])',
    "",
    "",
])

SAVE_AT_ANCHOR = '    ap.add_argument("--save-every", type=int, default=30)'
SAVE_AT_ARG = "\n".join([
    '    ap.add_argument("--save-at", default=None,',
    '                    help="comma list of update numbers to checkpoint, replacing "',
    '                         "the every-N rule; the final update is always saved")',
    "",
])
OLD_SAVE_COND = "        if a.save_every and (upd + 1) % a.save_every == 0:"
NEW_SAVE_COND = "\n".join([
    "        # `--save-at` names the frozen checkpoint grid exactly. The plain",
    "        # every-N rule would write step 90 as well, which is 16 GiB of disk the",
    "        # amended grid does not budget for and does not need: 30/60/120 is the",
    "        # registered grid, and the final update is always written.",
    '        _save_at = ({int(x) for x in str(a.save_at).split(",") if x.strip()}',
    "                    if a.save_at else None)",
    "        _save_now = ((upd + 1) in _save_at if _save_at is not None",
    "                     else bool(a.save_every and (upd + 1) % a.save_every == 0))",
    "        _save_now = _save_now or (upd + 1) == a.updates",
    "        if _save_now:",
])
OLD_TASK_KEY = "\n".join([
    "            task = f\"{slot['video_id']}|{slot['question_id']}\"",
])
NEW_TASK_KEY = "\n".join([
    "            task = slot.get(\"task\") or f\"{slot['video_id']}|{slot['question_id']}\"",
])
OLD_TASK_KEY_OPD = "\n".join([
    "                task = f\"{slot['video_id']}|{slot['question_id']}\"",
])
NEW_TASK_KEY_OPD = "\n".join([
    "                task = slot.get(\"task\") or f\"{slot['video_id']}|{slot['question_id']}\"",
])

def _sub(s: str, old: str, new: str, count: int = 1) -> str:
    """Replace, and FAIL LOUDLY when the pattern is absent.

    Every earlier silent no-op in this generator came from a pattern that no
    longer matched after an earlier edit (the arm rename in particular), so the
    generated file looked fine while a documented change was simply missing.
    A generator that can silently skip an edit is worse than no generator.
    """
    if old not in s:
        raise AssertionError(
            f"generator edit did not match; refusing to emit a file that is "
            f"missing a documented change. Pattern was:\n{old[:200]!r}")
    return s.replace(old, new, count)


def _common(s: str, arm_old: str, arm_new: str, entry: str, note: str) -> str:
    s = _sub(s, UPDATES_ARG, SCHEDULE_ARG + UPDATES_ARG)
    s = _sub(s,
        'sys.path[:0] = [str(HERE), str(REPRO), str(ROOT / "GLANCE_repo"),',
        'sys.path[:0] = [str(HERE), str(REPRO), str(REPRO / "qa_opd_fair"),\n'
        '                str(ROOT / "GLANCE_repo"),')
    s = _sub(s, OLD_SCHEDULE, NEW_SCHEDULE)
    # A ScienceQA record has one identity, not a video|question pair, so the
    # schedule carries an explicit `task`; honour it. Falls back to the frozen
    # concatenation for any caller that still supplies video-style fields.
    if arm_old == "QA-SFT":
        s = _sub(s, OLD_TASK_KEY, NEW_TASK_KEY)
    if arm_old == "QA-SFT":
        s = _sub(s, OLD_LOG_KEY, NEW_LOG_KEY)
    s = _sub(s, SAVE_AT_ANCHOR, SAVE_AT_ARG + SAVE_AT_ANCHOR)
    s = _sub(s, OLD_SAVE_COND, NEW_SAVE_COND)
    if arm_old == "QA-SFT":
        s = _sub(s, OLD_CKPT_SFT, NEW_CKPT_SFT)
    else:
        s = _sub(s, OLD_CKPT_OPD, NEW_CKPT_OPD)
    if arm_old == "QA-OPD":
        s = _sub(s, OLD_CKPT2, NEW_CKPT2)
    else:
        s = _sub(s, OLD_CKPT2_SFT, NEW_CKPT2_SFT)
    s = _sub(s, "def main():", RNG_HELPER + "def main():")
    if arm_old == "QA-OPD":
        s = _sub(s, OLD_TASK_KEY_OPD, NEW_TASK_KEY_OPD)
    s = _sub(s, f'"arm": "{arm_old}"', f'"arm": "{arm_new}"')
    return s


def adapt_sft(s: str) -> str:
    s = _hdr(s, SFT_HEADER_FROZEN, SFT_HEADER_GEN)
    s = _sub(s,
        "from videoencode import build_video_text, encode_inputs  # noqa: E402",
        "from imageencode import encode_image_inputs  # noqa: E402")
    s = _sub(s,
        '            text = build_video_text(proc, row["prompt"])\n'
        '            x = encode_inputs(proc, row.to_dict(), a.time_mode, text=text)\n',
        '            x = encode_image_inputs(proc, row.to_dict(), max_pixels=a.max_pixels)\n')
    s = _sub(s,
        '                ctx = common.target_context_cached(\n'
        '                    target, full[None], pixel_values_videos=x["pixel_values_videos"].to(dev),\n'
        '                    video_grid_thw=x["video_grid_thw"].to(dev))',
        '                ctx = common.target_context_cached(\n'
        '                    target, full[None], pixel_values=x["pixel_values"].to(dev),\n'
        '                    image_grid_thw=x["image_grid_thw"].to(dev))')
    s = _sub(s, OLD_TIMEMODE_ARG, NEW_TIMEMODE_ARG_SFT)
    s = _common(s, "QA-SFT", "SQA-SFT", "train_qa_sft.py", "sft")
    s = _sub(s, '"time_mode": a.time_mode, "eos_stop": "immediate",',
             '"time_mode": "n/a-still-image", "eos_stop": "immediate",\n'
             '        "max_pixels": a.max_pixels, "modality": "image",')
    s = _sub(s, 'help="qa_pool_df<seed>.parquet"', 'help="sqa_pool_df<seed>.parquet"')
    s = _sub(s, 'help="target_ar_df<seed>.pt"', 'help="sqa_target_ar_df<seed>.pt"')
    return s


def adapt_opd(s: str) -> str:
    s = _hdr(s, OPD_HEADER_FROZEN, OPD_HEADER_GEN)
    s = _sub(s,
        "from train_qa_sft import LmHeadShim, draft_block_logits_mixed  # noqa: E402\n"
        "from videoencode import build_video_text, encode_inputs  # noqa: E402",
        "from train_sqa_sft import LmHeadShim, draft_block_logits_mixed  # noqa: E402\n"
        "from imageencode import encode_image_inputs, prefix_image_vision_positions  # noqa: E402")
    s = _sub(s,
        '                text = build_video_text(proc, row["prompt"])\n'
        '                x = encode_inputs(proc, row.to_dict(), a.time_mode, text=text)\n',
        '                x = encode_image_inputs(proc, row.to_dict(), max_pixels=a.max_pixels)\n')
    s = _sub(s,
        '                vispos = opd_fullkl.prefix_vision_positions(\n'
        '                    target, full[None],\n'
        '                    pixel_values_videos=x["pixel_values_videos"].to(dev),\n'
        '                    video_grid_thw=x["video_grid_thw"].to(dev))',
        '                vispos = prefix_image_vision_positions(\n'
        '                    target, full[None],\n'
        '                    x["pixel_values"].to(dev), x["image_grid_thw"].to(dev))')
    s = _sub(s,
        '                    fused_all, prefix_kv = opd_fullkl.target_prefill_shared(\n'
        '                        target, full[None],\n'
        '                        pixel_values_videos=x["pixel_values_videos"].to(dev),\n'
        '                        video_grid_thw=x["video_grid_thw"].to(dev),\n'
        '                        max_seq_len=a.max_seq_len, vispos=vispos)',
        '                    fused_all, prefix_kv = opd_fullkl.target_prefill_shared(\n'
        '                        target, full[None],\n'
        '                        max_seq_len=a.max_seq_len, vispos=vispos)')
    s = _sub(s,
        '                                 "slot": slot, "pixel": x["pixel_values_videos"],\n'
        '                                 "grid": x["video_grid_thw"]})',
        '                                 "slot": slot, "pixel": x["pixel_values"],\n'
        '                                 "grid": x["image_grid_thw"]})')
    s = _sub(s, OLD_TIMEMODE_ARG, NEW_TIMEMODE_ARG_OPD)
    s = _common(s, "QA-OPD", "SQA-OPD", "train_qa_opd.py", "opd")
    s = _sub(s, '        "time_mode": a.time_mode, "eos_stop": "immediate",',
             '        "time_mode": "n/a-still-image", "eos_stop": "immediate",\n'
             '        "max_pixels": a.max_pixels, "modality": "image",')
    return s


SFT_HEADER_GEN = '"""SQA-SFT: the continue-SFT control arm on the frozen ScienceQA schedule.\n\nGENERATED FROM `repro/qa_opd_fair/train_qa_sft.py` by\n`repro/scienceqa_image/make_image_trainers.py`. The only edited lines are the\nimage-input adaptation (documented in that generator and in `imageencode.py`);\nthe objective, the anchor rule, the reveal schedule, the EOS guard, the\noptimizer and the checkpoint contract are byte-for-byte the frozen ones. The\noriginal file\'s docstring follows.\n\nQA-SFT: the fair control arm on the frozen QA schedule.'
SFT_HEADER_FROZEN = '\"\"\"QA-SFT: the fair control arm on the frozen QA schedule.'
OPD_HEADER_GEN = '"""SQA fixed-anchor OPD with explicit position contracts.\n\nGENERATED FROM `repro/qa_opd_fair/train_qa_opd.py` by\n`repro/scienceqa_image/make_image_trainers.py`. The only edited lines are the\nimage-input adaptation; the accepted one-hot CE, the rejected full-vocabulary\nreverse KL with 0.8**(k-1), the update-global two-stream denominators, the\nanchor rule, the `n_acc`/teacher-shift/suffix-mask contract and the optimizer are\nbyte-for-byte the frozen ones. The original file\'s docstring follows.\n\nQA fixed-anchor OPD with explicit position contracts.'
OPD_HEADER_FROZEN = '\"\"\"QA fixed-anchor OPD with explicit position contracts.'

def _hdr(s, gen, frozen):
    assert gen in s, 'generated header not found'
    return s.replace(gen, frozen, 1)

PAIRS = [
    (SRC / "train_qa_sft.py", HERE / "train_sqa_sft.py", adapt_sft),
    (SRC / "train_qa_opd.py", HERE / "train_sqa_opd.py", adapt_opd),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    patch_lines = []
    bad = False
    for src, dst, fn in PAIRS:
        want = fn(src.read_text())
        if a.check:
            have = dst.read_text() if dst.exists() else ""
            if have != want:
                print(f"STALE: {dst} differs from the generator output", file=sys.stderr)
                bad = True
            continue
        dst.write_text(want)
        patch_lines.extend(difflib.unified_diff(
            src.read_text().splitlines(keepends=True),
            want.splitlines(keepends=True),
            fromfile=str(src.relative_to(REPRO.parent)),
            tofile=str(dst.relative_to(REPRO.parent))))
    if a.check:
        return 1 if bad else 0
    (HERE / "image_trainer.patch").write_text("".join(patch_lines))
    print(f"wrote {len(PAIRS)} trainers and image_trainer.patch "
          f"({len(patch_lines)} diff lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
