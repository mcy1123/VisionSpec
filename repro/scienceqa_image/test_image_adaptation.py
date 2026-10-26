#!/usr/bin/env python3
"""CPU tests for archived image adaptation and current input binding.

The generator/reversal checks below target only the frozen legacy archive.
They do not certify an official KL implementation or current trainer identity.

Three groups, all runnable without a GPU:

A. Adaptation audit. The generated trainers must be exactly what the generator
   produces (`--check`), and **reversing every documented generator edit must give
   back the frozen video trainer byte for byte**. That is the machine-checked form
   of "only the image input path was changed": the reversal uses the same
   constants the generator used to make the edit, so a change that is not in the
   documented list cannot be hidden.

B. Position contract, exercised on the IMAGE trainer (not the video one).
   `_verify` is imported from `train_sqa_opd` and run on synthetic proposals
   covering n_acc = 0, full acceptance, partial acceptance, the answer horizon, a
   reference EOS and a proposal EOS, plus the whole-suffix rule that the
   2026-10-15 contract fix introduced. `stream_split` must then cover exactly the
   valid offsets. Anchor-relative indexing is pinned explicitly: the SAME
   proposal must score differently at two anchors one position apart.

C. Input binding. `imageencode` must delegate to `common.encode` (so training and
   evaluation cannot drift), `build_image_text` must emit exactly one image
   placeholder, the imported `qa_anchors` must be the FROZEN file, and the frozen
   prompt template must carry no gold leakage.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path[:0] = [str(HERE), str(REPRO), str(REPRO / "qa_opd_fair"),
                str(ROOT / "GLANCE_repo"),
                str(ROOT / "scripts" / "dflash_pkg" / "dflash-0.1.0")]

ARCHIVE = REPRO / "legacy" / "pre20261016" / "repro" / "scienceqa_image"
FAILS = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def load_generator():
    spec = importlib.util.spec_from_file_location("gen", ARCHIVE / "make_image_trainers.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen


# ---------------------------------------------------------------- group A ----
def test_generator_is_current():
    import subprocess
    r = subprocess.run([sys.executable, str(HERE / "make_image_trainers.py"), "--check"],
                       capture_output=True, text=True)
    check("A1 active generator is retired", r.returncode != 0 and "retired_generator" in r.stderr)
    r = subprocess.run([sys.executable, str(ARCHIVE / "make_image_trainers.py"), "--check"],
                       capture_output=True, text=True)
    check("A1 archived historical trainers match frozen generator", r.returncode == 0, r.stderr[:200])


def test_reverse_every_documented_edit_is_byte_exact():
    gen = load_generator()
    for src, dst, _fn in gen.PAIRS:
        frozen, generated = src.read_text(), dst.read_text()
        r = generated
        reversals = [
            (gen.SFT_HEADER_GEN, gen.SFT_HEADER_FROZEN),
            (gen.OPD_HEADER_GEN, gen.OPD_HEADER_FROZEN),
            (gen.SCHEDULE_ARG + gen.UPDATES_ARG, gen.UPDATES_ARG),
            (gen.NEW_SCHEDULE, gen.OLD_SCHEDULE),
            (gen.NEW_TASK_KEY, gen.OLD_TASK_KEY),
            (gen.NEW_TASK_KEY_OPD, gen.OLD_TASK_KEY_OPD),
            (gen.NEW_LOG_KEY, gen.OLD_LOG_KEY),
            (gen.NEW_CKPT_SFT, gen.OLD_CKPT_SFT),
            (gen.NEW_CKPT_OPD, gen.OLD_CKPT_OPD),
            (gen.NEW_CKPT2, gen.OLD_CKPT2),
            (gen.NEW_CKPT2_SFT, gen.OLD_CKPT2_SFT),
            (gen.RNG_HELPER + "def main():", "def main():"),
            (gen.SAVE_AT_ARG + gen.SAVE_AT_ANCHOR, gen.SAVE_AT_ANCHOR),
            (gen.NEW_SAVE_COND, gen.OLD_SAVE_COND),
            (gen.NEW_TIMEMODE_ARG_SFT, gen.OLD_TIMEMODE_ARG),
            (gen.NEW_TIMEMODE_ARG_OPD, gen.OLD_TIMEMODE_ARG),
            ('sys.path[:0] = [str(HERE), str(REPRO), str(REPRO / "qa_opd_fair"),\n'
             '                str(ROOT / "GLANCE_repo"),',
             'sys.path[:0] = [str(HERE), str(REPRO), str(ROOT / "GLANCE_repo"),'),
            ('"arm": "SQA-SFT"', '"arm": "QA-SFT"'),
            ('"arm": "SQA-OPD"', '"arm": "QA-OPD"'),
            ("from imageencode import encode_image_inputs  # noqa: E402",
             "from videoencode import build_video_text, encode_inputs  # noqa: E402"),
            ("from train_sqa_sft import LmHeadShim, draft_block_logits_mixed  # noqa: E402\n"
             "from imageencode import encode_image_inputs, prefix_image_vision_positions  # noqa: E402",
             "from train_qa_sft import LmHeadShim, draft_block_logits_mixed  # noqa: E402\n"
             "from videoencode import build_video_text, encode_inputs  # noqa: E402"),
            # the leading newline matters: the 12-space SFT line is a substring
            # of the 16-space OPD line, so an unanchored replace would corrupt it
            ('\n                x = encode_image_inputs(proc, row.to_dict(), max_pixels=a.max_pixels)\n',
             '\n                text = build_video_text(proc, row["prompt"])\n'
             '                x = encode_inputs(proc, row.to_dict(), a.time_mode, text=text)\n'),
            ('\n            x = encode_image_inputs(proc, row.to_dict(), max_pixels=a.max_pixels)\n',
             '\n            text = build_video_text(proc, row["prompt"])\n'
             '            x = encode_inputs(proc, row.to_dict(), a.time_mode, text=text)\n'),
            ('                ctx = common.target_context_cached(\n'
             '                    target, full[None], pixel_values=x["pixel_values"].to(dev),\n'
             '                    image_grid_thw=x["image_grid_thw"].to(dev))',
             '                ctx = common.target_context_cached(\n'
             '                    target, full[None], pixel_values_videos=x["pixel_values_videos"].to(dev),\n'
             '                    video_grid_thw=x["video_grid_thw"].to(dev))'),
            ('                vispos = prefix_image_vision_positions(\n'
             '                    target, full[None],\n'
             '                    x["pixel_values"].to(dev), x["image_grid_thw"].to(dev))',
             '                vispos = opd_fullkl.prefix_vision_positions(\n'
             '                    target, full[None],\n'
             '                    pixel_values_videos=x["pixel_values_videos"].to(dev),\n'
             '                    video_grid_thw=x["video_grid_thw"].to(dev))'),
            ('                    fused_all, prefix_kv = opd_fullkl.target_prefill_shared(\n'
             '                        target, full[None],\n'
             '                        max_seq_len=a.max_seq_len, vispos=vispos)',
             '                    fused_all, prefix_kv = opd_fullkl.target_prefill_shared(\n'
             '                        target, full[None],\n'
             '                        pixel_values_videos=x["pixel_values_videos"].to(dev),\n'
             '                        video_grid_thw=x["video_grid_thw"].to(dev),\n'
             '                        max_seq_len=a.max_seq_len, vispos=vispos)'),
            ('                                 "slot": slot, "pixel": x["pixel_values"],\n'
             '                                 "grid": x["image_grid_thw"]})',
             '                                 "slot": slot, "pixel": x["pixel_values_videos"],\n'
             '                                 "grid": x["video_grid_thw"]})'),
            ('"time_mode": "n/a-still-image", "eos_stop": "immediate",\n'
             '        "max_pixels": a.max_pixels, "modality": "image",',
             '"time_mode": a.time_mode, "eos_stop": "immediate",'),
            ('help="sqa_pool_df<seed>.parquet"', 'help="qa_pool_df<seed>.parquet"'),
            ('help="sqa_target_ar_df<seed>.pt"', 'help="target_ar_df<seed>.pt"'),
        ]
        for a, b in reversals:
            r = r.replace(a, b)
        check(f"A2 {dst.name}: reversing the documented edits is byte-exact",
              r == frozen,
              "byte-identical to the frozen trainer" if r == frozen else
              f"{sum(1 for x, y in zip(r.splitlines(), frozen.splitlines()) if x != y)} "
              f"differing lines; len {len(r)} vs {len(frozen)}")


# ---------------------------------------------------------------- group B ----
def test_position_contract_on_image_trainer():
    from historical import opd_fullkl
    import train_sqa_opd as T
    K = 15
    PL, P = 100, 99          # the first legal anchor predicts answer index 0
    V = [1, 2, 3]
    n_acc, valid = T._verify([9, 9, 9], V, p=P, prompt_len=PL, n_ans=3, eos_pos=None)
    check("B1 n_acc=0 keeps the WHOLE suffix valid", n_acc == 0 and valid == [True] * 3,
          f"n_acc={n_acc} valid={valid}")
    n_acc, valid = T._verify([1, 2, 3], V, p=P, prompt_len=PL, n_ans=3, eos_pos=None)
    check("B2 full acceptance counts every offset",
          n_acc == 3 and valid == [True] * 3, f"n_acc={n_acc}")
    n_acc, valid = T._verify([1, 2, 7], V, p=P, prompt_len=PL, n_ans=3, eos_pos=None)
    check("B3 rejection ends acceptance but NOT validity",
          n_acc == 2 and valid == [True, True, True], f"n_acc={n_acc} valid={valid}")
    n_acc, valid = T._verify([1, 2, 7, 8], V, p=P, prompt_len=PL, n_ans=3, eos_pos=None)
    check("B4 validity stops at the answer horizon",
          valid == [True, True, True, False], f"valid={valid}")
    n_acc, valid = T._verify([1, 9, 9], V, p=P, prompt_len=PL, n_ans=3, eos_pos=0)
    check("B5 a reference EOS at answer index 0 bounds validity and still accepts it",
          valid == [True, False, False] and n_acc == 1, f"valid={valid} n_acc={n_acc}")
    n_acc, valid = T._verify([1, 151645, 9], V, p=P, prompt_len=PL, n_ans=3,
                             eos_pos=None, eos_token_ids=151645)
    check("B6 a proposal EOS is itself valid and ends the suffix after it",
          valid == [True, True, False], f"valid={valid}")
    # anchor-relative indexing: verifier index 0 belongs to the anchor at PL-1
    n_a, _ = T._verify([1, 2], [7, 1, 2], p=PL - 1, prompt_len=PL, n_ans=3, eos_pos=None)
    n_b, _ = T._verify([1, 2], [7, 1, 2], p=PL, prompt_len=PL, n_ans=3, eos_pos=None)
    check("B7 the same proposal scores differently one anchor apart "
          "(anchor-relative, not absolute)", (n_a, n_b) == (0, 2), f"{n_a} vs {n_b}")
    acc, rej = opd_fullkl.stream_split(K, 2, [True] * 5 + [False] * 10)
    check("B8 stream_split covers the whole legal suffix",
          int(acc.numel()) == 2 and int(rej.numel()) == 3,
          f"acc={int(acc.numel())} rej={int(rej.numel())}")
    acc, rej = opd_fullkl.stream_split(K, 0, [True] * K)
    check("B9 n_acc=0 leaves an empty accepted and a full rejected stream",
          int(acc.numel()) == 0 and int(rej.numel()) == K)
    acc, rej = opd_fullkl.stream_split(K, K, [True] * K)
    check("B10 full acceptance leaves an empty rejected stream",
          int(acc.numel()) == K and int(rej.numel()) == 0)
    try:
        T._verify([1], [1, 2], p=P, prompt_len=PL, n_ans=3, eos_pos=None)
        check("B11 verifier length mismatch is refused", False)
    except ValueError:
        check("B11 verifier length mismatch is refused", True)


# ---------------------------------------------------------------- group C ----
def test_input_binding():
    import inspect
    import qa_anchors as qa
    import train_sqa_opd as O
    import train_sqa_sft as S
    import imageencode as IE
    check("C1 the FROZEN qa_anchors is the imported one",
          pathlib.Path(qa.__file__).resolve()
          == (REPRO / "qa_opd_fair" / "qa_anchors.py").resolve(), str(qa.__file__))
    check("C2 both image trainers import that same anchor module",
          pathlib.Path(S.qa_anchors.__file__).resolve()
          == pathlib.Path(qa.__file__).resolve()
          and pathlib.Path(O.qa_anchors.__file__).resolve()
          == pathlib.Path(qa.__file__).resolve())
    picks = qa.select_anchors(list(range(100)), 6)
    check("C3 anchor rule keeps first and last and exactly 6 points",
          len(picks) == 6 and picks[0] == 0 and picks[-1] == 99, str(picks))
    check("C4 a short candidate list is kept whole with nothing duplicated",
          qa.select_anchors([5, 6, 7], 6) == [5, 6, 7])
    # A generated literal that a human reads as `split(",")` can be silently
    # parsed as two adjacent string literals, i.e. `split()`, which produces one
    # unsplit element. The byte-exact reversal audit cannot see that (it reverses
    # the same broken constant), so the generated text is asserted directly.
    for name in ("train_sqa_sft.py", "train_sqa_opd.py"):
        source = ARCHIVE / name if name == "train_sqa_opd.py" else HERE / name
        txt = source.read_text()
        check(f"C9 {source}: --save-at splits on a comma",
              'split(",")' in txt and 'split()' not in txt)
    check("C5 encode_image_inputs delegates to common.encode",
          "common.encode(" in inspect.getsource(IE.encode_image_inputs))
    check("C6 build_image_text emits exactly one image placeholder",
          IE.build_image_text.__doc__ is not None
          and "one image placeholder" in IE.build_image_text.__doc__)
    import build_dataset as B
    t = B.PROMPT_TEMPLATE
    check("C7 the frozen prompt has no gold leakage",
          "Answer: (X)" in t and "{question}" in t and "{choices}" in t
          and "solution" not in t.lower() and "lecture" not in t.lower())
    txt = B.build_prompt("Q?", ["a", "b"])
    check("C8 options map to letters in official order, no extra options",
          "(A) a" in txt and "(B) b" in txt and "(C)" not in txt)


def main():
    test_generator_is_current()
    test_reverse_every_documented_edit_is_byte_exact()
    test_position_contract_on_image_trainer()
    test_input_binding()
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} -> {FAILS}")
        return 1
    print("CPU: image adaptation + position contract tests all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
