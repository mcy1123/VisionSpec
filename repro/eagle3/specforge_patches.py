#!/usr/bin/env python3
"""Apply the small SpecForge patches EAGLE-3-on-Qwen3-VL needs. Idempotent.

SpecForge is text-and-Llama-shaped. Its EAGLE-3 draft model genuinely implements
mRoPE and genuinely accepts 3-axis position ids, but for a Qwen3-VL target three
things are wrong or missing -- and every one of them fails *silently*, which is
the reason this file exists as a checked, revertible patch rather than three
hand edits:

1. `_OFFLINE_EAGLE3_KEYS` gates what the feature store will even fetch. Without
   `"position_ids"` in that tuple an on-disk `position_ids` is discarded without
   a warning, and the draft trains with plain 1D RoPE against a multimodal
   target.

2. `normalize_offline_sample` builds the trainer's tensor dict and never emits
   `position_ids`, so the 3-axis tensor has no route into the model.

3. `apply_multimodal_rotary_pos_emb` implements only the *chunked* layout
   (`[TTT...HHH...WWW]`). Qwen3-VL sets `mrope_interleaved: true`, so the correct
   frequencies are laid out `[THTHWH...TT]`. Feeding non-interleaved sections to
   an interleaved model does not raise -- it produces wrong rotations and a
   quietly degraded acceptance rate. transformers' own
   `Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope` is the reference.

4. `_init_rope` reads `rope_type` before `type`. Qwen3-VL ships
   `rope_type: "default"` *together with* `mrope_section` and
   `mrope_interleaved`, so copying the target's rope_scaling verbatim hits the
   `"default"` early-return and silently installs 1D RoPE. The patch prefers
   mrope when a section is declared, and raises rather than guessing when
   `mrope_section` appears without `mrope_interleaved`.

Patch 3 is verified numerically against transformers by
`test_mrope_interleaved.py`, on CPU, before any GPU work happens.

    python specforge_patches.py --check      # report, exit 1 if incomplete
    python specforge_patches.py --apply
    python specforge_patches.py --revert
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

SPECFORGE = os.environ.get("SPECFORGE_ROOT", "/home/mcy/Projects/SpecForge")
BACKUP_SUFFIX = ".eagle3qwen3vl.orig"

READER = "specforge/runtime/data_plane/offline_reader.py"
NORMALIZER = "specforge/algorithms/eagle3/data.py"
DRAFT = "specforge/modeling/draft/llama3_eagle.py"

MARKER = "EAGLE3QWEN3VL-PATCH"


# ---------------------------------------------------------------------------
# patch bodies
# ---------------------------------------------------------------------------

READER_OLD = (
    '_OFFLINE_EAGLE3_KEYS = ("input_ids", "loss_mask", "hidden_state", "aux_hidden_state")'
)
READER_NEW = (
    "# " + MARKER + ": position_ids added for multimodal (Qwen3-VL) targets.\n"
    "# Without it the feature store drops the 3-axis mRoPE tensor silently.\n"
    '_OFFLINE_EAGLE3_KEYS = (\n'
    '    "input_ids",\n'
    '    "loss_mask",\n'
    '    "hidden_state",\n'
    '    "aux_hidden_state",\n'
    '    "position_ids",\n'
    ')'
)

NORMALIZER_OLD = """    return {
        "attention_mask": torch.ones_like(loss_mask, dtype=torch.long),
        "loss_mask": loss_mask,
        "target": target,
        "hidden_state": hidden_state,
        "input_ids": input_ids,
    }"""
NORMALIZER_NEW = """    out = {
        "attention_mask": torch.ones_like(loss_mask, dtype=torch.long),
        "loss_mask": loss_mask,
        "target": target,
        "hidden_state": hidden_state,
        "input_ids": input_ids,
    }
    # """ + MARKER + """: carry 3-axis mRoPE positions through to the model.
    # Shape stays [axes, batch, seq]; DataCollatorWithPadding pads dim -1 via
    # padding_position_ids_3d, and OnlineEagle3Model._prepare_position_ids
    # accepts ndim == 3 unchanged. The offline reader must also list
    # "position_ids" in _OFFLINE_EAGLE3_KEYS or this never runs.
    positions = raw.get("position_ids")
    if positions is not None:
        out["position_ids"] = positions[:, :, :max_len].long()
    return out"""

# NOTE: this block is ONE triple-quoted literal, so it must not try to
# interpolate MARKER with `+` -- that would be swallowed as inert text (which is
# exactly the bug this comment replaces). The marker is written out literally.
HELPER = '''def _mrope_interleaved_axis_index(head_dim, mrope_section, device):
    """Channel -> axis assignment for Qwen3-VL's interleaved mRoPE.

    EAGLE3QWEN3VL-PATCH port of
    ``transformers.Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope``: start
    from the all-temporal layout, then overwrite the strided H and W slices. On
    the ``head_dim // 2`` frequency block, channel ``c`` therefore comes from
    temporal for ``c % 3 == 0``, from height for ``c % 3 == 1`` below
    ``3 * mrope_section[1]``, and from width for ``c % 3 == 2`` below
    ``3 * mrope_section[2]``. The doubled half repeats the same assignment.
    """
    import torch

    half = head_dim // 2
    axis = torch.zeros(half, dtype=torch.long, device=device)
    for dim, offset in enumerate((1, 2), start=1):
        length = min(mrope_section[dim] * 3, half)
        index = torch.arange(offset, length, 3, dtype=torch.long, device=device)
        axis[index] = dim
    return torch.cat([axis, axis])


def _select_mrope_axes(t, mrope_section, interleaved):
    """Reduce a (3, bs, seq, head_dim) cos/sin block to (bs, seq, head_dim)."""
    import torch

    if interleaved:
        index = _mrope_interleaved_axis_index(
            t.shape[-1], mrope_section, t.device)
        index = index.view(1, 1, 1, -1).expand(1, t.shape[1], t.shape[2], -1)
        return t.gather(0, index).squeeze(0)
    section = [s * 2 for s in mrope_section]
    return torch.cat(
        [m[i % 3] for i, m in enumerate(t.split(section, dim=-1))], dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section,
                                    unsqueeze_dim=1, mrope_interleaved=False):'''

DRAFT_SIG_OLD = (
    "def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):"
)

DRAFT_BODY_OLD = """    mrope_section = mrope_section * 2
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)"""
DRAFT_BODY_NEW = """    # """ + MARKER + """: chunked (Qwen2.5-VL) and interleaved (Qwen3-VL) layouts.
    cos = _select_mrope_axes(cos, mrope_section, mrope_interleaved).unsqueeze(
        unsqueeze_dim)
    sin = _select_mrope_axes(sin, mrope_section, mrope_interleaved).unsqueeze(
        unsqueeze_dim)"""

INITROPE_OLD = """    def _init_rope(self):
        rope_theta, rope_scaling = get_rope_config(self.config)
        self.rope_scaling = rope_scaling
"""
INITROPE_NEW = """    def _init_rope(self):
        rope_theta, rope_scaling = get_rope_config(self.config)
        self.rope_scaling = rope_scaling
        # """ + MARKER + """: default False; the mrope branch sets it from config.
        self.mrope_interleaved = False
"""

DISPATCH_OLD = '            scaling_type = rope_get("rope_type", rope_get("type"))'
DISPATCH_NEW = '''            scaling_type = rope_get("rope_type", rope_get("type"))
            # ''' + MARKER + ''': Qwen3-VL ships rope_type="default" together with
            # mrope_section / mrope_interleaved. Reading rope_type first would take
            # the "default" early-return below and silently install 1D RoPE on a
            # multimodal target. Prefer mrope when a section is declared, and
            # refuse rather than guess when the layout is not stated.
            if scaling_type == "default" and rope_get("mrope_section") is not None:
                if rope_get("mrope_interleaved") is None:
                    raise ValueError(
                        "rope_scaling declares mrope_section but not "
                        "mrope_interleaved; refusing to guess between the chunked "
                        "(Qwen2.5-VL) and interleaved (Qwen3-VL) mRoPE layouts"
                    )
                scaling_type = "mrope"'''

MROPE_BRANCH_OLD = """            elif scaling_type == "mrope":
                self.rotary_emb = LlamaMutiRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=rope_theta,
                )"""
MROPE_BRANCH_NEW = MROPE_BRANCH_OLD + """
                # """ + MARKER + """: Qwen3-VL needs the interleaved channel layout.
                self.mrope_interleaved = bool(
                    rope_get("mrope_interleaved", False))"""

# The four attention call sites all pass the section positionally; the flag is
# appended after it. Matched by regex because the surrounding indentation and the
# trailing `unsqueeze_dim=2` differ between sites.
CALLSITE_RE = re.compile(
    r'^(?P<indent>[ \t]*)self\.rope_scaling\["mrope_section"\],$', re.M)
CALLSITE_N = 4


def _path(rel):
    return os.path.join(SPECFORGE, rel)


def _self_test_patch_text():
    """Catch unresolved `+ MARKER +` inside a triple-quoted patch body.

    A single triple-quoted literal cannot interpolate; writing `"..." + MARKER +
    "..."` inside one leaves the concatenation as inert text and ships `MARKER`
    straight into SpecForge as an undefined name. That happened once here, so it
    is now checked before any file is touched.
    """
    bodies = {
        "READER_NEW": READER_NEW, "NORMALIZER_NEW": NORMALIZER_NEW,
        "HELPER": HELPER, "DRAFT_BODY_NEW": DRAFT_BODY_NEW,
        "INITROPE_NEW": INITROPE_NEW, "DISPATCH_NEW": DISPATCH_NEW,
        "MROPE_BRANCH_NEW": MROPE_BRANCH_NEW,
    }
    for name, body in bodies.items():
        if "MARKER" in body:
            raise AssertionError(
                f"patch body {name} contains the bare token 'MARKER'; a "
                f"triple-quoted literal swallowed a concatenation and would "
                f"ship an undefined name into SpecForge")
        if MARKER not in body:
            raise AssertionError(f"patch body {name} is missing the {MARKER} tag")


def _read(rel):
    with open(_path(rel)) as handle:
        return handle.read()


def _write(rel, text):
    path = _path(rel)
    backup = path + BACKUP_SUFFIX
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    with open(path, "w") as handle:
        handle.write(text)


def add_callsite_flags(text):
    """Append `mrope_interleaved=...` to every mRoPE attention call site."""
    def repl(match):
        indent = match.group("indent")
        return (f'{indent}self.rope_scaling["mrope_section"],\n'
                f'{indent}mrope_interleaved=getattr(self, "mrope_interleaved", False),')

    new, count = CALLSITE_RE.subn(repl, text)
    if count != CALLSITE_N:
        raise AssertionError(
            f"expected {CALLSITE_N} mrope call sites, matched {count}; upstream "
            f"changed shape, review the patch")
    return new


def apply_one(rel, old, new, *, regex=None, label=""):
    text = _read(rel)
    if new in text or (MARKER in text and label and label in text):
        return "already"
    if regex is not None:
        return "regex"
    if text.count(old) != 1:
        raise AssertionError(
            f"{rel}: expected exactly 1 occurrence of the anchor for {label!r}, "
            f"found {text.count(old)}")
    _write(rel, text.replace(old, new, 1))
    return "applied"


def apply_all():
    _self_test_patch_text()
    status = {}

    # --- 1. offline reader key allowlist -----------------------------------
    text = _read(READER)
    if '"position_ids"' in text and MARKER in text:
        status["reader.position_ids"] = "already"
    else:
        if text.count(READER_OLD) != 1:
            raise AssertionError(f"{READER}: anchor not found exactly once")
        _write(READER, text.replace(READER_OLD, READER_NEW, 1))
        status["reader.position_ids"] = "applied"

    # --- 2. normalizer emits position_ids ----------------------------------
    text = _read(NORMALIZER)
    if "position_ids" in text and MARKER in text:
        status["normalizer.position_ids"] = "already"
    else:
        if text.count(NORMALIZER_OLD) != 1:
            raise AssertionError(f"{NORMALIZER}: anchor not found exactly once")
        _write(NORMALIZER, text.replace(NORMALIZER_OLD, NORMALIZER_NEW, 1))
        status["normalizer.position_ids"] = "applied"

    # --- 3. interleaved mRoPE ----------------------------------------------
    text = _read(DRAFT)
    if "_select_mrope_axes" in text:
        status["draft.interleaved"] = "already"
    else:
        if text.count(DRAFT_SIG_OLD) != 1:
            raise AssertionError(f"{DRAFT}: signature anchor not found exactly once")
        text = text.replace(DRAFT_SIG_OLD, HELPER, 1)
        if text.count(DRAFT_BODY_OLD) != 1:
            raise AssertionError(f"{DRAFT}: body anchor not found exactly once")
        text = text.replace(DRAFT_BODY_OLD, DRAFT_BODY_NEW, 1)
        status["draft.interleaved"] = "applied"

    # --- 4a. mrope dispatch guard ------------------------------------------
    if DISPATCH_NEW in text:
        status["draft.dispatch_guard"] = "already"
    else:
        if text.count(DISPATCH_OLD) != 1:
            raise AssertionError(f"{DRAFT}: dispatch anchor not found exactly once")
        text = text.replace(DISPATCH_OLD, DISPATCH_NEW, 1)
        status["draft.dispatch_guard"] = "applied"

    # --- 4b. interleaved flag on the module + mrope branch ------------------
    if "self.mrope_interleaved = False" in text:
        status["draft.flag_init"] = "already"
    else:
        if text.count(INITROPE_OLD) != 1:
            raise AssertionError(f"{DRAFT}: _init_rope anchor not found exactly once")
        text = text.replace(INITROPE_OLD, INITROPE_NEW, 1)
        status["draft.flag_init"] = "applied"

    if 'rope_get("mrope_interleaved", False)' in text:
        status["draft.flag_mrope_branch"] = "already"
    else:
        if text.count(MROPE_BRANCH_OLD) != 1:
            raise AssertionError(f"{DRAFT}: mrope branch anchor not found exactly once")
        text = text.replace(MROPE_BRANCH_OLD, MROPE_BRANCH_NEW, 1)
        status["draft.flag_mrope_branch"] = "applied"

    # --- 4c. thread the flag through the four attention call sites ----------
    if text.count("mrope_interleaved=getattr(self,") == CALLSITE_N:
        status["draft.callsites"] = "already"
    else:
        text = add_callsite_flags(text)
        status["draft.callsites"] = "applied"

    _write(DRAFT, text)
    return status


def check():
    """Static verification: every patch present, and the file still compiles."""
    import ast

    _self_test_patch_text()
    problems = []
    reader = _read(READER)
    if '"position_ids"' not in reader:
        problems.append(f"{READER}: position_ids not in _OFFLINE_EAGLE3_KEYS")
    normalizer = _read(NORMALIZER)
    if 'raw.get("position_ids")' not in normalizer:
        problems.append(f"{NORMALIZER}: normalize_offline_sample does not emit "
                        f"position_ids")
    draft = _read(DRAFT)
    for needle, label in (
        ("_mrope_interleaved_axis_index", "interleaved axis index helper"),
        ("_select_mrope_axes", "axis selection helper"),
        ("mrope_interleaved=False", "signature keyword"),
        ("self.mrope_interleaved = False", "flag init"),
        ('rope_get("mrope_interleaved", False)', "flag from config"),
        ("refusing to guess between the chunked", "dispatch guard"),
    ):
        if needle not in draft:
            problems.append(f"{DRAFT}: missing {label} ({needle!r})")
    n_calls = draft.count("mrope_interleaved=getattr(self,")
    if n_calls != CALLSITE_N:
        problems.append(f"{DRAFT}: {n_calls} call sites carry the flag, "
                        f"expected {CALLSITE_N}")

    for rel in (READER, NORMALIZER, DRAFT):
        try:
            ast.parse(_read(rel))
        except SyntaxError as exc:
            problems.append(f"{rel}: syntax error after patching: {exc}")
    return problems


def revert():
    reverted = []
    for rel in (READER, NORMALIZER, DRAFT):
        backup = _path(rel) + BACKUP_SUFFIX
        if os.path.exists(backup):
            shutil.copy2(backup, _path(rel))
            reverted.append(rel)
    return reverted


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    if not os.path.isdir(SPECFORGE):
        print(f"SpecForge checkout not found: {SPECFORGE}", file=sys.stderr)
        return 2
    print(f"SpecForge root: {SPECFORGE}")

    if a.revert:
        for rel in revert():
            print(f"  reverted {rel}")
        return 0

    if a.apply:
        for name, state in apply_all().items():
            print(f"  {state:8s} {name}")

    problems = check()
    if problems:
        print("\nPATCH CHECK FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\npatch check: all 6 patch points present, 4/4 call sites threaded, "
          "files parse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
