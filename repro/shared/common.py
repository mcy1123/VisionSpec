"""Shared paths, loaders and encoding for the GLANCE reproduction.

Every constant here is annotated with where it came from. `CONFIG.md` at the
reproduction root carries the same table with the paper/repo citations behind
it; anything marked UNKNOWN there is a documented deviation, not an invention
dressed up as the recipe.
"""

from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import json
import os
import sys

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

WORKSPACE = str(_REPRO_ROOT.parent)
ROOT = os.environ.get("GLANCE_ROOT", "/mnt/local/mcy/GLANCE")

REPO = os.path.join(WORKSPACE, "GLANCE_repo")
DFLASH_PKG = os.path.join(WORKSPACE, "scripts", "dflash_pkg", "dflash-0.1.0")
for _p in (REPO, DFLASH_PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TARGET_PATH = f"{ROOT}/models/Qwen3-VL-8B-Instruct"
HEAD_WARM_PATH = f"{ROOT}/models/Qwen3-8B-DFlash-b16"
DATASETS = f"{ROOT}/datasets"
DATA = f"{ROOT}/data"
CKPT = f"{ROOT}/checkpoints"
RESULTS = f"{ROOT}/results"

# --------------------------------------------------------------------------
# Recipe constants
# --------------------------------------------------------------------------

BLOCK_SIZE = 16          # [Table 5] block size B
HORIZON = 15             # [App.D] "block horizon L=15"
KEPT_LAYERS = (1, 9, 17, 25, 33)   # [Table 5] / [head cfg] target_layer_ids
MASK_ID = 151669         # [head cfg] dflash_config.mask_token_id
IMAGE_LONG_SIDE = 768    # [App.D] "image long side 768 px"
MAX_SEQ_LEN = 2048       # [Table 5]

LR = 4e-5                # [Table 5] AdamW
BETAS = (0.9, 0.95)      # [App.D]
WEIGHT_DECAY = 0.0       # [App.D] "no weight decay"
GRAD_CLIP = 1.0          # [App.D] "gradient clipping 1.0"
EPOCHS = 1               # [Table 5]
ACCUM = 8                # [Table 5] gradient accumulation
SEED = 0                 # [App.D] "seed 0"

DECAY = 0.8              # [Table 5] "coverage CE, 0.8^offset weight"
# [UNKNOWN] the paper says only "exposes a random prefix so the head learns
# every reveal length"; the DFlash/D2SD reference implementation (NeMo
# `prefix_weight_base`) defaults to 0.9.
PREFIX_BASE = 0.9
# The reveal length must reach 1, because 1 is what decode actually uses:
# `glance/decode.py` sets `block_ids[:, 1:] = mask_id`, so only the root is
# visible at inference. With reveal = 1 the weighted objective collapses to the
# fixed-anchor DFlash loss `0.8 ** (k - 1)`, which is the objective the
# warm-start head was trained under -- so covering it costs nothing and leaving
# it out would train a head that never saw the decode-time mask.
PREFIX_MIN = 1

N_COCO_PROMPTS = 8000    # [Table 5] "8000 COCO + 8000 TextVQA"
N_TEXTVQA_PROMPTS = 8000
GEN_NEW_TOKENS = 256     # [Table 5] "target greedy, 256 new tokens"

# [UNKNOWN] the paper never prints the prompt strings; the repo ships the task
# JSONLs separately and they are not in the released tarball. These are the
# community-standard wordings and are flagged as a deviation in CONFIG.md §8.
PROMPT_CAPTION = "Describe this image in detail."
PROMPT_CHARTQA_TEMPLATE = "Answer the question using a single word or phrase. {q}"


def ensure_dirs():
    for d in (DATA, CKPT, RESULTS):
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------

def load_image(path, long_side=IMAGE_LONG_SIDE):
    """Load an image, capping its long side. [App.D] 768 px for training."""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if long_side and max(image.size) > long_side:
        w, h = image.size
        scale = long_side / max(w, h)
        image = image.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                             Image.BICUBIC)
    return image


def read_jsonl(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, rows):
    with open(path, "a") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def load_target(path=TARGET_PATH, device="cuda", attn_implementation="sdpa", dtype=None):
    """The frozen Qwen3-VL target and its processor. [§5.1] never modified."""
    import torch
    from transformers import AutoProcessor

    from glance.vl import target_class_for

    use_fast_patch_embed()
    cls = target_class_for(path)
    target = cls.from_pretrained(
        path, dtype=dtype or torch.bfloat16,
        attn_implementation=attn_implementation).to(device).eval()
    target.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(path)
    return target, processor


def load_head(path, dtype=None, device="cuda"):
    """A trained (or warm-start) block head. [repo glance/head.py]"""
    import torch
    from dflash.model import DFlashDraftModel

    head = DFlashDraftModel.from_pretrained(
        path, dtype=dtype or torch.bfloat16)
    return head.to(device)


def build_processor_messages(prompt, image):
    """One user turn: image first, then text. [repo main_table.py:102]"""
    return [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}]}]


def encode(processor, prompt, image):
    """Tokenise one (image, prompt) pair. Returns cpu tensors."""
    inputs = processor.apply_chat_template(
        build_processor_messages(prompt, image),
        add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt")
    return (inputs["input_ids"],
            inputs.get("pixel_values"),
            inputs.get("image_grid_thw"))


def context_feature(hidden_states, layer_ids=KEPT_LAYERS):
    """Concatenate the target's fused hidden states at the kept layers.

    [repo glance/head.py] `hidden_states[0]` is the embedding output, so layer
    `i` sits at index `i + 1`.
    """
    import torch

    return torch.cat([hidden_states[i + 1] for i in layer_ids], dim=-1)


# --------------------------------------------------------------------------
# The variable-prefix (VP) reveal schedule -- see CONFIG.md §5
# --------------------------------------------------------------------------

def prefix_weights(block_size=BLOCK_SIZE, base=None, min_prefix=PREFIX_MIN):
    """The truncated-geometric prior over visible-prefix lengths.

    `Pr(l) ~ base ** l` over `[min_prefix, block_size - 1]`, which is exactly
    what the DFlash variable-prefix objective samples (NeMo
    `_sample_prefix_lengths`). The lower bound keeps at least one masked
    target per block and skips the degenerate fixed-anchor case.
    """
    import torch

    base = PREFIX_BASE if base is None else float(base)
    lengths = torch.arange(min_prefix, block_size)
    weights = torch.tensor([base ** int(l) for l in lengths], dtype=torch.float64)
    return lengths, weights / weights.sum()


def offset_weights(block_size=BLOCK_SIZE, decay=DECAY):
    """[Table 5] `0.8^offset`, anchored at the reveal boundary.

    D2SD (arXiv:2606.04446) Eq. 7 defines the VP loss as

        w_k = exp(-(k - l - 1) / tau),   k = l+1, ..., gamma-1

    over its own indexing, where `l` counts the re-drafted boundary and the
    visible prefix is positions `0..l`. Writing `reveal = l + 1` for the number
    of visible positions, its first masked position `k = l + 1 = reveal` has
    exponent `0`, so the decay is measured from the first masked position:

        w_k = exp(-(k - reveal) / tau) = 0.8 ** (k - reveal)

    with `exp(-1/tau) = 0.8`, i.e. `tau = 1 / ln(1/0.8) = 4.4814`. That is
    exactly the `0.8^offset` of [Table 5] once "offset" is read as the distance
    past the reveal boundary -- which is the only reading under which the head
    can "learn every reveal length" ([§4.3]).
    """
    import torch

    offsets = torch.arange(block_size)
    return offsets, torch.tensor([float(decay) ** int(k) for k in offsets], dtype=torch.float32)


def d2sd_tau(decay=DECAY):
    """The decay rate implied by a per-offset factor: tau = 1 / ln(1/decay)."""
    import math

    return 1.0 / math.log(1.0 / float(decay))


# --------------------------------------------------------------------------
# Vision-tower patch embedding: identical math, usable speed
# --------------------------------------------------------------------------

def _linear_patch_embed(self, hidden_states):
    """`Qwen3VLVisionPatchEmbed.forward`, routed through cuBLAS.

    The layer is a Conv3d whose kernel equals its stride equals the input's
    spatial size, so every output is a plain reduction over one flattened
    patch: mathematically a `Linear` with the conv weight viewed as a matrix.
    """
    import torch.nn.functional as F

    weight = self.proj.weight
    flat = hidden_states.to(dtype=weight.dtype).reshape(hidden_states.shape[0], -1)
    return F.linear(flat, weight.view(weight.shape[0], -1), self.proj.bias)


def use_fast_patch_embed(verify=True):
    """Replace the vision patch-embed conv with its exact Linear form.

    Why this exists. `Qwen3VLVisionPatchEmbed` reshapes its input to
    `(N, 3, 2, 16, 16)` and runs a Conv3d whose kernel equals its stride equals
    the patch size. On this box's `fin_research` environment (torch
    2.9.1+cu128, cuDNN 9.10.2) cuDNN picks a pathological algorithm for that
    shape and the convolution costs **9.4 seconds** for 1040 patches, where the
    same weights through cuBLAS cost 0.03 ms. The vision tower is 27 layers of
    small matmuls plus this one op, so the convolution *is* the forward pass:
    84 s for a batch of eight images instead of about one second. Two other
    environments on the same card (torch 2.8.0+cu128 with the same cuDNN, and
    torch 2.5.1+cu124) run the conv at 0.14 ms, which clears the model and
    points at this build.

    The replacement is not an approximation. kernel = stride = input size means
    the conv is a single matrix product over the flattened patch; `verify`
    checks the two forms against each other on a small probe before the swap
    and refuses to patch if they disagree.
    """
    import torch

    from transformers.models.qwen3_vl import modeling_qwen3_vl as q3

    cls = q3.Qwen3VLVisionPatchEmbed
    if getattr(cls, "_glance_fast_embed", False):
        return cls
    original = cls.forward

    if verify:
        torch.manual_seed(0)
        probe = cls(q3.Qwen3VLVisionConfig(hidden_size=64, in_channels=3,
                                           patch_size=16, temporal_patch_size=2))
        x = torch.randn(8, 3 * 2 * 16 * 16, dtype=probe.proj.weight.dtype)
        with torch.no_grad():
            before = original(probe, x)
            after = _linear_patch_embed(probe, x)
        diff = (before.float() - after.float()).abs().max().item()
        scale = before.float().abs().max().item()
        if diff > 0.02 * max(scale, 1.0):
            raise RuntimeError(
                f"patch-embed replacement is not equivalent (diff {diff:.3e}, "
                f"scale {scale:.3e}); refusing to patch")

    cls.forward = _linear_patch_embed
    cls._glance_fast_embed = True
    return cls


def target_positions(target, input_ids, image_grid_thw):
    """M-RoPE positions for a full image+text sequence, stated explicitly.

    This does not change any number -- it is here so the training pass makes the
    same dependency visible that the decode pass already makes.

    The suspicion that prompted it was that a training forward without
    `position_ids` would silently mis-position an image sequence, because
    `Qwen3VLModel.forward` contains a plain fallback:

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, batch, -1)

    That suspicion is **wrong**, and the measurement is worth recording: on a
    532-token COCO row with a 26x40 image grid the M-RoPE positions run 0..291
    against plain 0..531 -- a 240-position gap at the last token -- yet the
    hidden states come out *identical* (cosine 1.000000, max |diff| 0.0). The
    reason is a second `get_rope_index` call further down in
    `Qwen3VLModel.forward` (transformers 4.57.3, line ~1201) that fires on the
    prefill stage when `image_grid_thw` is present, so the fallback never
    applies to an image sequence. A control confirms positions are used at all:
    all-zero positions give cosine 0.231, and a uniform +1000 shift gives
    0.999991, as relative RoPE implies.

    So the helper is a clarity change, verified behaviour-preserving, not a fix.
    """
    positions, _ = target.model.get_rope_index(
        input_ids, image_grid_thw, None, None)
    return positions


def target_context(target, input_ids, pixel_values=None, image_grid_thw=None,
                   layer_ids=KEPT_LAYERS, pixel_values_videos=None,
                   video_grid_thw=None):
    """Run the frozen target and return its fused hidden states at `layer_ids`.

    `input_ids` is `(1, n)`. Runs under `no_grad`: the target is frozen and this
    is the teacher pass, not part of the graph the head is trained through.
    """
    import torch

    kwargs = dict(input_ids=input_ids, use_cache=False,
                  output_hidden_states=True, logits_to_keep=1)
    if pixel_values is not None:
        kwargs.update(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        kwargs["position_ids"] = target_positions(target, input_ids, image_grid_thw)
    # Video is a pure addition: when no video is passed the image branch above is
    # byte-for-byte what it was, so the image regression stays bit-identical.
    if pixel_values_videos is not None:
        kwargs.update(pixel_values_videos=pixel_values_videos,
                      video_grid_thw=video_grid_thw)
        positions, _ = target.model.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, None)
        kwargs["position_ids"] = positions
    with torch.no_grad():
        out = target(**kwargs)
    return context_feature(out.hidden_states, layer_ids).detach()


def target_context_cached(target, input_ids, pixel_values=None, image_grid_thw=None,
                          layer_ids=KEPT_LAYERS, pixel_values_videos=None,
                          video_grid_thw=None):
    """`target_context`, but on the CACHE path that every decode actually uses.

    Why this exists (2026-10-05). `target_context` above passes `use_cache=False`,
    and on Qwen3-VL that is not a numerical detail. With `use_cache=False` the
    attention mask is built from the M-RoPE `position_ids`, and the TEXT positions
    come out independent of the visual input: zeroing or even negating
    `pixel_values_videos` leaves the text-position hidden states and logits
    bit-identical, while the visual positions move by ~2e3. With `use_cache=True`
    -- what `glance.decode.generate` does at prefill, and therefore what every
    reported tau is measured under -- the same comparison moves the logits by ~15
    and changes the greedy continuation outright.

    Consequence: a head trained on `target_context` features has never seen a
    text-position context row that depends on the video, yet at decode time it
    receives exactly that. Training and evaluation disagree about where the visual
    information lives. This function removes the disagreement. It is additive and
    leaves `target_context` untouched, so the image-side regression stays
    bit-identical.

    Verified by `repro/diagnostics/probe_context_cache_path.py`.
    """
    import torch
    from transformers import DynamicCache

    kwargs = dict(input_ids=input_ids, use_cache=True, output_hidden_states=True,
                  logits_to_keep=1, past_key_values=DynamicCache())
    if pixel_values is not None:
        kwargs.update(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        kwargs["position_ids"] = target_positions(target, input_ids, image_grid_thw)
    if pixel_values_videos is not None:
        kwargs.update(pixel_values_videos=pixel_values_videos,
                      video_grid_thw=video_grid_thw)
        positions, _ = target.model.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, None)
        kwargs["position_ids"] = positions
    with torch.no_grad():
        out = target(**kwargs)
    return context_feature(out.hidden_states, layer_ids).detach()
