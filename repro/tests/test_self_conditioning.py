#!/usr/bin/env python3
"""Does self-conditioned training really use the model's own proposals?

The failure this guards against is silent. `self_conditioned_source` shifts a
mask and a token tensor by one slot; if either is wrong, every condition falls
back to the block's given tokens and the arm quietly degenerates into
`--markov-input block` while still reporting losses that look plausible. The
first smoke run could not tell the difference: `self` and `gt` produced
byte-identical losses, because at step 1 the bias is ~0.006 against logits of
order 1 and its effect is below the printed precision. So the branch needs a
direct test on synthetic inputs where the expected answer is known exactly.

    python test_self_conditioning.py
"""

from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import os
import sys

import torch

sys.path.insert(0, str(_REPRO_ROOT))

from historical import train_head


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}: {name}\n      got  {got}\n      want {want}")
    return ok


def main():
    ok = True

    # 5 slots, slots 2 and 3 supervised (i.e. the reveal boundary is at 2).
    # The model's own proposals differ from the block's true tokens at exactly
    # those supervised slots -- that difference is the whole point.
    blocks = torch.tensor([[10, 11, 12, 13, 14]])
    super_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]])
    own_full = torch.tensor([[0, 0, 21, 22, 0]])       # 0 at unsupervised slots

    src = train_head.self_conditioned_source(blocks, super_mask, own_full)[0].tolist()

    # slot 0 has no previous slot -> keeps the root
    # slot 1's previous slot (0) was given   -> block token 10
    # slot 2's previous slot (1) was given   -> block token 11
    # slot 3's previous slot (2) was PREDICTED -> the model's own 21
    # slot 4's previous slot (3) was PREDICTED -> the model's own 22
    ok &= check("self-conditioned source", src, [10, 10, 11, 21, 22])

    # Every slot supervised: then every condition past the root must be the
    # model's own proposal, and none may come from the block.
    all_sup = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]])
    own2 = torch.tensor([[0, 31, 32, 33, 34]])
    src2 = train_head.self_conditioned_source(blocks, all_sup, own2)[0].tolist()
    ok &= check("all supervised -> own tokens", src2, [10, 10, 31, 32, 33])

    # Nothing supervised: it must reduce exactly to the block-shift ("block"),
    # which is the degenerate case the test above is designed to detect.
    no_sup = torch.zeros(1, 5)
    src3 = train_head.self_conditioned_source(blocks, no_sup, own_full)[0].tolist()
    ok &= check("nothing supervised -> block shift", src3, [10, 10, 11, 12, 13])

    # And the key negative: with slots 2,3 supervised the result must NOT be the
    # plain block shift, or the branch is dead.
    block_shift = [10, 10, 11, 12, 13]
    differs = src != block_shift
    print(f"{'PASS' if differs else 'FAIL'}: self differs from block shift "
          f"(this is what proves the branch is live)")
    ok &= differs

    print("\nall self-conditioning tests passed" if ok else "\nTHERE WERE FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
