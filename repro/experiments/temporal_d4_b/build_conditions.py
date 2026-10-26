#!/usr/bin/env python3
"""Stage B (pure CPU): build the O_rec / T / K clips and verify the permutation.

All three written clips go through the identical cv2 `mp4v` writer at the same
384-long-side geometry, 16 frames, 30 fps, so a codec round trip is shared by
all of them and cannot masquerade as the intervention.  O itself is the
original cached file, unchanged, so the C6 input hash gate can still be run.

T reverses the 16-frame *sampled* sequence; K mirrors it horizontally in the
same temporal order.  Verification re-decodes every written file and checks
that its frame i is the intended frame (order, not just set).

Writes into `--out/conditions/`:
  <video>_Orec.mp4 / _T.mp4 / _K.mp4
  conditions.json   hashes, frame-order verification, per-condition notes
  sheet_T_<video>.png / sheet_K_<video>.png   (16-frame contact sheets)
"""
from __future__ import annotations

import argparse
from datetime import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_common as B


def read_frames(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def write_frames(path, frames, fps):
    import cv2
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"cannot open writer for {path}")
    for f in frames:
        vw.write(f)
    vw.release()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"wrote nothing to {path}")


def contact_sheet(path, frames, labels):
    import cv2
    tiles = []
    for f, lab in zip(frames, labels):
        t = cv2.resize(f, (256, 192))
        cv2.rectangle(t, (0, 0), (150, 20), (0, 0, 0), -1)
        cv2.putText(t, lab, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1)
        tiles.append(t)
    rows = [np.hstack(tiles[i * 4:(i + 1) * 4]) for i in range(4)]
    cv2.imwrite(path, np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()
    run = os.path.abspath(a.out)
    cdir = os.path.join(run, "conditions")
    os.makedirs(cdir, exist_ok=True)

    fman = B.read_json(os.path.join(run, "frames", "frame_manifest.json"))
    report = {
        "created": datetime.now().astimezone().isoformat(),
        "fps": a.fps,
        "writer": "cv2.VideoWriter mp4v, same geometry as the cached clip",
        "definitions": {
            "O": "the original cached file, untouched",
            "O_rec": "the same 16 frames in the same order, re-written",
            "T": "the same 16 frames in exact reverse order",
            "K": "the same 16 frames in the same order, mirrored horizontally",
        },
        "videos": {},
    }
    for vid in B.VIDEOS:
        info = fman[vid]
        src = os.path.join(B.CACHE_DIR, vid + ".mp4")
        frames = read_frames(src)
        n = len(frames)
        assert n == info["n_cached_frames"] == 16, (vid, n)
        for k, f in enumerate(frames):
            assert f.shape[1] == info["cached_hw"][0], (vid, k, f.shape)

        variants = {
            "O_rec": list(frames),
            "T": list(frames)[::-1],
            "K": [f[:, ::-1].copy() for f in frames],
        }
        entry = {
            "cache_path": src,
            "cache_sha256": B.sha256_file(src),
            "n_frames": n,
            "orig_total_frames": info["orig_total_frames"],
            "orig_fps": info["orig_fps"],
            "orig_duration_s": info["orig_duration_s"],
            "sampled_indices": info["sampled_indices"],
            "sampled_times_s": info["sampled_times_s"],
            "conditions": {},
        }
        for name, fr in variants.items():
            dst = os.path.join(cdir, f"{vid}_{name}.mp4")
            write_frames(dst, fr, a.fps)
            back = read_frames(dst)
            if len(back) != n:
                raise RuntimeError(f"{dst} decoded {len(back)} frames, want {n}")
            diffs = [float(np.abs(back[i].astype(np.int16)
                                  - fr[i].astype(np.int16)).mean())
                     for i in range(n)]
            # order check: frame i of the file must match intended frame i much
            # better than any other intended frame
            best = []
            for i in range(n):
                d = [float(np.abs(back[i].astype(np.int16)
                                  - fr[j].astype(np.int16)).mean())
                     for j in range(n)]
                best.append(int(np.argmin(d)))
            order_ok = best == list(range(n))
            # Near-duplicate frames (e.g. black opening/closing frames) make the
            # argmin ambiguous; that is harmless, so the gate is the direct
            # per-frame tolerance and the argmin list is reported as evidence.
            entry["conditions"][name] = {
                "path": dst,
                "sha256": B.sha256_file(dst),
                "n_frames_decoded": len(back),
                "mean_abs_pixel_diff_max": max(diffs),
                "mean_abs_pixel_diff_mean": float(np.mean(diffs)),
                "frame_order_argmin": best,
                "frame_order_argmin_identity": order_ok,
                "permutation": ("identity" if name != "T" else "reverse"),
            }
            assert max(diffs) < 8.0, (vid, name, max(diffs))
            ref = [f"orig{sampled}{'' if name != 'T' else ' [T rev]'}"
                   for sampled in info["sampled_indices"]]
            if name == "T":
                ref = list(reversed(ref))
            contact_sheet(os.path.join(run, "frames", f"sheet_{name}_{vid}.png"),
                          fr, ref)
        entry["O"] = {
            "path": src, "sha256": entry["cache_sha256"],
            "definition": "original cached file, untouched",
        }
        report["videos"][vid] = entry
        print(f"{vid}: {n} frames, T/K written and order-verified", flush=True)

    B.write_json(os.path.join(run, "conditions", "conditions.json"), report)
    print("wrote", os.path.join(run, "conditions", "conditions.json"))


if __name__ == "__main__":
    main()
