#!/usr/bin/env python3
"""Build Video-MME-derived long descriptions from distinct source video frames.

No model or GPU is used. --dry-run probes metadata and reports the split without
writing. Normal builds require --cache-root and store lossless uint8 NPZ frames;
--metadata-only is an explicit, non-trainable alternative. Run this on the data
server: absolute source/cache paths in a manifest belong to that server.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))
from qa_opd_fair.videoencode import validate_real_metadata

PROMPT = (
    "Describe the main visible events in this video in chronological order. "
    "For each distinct event, give its approximate time range and describe the "
    "people or objects involved, their actions, and visible changes in state. "
    "Describe only evidence visible in the frames. Do not infer dialogue or "
    "unseen events, repeat descriptions to add length, or add a separate reasoning trace."
)
ALLOWED_CACHE_ROOTS = (Path("/mnt/local"), Path("/mnt/local2"), Path("/mnt/local3"))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_id(row):
    value = row.get("videoID") or row.get("video_name") or row.get("video_id")
    if value is None or not str(value).strip():
        raise ValueError("every source row needs videoID, video_name, or video_id")
    return str(value)


def load_exclusions(path):
    """Accept an explicit old-pilot JSONL, JSON list, or one source id per line."""
    data = Path(path).read_text().strip()
    if not data:
        return set()
    if data.startswith("["):
        values = json.loads(data)
    else:
        values = [json.loads(line) if line.lstrip().startswith("{") else line.strip()
                  for line in data.splitlines() if line.strip() and not line.startswith("#")]
    result = set()
    for value in values:
        if isinstance(value, dict):
            result.add(source_id(value))
            # Official numeric ids and YouTube ids both occur in old manifests.
            result.update(str(value[k]) for k in ("video_id", "videoID", "video_name")
                          if value.get(k) is not None)
        else:
            result.add(str(value))
    return result


def unique_sources(rows, exclusions, source_root=None):
    """Group QA rows before splitting; reject inconsistent mappings of an id."""
    videos = {}
    paths = {}
    for row in rows:
        key = source_id(row)
        aliases = {str(row[k]) for k in ("video_id", "videoID", "video_name")
                   if row.get(k) is not None}
        if aliases & exclusions:
            continue
        raw_path = row.get("local_video_path") or row.get("source_path")
        if not raw_path:
            raise ValueError(f"{key}: missing local_video_path/source_path")
        path = ((Path(source_root) / Path(raw_path).name) if source_root
                else Path(raw_path)).expanduser().resolve()
        if key in videos:
            if videos[key]["source_path"] != str(path):
                raise ValueError(f"source id {key!r} maps to conflicting media paths")
            continue
        if str(path) in paths and paths[str(path)] != key:
            raise ValueError(f"one source path maps to multiple video ids: {path}")
        paths[str(path)] = key
        videos[key] = {"video_id": key, "source_path": str(path),
                       "source_url": row.get("url"),
                       "official_video_id": str(row.get("video_id", key)),
                       "duration_category": row.get("duration"),
                       "domain": row.get("domain"), "sub_category": row.get("sub_category")}
    return [videos[k] for k in sorted(videos)]


def probe_video(path):
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_frames,duration:format=duration",
         "-of", "json", str(path)], check=True, capture_output=True, text=True)
    data = json.loads(proc.stdout)
    stream = data["streams"][0]
    raw_duration = stream.get("duration")
    if raw_duration in (None, "N/A"):
        raw_duration = data.get("format", {}).get("duration", "nan")
    duration = float(raw_duration)
    fps = float(Fraction(stream.get("avg_frame_rate", "0/1")))
    if not math.isfinite(duration) or duration <= 0 or fps <= 0:
        raise ValueError(f"invalid video duration/fps: {path}")
    return {"video_duration": duration, "source_fps": fps,
            "source_width": int(stream["width"]), "source_height": int(stream["height"])}


def assign_splits(videos, n_train, n_dev, seed):
    if n_train < 1 or n_dev < 1:
        raise ValueError("train and dev counts must both be positive")
    if len(videos) < n_train + n_dev:
        raise ValueError(f"need {n_train + n_dev} eligible videos, found {len(videos)}")
    ordered = sorted(videos, key=lambda x: x["video_id"])
    random.Random(seed).shuffle(ordered)
    return [dict(row, split="train" if i < n_train else "dev")
            for i, row in enumerate(ordered[:n_train + n_dev])]


def decode_timeline(path):
    """Read actual presentation timestamps, never substitute index/fps for PTS."""
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        pts = []
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"source has a decoded frame without a timestamp: {path}")
            pts.append(float(frame.pts * frame.time_base))
    if not pts or fps <= 0 or any(a >= b for a, b in zip(pts, pts[1:])):
        raise ValueError(f"source needs positive fps and strictly increasing PTS: {path}")
    origin = pts[0]
    return [t - origin for t in pts], fps, origin


def sample_unique_timestamps(timestamps, count, candidates=None):
    """Nearest uniform-time samples, with room reserved for remaining ids."""
    ids = list(range(len(timestamps))) if candidates is None else list(candidates)
    if count < 1 or len(ids) < count:
        raise ValueError("not enough distinct source frames; repeating frames is forbidden")
    if any(a >= b for a, b in zip(ids, ids[1:])):
        raise ValueError("candidate frame ids must be strictly increasing")
    times = [timestamps[i] for i in ids]
    if count == 1:
        return [ids[len(ids) // 2]]
    selected, lower = [], 0
    for i in range(count):
        target = times[0] + (times[-1] - times[0]) * i / (count - 1)
        upper = len(ids) - (count - i)
        pos = bisect.bisect_left(times, target, lower, upper + 1)
        options = {max(lower, min(upper, pos)), max(lower, min(upper, pos - 1))}
        choice = min(options, key=lambda j: (abs(times[j] - target), j))
        selected.append(ids[choice])
        lower = choice + 1
    return selected


def nested_samples(timestamps, counts):
    result, candidates = {}, None
    for count in sorted(set(counts), reverse=True):
        candidates = sample_unique_timestamps(timestamps, count, candidates)
        result[count] = candidates
    return result


def resized_dimensions(width, height, long_side):
    scale = min(1.0, long_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def materialize_frames(path, sample_ids, width, height):
    """Decode only selected pixels; return uint8 TCHW without temporal padding."""
    import av
    import numpy as np
    from PIL import Image
    wanted = set(sample_ids)
    frames = {}
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                pixels = frame.to_image().convert("RGB")
                if pixels.size != (width, height):
                    pixels = pixels.resize((width, height), Image.Resampling.BILINEAR)
                frames[i] = np.asarray(pixels, dtype=np.uint8).transpose(2, 0, 1).copy()
            if i >= max(wanted):
                break
    if set(frames) != wanted:
        raise ValueError("source did not decode every requested frame")
    return frames


def require_cache_location(path):
    resolved = Path(path).expanduser().resolve()
    if not any(root.exists() and resolved.is_relative_to(root.resolve())
               for root in ALLOWED_CACHE_ROOTS):
        raise ValueError("frame caches must resolve under an existing /mnt/local, "
                         "/mnt/local2, or /mnt/local3 mount")
    return resolved


def write_jsonl(path, rows):
    with Path(path).open("x") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-jsonl", required=True)
    p.add_argument("--exclude-video-list", required=True,
                   help="explicit historical JSONL/JSON/id list; an empty file is allowed")
    p.add_argument("--source-root", help="relocate source files by original basename")
    p.add_argument("--out-dir", required=True, help="new/empty manifest directory")
    p.add_argument("--cache-root", help="new/empty cache directory on an allowed data mount")
    p.add_argument("--metadata-only", action="store_true",
                   help="write source PTS and frame selections only; NOT trainable")
    p.add_argument("--dry-run", action="store_true", help="probe and plan only; write nothing")
    p.add_argument("--n-train", type=int, default=80)
    p.add_argument("--n-dev", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--min-duration", type=float, default=240.0)
    p.add_argument("--max-duration", type=float, default=600.0)
    p.add_argument("--duration-category", default="medium", choices=("short", "medium", "long", "all"))
    p.add_argument("--frame-counts", type=int, nargs="+", default=[64, 128])
    p.add_argument("--long-side", type=int, default=384)
    p.add_argument("--prompt-file", help="UTF-8 file replacing the structured description prompt")
    p.add_argument("--allow-approximate-timeline", action="store_true",
                   help="diagnostic manifests only; strict real encoding still rejects irregular PTS")
    p.add_argument("--write-parquet", action="store_true", help="also write per-budget train/dev parquet")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 0 < args.min_duration <= args.max_duration or args.long_side < 1:
        raise ValueError("duration range and long-side must be positive")
    if any(n < 2 or n % 2 for n in args.frame_counts):
        raise ValueError("frame counts must be positive even numbers (no temporal padding)")
    if args.metadata_only and args.cache_root:
        raise ValueError("choose --metadata-only or --cache-root")
    if not args.dry_run and not (args.metadata_only or args.cache_root):
        raise ValueError("normal builds require --cache-root; use --metadata-only explicitly otherwise")
    exclusions = load_exclusions(args.exclude_video_list)
    rows = [json.loads(line) for line in Path(args.source_jsonl).read_text().splitlines() if line.strip()]
    sources = unique_sources(rows, exclusions, args.source_root)
    eligible, rejected = [], []
    for source in sources:
        if args.duration_category != "all" and source["duration_category"] != args.duration_category:
            continue
        info = probe_video(source["source_path"])
        if args.min_duration <= info["video_duration"] <= args.max_duration:
            eligible.append(dict(source, **info))
        else:
            rejected.append(source["video_id"])
    selected = assign_splits(eligible, args.n_train, args.n_dev, args.seed)
    summary = {"schema_version": "real_video_pilot_v1", "dataset": "Video-MME-derived",
               "official_benchmark": False, "n_source_qa_rows": len(rows),
               "n_unique_after_exclusions": len(sources), "n_eligible": len(eligible),
               "n_rejected_duration": len(rejected), "n_train": args.n_train, "n_dev": args.n_dev,
               "seed": args.seed, "frame_counts": sorted(set(args.frame_counts)),
               "selection_config": {"min_duration_s": args.min_duration,
                                    "max_duration_s": args.max_duration,
                                    "duration_category": args.duration_category,
                                    "n_excluded_id_aliases": len(exclusions)},
               "preprocessing_config": {"long_side": args.long_side,
                                        "sampling": "uniform_pts_nested_v1",
                                        "allow_approximate_timeline": args.allow_approximate_timeline},
               "source_jsonl_sha256": sha256_file(args.source_jsonl),
               "exclusion_file_sha256": sha256_file(args.exclude_video_list),
               "dry_run": args.dry_run, "materialized": not args.metadata_only and not args.dry_run,
               "selection": [{k: x[k] for k in ("video_id", "split", "video_duration")} for x in selected],
               "limitations": ["No model was run; visual token and generated token counts are unmeasured.",
                               "Dry-run checks container metadata, not decoded source PTS or cache pixels."]}
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return summary
    out = Path(args.out_dir).expanduser().resolve()
    cache_root = require_cache_location(args.cache_root) if args.cache_root else None
    for path in (out, cache_root):
        if path and path.exists() and any(path.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty output directory: {path}")
    if args.write_parquet:
        import pandas as pd
        import pyarrow  # Fail before decoding if the requested writer is unavailable.
    prompt = Path(args.prompt_file).read_text().strip() if args.prompt_file else PROMPT
    if not prompt:
        raise ValueError("prompt must not be empty")
    manifests = {n: [] for n in sorted(set(args.frame_counts))}
    seen_hashes = set()
    for source in selected:
        path = source["source_path"]
        timeline, fps, origin = decode_timeline(path)
        samples = nested_samples(timeline, args.frame_counts)
        digest = sha256_file(path)
        if digest in seen_hashes:
            raise ValueError("identical source bytes under different video ids; deduplicate before splitting")
        seen_hashes.add(digest)
        width, height = resized_dimensions(source["source_width"], source["source_height"], args.long_side)
        per_source = []
        for n in manifests:
            ids = samples[n]
            row = dict(source, task=source["video_id"], video_name=source["video_id"],
                       dataset="Video-MME-derived", prompt=prompt, question=prompt, answer="",
                       hash=digest, source_sha256=digest, source_fps=fps, fps=fps, source_frame_count=len(timeline),
                       time_origin="first_source_frame_pts", source_pts_origin_s=origin,
                       sample_frame_ids=ids, sample_timestamps=[timeline[i] for i in ids],
                       sample_source_pts=[timeline[i] + origin for i in ids],
                       n_frames=n, n_unique_frames=n, width=width, height=height,
                       cache_path="", cache_hash="", materialized=cache_root is not None,
                       input_ids_len=None, visual_token_count=None,
                       processing_config={"sampling": "uniform_pts_nested_v1", "long_side": args.long_side,
                                          "resize": "PIL_BILINEAR_no_upscale", "color": "RGB",
                                          "cache_layout": "uint8_TCHW", "cache_format": "npz_lossless",
                                          "processor_resamples_frames": False,
                                          "requested_frame_counts": sorted(manifests)})
            row.update(validate_real_metadata(row, require_cache=False,
                       require_exact=not args.allow_approximate_timeline))
            per_source.append(row)
        if cache_root:
            import numpy as np
            all_ids = sorted(set(i for ids in samples.values() for i in ids))
            pixels = materialize_frames(path, all_ids, width, height)
            cache_root.mkdir(parents=True, exist_ok=True)
            for row in per_source:
                cache_path = cache_root / f"{digest[:24]}_{row['n_frames']}f.npz"
                with cache_path.open("xb") as f:
                    np.savez_compressed(f, frames=np.stack([pixels[i] for i in row["sample_frame_ids"]]))
                row["cache_path"], row["cache_hash"] = str(cache_path), sha256_file(cache_path)
        for row in per_source:
            manifests[row["n_frames"]].append(row)
    out.mkdir(parents=True, exist_ok=True)
    for n, records in manifests.items():
        write_jsonl(out / f"manifest_{n}.jsonl", records)
        for split in ("train", "dev"):
            subset = [r for r in records if r["split"] == split]
            write_jsonl(out / f"{split}_{n}.jsonl", subset)
            if args.write_parquet:
                pd.DataFrame(subset).to_parquet(out / f"{split}_{n}.parquet", index=False)
    summary.update(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                   materialized=cache_root is not None,
                   all_timestamps_exact=all(r["real_time_exact"] for rows in manifests.values() for r in rows),
                   cache_root=str(cache_root) if cache_root else None,
                   trainable=cache_root is not None and all(r["real_time_exact"] for rows in manifests.values() for r in rows))
    (out / "PROTOCOL.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
