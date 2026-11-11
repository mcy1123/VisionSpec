"""CPU contracts for source-level splits, real timestamps and frame caches."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from video_caption import build_real_video_pilot as pilot
from qa_opd_fair import videoencode


def metadata(**changes):
    row = dict(n_frames=4, source_fps=10.0, video_duration=10.0,
               sample_frame_ids=[0, 30, 60, 90], sample_timestamps=[0.0, 3.0, 6.0, 9.0],
               height=8, width=12, source_frame_count=100)
    row.update(changes)
    return row


def test_source_dedup_exclusions_and_split(tmp_path):
    rows = [{"videoID": key, "video_id": str(i), "local_video_path": f"/old/{key}.mp4"}
            for i, key in enumerate(("a", "a", "b", "c", "d"))]
    exclusions = tmp_path / "used.jsonl"
    exclusions.write_text(json.dumps({"videoID": "b", "video_id": "002"}) + "\n")
    videos = pilot.unique_sources(rows, pilot.load_exclusions(exclusions), tmp_path)
    assert {r["video_id"] for r in videos} == {"a", "c", "d"}
    assert all(Path(r["source_path"]).parent == tmp_path for r in videos)
    split = pilot.assign_splits(videos, 2, 1, 7)
    assert split == pilot.assign_splits(list(reversed(videos)), 2, 1, 7)
    train = {r["video_id"] for r in split if r["split"] == "train"}
    dev = {r["video_id"] for r in split if r["split"] == "dev"}
    assert len(train) == 2 and len(dev) == 1 and train.isdisjoint(dev)
    with pytest.raises(ValueError, match="conflicting"):
        pilot.unique_sources([rows[0], dict(rows[0], local_video_path="/different/a.mp4")], set())


def test_uniform_real_frames_are_nested_and_never_repeated():
    timeline = np.cumsum(np.linspace(0.03, 0.06, 300)).tolist()
    samples = pilot.nested_samples(timeline, [64, 128])
    assert len(set(samples[64])) == 64
    assert len(set(samples[128])) == 128
    assert set(samples[64]) <= set(samples[128])
    assert samples[128][0] == 0 and samples[128][-1] == 299
    assert all(a < b for a, b in zip(samples[128], samples[128][1:]))
    with pytest.raises(ValueError, match="repeating frames is forbidden"):
        pilot.nested_samples(timeline[:16], [128])


@pytest.mark.parametrize("changes,match", [
    ({"sample_frame_ids": [0, 30, 30, 90]}, "unique"),
    ({"sample_timestamps": [0, 3, 2, 9]}, "increasing"),
    ({"sample_timestamps": [0, 3, float("nan"), 9]}, "finite"),
    ({"sample_timestamps": [0, 3, 6]}, "n_frames"),
    ({"source_frame_count": 90}, "exceeds"),
    ({"source_fps": 0}, "positive"),
])
def test_metadata_rejects_invalid_frame_and_timeline_contract(changes, match):
    with pytest.raises(ValueError, match=match):
        videoencode.validate_real_metadata(metadata(**changes), require_cache=False)


def test_irregular_pts_are_reported_and_strict_real_rejects():
    row = metadata(sample_timestamps=[0, 3.02, 6.01, 9])
    audit = videoencode.validate_real_metadata(row, require_cache=False)
    assert not audit["real_time_exact"]
    assert audit["max_timestamp_error_s"] == pytest.approx(0.02)
    with pytest.raises(ValueError, match="strict real mode"):
        videoencode.validate_real_metadata(row, require_cache=False, require_exact=True)
    with pytest.raises(ValueError, match="metadata-only"):
        videoencode.validate_real_metadata(metadata())


def test_npz_cache_checks_pixels_shape_count_and_dimensions(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "frames.npz"
    row = metadata(cache_path=str(path))
    pixels = np.zeros((4, 3, 8, 12), dtype=np.uint8)
    np.savez_compressed(path, frames=pixels)
    assert videoencode._cache_frames(row).shape == (4, 3, 8, 12)
    np.savez_compressed(path, frames=pixels[:3])
    with pytest.raises(RuntimeError, match="manifest says"):
        videoencode._cache_frames(row)
    np.savez_compressed(path, frames=np.zeros((4, 3, 9, 12), dtype=np.uint8))
    with pytest.raises(ValueError, match="dimensions"):
        videoencode._cache_frames(row)
    np.savez_compressed(path, frames=pixels.astype(np.float32))
    with pytest.raises(ValueError, match="uint8"):
        videoencode._cache_frames(row)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe") or
                    not importlib.util.find_spec("av"), reason="needs CPU ffmpeg and PyAV")
def test_builder_cpu_round_trip_and_dry_run(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    inputs = []
    for i, color in enumerate(("red", "blue", "green")):
        path = tmp_path / f"source{i}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        f"color=c={color}:s=32x24:r=8:d=4", "-c:v", "mpeg4",
                        str(path)], check=True)
        for q in range(3):
            inputs.append({"videoID": f"source{i}", "video_id": str(i),
                           "question_id": f"{i}-{q}", "duration": "short",
                           "local_video_path": str(path)})
    source = tmp_path / "input.jsonl"
    source.write_text("\n".join(json.dumps(x) for x in inputs) + "\n")
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("source2\n")
    out, caches = tmp_path / "manifest", tmp_path / "caches"
    args = ["--source-jsonl", str(source), "--exclude-video-list", str(excluded),
            "--out-dir", str(out), "--n-train", "1", "--n-dev", "1",
            "--duration-category", "all", "--min-duration", "3", "--max-duration", "5",
            "--frame-counts", "4", "8", "--long-side", "16"]
    result = pilot.main(args + ["--dry-run"])
    assert result["n_eligible"] == 2 and not out.exists() and not caches.exists()
    monkeypatch.setattr(pilot, "ALLOWED_CACHE_ROOTS", (tmp_path,))
    result = pilot.main(args + ["--cache-root", str(caches), "--write-parquet"])
    assert result["trainable"] and result["all_timestamps_exact"]
    small = [json.loads(x) for x in (out / "manifest_4.jsonl").read_text().splitlines()]
    large = [json.loads(x) for x in (out / "manifest_8.jsonl").read_text().splitlines()]
    assert {r["video_id"] for r in large} == {"source0", "source1"}
    assert {r["split"] for r in large} == {"train", "dev"}
    parquet_row = pd.read_parquet(out / "train_8.parquet").iloc[0].to_dict()
    assert videoencode.validate_real_metadata(parquet_row, require_exact=True)["real_time_exact"]
    for lo, hi in zip(small, large):
        assert lo["video_id"] == hi["video_id"] and lo["split"] == hi["split"]
        assert set(lo["sample_frame_ids"]) <= set(hi["sample_frame_ids"])
        assert hi["source_frame_count"] == 32
        assert hi["sample_timestamps"] == [i / 8 for i in hi["sample_frame_ids"]]
        assert hi["sample_timestamps"][-1] > 3
        assert hi["width"] == 16 and hi["height"] == 12
        assert videoencode._cache_frames(hi).shape == (8, 3, 12, 16)
        assert hi["cache_hash"] == pilot.sha256_file(hi["cache_path"])
        assert hi["input_ids_len"] is None and hi["visual_token_count"] is None
    with pytest.raises(FileExistsError):
        pilot.main(args + ["--cache-root", str(caches)])


def test_cache_mount_guard_checks_realpath(tmp_path):
    with pytest.raises(ValueError, match="must resolve under"):
        pilot.require_cache_location(tmp_path / "large-cache")
