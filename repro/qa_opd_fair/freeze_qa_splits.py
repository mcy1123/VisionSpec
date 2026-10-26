#!/usr/bin/env python3
"""D3: freeze the QA train schedule, the confirm split and the dataset manifests.

Everything here is a pure function of the official annotations and a fixed seed.
No model is loaded, no model output is read. Run once, before any training.

Inputs (read-only):
  * `/mnt/local2/mcy/datasets/msvd_qa_full/official/qa_{train,val}.json` -- LAVIS
    official MSVD-QA annotations.
  * the 2026-10-13 transfer run's frozen 20-video dev manifest -- the existing
    development set, reused as `dev`.
  * `/mnt/local2/mcy/datasets/msvd_qa_full/inventory.jsonl` -- acquired train
    videos and their cache/real-frame metadata.

Outputs (all under the run's `dataset/`):
  * `confirm_manifest.jsonl`  -- the frozen MSVD confirmation set (video-level
    disjoint from dev and from every training video).
  * `train_pool.jsonl`        -- the eligible MSVD train pool, one row per video.
  * `train_schedule_seed_<seed>.jsonl` -- 480 slots (4 QA x 120 updates), the
    SAME video order and the same (video, question) pairs in both files except
    for the seeded shuffle; a video never appears twice.
  * `dev_manifest.jsonl`      -- the frozen development inputs (from the
    historical run; copy of the source rows with their cache paths).
  * `split_overlap_audit.csv` -- the overlap matrix between train / dev /
    confirm, with the evidence class for each comparison.
  * `dataset_summary.json`    -- counts and the exact selection rule.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path[:0] = [str(HERE), str(REPRO),
                str(REPRO / "videoqa_transfer")]

from protocol import prompt, source_id  # noqa: E402  (historical templates, reused)

SEED_BASE = 20261013
SEEDS = [20261013, 20261014]
UPDATES = 120
QA_PER_UPDATE = 4
CONFIRM_VIDEOS = 60
CONFIRM_QA_PER_VIDEO = 2
DEV_MANIFEST = pathlib.Path(
    "/mnt/local/mcy/GLANCE/results/videoqa_transfer/20261013T134000/"
    "dataset_manifest_20videos.jsonl")


def load_qa(path):
    rows = json.loads(pathlib.Path(path).read_text())
    out = []
    for r in rows:
        out.append({"video_id": str(r["video"]), "question_id": str(r["question_id"]),
                    "question": str(r["question"]), "gold_answer": str(r["answer"]),
                    "question_type": str(r["answer_type"]), "options": None,
                    "split": pathlib.Path(path).stem.replace("qa_", "")})
    return out


def by_video(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["video_id"]].append(r)
    for v in groups:
        groups[v].sort(key=lambda x: x["question_id"])
    return groups


def listify_schedule(slots, videos, pool):
    """Turn (video, question) slots into the frozen schedule records."""
    out = []
    for i, (vid, qa) in enumerate(slots):
        out.append({
            "update": i // QA_PER_UPDATE,
            "slot": i % QA_PER_UPDATE,
            "order": i,
            "video_id": vid,
            "source_id": source_id(vid),
            "question_id": qa["question_id"],
            "question": qa["question"],
            "question_type": qa["question_type"],
            "gold_answer_for_audit_only": qa["gold_answer"],
            "prompt": prompt(qa, "B"),
            "prompt_mode": "B",
            "split": "train",
            "cache_path": pool.get(vid, {}).get("cache_path"),
            "fps": pool.get(vid, {}).get("fps"),
            "source_fps": pool.get(vid, {}).get("source_fps"),
            "raw_path": pool.get(vid, {}).get("raw_path"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--official", default="/mnt/local2/mcy/datasets/msvd_qa_full/official")
    ap.add_argument("--inventory", default="/mnt/local2/mcy/datasets/msvd_qa_full/inventory.jsonl")
    a = ap.parse_args()
    R = pathlib.Path(a.run)
    D = R / "dataset"
    D.mkdir(parents=True, exist_ok=True)

    train = load_qa(pathlib.Path(a.official) / "qa_train.json")
    val = load_qa(pathlib.Path(a.official) / "qa_val.json")
    test = load_qa(pathlib.Path(a.official) / "qa_test.json")
    gtr, gva, gte = by_video(train), by_video(val), by_video(test)

    # ---- the historical development set (already published, never a test)
    dev_rows = [json.loads(l) for l in DEV_MANIFEST.read_text().splitlines() if l.strip()]
    dev_videos = sorted({r["video_id"] for r in dev_rows})
    dev_videos_set = set(dev_videos)

    # ---- confirm: video-disjoint from dev, drawn from the official val split
    candidates = sorted(set(gva) - dev_videos_set)
    rng = random.Random(SEED_BASE)
    rng.shuffle(candidates)
    confirm_videos = sorted(candidates[:CONFIRM_VIDEOS])
    missing = CONFIRM_VIDEOS - len(confirm_videos)
    if missing:
        raise SystemExit(f"official val has only {len(confirm_videos)} videos left "
                         f"after dev; {missing} short of the {CONFIRM_VIDEOS} target")
    confirm_rows = []
    for v in confirm_videos:
        qs = gva[v][:CONFIRM_QA_PER_VIDEO]
        for q in qs:
            confirm_rows.append({**q, "dataset": "MSVD-QA", "prompt_mode": "B",
                                 "prompt": prompt(q, "B"),
                                 "selection": "confirm", "source_id": source_id(v)})

    # ---- train pool: official train, minus any dev identity (there must be none)
    overlap_dev = sorted(set(gtr) & dev_videos_set)
    overlap_confirm = sorted(set(gtr) & set(confirm_videos))
    pool_rows = []
    for v in sorted(gtr):
        qs = gtr[v]
        pool_rows.append({"video_id": v, "source_id": source_id(v),
                          "split": "train", "n_qa": len(qs),
                          "question_ids": [q["question_id"] for q in qs],
                          "question_types": sorted({q["question_type"] for q in qs}),
                          "in_dev": v in dev_videos_set,
                          "in_confirm": v in set(confirm_videos)})

    # ---- the acquired-video inventory, if it already exists (may be partial)
    inv = {}
    inv_path = pathlib.Path(a.inventory)
    if inv_path.exists():
        for line in inv_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                inv[rec["video_id"]] = rec

    # ---- the training unit: a fixed set of (video, question) QA slots.
    # Both seeds train on the SAME slots and differ only in the ORDER they are
    # visited. That is what makes the two seeds a paired comparison (same data,
    # same budget, different seed); a per-seed draw would instead compare two
    # different video sets and call the difference "seed variance". Each video
    # contributes exactly one question and a video never repeats, so 480 slots
    # equals 480 distinct videos out of the 1161-video pool.
    video_order = sorted(gtr)
    random.Random(SEED_BASE).shuffle(video_order)
    slots_fixed = []
    for vid in video_order:
        qa = sorted(gtr[vid], key=lambda q: q["question_id"])[0]
        slots_fixed.append((vid, qa))
        if len(slots_fixed) >= UPDATES * QA_PER_UPDATE:
            break
    if len(slots_fixed) < UPDATES * QA_PER_UPDATE:
        raise SystemExit("train pool smaller than the schedule")
    if any(n > 1 for n in collections.Counter(v for v, _ in slots_fixed).values()):
        raise SystemExit("a video repeats inside the frozen slot set")

    schedules = {}
    for seed in SEEDS:
        order = list(range(len(slots_fixed)))
        random.Random(seed * 1_000_003).shuffle(order)
        schedules[seed] = [slots_fixed[i] for i in order]

    # Both seeds must reference the same QA multiset; only the order differs.
    qa_ids = [sorted(q["question_id"] for _, q in schedules[s]) for s in SEEDS]
    if qa_ids[0] != qa_ids[1]:
        raise SystemExit("the two seeds do not share the same QA slot set")
    assert all(len(s) == UPDATES * QA_PER_UPDATE for s in schedules.values())

    for seed, slots in schedules.items():
        recs = listify_schedule(slots, [v for v, _ in slots_fixed], inv)
        with (D / f"train_schedule_seed_{seed}.jsonl").open("w") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with (D / "train_pool.jsonl").open("w") as fh:
        for rec in pool_rows:
            fh.write(json.dumps({**rec, "acquired": rec["video_id"] in inv},
                                ensure_ascii=False) + "\n")
    with (D / "confirm_manifest.jsonl").open("w") as fh:
        for rec in confirm_rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (D / "dev_manifest.jsonl").open("w") as fh:
        for rec in dev_rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (D / "qa_prompt_templates.json").open("w") as fh:
        json.dump({"A": prompt({"question": "{question}"}, "A"),
                   "B": prompt({"question": "{question}"}, "B"),
                   "source": "repro/videoqa_transfer/protocol.py TEMPLATES (reused "
                             "unchanged so the new training shares the historical "
                             "format-B prompt exactly)"},
                  fh, indent=2, ensure_ascii=False)

    # ---- overlap audit
    with (D / "split_overlap_audit.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["left", "right", "n_left", "n_right", "n_overlap",
                    "overlap_videos", "evidence"])
        pairs = [
            ("train", "dev", set(gtr), dev_videos_set, "video id"),
            ("train", "confirm", set(gtr), set(confirm_videos), "video id"),
            ("train", "test", set(gtr), set(gte), "video id"),
            ("dev", "confirm", dev_videos_set, set(confirm_videos), "video id"),
            ("dev", "test", dev_videos_set, set(gte), "video id"),
            ("confirm", "test", set(confirm_videos), set(gte), "video id"),
        ]
        for name_l, name_r, l, r, ev in pairs:
            ov = sorted(l & r)
            w.writerow([name_l, name_r, len(l), len(r), len(ov),
                        ";".join(ov[:10]), ev])
        # Source-id level (the 11-character YouTube id). MSVD splits one source
        # video into several clips, and the official split does NOT keep the
        # source together -- `bQJQGoJF7_k` is a train clip at 109-153 s and a val
        # clip at 162-169 s -- so a source-level collision is a *same-source*
        # flag, not a label leak. It is reported with the exact clip names so the
        # overlap can be judged instead of being hidden by the id-level count.
        src = collections.defaultdict(set)
        groups_of = {"train": gtr, "val": gva, "test": gte}
        for name, groups in groups_of.items():
            for v in groups:
                sid = source_id(v)
                if sid:
                    src[name].add(sid)
        src["dev"] = {source_id(v) for v in dev_videos} - {None}
        src["confirm"] = {source_id(v) for v in confirm_videos} - {None}
        for name_l, name_r in (("train", "dev"), ("train", "confirm"),
                               ("train", "test"), ("dev", "confirm")):
            ov = sorted(src[name_l] & src[name_r])
            clips_l = sorted(v for v in groups_of.get(name_l, {})
                             if source_id(v) in ov) if name_l in groups_of \
                else sorted(v for v in ([dev_videos, confirm_videos][name_l == "confirm"])
                            if source_id(v) in ov)
            clips_r = sorted(v for v in groups_of.get(name_r, {})
                             if source_id(v) in ov) if name_r in groups_of \
                else sorted(v for v in ([dev_videos, confirm_videos][name_r == "confirm"])
                            if source_id(v) in ov)
            w.writerow([name_l + "_source", name_r + "_source",
                        len(src[name_l]), len(src[name_r]), len(ov),
                        ";".join(f"{k}:[{','.join(x for x in clips_l if source_id(x)==k)}]"
                                 f"vs[{','.join(x for x in clips_r if source_id(x)==k)}]"
                                 for k in ov[:10]),
                        "youtube source id (11 chars); same-source-different-clip "
                        "is expected and is NOT a label leak"])

    with (D / "dataset_summary.json").open("w") as fh:
        json.dump({
            "official_counts": {
                "train": {"qa": len(train), "videos": len(gtr)},
                "val": {"qa": len(val), "videos": len(gva)},
                "test": {"qa": len(test), "videos": len(gte)},
            },
            "dev": {"videos": len(dev_videos), "source": str(DEV_MANIFEST),
                    "note": "the 2026-10-13 transfer run's frozen 20-video "
                            "development set, already used for method selection"},
            "confirm": {"videos": len(confirm_videos), "qa": len(confirm_rows),
                        "qa_per_video": CONFIRM_QA_PER_VIDEO,
                        "selection": "official val videos disjoint from dev, "
                                     f"shuffled with random.Random({SEED_BASE}), "
                                     "lowest 60 ids kept"},
            "train_pool": {"videos": len(gtr), "per_update": QA_PER_UPDATE,
                           "updates": UPDATES,
                           "slots": UPDATES * QA_PER_UPDATE,
                           "rule": "one question per distinct video; 4 distinct "
                                   "videos per update; a video never repeats"},
            "seeds": SEEDS,
            "identity_overlap": {"train_dev": len(overlap_dev),
                                 "train_confirm": len(overlap_confirm)},
            "gold_answers_in_prompts": False,
            "inventory_videos_present": len(inv),
        }, fh, indent=2, ensure_ascii=False)

    print(json.dumps({"dev_videos": len(dev_videos),
                      "confirm_videos": len(confirm_videos),
                      "confirm_qa": len(confirm_rows),
                      "train_pool_videos": len(gtr),
                      "train_dev_overlap": len(overlap_dev),
                      "train_confirm_overlap": len(overlap_confirm),
                      "inventory_present": len(inv)}, indent=2))


if __name__ == "__main__":
    main()
