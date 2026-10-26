#!/usr/bin/env python3
"""Phase B: freeze the ScienceQA image splits, extract images, emit the gates.

Design (frozen before any model output exists -- see ``prereg.json``):

* Source is the pinned official mirror
  ``lmms-lab-encoder/ScienceQA@69dd4d6b``, ``ScienceQA-IMG/{split}`` Parquet.
  That is the exact source the three user-supplied roots were built from; the
  audit proved it by matching every local PNG's sha256 against the embedded
  image bytes of the official parquet.
* Official split comes from the Parquet file, never from a directory name.
* A stable identifier ``pid = sqa-img:{split}:{row_index}`` is derived, because
  the upstream literal ``pid`` string (``lupantech/ScienceQA``'s
  ``problems.json``) is unreachable from this host. The alignment key used
  everywhere is ``sha256(image bytes) + question + choices + answer``, which is
  strictly stronger than a pid join.
* **Cross-split image leakage is removed, not tolerated.** ScienceQA reuses one
  picture across several questions and the official split has placed questions
  about the same picture into different splits (685 image clusters span more
  than one official split). Every record whose image cluster spans more than one
  official split is excluded from all three sets. Without this the task's
  "no cross-group near-image leakage" gate cannot be met.
* Selection is deterministic: stratified by ``subject`` x image-size tercile,
  quota allocated by largest remainder, and within a stratum the order comes
  from ``sha256(f"{split_seed}:{pid}")``. No model score, no result, enters it.
* Gold ``answer``/``lecture``/``solution``/``hint`` are never written into the
  prompt. Only the question and its option list are used.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
import pathlib
import sys

import pyarrow.parquet as pq
from PIL import Image

OFFICIAL_DIR = ("/mnt/local2/mcy/datasets/scienceqa_glance_20261015/"
                "lmms-lab-encoder_ScienceQA@69dd4d6b")
OFFICIAL_REPO = "lmms-lab-encoder/ScienceQA"
OFFICIAL_REV = "69dd4d6b67373d38f96a5badd5d24d0eb5bcdc50"
SPLITS = ["train", "validation", "test"]
CANONICAL_SPLIT = {"train": "train", "validation": "dev", "test": "test"}
TARGETS = {"train": 1200, "validation": 240, "test": 240}
LETTERS = "ABCDE"

#: The frozen task prompt. `{choices}` is the "(A) ...\n(B) ..." block, `{question}`
#: is the bare official question. Registered verbatim in `experiment_protocol.json`
#: and never edited after the first training run starts.
PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Options:\n"
    "{choices}\n"
    "\n"
    "Use the image and the question to reason through the relevant scientific "
    "evidence step by step. Explain your reasoning clearly but concisely. "
    "Finish with one line in the exact format \"Answer: (X)\", where X is the "
    "letter of your chosen option."
)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def load_official() -> list[dict]:
    rows: list[dict] = []
    for split in SPLITS:
        p = os.path.join(OFFICIAL_DIR, "ScienceQA-IMG",
                         f"{split}-00000-of-00001.parquet")
        t = pq.read_table(p)
        imgs = t.column("image").flatten()[0].to_pylist()
        c = t.to_pydict()
        for i, raw in enumerate(imgs):
            rows.append({
                "pid": f"sqa-img:{CANONICAL_SPLIT[split]}:{i}",
                "official_split": split,
                "official_row_index": i,
                "canonical_split": CANONICAL_SPLIT[split],
                "image_sha256": sha_bytes(raw),
                "image_bytes": raw,
                "image_nbytes": len(raw),
                "question": c["question"][i],
                "choices": list(c["choices"][i]),
                "answer": int(c["answer"][i]),
                "subject": c["subject"][i],
                "topic": c["topic"][i],
                "category": c["category"][i],
                "grade": c["grade"][i],
                "task": c["task"][i],
                "has_hint": bool((c["hint"][i] or "").strip()),
                "has_lecture": bool((c["lecture"][i] or "").strip()),
                "has_solution": bool((c["solution"][i] or "").strip()),
            })
    return rows


def image_area(raw: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(raw)) as im:
        return im.size


def build_prompt(question: str, choices: list[str]) -> str:
    block = "\n".join(f"({LETTERS[j]}) {c}" for j, c in enumerate(choices))
    return PROMPT_TEMPLATE.format(question=question.strip(), choices=block)


def largest_remainder(counts: dict, total: int) -> dict:
    """Allocate `total` seats across strata proportionally, by largest remainder."""
    n = sum(counts.values())
    if n == 0:
        raise ValueError("no eligible records")
    total = min(total, n)
    exact = {k: v * total / n for k, v in counts.items()}
    base = {k: int(v) for k, v in exact.items()}
    rem = total - sum(base.values())
    order = sorted(exact, key=lambda k: (-(exact[k] - base[k]), k))
    for k in order[:rem]:
        base[k] += 1
    # A stratum cannot give more than it has; re-balance any overflow.
    overflow = {k: base[k] - counts[k] for k in base if base[k] > counts[k]}
    if overflow:
        for k, over in overflow.items():
            base[k] = counts[k]
        deficit = total - sum(base.values())
        pool = sorted((k for k in counts if base[k] < counts[k]),
                      key=lambda k: (-counts[k], k))
        idx = 0
        while deficit > 0 and pool:
            k = pool[idx % len(pool)]
            if base[k] < counts[k]:
                base[k] += 1
                deficit -= 1
            idx += 1
            if idx > 10 * len(pool) + 10:
                break
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-seed", type=int, default=20261015)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    (out / "dataset").mkdir(parents=True, exist_ok=True)

    rows = load_official()
    print(f"[data] official records: {len(rows)}", flush=True)

    # ---- cross-split image clusters are excluded everywhere
    by_sha: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_sha[r["image_sha256"]].append(r)
    cross = {s for s, v in by_sha.items() if len({x["official_split"] for x in v}) > 1}
    excl = {"cross_split_image_records": sum(len(by_sha[s]) for s in cross),
            "cross_split_image_clusters": len(cross),
            "per_split_excluded": dict(collections.Counter(
                x["official_split"] for s in cross for x in by_sha[s]))}
    print(f"[data] excluded cross-split image records: {excl['cross_split_image_records']}",
          flush=True)

    clean = [r for r in rows if r["image_sha256"] not in cross]
    # group id: an image shared by several questions inside ONE official split
    grp: dict[str, str] = {}
    for r in clean:
        grp.setdefault(r["image_sha256"], f"img:{r['image_sha256'][:16]}")
    for r in clean:
        r["image_group_id"] = grp[r["image_sha256"]]
    for r in clean:
        r["image_size"] = list(image_area(r["image_bytes"]))
        r["area"] = r["image_size"][0] * r["image_size"][1]
        r["n_choices"] = len(r["choices"])
        r["gold_letter"] = LETTERS[r["answer"]]
        r["prompt"] = build_prompt(r["question"], r["choices"])

    # ---- size tercile per split, then stratify by (subject, size_bucket)
    selected: dict[str, list[dict]] = {}
    selection_log = {}
    for split in SPLITS:
        pool = [r for r in clean if r["official_split"] == split]
        pool.sort(key=lambda r: r["official_row_index"])
        areas = sorted(r["area"] for r in pool)
        if len(pool) < 3:
            raise SystemExit(f"{split}: too few eligible records")
        q1, q2 = areas[len(areas) // 3], areas[2 * len(areas) // 3]
        for r in pool:
            r["size_bucket"] = ("small" if r["area"] <= q1 else
                                "medium" if r["area"] <= q2 else "large")
            r["stratum"] = f"{r['subject']}|{r['size_bucket']}"
        counts = collections.Counter(r["stratum"] for r in pool)
        alloc = largest_remainder(dict(counts), TARGETS[split])
        pick: list[dict] = []
        for stratum, k in sorted(alloc.items()):
            if k == 0:
                continue
            cands = [r for r in pool if r["stratum"] == stratum]
            cands.sort(key=lambda r: sha_text(f"{a.split_seed}:{r['pid']}"))
            pick.extend(cands[:k])
        pick.sort(key=lambda r: r["official_row_index"])
        selected[split] = pick
        selection_log[split] = {
            "eligible_records": len(pool),
            "eligible_unique_images": len({r["image_sha256"] for r in pool}),
            "quota": TARGETS[split],
            "selected_records": len(pick),
            "selected_unique_images": len({r["image_sha256"] for r in pick}),
            "size_tercile_cutoffs": {"p33": q1, "p67": q2},
            "stratum_pool": dict(counts),
            "stratum_alloc": {k: v for k, v in sorted(alloc.items()) if v},
            "stratum_selected": dict(collections.Counter(r["stratum"] for r in pick)),
        }
        print(f"[data] {split}: pool {len(pool)} -> selected {len(pick)} "
              f"({len({r['image_sha256'] for r in pick})} unique images)",
              flush=True)

    gate = {s: len(selected[s]) >= TARGETS[s] for s in SPLITS}
    if not all(gate.values()):
        raise SystemExit(f"readiness gate failed: {gate}")

    # ---- image groups may not be split across sets (must be empty by construction)
    set_of_group: dict[str, set] = collections.defaultdict(set)
    for split in SPLITS:
        for r in selected[split]:
            set_of_group[r["image_group_id"]].add(split)
    shared_groups = {g: sorted(v) for g, v in set_of_group.items() if len(v) > 1}
    # ---- no duplicated pid, and no exact record in two sets
    pid_sets = {s: {r["pid"] for r in selected[s]} for s in SPLITS}
    record_sets = {s: {(r["image_sha256"], r["question"],
                        tuple(r["choices"]), r["answer"]) for r in selected[s]}
                   for s in SPLITS}

    # ---- extract images to the canonical mnt dataset dir
    img_root = pathlib.Path(OFFICIAL_DIR) / "images"
    img_root.mkdir(parents=True, exist_ok=True)
    image_hashes = []
    manifest_rows = []
    for split in SPLITS:
        d = img_root / CANONICAL_SPLIT[split]
        d.mkdir(parents=True, exist_ok=True)
        for r in selected[split]:
            p = d / f"{r['official_row_index']:05d}.png"
            if not p.exists() or sha_bytes(p.read_bytes()) != r["image_sha256"]:
                with Image.open(io.BytesIO(r["image_bytes"])) as im:
                    im.convert("RGB").save(p, format="PNG")
            b = p.read_bytes()
            h = sha_bytes(b)
            if h != r["image_sha256"]:
                raise SystemExit(
                    f"{r['pid']}: re-encoded PNG {h[:16]} != source image "
                    f"{r['image_sha256'][:16]}; refusing to substitute pixels")
            with Image.open(p) as im:
                im.verify()
            image_hashes.append((f"{CANONICAL_SPLIT[split]}/{p.name}", h))
            manifest_rows.append({
                "pid": r["pid"], "canonical_split": CANONICAL_SPLIT[split],
                "official_split": r["official_split"],
                "official_row_index": r["official_row_index"],
                "image_path": str(p), "image_sha256": h,
                "image_nbytes": len(b), "image_size": f"{r['image_size'][0]}x{r['image_size'][1]}",
                "image_group_id": r["image_group_id"],
                "subject": r["subject"], "topic": r["topic"],
                "category": r["category"], "grade": r["grade"],
                "n_choices": r["n_choices"], "answer_index": r["answer"],
                "gold_letter": r["gold_letter"],
                "has_hint": r["has_hint"], "has_lecture": r["has_lecture"],
                "has_solution": r["has_solution"],
                "question": r["question"],
                "choices": " || ".join(r["choices"]),
                "prompt_sha256": sha_text(r["prompt"]),
                "prompt": r["prompt"],
            })
    print(f"[data] extracted {len(image_hashes)} images -> {img_root}", flush=True)

    # ---- deliverables
    for split in SPLITS:
        json.dump([r["pid"] for r in selected[split]],
                  open(out / "dataset" / f"final_{CANONICAL_SPLIT[split]}_ids.json", "w"),
                  indent=2)
    with open(out / "dataset" / "image_hashes.sha256", "w") as fh:
        for name, h in sorted(image_hashes):
            fh.write(f"{h}  {name}\n")
    mcols = list(manifest_rows[0].keys())
    with open(out / "dataset" / "records_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=mcols)
        w.writeheader()
        w.writerows(manifest_rows)

    # ---- duplicate groups (the clusters used as the statistics unit)
    with open(out / "dataset" / "duplicate_groups.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_group_id", "canonical_split", "group_size", "pids"])
        g = collections.defaultdict(list)
        for r in manifest_rows:
            g[(r["image_group_id"], r["canonical_split"])].append(r["pid"])
        for (gid, split), pids in sorted(g.items()):
            w.writerow([gid, split, len(pids), "|".join(pids)])

    readiness = {
        "split_seed": a.split_seed,
        "source": {"repo": OFFICIAL_REPO, "revision": OFFICIAL_REV},
        "targets": TARGETS,
        "achieved": {s: len(selected[s]) for s in SPLITS},
        "distinct_images": {s: len({r["image_sha256"] for r in selected[s]})
                            for s in SPLITS},
        "distinct_image_groups": {s: len({r["image_group_id"] for r in selected[s]})
                                  for s in SPLITS},
        "gate_pass": gate,
        "cross_split_exclusion": excl,
        "cross_set_groups": shared_groups,
        "cross_set_duplicate_pids": {
            "train∩dev": sorted(pid_sets["train"] & pid_sets["validation"]),
            "train∩test": sorted(pid_sets["train"] & pid_sets["test"]),
            "dev∩test": sorted(pid_sets["validation"] & pid_sets["test"]),
        },
        "cross_set_duplicate_records": {
            "train∩dev": len(record_sets["train"] & record_sets["validation"]),
            "train∩test": len(record_sets["train"] & record_sets["test"]),
            "dev∩test": len(record_sets["validation"] & record_sets["test"]),
        },
        "every_row_has_real_image": True,
        "every_image_deep_verified": True,
        "selection_log": selection_log,
        "gold_isolation": {
            "answer_in_prompt": False, "lecture_in_prompt": False,
            "solution_in_prompt": False, "hint_in_prompt": False,
            "note": ("the frozen prompt is question + lettered options + one "
                     "instruction sentence; `records_manifest.csv` keeps the gold "
                     "letter only as metadata for the separate accuracy appendix"),
        },
    }
    json.dump(readiness, open(out / "dataset" / "readiness_gate.json", "w"),
              indent=2, ensure_ascii=False)
    json.dump({
        "repo": OFFICIAL_REPO, "revision": OFFICIAL_REV,
        "files": json.load(open(os.path.join(
            OFFICIAL_DIR, "download_manifest.json")))["files"]
        if os.path.exists(os.path.join(OFFICIAL_DIR, "download_manifest.json")) else {},
        "extracted_images": len(image_hashes),
        "extracted_bytes": sum(r["image_nbytes"] for r in manifest_rows),
        "local_roots_reused_readonly": [
            "/home/mcy/Projects/Draft-OPD/opd_multimodal/data_sqa_train",
            "/home/mcy/Projects/Draft-OPD/opd_multimodal/data_sqa_test",
            "/home/mcy/Projects/Draft-OPD/opd_multimodal/data_sqa_full"],
        "note": ("official validation (2097 image questions) had to be downloaded: "
                 "it existed in none of the three supplied roots. train was "
                 "available only as a 2400-record random subset of the official "
                 "6218, so the remaining official train records were also needed."),
    }, open(out / "dataset" / "download_manifest.json", "w"), indent=2,
        ensure_ascii=False)

    print(json.dumps({k: readiness[k] for k in
                      ("achieved", "distinct_images", "gate_pass",
                       "cross_set_groups", "cross_set_duplicate_records")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
