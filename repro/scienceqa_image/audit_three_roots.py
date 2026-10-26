#!/usr/bin/env python3
"""Phase A: read-only audit of the three ScienceQA roots the user supplied.

Facts this script establishes, with no writes to the audited roots:

1. What each of ``data_sqa_train`` / ``data_sqa_test`` / ``data_sqa_full``
   physically contains: file counts, byte totals, suffix distribution, symlinks,
   broken links, and the shallow tree.
2. What each *record* is. The three roots are produced by
   ``Draft-OPD/opd_multimodal/fetch_sqa_data.py``, which reads
   ``lmms-lab/ScienceQA :: ScienceQA-IMG/<split>-00000-of-00001.parquet``,
   filters to rows that have an embedded image, shuffles with ``seed=0`` and
   truncates to ``--n``. The records are therefore *official question text plus
   a re-encoded PNG*, not trajectories or model output.
3. Which official split each record actually belongs to. The split is NOT taken
   from the directory name: the image bytes are hashed and matched against the
   sha256 of the embedded image of every row of the pinned official parquet.
   That is a stronger key than a ``pid`` string, and it is the only reliable one
   here because no reachable mirror of ScienceQA carries a ``pid`` column (see
   ``source_and_license.md``, which records the verified network limitation).

Outputs go to ``--out`` (a run directory under /mnt). The audited roots are
opened read-only.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import pathlib
import re
import sys

import pyarrow.parquet as pq

LOCAL_ROOTS = {
    "data_sqa_train": "/home/mcy/Projects/Draft-OPD/opd_multimodal/data_sqa_train",
    "data_sqa_test": "/home/mcy/Projects/Draft-OPD/opd_multimodal/data_sqa_test",
    "data_sqa_full": "/home/mcy/Projects/Draft-OPD/opd_multimodal/data_sqa_full",
}
OFFICIAL_DIR = ("/mnt/local2/mcy/datasets/scienceqa_glance_20261015/"
                "lmms-lab-encoder_ScienceQA@69dd4d6b")
OFFICIAL_REPO = "lmms-lab-encoder/ScienceQA"
OFFICIAL_REV = "69dd4d6b67373d38f96a5badd5d24d0eb5bcdc50"
PARQUET_SPLIT = {"train": "train", "validation": "validation", "test": "test"}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def norm_question(q: str) -> str:
    """The bare official question: everything before the `Choices:` block."""
    return q.split("\n\nChoices:")[0].strip()


def choices_block(q: str) -> str:
    try:
        return q.split("\n\nChoices:\n", 1)[1].split("\n\nThink step")[0].strip()
    except Exception:
        return ""


# ---------------------------------------------------------------- inventory ---
def inventory(path: str) -> dict:
    p = pathlib.Path(path)
    info: dict = {"path": path, "exists": p.exists()}
    if not p.exists():
        return info
    info["realpath"] = os.path.realpath(path)
    info["is_symlink"] = p.is_symlink()
    info["is_dir"] = p.is_dir()
    files = [f for f in p.rglob("*") if f.is_file()]
    info["n_files"] = len(files)
    info["bytes"] = sum(f.stat().st_size for f in files)
    info["bytes_human"] = f"{info['bytes'] / 2**20:.1f} MiB"
    info["suffixes"] = dict(collections.Counter(
        (f.suffix.lower() or "<none>") for f in files))
    links = [f for f in p.rglob("*") if f.is_symlink()]
    info["n_symlinks"] = len(links)
    info["broken_symlinks"] = [str(f) for f in links if not f.exists()]
    info["top_level"] = sorted(x.name for x in p.iterdir())
    info["shallow_tree"] = sorted(
        str(x.relative_to(p)) for x in p.rglob("*") if x.is_dir())[:20]
    return info


# ------------------------------------------------------- official parquet ----
def official_index(verbose=True) -> tuple[dict, dict, list[dict]]:
    """Two alignment indices plus a per-row manifest.

    ``by_content`` maps ``(sha256(image bytes), question, choices, answer)`` to
    the list of ``(split, row_index)`` that carry it. The compound key is
    required: ScienceQA legitimately reuses one image across several questions,
    and the official split has placed different questions about the same image
    into different splits. Keying on the image alone therefore produced 235
    false "cross-split" hits in the first pass of this audit; only the full
    record identifies a row.
    """
    by_content: dict[tuple, list[tuple[str, int]]] = {}
    img_rows: dict[str, list[tuple[str, int]]] = {}
    rows_meta: list[dict] = []
    for split in PARQUET_SPLIT:
        path = os.path.join(OFFICIAL_DIR, "ScienceQA-IMG",
                            f"{split}-00000-of-00001.parquet")
        table = pq.read_table(path)
        cols = table.to_pydict()
        img_bytes = table.column("image").flatten()[0].to_pylist()
        n = table.num_rows
        if verbose:
            print(f"[official] {split}: {n} rows", flush=True)
        for i in range(n):
            raw = img_bytes[i]
            h = sha(raw)
            q = cols["question"][i]
            ch = tuple(cols["choices"][i])
            ans = int(cols["answer"][i])
            key = (h, q, ch, ans)
            by_content.setdefault(key, []).append((split, i))
            img_rows.setdefault(h, []).append((split, i))
            rows_meta.append({"official_split": split, "row_index": i,
                              "image_sha256": h,
                              "question": q, "choices": list(ch), "answer": ans,
                              "subject": cols["subject"][i],
                              "topic": cols["topic"][i],
                              "category": cols["category"][i],
                              "grade": cols["grade"][i],
                              "task": cols["task"][i],
                              "has_hint": bool((cols["hint"][i] or "").strip()),
                              "has_lecture": bool((cols["lecture"][i] or "").strip()),
                              "has_solution": bool((cols["solution"][i] or "").strip()),
                              "image_bytes": len(raw)})
    return by_content, img_rows, rows_meta


# ------------------------------------------------------------- local rows ----
def read_local(root: str) -> list[dict]:
    rows = []
    with open(os.path.join(root, "qa.jsonl")) as fh:
        for lineno, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({"lineno": lineno, "raw": r})
    return rows


def audit_local(name: str, root: str, by_content: dict,
                img_rows: dict) -> list[dict]:
    rows = read_local(root)
    out = []
    for r in rows:
        raw = r["raw"]
        img_rel = raw.get("image", "")
        # fetch_sqa_data.py wrote `str(out / "images" / f"{i:05d}.png")` where
        # `out` was the relative output dir passed on the command line, so the
        # stored string is like "data_sqa_train/images/00000.png". Resolve it
        # against the audited root by basename: the basename is the only part
        # that is stable across the three roots (full/ renames with a prefix).
        base = os.path.basename(img_rel)
        img_abs = os.path.join(root, "images", base)
        if not os.path.exists(img_abs):
            # Fallback: honour the stored relative tail if it resolves.
            cand = os.path.join(os.path.dirname(root), img_rel)
            if os.path.exists(cand):
                img_abs = cand
        rec = {
            "local_root": name,
            "lineno": r["lineno"],
            "image_rel": img_rel,
            "image_basename": os.path.basename(img_rel),
            "image_exists": os.path.exists(img_abs),
            "image_sha256": None,
            "image_bytes": None,
            "image_decodable": None,
            "image_size": None,
            "official_split": "unknown",
            "official_row_index": None,
            "official_splits_sharing_this_image": "",
            "kind": raw.get("kind"),
            "answer": raw.get("answer"),
            "n_choices": len([l for l in choices_block(raw.get("question", "")).splitlines()
                              if re.match(r"^\d+\. ", l)]),
            "has_solution": bool((raw.get("solution") or "").strip()),
            "question_norm": norm_question(raw.get("question", "")),
            "choices_norm": choices_block(raw.get("question", "")),
            "n_image_placeholders": raw.get("question", "").count("<image>"),
            "answer_index_valid": None,
        }
        if rec["image_exists"]:
            b = open(img_abs, "rb").read()
            rec["image_bytes"] = len(b)
            rec["image_sha256"] = sha(b)
            # Recover the official row from the full record, not the image alone.
            official_q = rec["question_norm"]
            opts = []
            for line in rec["choices_norm"].splitlines():
                m = re.match(r"^\d+\. (.*)$", line)
                if m:
                    opts.append(m.group(1))
            try:
                ans_i = int(rec["answer"])
            except Exception:
                ans_i = -1
            key = (rec["image_sha256"], official_q, tuple(opts), ans_i)
            hits = by_content.get(key)
            if hits:
                splits = sorted({s for s, _ in hits})
                rec["official_splits_sharing_this_image"] = "|".join(splits)
                if len(hits) == 1:
                    rec["official_split"], rec["official_row_index"] = hits[0]
                else:
                    # The identical record exists in more than one official
                    # split. It is real official duplication; it cannot be
                    # assigned to one split and is excluded downstream.
                    rec["official_split"] = "split_ambiguous"
                    rec["official_row_index"] = hits[0][1]
            else:
                sharing = sorted({s for s, _ in img_rows.get(rec["image_sha256"], [])})
                rec["official_splits_sharing_this_image"] = "|".join(sharing)
            try:
                from PIL import Image
                with Image.open(io.BytesIO(b)) as im:
                    im.verify()
                with Image.open(io.BytesIO(b)) as im:
                    rec["image_size"] = list(im.size)
                rec["image_decodable"] = True
            except Exception as e:      # noqa: BLE001
                rec["image_decodable"] = False
                rec["image_error"] = str(e)[:120]
        try:
            ai = int(rec["answer"])
            rec["answer_index_valid"] = bool(0 <= ai < rec["n_choices"])
        except Exception:
            rec["answer_index_valid"] = False
        out.append(rec)
    return out


# ------------------------------------------------------------------- main ----
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    (out / "audit").mkdir(parents=True, exist_ok=True)

    print("== 1/4 inventory of the three roots", flush=True)
    inv = {name: inventory(root) for name, root in LOCAL_ROOTS.items()}
    json.dump(inv, open(out / "audit" / "three_roots_inventory.json", "w"),
              indent=2, ensure_ascii=False)

    print("== 2/4 official pid/image index", flush=True)
    by_content, img_rows, official_rows = official_index()
    json.dump({
        "repo": OFFICIAL_REPO, "revision": OFFICIAL_REV, "dir": OFFICIAL_DIR,
        "n_official_rows": len(official_rows),
        "n_unique_image_sha": len(img_rows),
        "n_unique_records": len(by_content),
        "per_split_rows": dict(collections.Counter(
            r["official_split"] for r in official_rows)),
        "per_split_unique_records": {
            s: len({k for k, v in by_content.items()
                    if any(x[0] == s for x in v)})
            for s in PARQUET_SPLIT},
        "records_in_more_than_one_split": sum(
            1 for v in by_content.values() if len({s for s, _ in v}) > 1),
        "images_shared_across_splits": sum(
            1 for v in img_rows.values() if len({s for s, _ in v}) > 1),
        "pid_note": ("the official 'pid' string (lupantech/ScienceQA problems.json) "
                     "is not recoverable on this host: github.com/raw.githubusercontent.com "
                     "are unreachable and no reachable HF mirror carries a pid column; "
                     "the alignment key is therefore (sha256(image bytes) + question + "
                     "choices + answer) -> (official split, official row index), which is "
                     "strictly stronger than a pid join"),
    }, open(out / "audit" / "official_pid_alignment_summary.json", "w"),
        indent=2, ensure_ascii=False)

    print("== 3/4 align local records", flush=True)
    aligned: dict[str, list[dict]] = {}
    for name, root in LOCAL_ROOTS.items():
        aligned[name] = audit_local(name, root, by_content, img_rows)
        print(f"   {name}: {len(aligned[name])} rows", flush=True)

    # ---- records_manifest.csv (all rows, all roots)
    import csv
    cols = list(aligned["data_sqa_train"][0].keys())
    for name in aligned:
        for rec in aligned[name]:
            rec.setdefault("image_error", "")
    cols = list(dict.fromkeys([c for name in aligned for rec in aligned[name]
                               for c in rec]))
    with open(out / "audit" / "three_roots_records.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for name in aligned:
            for rec in aligned[name]:
                w.writerow({k: rec.get(k) for k in cols})

    # ---- official_pid_alignment.csv
    with open(out / "audit" / "official_pid_alignment.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["local_root", "lineno", "image_basename", "image_sha256",
                    "image_bytes", "image_decodable", "official_split",
                    "official_row_index", "derived_pid", "n_choices",
                    "answer", "answer_index_valid", "has_solution",
                    "question_norm_sha256"])
        for name in aligned:
            for rec in aligned[name]:
                pid = (f"sqa-img:{rec['official_split']}:{rec['official_row_index']}"
                       if rec["official_row_index"] is not None else "unmatched")
                w.writerow([name, rec["lineno"], rec["image_basename"],
                            rec["image_sha256"], rec["image_bytes"],
                            rec["image_decodable"], rec["official_split"],
                            rec["official_row_index"], pid, rec["n_choices"],
                            rec["answer"], rec["answer_index_valid"],
                            rec["has_solution"],
                            sha(rec["question_norm"].encode())])

    # ---- intersections
    def keyset(name):
        return {(rec["official_split"], rec["official_row_index"])
                for rec in aligned[name]
                if rec["official_split"] not in ("unknown", "split_ambiguous")}

    def shaset(name):
        return {rec["image_sha256"] for rec in aligned[name]
                if rec["image_sha256"]}

    tri, tei, fui = (keyset("data_sqa_train"), keyset("data_sqa_test"),
                     keyset("data_sqa_full"))
    rows = [
        {"relation": "rows", "data_sqa_train": len(aligned["data_sqa_train"]),
         "data_sqa_test": len(aligned["data_sqa_test"]),
         "data_sqa_full": len(aligned["data_sqa_full"])},
        {"relation": "unique_official_record", "data_sqa_train": len(tri),
         "data_sqa_test": len(tei), "data_sqa_full": len(fui)},
        {"relation": "split_ambiguous_rows",
         "data_sqa_train": sum(1 for r in aligned["data_sqa_train"]
                               if r["official_split"] == "split_ambiguous"),
         "data_sqa_test": sum(1 for r in aligned["data_sqa_test"]
                              if r["official_split"] == "split_ambiguous"),
         "data_sqa_full": sum(1 for r in aligned["data_sqa_full"]
                              if r["official_split"] == "split_ambiguous")},
        {"relation": "unmatched_rows",
         "data_sqa_train": sum(1 for r in aligned["data_sqa_train"]
                               if r["official_split"] == "unknown"),
         "data_sqa_test": sum(1 for r in aligned["data_sqa_test"]
                              if r["official_split"] == "unknown"),
         "data_sqa_full": sum(1 for r in aligned["data_sqa_full"]
                              if r["official_split"] == "unknown")},
        {"relation": "unique_image_sha", "data_sqa_train": len(shaset("data_sqa_train")),
         "data_sqa_test": len(shaset("data_sqa_test")),
         "data_sqa_full": len(shaset("data_sqa_full"))},
        {"relation": "official_train_hits",
         "data_sqa_train": sum(1 for s, _ in tri if s == "train"),
         "data_sqa_test": sum(1 for s, _ in tei if s == "train"),
         "data_sqa_full": sum(1 for s, _ in fui if s == "train")},
        {"relation": "official_validation_hits",
         "data_sqa_train": sum(1 for s, _ in tri if s == "validation"),
         "data_sqa_test": sum(1 for s, _ in tei if s == "validation"),
         "data_sqa_full": sum(1 for s, _ in fui if s == "validation")},
        {"relation": "official_test_hits",
         "data_sqa_train": sum(1 for s, _ in tri if s == "test"),
         "data_sqa_test": sum(1 for s, _ in tei if s == "test"),
         "data_sqa_full": sum(1 for s, _ in fui if s == "test")},
    ]
    inter = {
        "train∩test (official record)": len(tri & tei),
        "train∩full (official record)": len(tri & fui),
        "test∩full (official record)": len(tei & fui),
        "full == train ∪ test (official record)": fui == (tri | tei),
        "train∩test (image sha, over-counts shared pictures)":
            len(shaset("data_sqa_train") & shaset("data_sqa_test")),
        "full == train ∪ test (image sha)":
            shaset("data_sqa_full") == (shaset("data_sqa_train") | shaset("data_sqa_test")),
        "official splits present in data_sqa_train":
            sorted({s for s, _ in tri}),
        "official splits present in data_sqa_test":
            sorted({s for s, _ in tei}),
        "official splits present in data_sqa_full":
            sorted({s for s, _ in fui}),
    }
    with open(out / "audit" / "three_roots_intersections.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["relation", "data_sqa_train", "data_sqa_test", "data_sqa_full"])
        for r in rows:
            w.writerow([r["relation"], r["data_sqa_train"], r["data_sqa_test"],
                        r["data_sqa_full"]])
        w.writerow([])
        w.writerow(["intersection", "value"])
        for k, v in inter.items():
            w.writerow([k, v])

    # ---- per-split coverage vs the requested 1200/240/240
    n_off = collections.Counter(r["official_split"] for r in official_rows)
    cov = {
        "official_rows_per_split": dict(n_off),
        "official_unique_records_per_split": {
            s: len({k for k, v in by_content.items() if any(x[0] == s for x in v)})
            for s in PARQUET_SPLIT},
        "official_records_in_more_than_one_split": sum(
            1 for v in by_content.values() if len({s for s, _ in v}) > 1),
        "official_records_sharing_an_image_with_another_split": sum(
            1 for r in official_rows
            if len({s for s, _ in img_rows[r["image_sha256"]]}) > 1),
        "local_split_assignment_per_root": {
            name: dict(collections.Counter(rec["official_split"] for rec in aligned[name]))
            for name in aligned},
        "local_unique_records_per_root_and_split": {
            name: dict(collections.Counter(
                rec["official_split"] for rec in aligned[name]
                if rec["official_split"] not in ("unknown", "split_ambiguous")))
            for name in aligned},
        "requested": {"train": 1200, "dev(official val)": 240, "test": 240},
        "local_capacity_after_dedup": {
            "data_sqa_train_root_official_train_unique":
                len({(r["official_row_index"]) for r in aligned["data_sqa_train"]
                     if r["official_split"] == "train"}),
            "data_sqa_test_root_official_test_unique":
                len({(r["official_row_index"]) for r in aligned["data_sqa_test"]
                     if r["official_split"] == "test"}),
            "any_root_official_validation_unique":
                len({(r["official_row_index"]) for name in aligned
                     for r in aligned[name] if r["official_split"] == "validation"}),
            "official_train_available_total": n_off["train"],
            "official_validation_available_total": n_off["validation"],
            "official_test_available_total": n_off["test"],
        },
    }
    json.dump(cov, open(out / "audit" / "missing_by_official_split.json", "w"),
              indent=2, ensure_ascii=False)

    # ---- duplicate groups (within root, by normalized question)
    with open(out / "dataset" / "duplicate_groups.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "group_key", "group_size", "members"])
        for name in aligned:
            g: dict[str, list[str]] = collections.defaultdict(list)
            for rec in aligned[name]:
                g[rec["question_norm"] + "||" + rec["choices_norm"]].append(
                    f"{name}:{rec['lineno']}:{rec['image_basename']}")
            for k, members in g.items():
                if len(members) > 1:
                    w.writerow([f"within:{name}", sha(k.encode())[:16],
                                len(members), "|".join(members)])
        g2: dict[str, list[str]] = collections.defaultdict(list)
        for name in aligned:
            for rec in aligned[name]:
                g2[rec["question_norm"] + "||" + rec["choices_norm"]].append(
                    f"{name}:{rec['lineno']}:{rec['image_basename']}")
        for k, members in g2.items():
            roots = {m.split(":", 1)[0] for m in members}
            if len(roots) > 1:
                w.writerow(["cross-root", sha(k.encode())[:16], len(members),
                            "|".join(members)])

    # ---- inventory summary of image health
    health = {}
    for name in aligned:
        rs = aligned[name]
        health[name] = {
            "rows": len(rs),
            "image_exists": sum(1 for r in rs if r["image_exists"]),
            "image_decodable": sum(1 for r in rs if r["image_decodable"] is True),
            "image_undecodable": [r["image_basename"] for r in rs
                                  if r["image_decodable"] is False],
            "answer_index_valid": sum(1 for r in rs if r["answer_index_valid"]),
            "rows_with_<image>_placeholder_in_prompt": sum(
                1 for r in rs if r["n_image_placeholders"]),
            "rows_with_solution": sum(1 for r in rs if r["has_solution"]),
            "n_choices_histogram": dict(collections.Counter(r["n_choices"] for r in rs)),
        }
    json.dump(health, open(out / "audit" / "local_image_health.json", "w"),
              indent=2, ensure_ascii=False)

    print("== 4/4 done", flush=True)
    print(json.dumps(inter, indent=2, ensure_ascii=False))
    print(json.dumps(health, indent=2, ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
