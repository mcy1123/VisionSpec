#!/usr/bin/env python3
"""Build the 16K training prompts: 8000 COCO-Caption2017 + 8000 TextVQA.

[Table 5] "prompts: 8000 COCO + 8000 TextVQA".
[App.D] the images are read at long side 768 px.

Both prompt *wordings* are a documented deviation (CONFIG.md §8): the paper
prints neither the COCO captioning instruction nor the TextVQA phrasing, and
the task JSONLs are not part of the released tarball.

    python build_train_prompts.py --out $GLANCE_ROOT/data/train_prompts.jsonl
"""

from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import argparse
import json
import os
import random
import zipfile

from shared import common


def coco_rows(anno_zip, data_root, n, rng):
    """COCO-Caption2017: one image per prompt, the standard captioning ask.

    COCO's own `annotations_train2017.zip` is 404 from this network, so the
    image list comes from `ChristophSchuhmann/MS_COCO_2017_URL_TEXT`, which is
    the same 591,753 captions over the same 118,287 train2017 images with the
    canonical `images.cocodataset.org/train2017/<id>.jpg` URLs.
    """
    if anno_zip.endswith(".parquet"):
        import pyarrow.parquet as pq

        table = pq.read_table(anno_zip, columns=["URL"])
        urls = sorted(set(table.column("URL").to_pylist()))
        rng.shuffle(urls)
        chosen = urls[:n]
        return [{
            "task": "coco",
            "image": os.path.join(data_root, "coco", "train2017",
                                  os.path.basename(url)),
            "prompt": common.PROMPT_CAPTION,
            "coco_url": url,
        } for url in chosen]

    with zipfile.ZipFile(anno_zip) as zf:
        with zf.open("annotations/captions_train2017.json") as handle:
            anno = json.load(handle)

    images = anno["images"]
    ids = sorted(int(im["id"]) for im in images)
    rng.shuffle(ids)
    chosen = ids[:n]
    by_id = {int(im["id"]): im["file_name"] for im in images}

    rows = []
    for image_id in chosen:
        rows.append({
            "task": "coco",
            "image": os.path.join(data_root, "coco", "train2017", by_id[image_id]),
            "prompt": common.PROMPT_CAPTION,
            "coco_id": image_id,
        })
    return rows


def textvqa_rows(anno_json, data_root, n, rng):
    """TextVQA train: 34,602 questions over 21,953 images; take n questions."""
    with open(anno_json) as handle:
        anno = json.load(handle)

    data = anno["data"]
    order = list(range(len(data)))
    rng.shuffle(order)
    rows = []
    for index in order[:n]:
        record = data[index]
        rows.append({
            "task": "textvqa",
            "image": os.path.join(data_root, "textvqa", "train_images",
                                  f"{record['image_id']}.jpg"),
            "prompt": record["question"],
            "textvqa_image_id": record["image_id"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=common.ROOT)
    parser.add_argument("--out", default=os.path.join(common.DATA, "train_prompts.jsonl"))
    parser.add_argument("--n-coco", type=int, default=common.N_COCO_PROMPTS)
    parser.add_argument("--n-textvqa", type=int, default=common.N_TEXTVQA_PROMPTS)
    parser.add_argument("--seed", type=int, default=common.SEED)
    args = parser.parse_args()

    datasets = os.path.join(args.root, "datasets")
    rng = random.Random(args.seed)

    rows = []
    coco_sources = [
        os.path.join(datasets, "coco_urls", "mscoco.parquet"),
        os.path.join(datasets, "coco", "annotations_train2017.zip"),
    ]
    for source in coco_sources:
        if os.path.exists(source):
            rows += coco_rows(source, datasets, args.n_coco, rng)
            print(f"COCO prompts from {source}")
            break
    else:
        print("[skip] no COCO annotation source found")

    tv_anno = os.path.join(datasets, "textvqa", "TextVQA_0.5.1_train.json")
    if os.path.exists(tv_anno):
        rows += textvqa_rows(tv_anno, datasets, args.n_textvqa, rng)
    else:
        print(f"[skip] TextVQA annotations missing: {tv_anno}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {}
    for row in rows:
        counts[row["task"]] = counts.get(row["task"], 0) + 1
    print(f"wrote {len(rows)} prompts -> {args.out}  {counts}")


if __name__ == "__main__":
    main()
