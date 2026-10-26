#!/usr/bin/env python3
"""Build the ChartQA eval set in the layout `experiments/main_table.py` expects.

The paper reports ChartQA "under its templated prompts" (Figure 4, Table 12)
but never prints the template, and the task JSONL is not part of the released
tarball. CONFIG.md §8 records that gap; this script therefore makes the
template a flag so the choice can be measured rather than assumed.

`ahmed-masry/ChartQA`'s test parquet carries the image bytes, so no separate
image archive is needed: 2500 questions (1250 human + 1250 augmented).

    python build_eval_data.py --out $GLANCE_ROOT/data/vl_eval
    python build_eval_data.py --out .../vl_eval_plain --template "{q}"
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
import io
import json
import os

from shared import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=os.path.join(
        common.DATASETS, "chartqa", "data", "test-00000-of-00001.parquet"))
    parser.add_argument("--out", default=os.path.join(common.DATA, "vl_eval"))
    parser.add_argument("--template", default=common.PROMPT_CHARTQA_TEMPLATE,
                        help="'{q}' is replaced by the ChartQA question")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(args.parquet,
                          columns=["imgname", "query", "label", "type", "image"])
    data = table.to_pydict()

    image_dir = os.path.join(args.out, "chartqa")
    os.makedirs(image_dir, exist_ok=True)

    rows = []
    written = {}
    n = table.num_rows if not args.limit else min(args.limit, table.num_rows)
    for i in range(n):
        name = data["imgname"][i]
        if name not in written:
            image = Image.open(io.BytesIO(data["image"][i])).convert("RGB")
            image.save(os.path.join(image_dir, name))
            written[name] = True
        rows.append({
            "image": os.path.join("chartqa", name),
            "prompt": args.template.format(q=data["query"][i]),
            "label": data["label"][i],
            "type": data["type"][i],
        })

    path = os.path.join(args.out, "chartqa.jsonl")
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} questions over {len(written)} images -> {path}")
    print(f"template: {args.template!r}")
    print(f"example : {rows[0]['prompt']!r}")


if __name__ == "__main__":
    main()
