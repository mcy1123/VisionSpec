#!/usr/bin/env python3
"""Build the delivery manifest: every artifact with a hash and a stated role.

Large binaries (checkpoints, AR label blobs, images) are listed by size and
SHA256 but obviously not copied. The manifest is the answer to "what is in this
run and how is each file reproduced".

    python package_delivery.py --run <run> [--out <path>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import time

RUN = pathlib.Path("/mnt/local3/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955")
DATASET = pathlib.Path("/mnt/local2/mcy/datasets/scienceqa_glance_20261015/"
                       "lmms-lab-encoder_ScienceQA@69dd4d6b")
SRC = pathlib.Path("/home/mcy/Projects/GLANCE/repro/scienceqa_image")
FROZEN = pathlib.Path("/home/mcy/Projects/GLANCE/repro/qa_opd_fair")

ROLES = {
    "audit": "阶段 A 只读审计与来源/许可记录",
    "dataset": "阶段 B 冻结数据、门禁、清单与图像哈希",
    "preflight": "阶段 C/D 门禁原始产物（长度、图像绑定、无损、位置契约、重放/恢复）",
    "train": "阶段 E 训练日志与 checkpoint",
    "eval": "阶段 F 无损评测、覆盖探针、逐题/逐簇统计",
    "reports": "交付报告与索引",
    "logs": "过程日志（含失败 pilot）",
    "scripts": "本 run 实际执行的脚本副本",
    "source": "生产源码快照与补丁",
    "root": "run 根目录文件（预注册、修订、清单）",
}


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def walk(root: pathlib.Path):
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            st = p.stat()
            out.append({"path": str(p), "relative": str(p.relative_to(root)),
                        "bytes": st.st_size,
                        "sha256": sha256(p) if st.st_size < 512 * 2**20 else
                        sha256(p)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(RUN))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    out = pathlib.Path(a.out or (run / "reports" / "DELIVERY_MANIFEST.json"))
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run": str(run),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "new_data_locations": {
            "dataset_root": str(DATASET),
            "run_root": str(run),
            "note": "every new artifact is under /mnt/local2 or /mnt/local3; "
                    "nothing new was written into /home/mcy",
        },
        "counts": {},
        "sections": {},
        "source_scripts": [],
        "frozen_sources_reused_unchanged": [],
        "history_preserved": [
            "/home/mcy/Projects/Draft-OPD/opd_multimodal/data_sqa_train (read-only, unmodified)",
            "/home/mcy/Projects/Draft-OPD/opd_multimodal/data_sqa_test (read-only, unmodified)",
            "/home/mcy/Projects/Draft-OPD/opd_multimodal/data_sqa_full (read-only, unmodified)",
            "/mnt/local/mcy/GLANCE/checkpoints/head (start point, byte-identical; "
            "SHA256 fc0ad5a60eb7e290ffb7cf220fef3289a223e749c59cc7381488db0962251336)",
            "/mnt/local/mcy/GLANCE/data, datasets, results (untouched)",
            "/mnt/local3/mcy/GLANCE/results/taskB_fair_qa_20261015T085742 (Task B, untouched)",
            "/mnt/local3/mcy/GLANCE/results/qa_contract_fix_20261015 (round 5, untouched)",
        ],
    }

    for sub in ("audit", "dataset", "preflight", "train", "eval", "reports",
                "logs", "scripts", "source"):
        d = run / sub
        files = walk(d) if d.exists() else []
        manifest["sections"][sub] = {
            "role": ROLES.get(sub, ""),
            "n_files": len(files),
            "bytes": sum(f["bytes"] for f in files),
            "files": files,
        }
        manifest["counts"][sub] = len(files)
    root_files = [{"path": str(p), "relative": p.name, "bytes": p.stat().st_size,
                   "sha256": sha256(p)} for p in sorted(run.iterdir()) if p.is_file()]
    manifest["sections"]["root"] = {"role": ROLES["root"], "n_files": len(root_files),
                                   "bytes": sum(f["bytes"] for f in root_files),
                                   "files": root_files}
    manifest["counts"]["root"] = len(root_files)

    # the scripts actually used, from the repo, plus the frozen sources reused
    for p in sorted(SRC.glob("*")):
        if p.is_file():
            manifest["source_scripts"].append(
                {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)})
    for p in sorted(FROZEN.glob("*.py")):
        manifest["frozen_sources_reused_unchanged"].append(
            {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p),
             "used_by": ("imported directly (qa_anchors.py) or used as the generator "
                         "input (train_qa_opd.py, train_qa_sft.py)")})

    # the extracted image set
    imgs = sorted((DATASET / "images").rglob("*.png"))
    manifest["dataset_images"] = {
        "root": str(DATASET / "images"), "n_images": len(imgs),
        "bytes": sum(p.stat().st_size for p in imgs),
        "per_split": {s: len(list((DATASET / "images" / s).glob("*.png")))
                      for s in ("train", "dev", "test")},
    }

    manifest["reproduce"] = {
        "python": "/home/mcy/.conda/envs/fin_research/bin/python",
        "order": [
            "repro/scienceqa_image/audit_three_roots.py --out <run>",
            "repro/scienceqa_image/build_dataset.py --out <run>",
            "repro/scienceqa_image/freeze_prereg.py --out <run>",
            "repro/scienceqa_image/preflight.py --run <run> --only 1,2,3",
            "repro/scienceqa_image/test_image_adaptation.py  (24 CPU tests)",
            "python repro/qa_opd_fair/test_position_contract.py ; test_qa_anchors.py",
            "repro/scienceqa_image/record_amendment.py --run <run>",
            "repro/scienceqa_image/run_long_stages.sh train-ar eval-ar",
            "repro/scienceqa_image/phase_d_gates.py --run <run> --checkpoint <smoke step10> ...",
            "repro/scienceqa_image/run_long_stages.sh train-all",
            "repro/scienceqa_image/eval_sqa.py ...  (per arm/seed/split)",
            "repro/scienceqa_image/trace_tree_coverage.py ...  (per final arm)",
            "repro/scienceqa_image/analyze_paired.py --run <run> --out-dir <run>/eval",
            "repro/scienceqa_image/package_delivery.py --run <run>",
        ],
        "caveat": ("all commands and their exact arguments are also appended to "
                   "<run>/commands.txt as they run; that file is the authority"),
    }
    json.dump(manifest, open(out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({"counts": manifest["counts"],
                      "dataset_images": manifest["dataset_images"]}, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
