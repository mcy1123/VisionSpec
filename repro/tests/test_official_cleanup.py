"""CPU regressions for entry routing, resume identity, and historical metrics."""

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import copy
import hashlib
import io
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from shared import training_resume as R
from shared.master_optim import MasterWeightAdamW
from scienceqa_image.metrics_contract import (
    classify_lengths,
    correct_saved_anchor,
    answer_metrics,
)

ROOT = _REPRO_ROOT.parent


class ResumeTests(unittest.TestCase):
    def run_steps(self, head, opt, begin, end):
        trace = []
        for step in range(begin, end):
            data_id = random.randrange(100)
            reveal = int(np.random.randint(1, 16))
            x = torch.rand(1, 2, dtype=next(head.parameters()).dtype)
            if getattr(self, "official_objective", False):
                from opd_official_kl.loss import pool

                q = torch.nn.functional.logsigmoid(head(x).float()).reshape(-1)
                p = q.detach().new_tensor([-(data_id + reveal + 1) / 100])
                loss = pool(q, p, q, p, torch.tensor([reveal])).loss()
            else:
                loss = (head(x) - (data_id + reveal) / 100).square().sum()
            opt.zero_grad()
            loss.backward()
            grads = [p.grad.clone() for p in head.parameters()]
            opt.sync_grads()
            opt.step()
            trace.append((data_id, reveal, loss.item(), grads))
        return trace

    def test_full_resume(self):
        for dtype in [torch.float32, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                self.check_full_resume(dtype)

    def check_full_resume(self, dtype):
        torch.manual_seed(123)
        random.seed(123)
        np.random.seed(123)
        h = torch.nn.Linear(2, 1).to(dtype)
        o = MasterWeightAdamW(
            h, lr=0.01, betas=(0.9, 0.95), weight_decay=0.0, where="cpu", clip=1.0
        )
        self.run_steps(h, o, 0, 2)
        state = dict(
            head=copy.deepcopy(h.state_dict()),
            master_optimizer=copy.deepcopy(o.state_dict()),
            update=2,
            **R.extras(2, {"schedule": "toy", "config": "toy"})
        )
        # Real serialization boundary, no checkpoint file or model data created.
        buf = io.BytesIO()
        torch.save(state, buf)
        buf.seek(0)
        state = torch.load(buf, weights_only=False)
        tail = self.run_steps(h, o, 2, 4)
        h2 = torch.nn.Linear(2, 1).to(dtype)
        o2 = MasterWeightAdamW(
            h2, lr=0.01, betas=(0.9, 0.95), weight_decay=0.0, where="cpu", clip=1.0
        )
        cursor = R.restore(state, h2, o2, {"schedule": "toy", "config": "toy"})
        self.assertEqual(cursor, 2)
        resumed = self.run_steps(h2, o2, cursor, 4)
        for a, b in zip(tail, resumed):
            self.assertEqual(a[:3], b[:3])
            for x, y in zip(a[3], b[3]):
                self.assertTrue(torch.equal(x, y))
        for k, v in h.state_dict().items():
            self.assertTrue(torch.equal(v, h2.state_dict()[k]))

        def equal(a, b):
            if isinstance(a, torch.Tensor):
                self.assertTrue(torch.equal(a, b))
            elif isinstance(a, dict):
                self.assertEqual(a.keys(), b.keys())
                for k in a:
                    equal(a[k], b[k])
            elif isinstance(a, (list, tuple)):
                self.assertEqual(len(a), len(b))
                for x, y in zip(a, b):
                    equal(x, y)
            else:
                self.assertEqual(a, b)

        equal(o.state_dict(), o2.state_dict())
        digest = hashlib.sha256(
            b"".join(
                v.detach().view(torch.uint8).numpy().tobytes()
                for v in h.state_dict().values()
            )
        ).hexdigest()
        print(
            json.dumps(
                dict(
                    objective=(
                        "official_kl"
                        if getattr(self, "official_objective", False)
                        else "toy_sft"
                    ),
                    dtype=str(dtype),
                    resume_final_parameter_sha256=digest,
                    resumed_data_and_reveals=[list(x[:3]) for x in resumed],
                    gradients_bitwise_equal=True,
                    optimizer_bitwise_equal=True,
                )
            )
        )
        for metadata, step in [
            ({"bad": 1}, None),
            ({"schedule": "toy", "config": "toy"}, 3),
        ]:
            with self.assertRaises(ValueError):
                R.restore(state, h2, o2, metadata, step)
        state["data_cursor"] = 1
        with self.assertRaises(ValueError):
            R.restore(state, h2, o2, {"schedule": "toy", "config": "toy"})

    def test_official_resume(self):
        self.official_objective = True
        self.test_full_resume()

    def test_storage_guard(self):
        with self.assertRaises(ValueError):
            R.validate_output("/home/mcy/checkpoints")
        R.validate_output("/mnt/local3/mcy/GLANCE/checkpoints/toy")


class MetricsTests(unittest.TestCase):
    def test_first_blocker(self):
        self.assertEqual(classify_lengths(9, 1, 15), ("in_topk_not_in_tree", 2))
        self.assertEqual(classify_lengths(1, 1, 15), ("not_in_topk", 2))
        self.assertEqual(classify_lengths(1, 1, 1), ("horizon_end", 2))
        self.assertEqual(classify_lengths(0, 0, 0), ("horizon_end", 1))
        with self.assertRaises(ValueError):
            classify_lengths(1, 2, 15)

    def test_saved_eos(self):
        r = dict(
            L_raw=1,
            L_tree=1,
            L_actual=10,
            L_correct_in_actual_path=1,
            horizon=1,
            first_blocker="past_block",
        )
        c = correct_saved_anchor(r)
        self.assertEqual(c["L_actual"], 1)
        self.assertEqual(r["L_actual"], 10)
        del r["L_correct_in_actual_path"]
        self.assertIsNone(correct_saved_anchor(r)["L_actual"])

    def test_accuracy(self):
        r = answer_metrics(
            [
                dict(pred_letter="A", answer_correct=True),
                dict(pred_letter=None, answer_correct=False),
            ]
        )
        self.assertEqual(r["parsed_only"], 1.0)
        self.assertEqual(r["all_questions"], 0.5)


class EntryTests(unittest.TestCase):
    def test_fail_closed(self):
        for entry in [
            "repro/scienceqa_image/train_sqa_opd.py",
            "repro/historical/opd_train_video_fullkl.py",
            "repro/qa_opd_fair/train_qa_opd.py",
            "repro/train_video_official_kl.py",
        ]:
            p = subprocess.run(
                [sys.executable, str(ROOT / entry), "--dry-run"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads(p.stdout)["status"], "explicit_override_recipe")
            p = subprocess.run(
                [sys.executable, str(ROOT / entry), "--updates", "120"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(p.returncode, 2)
            self.assertIn("required", p.stderr)

        # Imported entrypoints must preserve modality just like shell execution.
        import importlib
        from unittest.mock import patch

        for module in ["historical.opd_train_video", "train_video_official_kl",
                       "historical.opd_train_video_fullkl", "qa_opd_fair.train_qa_opd"]:
            entry = importlib.import_module(module)
            with patch("opd_official_kl.cli.main") as dispatch:
                entry.main(["--dry-run"])
                self.assertEqual(dispatch.call_args.args[0], ["--modality", "video", "--dry-run"])

    def test_retired_generator(self):
        p = subprocess.run(
            [
                sys.executable,
                str(ROOT / "repro/scienceqa_image/make_image_trainers.py"),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("retired_generator", p.stderr)

    def test_sft_resume_wiring(self):
        import ast

        for rel in [
            "repro/scienceqa_image/train_sqa_sft.py",
            "repro/qa_opd_fair/train_qa_sft.py",
        ]:
            tree = ast.parse((ROOT / rel).read_text())
            calls = [
                ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)
            ]
            self.assertIn("training_resume.restore", calls)
            self.assertEqual(calls.count("training_resume.extras"), 2)
            self.assertIn("F.cross_entropy", calls)
            self.assertFalse(any("bernoulli" in x or "k3" in x for x in calls))


if __name__ == "__main__":
    unittest.main()
