"""CPU checks for loss gradients, finite schedules and model/input identity."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from shared.model_contract import validate_model_contract
from video_caption import train_video_9k as trainer


class NormalizationTests(unittest.TestCase):
    def exercise(self, mode):
        parameter = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        events = []

        def collect():
            for index, denominator in enumerate((2.0, 6.0)):
                events.append(("collect", index))
                yield dict(index=index, denominator=denominator)

        def score(trace):
            index = trace["index"]
            events.append(("score", index))
            # Deliberately unrelated Pool denominator: additive TV-like terms
            # use the frozen base denominator, not the augmented score count.
            yield SimpleNamespace(numerator=parameter.square() * (index + 1), denominator=9999)
            yield SimpleNamespace(numerator=parameter * (index + 2), denominator=9999)

        result = trainer.backward_update(collect(), score, mode,
                                         denominator_fn=lambda trace: trace["denominator"])
        return parameter, result, events

    def test_global_pooled_matches_direct_objective_and_gradient(self):
        parameter, result, events = self.exercise("global_pooled")
        reference = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        objective = (3 * reference.square() + 5 * reference) / 8
        objective.backward()
        self.assertEqual(parameter.grad.item(), reference.grad.item())
        self.assertEqual(result["loss"], objective.item())
        self.assertEqual(result["pooled_loss"], result["optimized_loss"])
        self.assertEqual(events, [("collect", 0), ("collect", 1), ("score", 0), ("score", 1)])

    def test_legacy_preserves_sum_of_per_qa_objectives_and_interleaving(self):
        parameter, result, events = self.exercise("legacy_per_qa")
        reference = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        objective = (reference.square() + 2 * reference) / 2 + (2 * reference.square() + 3 * reference) / 6
        objective.backward()
        self.assertEqual(parameter.grad.item(), reference.grad.item())
        self.assertEqual(result["loss"], objective.item())
        self.assertNotEqual(result["pooled_loss"], result["optimized_loss"])
        self.assertEqual(events, [("collect", 0), ("score", 0), ("collect", 1), ("score", 1)])

    def test_global_stages_each_trace_once_and_restores_in_order(self):
        events = []
        parameter = torch.tensor(1.0, requires_grad=True)

        def stage(trace):
            events.append(("stage", trace["index"]))
            return dict(trace, staged=True)

        def restore(trace):
            self.assertTrue(trace["staged"])
            events.append(("restore", trace["index"]))
            return trace

        traces = [dict(index=i, denominator=1) for i in range(2)]
        trainer.backward_update(traces, lambda trace: [SimpleNamespace(numerator=parameter * 2)],
                                "global_pooled", denominator_fn=lambda trace: trace["denominator"],
                                stage_trace=stage, restore_trace=restore)
        self.assertEqual(events, [("stage", 0), ("stage", 1), ("restore", 0), ("restore", 1)])
        self.assertEqual(parameter.grad.item(), 2.0)

    def test_invalid_denominators_fail_before_backward(self):
        for mode in trainer.LOSS_NORMALIZATIONS:
            for bad in (0.0, -1.0, float("inf"), float("nan")):
                with self.subTest(mode=mode, denominator=bad), self.assertRaisesRegex(ValueError, "denominator"):
                    trainer.backward_update([dict(denominator=bad)], lambda trace: [], mode,
                                            denominator_fn=lambda trace: trace["denominator"])


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.head = self.root / "head"
        self.target = self.root / "target"
        self.head.mkdir()
        self.target.mkdir()
        self.head_config = dict(hidden_size=4096, vocab_size=151936, num_target_layers=36,
                                dflash_config=dict(target_layer_ids=[1, 9, 17, 25, 33]))
        self.target_config = dict(text_config=dict(hidden_size=4096, vocab_size=151936, num_hidden_layers=36))
        self.write_configs()
        (self.head / "model.safetensors").write_bytes(b"toy model identity; never loaded")
        self.cache = self.root / "frames.mp4"
        self.cache.write_bytes(b"toy cache; never decoded")
        self.rows = pd.DataFrame([dict(task=f"video{i}", prompt=f"Describe {i}",
                                       cache_path=str(self.cache), fps=30.0) for i in range(2)]).set_index("task")
        self.schedule = self.root / "schedule.jsonl"
        self.items = [dict(update=0, slot=i, task=f"video{i}") for i in range(2)]
        self.write_schedule()

    def write_configs(self):
        (self.head / "config.json").write_text(json.dumps(self.head_config))
        (self.target / "config.json").write_text(json.dumps(self.target_config))

    def write_schedule(self):
        self.schedule.write_text("\n".join(json.dumps(item) for item in self.items) + "\n")

    def test_schedule_requires_exact_qa_count_and_unique_slots(self):
        self.assertEqual(len(trainer.load_schedule(self.schedule, self.rows, 1, 2)[0]), 2)
        with self.assertRaisesRegex(ValueError, "slots"):
            trainer.load_schedule(self.schedule, self.rows, 1, 4)
        self.items[1]["slot"] = 0
        self.write_schedule()
        with self.assertRaisesRegex(ValueError, "slots"):
            trainer.load_schedule(self.schedule, self.rows, 1, 2)

    def test_schedule_checks_tasks_coverage_and_prompt_identity(self):
        self.items[0]["task"] = "missing"
        self.write_schedule()
        with self.assertRaisesRegex(ValueError, "unknown task"):
            trainer.load_schedule(self.schedule, self.rows, 1, 2)
        self.items[0]["task"] = "video0"
        self.items[0]["prompt_sha256"] = "wrong"
        self.write_schedule()
        with self.assertRaisesRegex(ValueError, "prompt hash"):
            trainer.load_schedule(self.schedule, self.rows, 1, 2)
        self.items[0]["prompt_sha256"] = hashlib.sha256(b"Describe 0").hexdigest()
        self.write_schedule()
        with self.assertRaisesRegex(ValueError, "exactly all updates"):
            trainer.load_schedule(self.schedule, self.rows, 2, 2)

    def test_config_contract_accepts_current_head_and_rejects_other_targets(self):
        dims = validate_model_contract(self.target, self.head)
        self.assertEqual(dims["hidden_size"], 4096)
        self.target_config["text_config"]["hidden_size"] = 5120
        self.write_configs()
        with self.assertRaisesRegex(ValueError, "hidden_size mismatch"):
            validate_model_contract(self.target, self.head)
        self.target_config["text_config"]["hidden_size"] = 4096
        self.head_config["dflash_config"]["target_layer_ids"] = [1, 5, 9]
        self.write_configs()
        with self.assertRaisesRegex(ValueError, "target_layer_ids"):
            validate_model_contract(self.target, self.head)

    def cli(self):
        pool = self.root / "pool.parquet"
        self.rows.reset_index().to_parquet(pool, index=False)
        return ["--pool", str(pool), "--schedule", str(self.schedule),
                "--init-from", str(self.head), "--target", str(self.target),
                "--out", "/mnt/local/video-protocol-unittest-output",
                "--updates", "1", "--qa-per-update", "2", "--dry-run"]

    def test_dry_run_is_cpu_only_and_records_actual_semantics(self):
        with patch.object(trainer.common, "load_target", side_effect=AssertionError("model load forbidden")), \
             patch.object(trainer.common, "load_head", side_effect=AssertionError("model load forbidden")), \
             patch.object(torch, "manual_seed", side_effect=AssertionError("RNG must not change")), \
             contextlib.redirect_stdout(io.StringIO()):
            legacy = trainer.main(self.cli())
            pooled = trainer.main(self.cli() + ["--loss-normalization", "global_pooled"])
        self.assertTrue(legacy["diagnostic_only"])
        self.assertFalse(legacy["resume_supported"])
        self.assertEqual(legacy["initialization"], "weights_only_fresh_optimizer")
        self.assertEqual(legacy["loss_contract"]["normalization"], "legacy_per_qa")
        self.assertNotEqual(legacy["protocol_sha256"], pooled["protocol_sha256"])
        self.assertFalse(Path("/mnt/local/video-protocol-unittest-output").exists())

    def test_missing_real_metadata_fails_before_model_load(self):
        with patch.object(trainer.common, "load_target", side_effect=AssertionError("model load forbidden")), \
             self.assertRaises(ValueError):
            trainer.main(self.cli() + ["--time-mode", "real"])

    def test_real_preflight_records_unique_frames_and_rejects_inexact_time(self):
        row = self.rows.loc["video0"].to_dict()
        row.update(n_frames=2, source_fps=30.0, video_duration=3.0,
                   sample_frame_ids=[0, 60], sample_timestamps=[0.0, 2.0], height=28, width=28)
        rows = pd.DataFrame([dict(row, task="video0")]).set_index("task")
        schedule = {0: [dict(task="video0", slot=0)]}
        diagnostic = trainer.validate_rows(rows, schedule, "real")["real_timeline_diagnostics"][0]
        self.assertEqual(diagnostic["n_unique_frames"], 2)
        self.assertTrue(diagnostic["real_time_exact"])
        rows.at["video0", "sample_timestamps"] = [0.0, 2.1]
        with self.assertRaisesRegex(ValueError, "cannot be represented exactly"):
            trainer.validate_rows(rows, schedule, "real")

    def test_cache_identity_is_checked_once_per_unique_file(self):
        schedule = trainer.load_schedule(self.schedule, self.rows, 1, 2)
        digest = hashlib.sha256(self.cache.read_bytes()).hexdigest()
        self.rows["cache_hash"] = digest
        with patch.object(trainer.training_resume, "file_hash", wraps=trainer.training_resume.file_hash) as hashed:
            result = trainer.validate_rows(self.rows, schedule, "legacy")
        self.assertEqual(hashed.call_count, 1)
        self.assertEqual(result["cache_sha256"], {str(self.cache): digest})
        self.cache.write_bytes(b"tampered cache")
        with self.assertRaisesRegex(ValueError, "cache_hash mismatch"):
            trainer.validate_rows(self.rows, schedule, "legacy")

    def test_legacy_preflight_rejects_npz_before_processor_load(self):
        npz = self.root / "frames.npz"
        npz.write_bytes(b"not decoded by preflight")
        self.rows["cache_path"] = str(npz)
        with patch.object(trainer.common, "load_target", side_effect=AssertionError("model load forbidden")), \
             self.assertRaisesRegex(ValueError, "NPZ frame caches require --time-mode real"):
            trainer.main(self.cli())

    def test_external_visual_head_fails_before_model_load(self):
        for flag in ("use_visual_fovea", "use_visual_film", "use_visual_xattn", "use_visual_perceiver"):
            with self.subTest(flag=flag):
                self.head_config["dflash_config"][flag] = True
                self.write_configs()
                with patch.object(trainer.common, "load_target", side_effect=AssertionError("model load forbidden")), \
                     self.assertRaisesRegex(ValueError, "no external visual-memory input"):
                    trainer.main(self.cli())
                self.head_config["dflash_config"].pop(flag)


if __name__ == "__main__":
    unittest.main()
