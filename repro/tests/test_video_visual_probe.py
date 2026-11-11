"""CPU regressions for visual masking and evaluation preflight contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from evaluation.visual_probe import VisualZeroProbe, zero_visual_context


class VisualMaskTests(unittest.TestCase):
    def setUp(self):
        self.mask = torch.tensor([False, True, True, False, False])

    def test_prefill_zeroes_only_visual_and_preserves_source(self):
        hidden = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3) + 1
        before = hidden.clone()
        out, count = zero_visual_context(
            hidden, self.mask, torch.arange(9)[None], cache_start=0)
        self.assertEqual(count, 2)
        self.assertTrue(torch.equal(hidden, before))
        self.assertTrue(torch.equal(out[:, ~self.mask], before[:, ~self.mask]))
        self.assertEqual(out[:, self.mask].count_nonzero().item(), 0)

    def test_incremental_generated_text_is_never_masked(self):
        # The old prompt-prefix slice would zero rows 1/2 in this text chunk.
        hidden = torch.ones(1, 3, 4)
        out, count = zero_visual_context(
            hidden, self.mask, torch.arange(5, 12)[None], cache_start=5)
        self.assertIs(out, hidden)
        self.assertEqual(count, 0)

    def test_partial_prompt_uses_absolute_positions(self):
        hidden = torch.ones(1, 4, 2)
        out, count = zero_visual_context(
            hidden, self.mask, torch.arange(2, 10)[None], cache_start=2)
        self.assertEqual(count, 1)
        self.assertTrue(torch.equal(out[0, 0], torch.zeros(2)))
        self.assertTrue(torch.equal(out[0, 1:], hidden[0, 1:]))

    def test_bad_positions_fail_closed(self):
        hidden = torch.ones(1, 3, 4)
        cases = [None, torch.arange(8)[None, None],
                 torch.tensor([[0, 2, 3, 4]]), torch.tensor([[-1, 0, 1]]),
                 torch.arange(8, dtype=torch.float32)[None], torch.tensor([[0, 1]])]
        for positions in cases:
            with self.subTest(positions=positions):
                with self.assertRaises(ValueError):
                    zero_visual_context(hidden, self.mask, positions)

    def test_cache_and_positions_disagreement_rejected(self):
        with self.assertRaisesRegex(ValueError, "draft cache"):
            zero_visual_context(torch.ones(1, 2, 4), self.mask,
                                torch.arange(5, 11)[None], cache_start=0)

    def test_zero_context_extension_is_noop(self):
        hidden = torch.empty(1, 0, 4)
        out, count = zero_visual_context(hidden, self.mask,
                                         torch.arange(20, 24)[None], cache_start=20)
        self.assertIs(out, hidden)
        self.assertEqual(count, 0)

    def test_probe_reset_does_not_reuse_previous_video_or_call_records(self):
        head = SimpleNamespace(forward=lambda **kw: kw["target_hidden"])
        probe = VisualZeroProbe(head)
        probe.reset(self.mask)
        out = probe(target_hidden=torch.ones(1, 5, 2),
                    position_ids=torch.arange(9)[None],
                    past_key_values=SimpleNamespace(get_seq_length=lambda: 0))
        self.assertEqual(probe.records[0]["zeroed_visual_rows"], 2)
        self.assertEqual(out[0, 1].sum().item(), 0)
        probe.reset(torch.zeros(5, dtype=torch.bool))
        out = probe(target_hidden=torch.ones(1, 5, 2),
                    position_ids=torch.arange(9)[None],
                    past_key_values=SimpleNamespace(get_seq_length=lambda: 0))
        self.assertEqual(len(probe.records), 1)
        self.assertEqual(probe.records[0]["zeroed_visual_rows"], 0)
        self.assertEqual(out.sum().item(), 10)


class EvalPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evaluation import eval_video_opd
        cls.eval = eval_video_opd

    def test_existing_result_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "result.json"
            path.write_text("historical result")
            with self.assertRaises(FileExistsError):
                self.eval.write_result(path, {"new": "result"})
            self.assertEqual(path.read_text(), "historical result")

    def test_legacy_rows_missing_cache_fail_before_model_execution(self):
        import pandas as pd
        df = pd.DataFrame([{"video_name": "v", "question": "Describe.",
                            "cache_path": "/nonexistent/video-cache.mp4", "fps": 30}])
        with self.assertRaisesRegex(ValueError, "cache_path"):
            self.eval.validate_rows(df, "legacy")

    def test_real_rows_cannot_silently_fall_back_to_cache_fps(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root) / "cache.mp4"
            cache.touch()
            df = pd.DataFrame([{"video_name": "v", "question": "Describe.",
                                "cache_path": str(cache), "fps": 30}])
            with self.assertRaises((ValueError, KeyError)):
                self.eval.validate_rows(df, "real")

    def test_real_cache_identity_is_verified_before_model_execution(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root) / "cache.npz"
            cache.write_bytes(b"frame-cache fixture")
            row = {"video_name": "v", "question": "Describe.", "cache_path": str(cache),
                   "cache_hash": hashlib.sha256(cache.read_bytes()).hexdigest(),
                   "n_frames": 2, "source_fps": 10, "video_duration": 2,
                   "sample_frame_ids": [0, 10], "sample_timestamps": [0., 1.],
                   "width": 2, "height": 2}
            identities = self.eval.validate_rows(pd.DataFrame([row]), "real")
            self.assertTrue(identities[0]["real_time_exact"])
            cache.write_bytes(b"different pixels, stale manifest")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                self.eval.validate_rows(pd.DataFrame([row]), "real")

    def test_real_preflight_rejects_approximate_timeline_before_loading(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root) / "cache.npz"
            cache.touch()
            row = {"video_name": "v", "question": "Describe.", "cache_path": str(cache),
                   "n_frames": 2, "source_fps": 10, "video_duration": 2,
                   "sample_frame_ids": [0, 10], "sample_timestamps": [0., 1.5],
                   "width": 2, "height": 2}
            with self.assertRaisesRegex(ValueError, "cannot be represented exactly"):
                self.eval.validate_rows(pd.DataFrame([row]), "real")

    def test_input_identity_changes_for_pixels_or_time_tokens(self):
        inputs = {"input_ids": torch.tensor([[1, 2, 3]]),
                  "pixel_values_videos": torch.ones(2, 3),
                  "video_grid_thw": torch.tensor([[1, 2, 2]])}
        before = self.eval.encoded_identity(inputs)
        inputs["input_ids"][0, 1] = 7
        after_time = self.eval.encoded_identity(inputs)
        self.assertNotEqual(before["input_ids"], after_time["input_ids"])
        self.assertEqual(before["pixel_values_videos"], after_time["pixel_values_videos"])
        inputs["pixel_values_videos"][0, 0] = 0
        self.assertNotEqual(after_time["pixel_values_videos"],
                            self.eval.encoded_identity(inputs)["pixel_values_videos"])

    def test_cli_preflight_never_loads_model_weights(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            cache = root / "cache.mp4"
            cache.touch()
            rows = root / "rows.parquet"
            rows.write_bytes(b"manifest fixture; read_parquet mocked below")
            target, head = root / "target", root / "head"
            target.mkdir()
            head.mkdir()
            text = {"hidden_size": 4096, "vocab_size": 160000, "num_hidden_layers": 36}
            (target / "config.json").write_text(json.dumps({"text_config": text}))
            (head / "config.json").write_text(json.dumps({
                "hidden_size": 4096, "vocab_size": 160000, "num_target_layers": 36,
                "dflash_config": {"target_layer_ids": [1, 9, 17, 25, 33]}}))
            (head / "model.safetensors").write_bytes(b"CPU identity fixture, never loaded")
            out = root / "preflight.json"
            df = pd.DataFrame([{"video_name": "v", "question": "Describe.",
                                "cache_path": str(cache), "fps": 30}])
            argv = ["eval_video_opd.py", "--rows", str(rows), "--head", str(head),
                    "--target", str(target), "--out", str(out), "--preflight-only", "--ar"]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(self.eval.pd, "read_parquet", return_value=df), \
                    mock.patch.object(self.eval.common, "load_target",
                                      side_effect=AssertionError("GPU model load forbidden")):
                self.eval.main()
            result = json.loads(out.read_text())
            self.assertEqual(result["status"], "CPU_PREFLIGHT_ONLY_NO_MODEL_EXECUTION")
            self.assertFalse(result["protocol"]["lossless_checked"])
            before = result["protocol"]["artifacts"]
            (head / "model.safetensors").write_bytes(b"replacement under the same path")
            after = self.eval.evaluation_artifacts(target, head)
            self.assertNotEqual(before["head_weights_sha256"], after["head_weights_sha256"])


if __name__ == "__main__":
    unittest.main()
