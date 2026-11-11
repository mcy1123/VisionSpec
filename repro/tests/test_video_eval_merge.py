"""CPU gates against mixing incompatible video-evaluation shards."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import merge_eval_jsons as merge


def fixture(*, protocol=True):
    cfg = {"rows": "same.parquet", "head": "head", "budget": 63, "branch": 8,
           "max_new": 1024, "ar": True, "warmup": 1, "dtype": "float32",
           "shards": 2}
    parts = [{"config": {**cfg, "shard": i, "gpu": i},
              "rows": [{"video": f"v{i}", "lossless": True}]} for i in range(2)]
    if protocol:
        for i, part in enumerate(parts):
            part["config"].update(time_mode="real", target="target", eos_stop="immediate",
                                  zero_visual=False, visual_memory=None)
            part["protocol"] = {"version": "video_eval_real_timeline_v1",
                                "time_mode": "real", "eos_stop": "immediate",
                                "rows_sha256": "abc", "target": "target",
                                "model_contract": {"hidden_size": 4096},
                                "target_device_map": {"": f"cuda:{i}"},
                                "lossless_checked": True}
    return parts


class MergeTests(unittest.TestCase):
    def test_same_protocol_allows_different_shard_devices(self):
        result = merge.merge_results(fixture(), expect=2)
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["status"], "LOSSLESS_PASSED")
        self.assertEqual(len(result["protocol"]["shard_target_device_maps"]), 2)
        self.assertNotIn("target_device_map", result["protocol"])

    def test_each_new_config_dimension_must_match(self):
        for key, value in [("time_mode", "legacy"), ("eos_stop", "legacy"),
                           ("target", "different-target"), ("zero_visual", True),
                           ("visual_memory", "another.pt")]:
            with self.subTest(key=key):
                parts = fixture()
                parts[1]["config"][key] = value
                with self.assertRaisesRegex(ValueError, f"config.{key}"):
                    merge.merge_results(parts)

    def test_protocol_input_hash_or_mask_mismatch_rejected(self):
        for key in ("version", "rows_sha256", "visual_mask", "model_contract", "artifacts",
                    "timing_scope", "uses_cpu_or_disk_offload"):
            with self.subTest(key=key):
                parts = fixture()
                parts[1]["protocol"][key] = "changed"
                with self.assertRaisesRegex(ValueError, f"protocol.{key}"):
                    merge.merge_results(parts)

    def test_identical_legacy_missing_fields_still_merge(self):
        result = merge.merge_results(fixture(protocol=False))
        self.assertEqual(len(result["rows"]), 2)
        self.assertNotIn("protocol", result)

    def test_versioned_and_unversioned_protocols_cannot_mix(self):
        parts = fixture()
        del parts[1]["protocol"]
        with self.assertRaisesRegex(ValueError, "versioned"):
            merge.merge_results(parts)

    def test_missing_or_repeated_shard_rejected(self):
        parts = fixture()
        with self.assertRaisesRegex(ValueError, "shard indices"):
            merge.merge_results(parts[:1])
        parts[1]["config"]["shard"] = 0
        with self.assertRaisesRegex(ValueError, "shard indices"):
            merge.merge_results(parts)

    def test_overlapping_videos_and_wrong_expected_count_rejected(self):
        parts = fixture()
        parts[1]["rows"][0]["video"] = "v0"
        with self.assertRaisesRegex(ValueError, "scored twice"):
            merge.merge_results(parts)
        with self.assertRaisesRegex(ValueError, "expected 3"):
            merge.merge_results(fixture(), expect=3)

    def test_failed_lossless_status_survives_merge(self):
        parts = fixture()
        parts[1]["rows"][0]["lossless"] = False
        self.assertEqual(merge.merge_results(parts)["status"], "LOSSLESS_FAILED")

    def test_cli_does_not_overwrite_historical_output(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "historical.json"
            path.write_text("historical output")
            argv = ["merge_eval_jsons.py", "--parts", "unused.json", "--out", str(path)]
            with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                merge.main()
            self.assertEqual(path.read_text(), "historical output")


if __name__ == "__main__":
    unittest.main()
