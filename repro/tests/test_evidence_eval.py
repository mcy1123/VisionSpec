"""CPU input identity / real timeline / intervention gates for evidence heads."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.evidence import (build_evidence_request, configure_evidence,
                             evidence_settings, FEATURE_PROTOCOL)
from evaluation.eval_video_opd import encoded_identity, validate_evidence_eval
from evaluation.merge_eval_jsons import merge_results


def fixture():
    row = {"n_frames": 4, "n_unique_frames": 4, "source_fps": 1.,
           "video_duration": 8., "source_frame_count": 8,
           "sample_frame_ids": [0, 2, 4, 6],
           "sample_timestamps": [0., 2., 4., 6.], "height": 16, "width": 16}
    ins = {"input_ids": torch.tensor([[1, 9, 9, 9, 9, 3, 9, 9, 9, 9, 4]]),
           "pixel_values_videos": torch.arange(384, dtype=torch.float32).reshape(32, 12),
           "video_grid_thw": torch.tensor([[2, 4, 4]])}
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 9, unk_token_id=0),
        video_processor=SimpleNamespace(temporal_patch_size=2, merge_size=2))
    return ins, row, processor


def cfg(mode="visual"):
    return {"dflash_config": {"evidence_mode": mode, "target_layer_ids": [1, 9, 17, 25, 33]}}


class EvidenceRequestTests(unittest.TestCase):
    def test_patch_times_use_real_pair_means_and_duration(self):
        ins, row, proc = fixture()
        request = build_evidence_request(ins, row, proc)
        torch.testing.assert_close(request["time_values"], torch.tensor([1 / 8, 5 / 8]))
        self.assertEqual(request["metadata"]["patch_times_s"], [1., 5.])
        self.assertEqual(request["grid_thw"], [2, 4, 4])
        self.assertEqual(request["metadata"]["feature_protocol"], FEATURE_PROTOCOL)
        self.assertEqual(request["prompt_visual_mask"].dtype, torch.bool)
        self.assertEqual(request["prompt_visual_mask"].numel(), ins["input_ids"].numel())
        self.assertEqual(request["prompt_visual_mask"].sum().item(), 8)

    def test_identity_binds_pixels_prompt_and_real_timeline(self):
        ins, row, proc = fixture()
        before = build_evidence_request(ins, row, proc)["identity"]
        for key in ("input_ids", "pixel_values_videos"):
            with self.subTest(key=key):
                altered = {k: v.clone() for k, v in ins.items()}
                altered[key][0, 0] += 1
                self.assertNotEqual(before, build_evidence_request(altered, row, proc)["identity"])
        later = dict(row, sample_frame_ids=[1, 3, 5, 7], sample_timestamps=[1., 3., 5., 7.])
        self.assertNotEqual(before, build_evidence_request(ins, later, proc)["identity"])
        self.assertNotEqual(before, build_evidence_request(ins, dict(row, video_duration=9.), proc)["identity"])

    def test_grid_layout_changes_identity_even_with_same_patch_count(self):
        ins, row, proc = fixture()
        before = build_evidence_request(ins, row, proc)["identity"]
        ins["video_grid_thw"] = torch.tensor([[2, 2, 8]])
        self.assertNotEqual(before, build_evidence_request(ins, row, proc)["identity"])

    def test_prehashed_actual_inputs_match_direct_hash_without_mutation(self):
        ins, row, proc = fixture()
        snapshots = {key: val.clone() for key, val in ins.items()}
        direct = build_evidence_request(ins, row, proc)
        reused = build_evidence_request(ins, row, proc, identity=encoded_identity(ins))
        self.assertEqual(direct["identity"], reused["identity"])
        for key in ins:
            self.assertTrue(torch.equal(ins[key], snapshots[key]))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            build_evidence_request(ins, row, proc, identity={"input_ids": "label"})

    def test_missing_or_inexact_time_and_duplicate_frames_fail(self):
        ins, row, proc = fixture()
        cases = [dict(row, sample_timestamps=[0., 2., 4., 6.2]),
                 dict(row, sample_frame_ids=[0, 2, 2, 6]),
                 dict(row, sample_timestamps=[0., 2., 4., 9.]),
                 dict(row, source_frame_count=6), dict(row, n_unique_frames=3)]
        missing = dict(row)
        del missing["sample_timestamps"]
        cases.append(missing)
        for altered in cases:
            with self.subTest(row=altered), self.assertRaises(ValueError):
                build_evidence_request(ins, altered, proc)

    def test_temporal_padding_resampling_and_visual_grid_mismatch_fail(self):
        ins, row, proc = fixture()
        proc.video_processor.temporal_patch_size = 1
        with self.assertRaisesRegex(ValueError, "temporal grid"):
            build_evidence_request(ins, row, proc)
        proc.video_processor.temporal_patch_size = 2
        del proc.video_processor.merge_size
        with self.assertRaisesRegex(ValueError, "merge_size"):
            build_evidence_request(ins, row, proc)
        proc.video_processor.merge_size = 2
        ins["input_ids"][0, 1] = 1
        with self.assertRaisesRegex(ValueError, "visual token count"):
            build_evidence_request(ins, row, proc)

    def test_batched_and_inconsistent_encoded_inputs_fail(self):
        ins, row, proc = fixture()
        for key, value in (("input_ids", ins["input_ids"].repeat(2, 1)),
                           ("video_grid_thw", torch.tensor([[2., 4., 4.]])),
                           ("pixel_values_videos", ins["pixel_values_videos"][:-1])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                build_evidence_request({**ins, key: value}, row, proc)

    def test_interventions_do_not_modify_target_or_base_memory_identity(self):
        ins, row, proc = fixture()
        before = build_evidence_request(ins, row, proc)
        pixels = ins["pixel_values_videos"].clone()
        for intervention in ("normal", "bypass", "zero", "time_shuffle"):
            request = build_evidence_request(ins, row, proc, intervention=intervention, seed=7)
            self.assertEqual(request["identity"], before["identity"])
            self.assertEqual(request["intervention"], intervention)
            self.assertEqual(request["seed"], 7)
            self.assertTrue(torch.equal(ins["pixel_values_videos"], pixels))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            build_evidence_request(ins, row, proc, intervention="unsupported_probe")

    def test_configure_rebinds_request_and_none_head_is_noop(self):
        ins, row, proc = fixture()
        seen = []
        head = SimpleNamespace(config=cfg(), set_evidence_request=seen.append)
        first = build_evidence_request(ins, row, proc)
        second = build_evidence_request(ins, dict(row, video_duration=9.), proc)
        configure_evidence(head, first)
        configure_evidence(head, second)
        self.assertIs(seen[0], first)
        self.assertIs(seen[1], second)
        configure_evidence(SimpleNamespace(config=cfg("none")), None)
        with self.assertRaisesRegex(ValueError, "identity-bound"):
            configure_evidence(head, None)


class EvidenceEvalProtocolTests(unittest.TestCase):
    def test_settings_bind_the_actual_retained_feature_layer(self):
        settings = evidence_settings(cfg())
        self.assertEqual(settings["feature_layer"], 33)
        altered = cfg()
        altered["dflash_config"]["target_layer_ids"][-1] = 34
        with self.assertRaisesRegex(ValueError, "retained target layer 33"):
            evidence_settings(altered)

    def test_eval_rejects_ambiguous_or_unsupported_interventions(self):
        for mode, options in (
                ("none", {"intervention": "zero", "time_mode": "real"}),
                ("visual", {"time_mode": "legacy"}),
                ("visual", {"time_mode": "real", "zero_visual": True}),
                ("visual", {"time_mode": "real", "visual_memory": "legacy.pt"}),
                ("text", {"time_mode": "real", "intervention": "time_shuffle"})):
            with self.subTest(mode=mode, options=options), self.assertRaises(ValueError):
                validate_evidence_eval(evidence_settings(cfg(mode)), **options)
        validate_evidence_eval(evidence_settings(cfg()), time_mode="real", intervention="time_shuffle")
        validate_evidence_eval(evidence_settings(cfg("none")), time_mode="legacy", zero_visual=True)

    def test_merge_never_combines_different_evidence_modes_or_probes(self):
        parts = [{"config": {"shards": 2, "shard": i, "evidence_mode": "visual",
                               "evidence_intervention": "normal", "evidence_intervention_seed": 0},
                  "protocol": {"evidence": {"feature_layer": 33}},
                  "rows": [{"video": f"v{i}"}]} for i in range(2)]
        self.assertEqual(len(merge_results(parts)["rows"]), 2)
        for key, value in (("evidence_mode", "text"), ("evidence_intervention", "zero"),
                           ("evidence_intervention_seed", 1)):
            altered = copy.deepcopy(parts)
            altered[1]["config"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                merge_results(altered)
        parts[1]["protocol"]["evidence"]["feature_layer"] = 34
        with self.assertRaisesRegex(ValueError, "protocol.evidence"):
            merge_results(parts)


if __name__ == "__main__":
    unittest.main()
