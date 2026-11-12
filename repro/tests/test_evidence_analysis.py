import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.summarize_evidence_eval import summarize


def payload(rounds, wall):
    return {"config": {"target": "/target", "dtype": "float32", "time_mode": "real",
                       "eos_stop": "immediate", "max_new": 4, "budget": 63, "branch": 8},
            "rows": [{"video": str(i), "lossless": True, "spec_ids": [1, 2, 3, 4],
                      "ar_ids": [1, 2, 3, 4], "input_identity": {"input_ids": str(i)},
                      "gen_tokens": 4, "rounds": rounds, "tau": 4 / rounds,
                      "spec_wall_s": wall * (100 if i == 0 else 1),
                      "spec_end_to_end_s": wall * (100 if i == 0 else 1) + 1,
                      "warmup": i == 0, "hit_eos": False, "truncated": True}
                     for i in range(4)]}


def inputs_for_tests():
    result = {name: payload(rounds, rounds) for name, rounds in (("A", 4), ("B", 2), ("C", 1))}
    for name, mode in (("A", "none"), ("B", "text"), ("C", "visual")):
        result[name]["protocol"] = {"evidence": {"mode": mode, "intervention": "normal",
                                                   "dim": 8, "max_tokens": 8, "spatial_bins": 2,
                                                   "feature_layer": 33, "feature_protocol": "test"},
                                    "artifacts": {"head_config_sha256": name,
                                                  "head_weights_sha256": {"model.safetensors": name}}}
    return result


def test_video_pairing_and_warmup_timing():
    inputs = inputs_for_tests()
    inputs["B"]["rows"].reverse()
    result = summarize(inputs, resamples=100)
    pair = result["comparisons"]["C_minus_A"]
    assert pair["effective_tokens_per_round"]["paired_mean_difference"] == 3
    assert pair["spec_wall_s"]["speedup_old_over_new"] == 4
    assert pair["spec_end_to_end_s"]["speedup_old_over_new"] == 2.5
    assert result["arms"]["C"]["n_timing_videos"] == 3
    assert result["arms"]["C"]["n_videos"] == 4


@pytest.mark.parametrize("field,value", [("input_identity", {"input_ids": "different"}),
                                         ("ar_ids", [5, 6, 7, 8]),
                                         ("warmup", False), ("lossless", False)])
def test_reject_nonpaired_or_failed_runs(field, value):
    inputs = inputs_for_tests()
    inputs["C"]["rows"][0][field] = value
    with pytest.raises(ValueError):
        summarize(inputs, resamples=10)


def test_duplicate_video_and_different_protocol_rejected():
    inputs = inputs_for_tests()
    bad = copy.deepcopy(inputs)
    bad["C"]["rows"].append(bad["C"]["rows"][0])
    with pytest.raises(ValueError, match="duplicate"):
        summarize(bad, resamples=10)
    inputs["C"]["config"]["budget"] = 31
    with pytest.raises(ValueError, match="budget"):
        summarize(inputs, resamples=10)


def test_intervention_checkpoint_and_mode_are_verified():
    inputs = inputs_for_tests()
    inputs["zero"] = copy.deepcopy(inputs["C"])
    inputs["zero"]["protocol"]["evidence"]["intervention"] = "zero"
    assert "C_minus_zero" in summarize(inputs, resamples=10)["comparisons"]
    inputs["zero"]["protocol"]["artifacts"]["head_config_sha256"] = "wrong"
    with pytest.raises(ValueError, match="identical C"):
        summarize(inputs, resamples=10)
    del inputs["zero"]
    inputs["B"]["protocol"]["evidence"]["mode"] = "visual"
    with pytest.raises(ValueError, match="mode/intervention"):
        summarize(inputs, resamples=10)


def test_checkpoint_training_budget_is_not_inferred_from_path():
    inputs = inputs_for_tests()
    assert not summarize(inputs, resamples=10)["training_identity"]["verified"]
    identity = {"schema": "video_opd_checkpoint_identity_v1", "optimizer_step": 60,
                "data_cursor": 60, "qa_exposures": 240, "max_new_tokens": 1024, "max_seq_len": 32768,
                "common_initial_head_weights_sha256": {"model": "a"},
                "pool_sha256": "pool", "schedule_sha256": "schedule", "target_config_sha256": "target",
                "seed": 42, "lr": 1e-5, "qa_per_update": 4, "time_mode": "real",
                "loss_normalization": "global_pooled", "e2e_tv_weight": 10,
                "optimizer": {"name": "AdamW"},
                "precision": {"target_dtype": "float32", "head_dtype": "bfloat16", "master_dtype": "float32",
                              "evidence_parameter_dtype": "float32"}}
    for name in ("A", "B", "C"):
        inputs[name]["protocol"]["artifacts"]["checkpoint_identity"] = copy.deepcopy(identity)
    assert summarize(inputs, resamples=10)["training_identity"]["optimizer_step"] == 60
    bad = copy.deepcopy(inputs)
    bad["C"]["protocol"]["artifacts"]["checkpoint_identity"]["precision"]["evidence_parameter_dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="must train in float32"):
        summarize(bad, resamples=10)
    inputs["C"]["protocol"]["artifacts"]["checkpoint_identity"]["optimizer_step"] = 20
    with pytest.raises(ValueError, match="optimizer_step"):
        summarize(inputs, resamples=10)


def test_wrong_video_is_mechanism_only_and_capacity_must_match():
    inputs = inputs_for_tests()
    inputs["wrong_video"] = copy.deepcopy(inputs["C"])
    inputs["wrong_video"]["protocol"]["evidence"]["intervention"] = "wrong_video"
    result = summarize(inputs, resamples=10)["comparisons"]["C_minus_wrong_video"]
    assert "timing_omitted" in result and "spec_wall_s" not in result
    inputs["B"]["protocol"]["evidence"]["max_tokens"] = 16
    with pytest.raises(ValueError, match="capacity/source"):
        summarize(inputs, resamples=10)
