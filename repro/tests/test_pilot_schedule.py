import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "video_caption/build_pilot_schedule.py"
spec = importlib.util.spec_from_file_location("pilot_schedule", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows():
    return [{"task": f"task{i}", "video_id": f"video{i}", "split": "train"}
            for i in range(8)]


def test_schedule_has_exact_exposure_and_slots_each_epoch():
    schedule = module.build_schedule(rows(), epochs=3, qa_per_update=4, seed=12)
    assert len(schedule) == 24
    assert schedule == module.build_schedule(rows(), epochs=3, qa_per_update=4, seed=12)
    for epoch in (1, 2, 3):
        assert {x["task"] for x in schedule if x["epoch"] == epoch} == {x["task"] for x in rows()}
    for update in range(6):
        assert [x["slot"] for x in schedule if x["update"] == update] == [0, 1, 2, 3]


def test_no_silent_padding_or_dev_leakage():
    with pytest.raises(ValueError, match="divisible"):
        module.build_schedule(rows()[:7])
    bad = rows()
    bad[0]["split"] = "dev"
    with pytest.raises(ValueError, match="split=train"):
        module.build_schedule(bad)
    bad = rows()
    bad[1]["video_id"] = bad[0]["video_id"]
    with pytest.raises(ValueError, match="one task"):
        module.build_schedule(bad)
