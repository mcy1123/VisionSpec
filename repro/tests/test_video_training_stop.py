"""CPU contracts for bounded unattended training; no model or optimizer loads."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_caption.train_video_9k import (checkpoint_and_status, parse_deadline,
                                         stop_reason)


def test_deadline_converts_local_offset_and_rejects_ambiguous_clock():
    local = parse_deadline("2026-11-12T08:00:00+08:00")
    utc = parse_deadline("2026-11-12T00:00:00Z")
    assert local == utc == datetime(2026, 11, 12, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="explicit timezone"):
        parse_deadline("2026-11-12T08:00:00")


def test_stop_file_and_deadline_boundaries(tmp_path):
    deadline = parse_deadline("2026-11-12T00:00:00Z")
    before = parse_deadline("2026-11-11T23:59:59Z")
    marker = tmp_path / "STOP"
    assert stop_reason(deadline, marker, now=before) is None
    assert stop_reason(deadline, marker, now=deadline) == "deadline"
    marker.touch()
    assert stop_reason(deadline, marker, now=before) == "stop_file"


def test_nonperiodic_stop_saves_last_completed_update(tmp_path):
    saved = []

    def save(step):
        saved.append(step)
        (tmp_path / f"step{step}").mkdir()

    result = checkpoint_and_status(tmp_path, 7, 60, "deadline", save)
    assert saved == [7]
    assert result["status"] == "STOPPED"
    assert result["completed_updates"] == 7 and result["planned_updates"] == 60
    assert result["resume_supported"] is False
    assert json.loads((tmp_path / "RUN_STATUS.json").read_text()) == result


def test_stop_does_not_overwrite_an_existing_periodic_checkpoint(tmp_path):
    (tmp_path / "step20").mkdir()
    sentinel = tmp_path / "step20" / "identity"
    sentinel.write_text("previous atomic checkpoint")

    def forbidden(_):
        raise AssertionError("must not overwrite checkpoint")

    checkpoint_and_status(tmp_path, 20, 60, "stop_file", forbidden)
    assert sentinel.read_text() == "previous atomic checkpoint"


def test_zero_update_stop_has_no_checkpoint(tmp_path):
    saved = []
    result = checkpoint_and_status(tmp_path, 0, 60, "deadline", saved.append)
    assert not saved and result["last_checkpoint"] is None


def test_fixed_budget_completion_and_save_failure_are_distinct(tmp_path):
    done = tmp_path / "done"
    done.mkdir()
    result = checkpoint_and_status(done, 60, 60, None,
                                   lambda step: (done / f"step{step}").mkdir())
    assert result["status"] == "COMPLETED" and result["stop_reason"] is None
    failed = tmp_path / "failed"
    failed.mkdir()

    def failure(step):
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        checkpoint_and_status(failed, 7, 60, "deadline", failure)
    assert not (failed / "RUN_STATUS.json").exists()
