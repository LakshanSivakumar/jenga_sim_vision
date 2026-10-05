"""generate.py survives being interrupted.

A full run takes hours, so it must never lose finished towers. These tests
swap the physics for an instant stand-in and exercise only the bookkeeping:
saving as it goes, resuming, and refusing to mix different runs.
"""

import sys

import pandas as pd
import pytest

import generate


def fake_tower(job):
    """What run_tower returns, without any physics. Tower 3 'would not stand'."""
    i = job["tower_id"]
    if i == 3:
        return {"tower_id": i, "seed": job["seed"], "ok": False, "tower": None,
                "labels": [], "views": [], "visibility": [], "seconds": 0.0}
    labels = [dict(tower_id=i, block_id=b, level=(b - 1) // 3, position_in_level=(b - 1) % 3,
                   legal=True, outcome="stable", max_displacement=0.001, max_tilt_deg=0.1,
                   tilt_margin_deg=5.0, margin_drop_deg=0.0, risk_level="low",
                   seed=job["seed"]) for b in (1, 2)]
    return {"tower_id": i, "seed": job["seed"], "ok": True,
            "tower": dict(tower_id=i, seed=job["seed"], num_blocks=54, num_gaps=0,
                          grid="1" * 54, base_tilt_deg=5.0, settle_seconds=0.3),
            "labels": labels,
            "views": [dict(tower_id=i, view=0, camera_params="az=0", look_params="default")],
            "visibility": [dict(tower_id=i, view=0, block_id=1, pixels=10)],
            "seconds": 0.0}


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["generate.py", "--workers", "1", *argv])
    generate.main()


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(generate, "run_tower", fake_tower)


def test_every_tower_is_saved_and_merged(tmp_path, monkeypatch, fake):
    run(monkeypatch, "--towers", "5", "--views", "1", "--out", str(tmp_path))
    assert len(list((tmp_path / "parts").glob("tower_*.json"))) == 5   # incl. the rejected one
    towers = pd.read_csv(tmp_path / "towers.csv")
    assert towers.tower_id.tolist() == [0, 1, 2, 4]                    # 3 was rejected
    assert len(pd.read_csv(tmp_path / "labels.csv")) == 8


def test_interrupted_run_keeps_its_towers_and_resumes(tmp_path, monkeypatch):
    def crashes_on_tower_2(job):
        if job["tower_id"] == 2:
            raise KeyboardInterrupt("lid closed")
        return fake_tower(job)

    monkeypatch.setattr(generate, "run_tower", crashes_on_tower_2)
    with pytest.raises(KeyboardInterrupt):
        run(monkeypatch, "--towers", "5", "--views", "1", "--out", str(tmp_path))
    kept = sorted(p.stem for p in (tmp_path / "parts").glob("tower_*.json"))
    assert kept == ["tower_0000", "tower_0001"]                       # nothing lost

    ran = []
    monkeypatch.setattr(generate, "run_tower", lambda job: ran.append(job["tower_id"]) or fake_tower(job))
    run(monkeypatch, "--towers", "5", "--views", "1", "--out", str(tmp_path), "--resume")
    assert ran == [2, 3, 4]                                            # only the missing ones
    assert pd.read_csv(tmp_path / "towers.csv").tower_id.tolist() == [0, 1, 2, 4]


def test_refuses_to_mix_runs_without_resume(tmp_path, monkeypatch, fake):
    run(monkeypatch, "--towers", "2", "--views", "1", "--out", str(tmp_path))
    with pytest.raises(SystemExit):
        run(monkeypatch, "--towers", "2", "--views", "1", "--out", str(tmp_path))


def test_resume_refuses_different_settings(tmp_path, monkeypatch, fake):
    run(monkeypatch, "--towers", "2", "--views", "1", "--out", str(tmp_path))
    with pytest.raises(SystemExit):
        run(monkeypatch, "--towers", "4", "--views", "3", "--out", str(tmp_path), "--resume")


def test_resume_can_extend_a_finished_run(tmp_path, monkeypatch, fake):
    run(monkeypatch, "--towers", "2", "--views", "1", "--out", str(tmp_path))
    run(monkeypatch, "--towers", "4", "--views", "1", "--out", str(tmp_path), "--resume")
    assert pd.read_csv(tmp_path / "towers.csv").tower_id.tolist() == [0, 1, 2]   # 3 rejected


def test_shards_split_the_work_and_merge(tmp_path, monkeypatch):
    """Two machines, each building half; copying one folder into the other and
    resuming gives the same dataset as building everything in one place."""
    import shutil
    ran = {0: [], 1: []}
    for k in (0, 1):
        monkeypatch.setattr(generate, "run_tower",
                            lambda job, k=k: ran[k].append(job["tower_id"]) or fake_tower(job))
        run(monkeypatch, "--towers", "6", "--views", "1", "--out", str(tmp_path / f"m{k}"),
            "--shard", f"{k}/2")
    assert ran == {0: [0, 2, 4], 1: [1, 3, 5]}

    shutil.copytree(tmp_path / "m1", tmp_path / "m0", dirs_exist_ok=True)
    rerun = []
    monkeypatch.setattr(generate, "run_tower", lambda job: rerun.append(job) or fake_tower(job))
    run(monkeypatch, "--towers", "6", "--views", "1", "--out", str(tmp_path / "m0"), "--resume")
    assert rerun == []                                                 # nothing rebuilt
    assert pd.read_csv(tmp_path / "m0" / "towers.csv").tower_id.tolist() == [0, 1, 2, 4, 5]


def test_bad_shard_is_refused(tmp_path, monkeypatch, fake):
    with pytest.raises(SystemExit):
        run(monkeypatch, "--towers", "2", "--out", str(tmp_path), "--shard", "2/2")
