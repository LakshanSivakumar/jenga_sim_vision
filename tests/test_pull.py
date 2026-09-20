"""Acceptance cases for the physical grip, including failures hidden by a weld."""
import random

import numpy as np
import pytest

from generate import build_one
from jenga_sim import config as C, removal, tower as T


@pytest.mark.parametrize("seed,index,end_seed", [(0, 51, 0), (1, 25, 1), (2, 51, 1)])
def test_unloaded_and_interior_pulls_stay_stable(seed, index, end_seed):
    tw = T.build_tower(seed)
    assert T.settle(tw)[0]
    out = removal.pull_block(tw, index, random.Random(end_seed), observe=1.0)
    assert out.outcome == "stable", out
    assert 0 < out.peak_force <= C.PULL_MAX_FORCE
    assert np.isfinite(tw.data.qpos).all()
    assert not tw.data.warning.number.any()


def test_lightly_loaded_gapped_tower_pulls():
    # Both orientations, two different gaps patterns. These were collapse
    # cases with the old weld despite surviving instantaneous deletion.
    for seed, indices in [(0, [40, 46]), (1, [43, 46])]:
        tw, ok, _ = build_one(seed, random.Random(seed))
        assert ok
        saved = T.save_state(tw)
        for index in indices:
            T.restore_state(tw, saved)
            out = removal.pull_block(tw, index, random.Random(index), observe=1.0)
            assert out.outcome == "stable", (seed, index, out)
            assert not tw.data.warning.number.any()


def test_force_cap_is_enforced_during_stuck_pull(monkeypatch):
    tw = T.build_tower(1)
    T.settle(tw)
    index = 25
    state = T.save_state(tw)
    bid = tw.body_id[index]
    _, axis, _ = removal.pull_path(tw, index, "end+")
    support = -tw.model.body_mass[bid] * tw.model.opt.gravity
    monkeypatch.setattr(C, "PULL_MAX_FORCE", 1e-6)
    forces = []

    def check_grip():
        applied = tw.data.xfrc_applied[bid]
        axial = applied[:3] - support
        forces.append(abs(float(np.dot(axial, axis))))
        assert np.linalg.norm(axial) <= C.PULL_MAX_FORCE + 1e-12
        # Horizontal guidance may push sideways, but never commands a lift.
        assert abs(axial[2]) < 1e-12
        assert np.linalg.norm(applied[3:]) <= C.PULL_MAX_TORQUE + 1e-12

    out = removal.pull_block(tw, index, random.Random(0), on_step=check_grip)
    assert out.outcome == "stuck", out
    assert len(forces) == int(C.PULL_MAX_SECONDS / C.TIME_STEP)
    assert out.peak_force == pytest.approx(max(forces))
    assert np.array_equal(tw.data.qpos, state[0])
    assert np.array_equal(tw.data.qvel, state[1])
    assert not tw.data.xfrc_applied[bid].any()
    assert tw.model.opt.timestep == C.TIME_STEP


def test_pull_last_support_still_collapses():
    tw = T.build_tower(1)
    T.settle(tw)
    # Keep the centre support of a high layer, then physically pull it out.
    for slot in (0, 2):
        index = tw.find(15, slot)
        T.park(tw, index, slot=slot)
        tw.gap_ids.append(index)
    T.settle(tw)
    out = removal.pull_block(tw, tw.find(15, 1), random.Random(0), observe=1.0)
    assert out.outcome == "collapse", out
    assert not tw.data.warning.number.any()


def test_interrupted_pull_releases_grip():
    tw = T.build_tower(1)
    T.settle(tw)
    index = 51

    def interrupt():
        raise RuntimeError("viewer closed")

    with pytest.raises(RuntimeError, match="viewer closed"):
        removal.pull_block(tw, index, on_step=interrupt)
    assert not tw.data.xfrc_applied[tw.body_id[index]].any()
    assert tw.model.opt.timestep == C.TIME_STEP


def test_saved_gaps_remain_parked_when_loaded(tmp_path):
    tw = T.build_tower(1)
    T.park(tw, 4)
    tw.gap_ids.append(4)
    path = tmp_path / "tower.npz"
    T.save_state_npz(tw, path)
    loaded = T.build_tower(1)
    T.load_state_npz(loaded, path)
    parked = loaded.pos(4)
    T.simulate(loaded, 0.1)
    assert np.allclose(loaded.pos(4), parked, atol=1e-10, rtol=0)
    assert loaded.model.geom_contype[loaded.geom_id[4]] == 0


def test_short_travel_does_not_teleport_a_block_out_of_contact(monkeypatch):
    tw = T.build_tower(1)
    T.settle(tw)
    monkeypatch.setattr(C, "PULL_DISTANCE", 0.001)
    monkeypatch.setattr(C, "PULL_MAX_SECONDS", 0.3)
    out = removal.pull_block(tw, 25, random.Random(0))
    assert out.outcome == "stuck", out


def test_numerical_reset_is_not_an_extraction(monkeypatch):
    import mujoco

    tw = T.build_tower(1)
    T.settle(tw)
    saved = T.save_state(tw)

    def failed_step(tower):
        tower.data.warning.number[int(mujoco.mjtWarning.mjWARN_BADQACC)] += 1

    monkeypatch.setattr(T, "step", failed_step)
    with pytest.raises(FloatingPointError, match="numerically unstable"):
        removal.pull_block(tw, 51)
    assert np.array_equal(tw.data.qpos, saved[0])
    assert not tw.data.xfrc_applied[tw.body_id[51]].any()
    assert tw.model.opt.timestep == C.TIME_STEP
