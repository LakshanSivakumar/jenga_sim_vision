"""Horizontal trajectories and outward side pulls, including alternating rows."""
import math
import random

import mujoco
import numpy as np
import pytest

from jenga_sim import config as C, removal, tower as T


@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("slot", [0, 2])
def test_side_direction_points_out_of_each_row(level, slot):
    tw = T.build_tower(1)
    index, middle = tw.find(level, slot), tw.find(level, 1)
    action, axis, distance = removal.pull_path(tw, index, "side")
    assert action == "side"
    assert axis[2] == 0
    assert np.linalg.norm(axis) == pytest.approx(1)
    assert np.dot(axis, tw.pos(index) - tw.pos(middle)) > 0.02
    _, end_axis, _ = removal.pull_path(tw, index, "end+")
    assert abs(np.dot(axis, end_axis)) < 1e-12
    assert distance == C.PULL_SIDE_DISTANCE < C.PULL_DISTANCE


def test_centre_has_only_two_end_directions():
    tw = T.build_tower(1)
    assert removal.pull_directions(tw, 25) == ("end+", "end-")
    before = T.save_state(tw)
    with pytest.raises(ValueError, match="not an available pull"):
        removal.pull_block(tw, 25, direction="side")
    assert np.array_equal(tw.data.qpos, before[0])
    assert np.array_equal(tw.data.qvel, before[1])
    assert not tw.data.xfrc_applied.any()


@pytest.mark.parametrize("level,action", [(16, "end+"), (17, "end-"),
                                          (16, "side"), (17, "side")])
def test_tilted_free_block_follows_horizontal_line(level, action):
    tw = T.build_tower(1)
    index = tw.find(level, 0)
    for other in tw.present_ids():
        if other != index:
            T.park(tw, other)
            tw.gap_ids.append(other)
    a, va = tw.qadr[index], tw.vadr[index]
    # A visible 15-degree pitch: the previous 3D long-axis target would lift
    # or lower this block by centimetres during an end pull.
    tilted = np.empty(4)
    pitch = np.array([math.cos(math.pi / 24), 0, math.sin(math.pi / 24), 0])
    mujoco.mju_mulQuat(tilted, tw.data.qpos[a + 3:a + 7].copy(), pitch)
    tw.data.qpos[a:a + 3] = (0, 0, 0.2)
    tw.data.qpos[a + 3:a + 7] = tilted
    mujoco.mj_forward(tw.model, tw.data)
    resolved, axis, _ = removal.pull_path(tw, index, action)
    cross_axis = np.array([-axis[1], axis[0], 0])
    tw.data.qvel[va:va + 3] = 0.002 * cross_axis
    before = tw.pos(index)
    samples = []

    def record():
        if tw.model.geom_contype[tw.geom_id[index]]:
            samples.append(tw.pos(index) - before)

    out = removal.pull_block(tw, index, direction=action, on_step=record)
    samples = np.array(samples)
    assert out.outcome == "stable", out
    assert out.pull_direction == resolved
    assert np.max(np.abs(samples[:, 2])) < 1e-8
    assert np.max(np.abs(samples @ cross_axis)) < 5e-5
    assert not tw.data.warning.number.any()


@pytest.mark.parametrize("index", [48, 50, 51, 53])
def test_outer_block_can_be_extracted_sideways(index):
    tw = T.build_tower(1)
    assert T.settle(tw)[0]
    out = removal.pull_block(tw, index, direction="side")
    assert out.outcome == "stable", (index, out)
    assert out.pull_direction == "side"
    assert out.extract_time < 1.0
    assert not tw.data.warning.number.any()
