"""Risk labelling: the presence grid, the tilt test and the three risk levels.

    pytest -q tests/test_risk.py
"""

import numpy as np
import pytest

from jenga_sim import camera, config as C, risk, tower as T


@pytest.fixture
def settled():
    tw = T.build_tower(seed=1)
    converged, _ = T.settle(tw)
    assert converged
    return tw


# ---------------------------------------------------------------------------
# Presence grid and the structural rule -- pure functions, fast
# ---------------------------------------------------------------------------

def test_grid_of_intact_tower(settled):
    grid = risk.grid_string(settled)
    assert len(grid) == C.NUM_LAYERS * C.BLOCKS_PER_LAYER == 54
    assert set(grid) == {"1"}


def test_grid_marks_gaps(settled):
    T.park(settled, 7)
    settled.gap_ids.append(7)
    grid = risk.grid_string(settled)
    assert grid[7] == "0"
    assert grid.count("0") == 1


@pytest.mark.parametrize("row, slot, left, collapses", [
    ("111", 1, (0, 2), False),   # middle out, both edges hold the layer
    ("111", 0, (1, 2), False),   # edge out, two offset blocks remain
    ("011", 2, (1,), False),     # middle alone is balanced
    ("110", 1, (0,), True),      # one EDGE block left: tips over
    ("010", 1, (), True),        # layer emptied
])
def test_structural_rule(row, slot, left, collapses):
    grid = "111" * 5 + row + "111" * 12
    index = 5 * 3 + slot
    assert risk.layer_left(grid, index) == left
    assert risk.structural_rule(grid, index) is collapses


@pytest.mark.parametrize("margin, drop, level", [
    (0.0, 5.0, "high"),                                    # collapsed
    (C.RISK_HIGH_BELOW_DEG - 0.01, 0.0, "high"),           # falls at a touch
    (5.0, C.RISK_MEDIUM_DROP_DEG, "medium"),               # removal cost margin
    (5.0, C.RISK_MEDIUM_DROP_DEG - 0.01, "low"),
    (5.0, 0.0, "low"),
    (5.0, -1.0, "low"),                                    # removal helped
])
def test_risk_level_thresholds(margin, drop, level):
    assert risk.level_for(margin, drop) == level


def test_fragile_tower_does_not_make_every_block_medium():
    """The bug the drop definition fixes: a tower already at 4.8 deg, with a
    removal that changes nothing, is low risk -- not medium by association."""
    assert risk.level_for(margin=4.8, drop=0.0) == "low"


def test_risk_map_codes():
    index_map = np.array([[0, 1, 2], [3, 3, 0]], dtype=np.uint8)
    out = risk.risk_map(index_map, {0: "low", 1: "medium", 2: "high"})
    assert out.tolist() == [[0, 1, 2], [3, 3, 0]]


# ---------------------------------------------------------------------------
# Tilt test
# ---------------------------------------------------------------------------

def test_tilt_margin_is_bounded_and_leaves_no_trace(settled):
    """Gravity and the tower must be exactly as they were afterwards --
    anything else would silently corrupt every label measured after it."""
    state = T.save_state(settled)
    g0 = settled.model.opt.gravity.copy()
    margin = risk.tilt_margin(settled, state)

    assert 0.0 < margin <= C.TILT_MAX_DEG
    assert np.allclose(settled.model.opt.gravity, g0)
    assert np.allclose(settled.data.qpos, state[0])


def test_gappier_tower_tolerates_less_tilt(settled):
    """Hollowing a layer down to its middle block must make the tower easier
    to tip, or the tilt test is not measuring fragility."""
    full = risk.tilt_margin(settled, T.save_state(settled))
    for i, slot in enumerate((0, 2)):
        block = settled.find(6, slot)
        T.park(settled, block, slot=i + 1)
        settled.gap_ids.append(block)
    T.settle(settled, max_seconds=1.0)
    hollow = risk.tilt_margin(settled, T.save_state(settled))
    assert hollow < full, (full, hollow)


# ---------------------------------------------------------------------------
# Whole-block assessment
# ---------------------------------------------------------------------------

def test_top_block_is_not_high_risk(settled):
    state = T.save_state(settled)
    base = risk.tilt_margin(settled, state)
    r = risk.assess_block(settled, settled.find(C.NUM_LAYERS - 1, 0), state, base)
    assert r.outcome == "stable"
    assert r.risk_level != "high"
    assert r.tilt_margin_deg > C.RISK_HIGH_BELOW_DEG
    assert abs(r.margin_drop_deg - (base - r.tilt_margin_deg)) < 1e-9


def test_leaving_one_edge_block_is_high_risk(settled):
    """Layer 3 already has one gap; taking its middle leaves a lone edge."""
    gap = settled.find(3, 0)
    T.park(settled, gap, slot=1)
    settled.gap_ids.append(gap)
    T.settle(settled, max_seconds=1.0)
    state = T.save_state(settled)

    target = settled.find(3, 1)
    assert risk.structural_rule(risk.grid_string(settled), target)
    r = risk.assess_block(settled, target, state, risk.tilt_margin(settled, state))
    assert r.outcome == "collapse"
    assert r.tilt_margin_deg == 0.0
    assert r.risk_level == "high"


def test_assessment_leaves_the_tower_unchanged(settled):
    state = T.save_state(settled)
    gaps = list(settled.gap_ids)
    risk.assess_block(settled, settled.find(8, 1), state, base_margin=5.0)
    assert settled.gap_ids == gaps
    assert np.allclose(settled.data.qpos, state[0])
    assert np.allclose(settled.data.qvel, state[1])


# ---------------------------------------------------------------------------
# Rendering views
# ---------------------------------------------------------------------------

def test_views_are_reproducible_and_restore_the_scene(settled):
    ambient = settled.model.vis.headlight.ambient.copy()
    a = camera.render_views(settled, seed=5, n_views=3, size=96)
    b = camera.render_views(settled, seed=5, n_views=3, size=96)
    camera.release_renderers()

    assert [v["cam"] for v in a] == [v["cam"] for v in b]
    assert all(np.array_equal(x["rgb"], y["rgb"]) for x, y in zip(a, b))
    assert a[0]["cam"] == camera.default_camera()          # view 0 is the fixed shot
    assert np.allclose(settled.model.vis.headlight.ambient, ambient)
