"""Physics sanity tests. These are the ones that catch a broken simulator.

    pytest -q
"""

import math
import random

import mujoco
import numpy as np
import pytest

from jenga_sim import config as C
from jenga_sim import removal, tower as T
from jenga_sim.labeling import classify, max_displacement, quat_angle_between


@pytest.fixture
def settled():
    """A freshly built, settled tower. Rebuilt per test so they cannot interact."""
    tw = T.build_tower(seed=1)
    converged, _ = T.settle(tw)
    assert converged, "tower never settled"
    return tw


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_tower_geometry(settled):
    assert settled.n_blocks == C.NUM_LAYERS * C.BLOCKS_PER_LAYER == 54
    assert settled.model.neq == 0          # grip applies bounded forces, no welds


def test_layers_alternate_orientation(settled):
    """Each layer is turned 90 degrees from the one below it."""
    for level in range(0, C.NUM_LAYERS - 1):
        a = settled.long_axis(settled.find(level, 1))
        b = settled.long_axis(settled.find(level + 1, 1))
        assert abs(float(np.dot(a, b))) < 0.1, f"layers {level}/{level+1} not crossed"


# ---------------------------------------------------------------------------
# The headline acceptance test
# ---------------------------------------------------------------------------

def test_untouched_tower_stands_for_ten_seconds(settled):
    """No drift, no creep, no explosion."""
    before = T.snapshot(settled, settled.present_ids())
    T.simulate(settled, 10.0)
    drift, tilt = max_displacement(settled, before)

    assert np.isfinite(settled.data.qpos).all(), "simulation went unstable"
    assert drift < C.COLLAPSE_DISPLACEMENT, f"drifted {drift*100:.3f} cm"
    assert tilt < C.COLLAPSE_TILT_DEG, f"tilted {tilt:.2f} deg"


def test_tower_does_not_sink(settled):
    """Blocks must not slowly penetrate each other under load."""
    T.simulate(settled, 5.0)
    top = settled.find(C.NUM_LAYERS - 1, 1)
    z = settled.pos(top)[2]
    ideal = C.BLOCK_HEIGHT * (C.NUM_LAYERS - 0.5)
    assert abs(z - ideal) < C.BLOCK_HEIGHT * 0.5, (
        f"top layer at z={z*100:.2f} cm, expected about {ideal*100:.2f} cm")


def test_contact_forces_match_tower_weight(settled):
    """Sanity-check the contact solver against statics.

    Summed over every layer interface, the vertical contact force must equal
    the total weight each interface carries. If this is wrong, nothing built
    on top of it means anything.
    """
    m, d = settled.model, settled.data
    g2b = {settled.geom_id[i]: i for i in range(settled.n_blocks)}
    total = 0.0
    f6 = np.zeros(6)
    for c in range(d.ncon):
        con = d.contact[c]
        b1, b2 = g2b.get(con.geom1), g2b.get(con.geom2)
        if b1 is None or b2 is None:
            continue
        mujoco.mj_contactForce(m, d, c, f6)
        total += abs(f6[0])

    # Interface k carries the weight of every layer above it.
    per_layer = C.BLOCKS_PER_LAYER * C.BLOCK_WEIGHT
    expected = per_layer * sum(range(1, C.NUM_LAYERS))
    assert abs(total - expected) / expected < 0.15, (
        f"contact forces sum to {total:.1f} N, statics says {expected:.1f} N")


# ---------------------------------------------------------------------------
# Removal outcomes
# ---------------------------------------------------------------------------

def test_delete_top_block_is_stable(settled):
    """Removing a block from the top layer cannot topple anything."""
    out = removal.delete_block(settled, settled.find(C.NUM_LAYERS - 1, 0))
    assert out.outcome == "stable", (
        f"top block gave {out.outcome} (disp={out.max_displacement*100:.3f} cm)")


def test_delete_last_support_collapses(settled):
    """Strip the bottom layer to one block, then remove it: the tower must fall.

    Both halves are asserted. A tower balanced on the CENTRE block of its
    bottom layer genuinely stands -- the support is 2.5 cm wide and the
    jitter only shifts the centre of mass by about a millimetre, so this is a
    real equilibrium, not a physics failure. Removing that last block leaves
    54 blocks with nothing underneath.
    """
    for i, slot in enumerate((0, 2)):
        block = settled.find(0, slot)
        T.park(settled, block, slot=i + 1)
        settled.gap_ids.append(block)
    T.settle(settled, max_seconds=1.0)

    # Half one: it really is standing on that single block.
    before = T.snapshot(settled, settled.present_ids())
    T.simulate(settled, 0.5)
    drift, _ = max_displacement(settled, before)
    assert drift < C.COLLAPSE_DISPLACEMENT, (
        f"tower was already falling before the removal (drift {drift*100:.3f} cm)")

    # Half two: taking that block away brings it down.
    out = removal.delete_block(settled, settled.find(0, 1), observe=1.0)
    assert out.outcome == "collapse", (
        f"expected collapse, got {out.outcome} "
        f"(disp={out.max_displacement*100:.3f} cm, tilt={out.max_tilt_deg:.2f})")


def test_pull_extracts_a_free_block(settled):
    """A top-layer block has nothing pressing on it, so it must pull out."""
    out = removal.pull_block(settled, settled.find(C.NUM_LAYERS - 1, 0),
                             random.Random(0))
    assert out.outcome == "stable", (
        f"free top block gave {out.outcome} at {out.peak_force:.2f} N "
        f"(budget {C.PULL_MAX_FORCE:.2f} N)")
    assert out.peak_force > 0


def test_stuck_pull_leaves_the_tower_untouched(settled):
    """A `stuck` trial must report zero disturbance, by construction."""
    saved = C.PULL_MAX_FORCE
    C.PULL_MAX_FORCE = 1e-6          # nothing can be pulled this gently
    try:
        out = removal.pull_block(settled, settled.find(4, 1), random.Random(0))
    finally:
        C.PULL_MAX_FORCE = saved
    assert out.outcome == "stuck"
    assert out.max_displacement == 0.0
    assert out.max_tilt_deg == 0.0


# ---------------------------------------------------------------------------
# State handling
# ---------------------------------------------------------------------------

def test_state_restore_is_exact(settled):
    """Trials are only comparable if restore puts the tower back exactly."""
    state = T.save_state(settled)
    T.simulate(settled, 0.5)
    T.restore_state(settled, state)
    assert np.allclose(settled.data.qpos, state[0])
    assert np.allclose(settled.data.qvel, state[1])


def test_parked_block_leaves_the_tower(settled):
    before = T.snapshot(settled, [i for i in settled.present_ids() if i != 7])
    T.park(settled, 7)
    assert np.allclose(settled.pos(7), C.PARK_POSITION)
    assert settled.model.geom_contype[settled.geom_id[7]] == 0
    disp, _ = max_displacement(settled, before)
    assert disp < 1e-6, "parking a block should not move anything else"
    parked = settled.pos(7)
    T.simulate(settled, 0.5)
    assert np.allclose(settled.pos(7), parked, atol=1e-10, rtol=0)


# ---------------------------------------------------------------------------
# Labelling arithmetic
# ---------------------------------------------------------------------------

def test_quat_angle_between():
    a = math.radians(30)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])                 # MuJoCo order: w x y z
    q1 = np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])
    assert abs(quat_angle_between(q0, q1) - 30.0) < 1e-3
    assert quat_angle_between(q0, q0) < 1e-6


def test_labels_respect_thresholds():
    assert classify(0.0, 0.0, extracted=True).outcome == "stable"
    assert classify(C.COLLAPSE_DISPLACEMENT * 2, 0.0, extracted=True).outcome == "collapse"
    assert classify(0.0, C.COLLAPSE_TILT_DEG * 2, extracted=True).outcome == "collapse"
    assert classify(0.0, 0.0, extracted=False).outcome == "stuck"
