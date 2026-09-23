"""Per-block risk: what happens to the tower if this block is taken out.

    high    removing the block collapses the tower
    medium  the tower survives, but is measurably more fragile than before
    low     the tower survives, about as robust as it was

Fragility is the tower's tilt margin: how far the table can be tilted, in the
tower's weakest direction, before it gives way. Each block gets
`tilt_margin_deg` (the margin left after removing it, 0 if it collapsed) and
`margin_drop_deg` (how much its removal lowered the tower's own margin). The
drop decides medium vs low; see the "Risk levels" section of config.py for
why the absolute margin cannot.

This module also owns the tower's presence GRID -- which of the 18 x 3 slots
still hold a block. The recommended model pipeline reads that grid rather
than pixels (photo -> CNN finds blocks -> grid -> risk), because a grid built
from a real photo is exactly the same kind of object as one from here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from . import config as C
from . import tower as T
from .labeling import max_displacement
from .removal import delete_block

RISK_LEVELS = ("low", "medium", "high")
RISK_CODES = {"low": 1, "medium": 2, "high": 3}     # pixel values in risk maps


# ---------------------------------------------------------------------------
# Presence grid
# ---------------------------------------------------------------------------

def grid_string(tw: T.Tower) -> str:
    """54 characters, '1' = block present, '0' = gap.

    Character i is block index i, which is level * 3 + slot: the first three
    characters are the bottom layer, the last three the top.
    """
    gone = set(tw.gap_ids)
    return "".join("0" if i in gone else "1" for i in range(tw.n_blocks))


def layer_left(grid: str, index: int) -> tuple[int, ...]:
    """Slots still filled in this block's layer once the block is removed."""
    level, slot = divmod(index, C.BLOCKS_PER_LAYER)
    row = grid[level * C.BLOCKS_PER_LAYER:(level + 1) * C.BLOCKS_PER_LAYER]
    return tuple(s for s, ch in enumerate(row) if ch == "1" and s != slot)


def structural_rule(grid: str, index: int) -> bool:
    """The real-Jenga rule: never leave a layer empty or on one EDGE block.

    Predicts collapse-on-removal from the grid alone. On the first 4155
    simulated removals it was right 98.0% of the time, which makes it both a
    sanity check on the physics and the baseline any learned model must beat.
    """
    left = layer_left(grid, index)
    edges = {(0,), (C.BLOCKS_PER_LAYER - 1,)}
    return len(left) == 0 or left in edges


# ---------------------------------------------------------------------------
# Tilt test
# ---------------------------------------------------------------------------

def tilt_margin(tw: T.Tower, state, cap: float | None = None) -> float:
    """Weakest-direction table tilt (degrees) the tower survives.

    Tilting the table is done by tilting gravity, which is equivalent and
    touches nothing else. The ramp is slow (TILT_RAMP_SECONDS for the full
    range) so the answer is close to the quasi-static tipping angle; a faster
    ramp would read slightly high, because the tower is already moving before
    it has travelled far enough to count as a failure.

    Once one direction has failed at some angle, the remaining directions only
    need ramping that far, since they can only lower the minimum. That makes
    the whole test roughly three times cheaper.

    Leaves the tower exactly as `state` describes, with gravity restored.
    """
    cap = C.TILT_MAX_DEG if cap is None else cap
    m, d = tw.model, tw.data
    g0 = m.opt.gravity.copy()
    g = float(np.linalg.norm(g0))
    total = max(1, int(round(C.TILT_RAMP_SECONDS / C.TIME_STEP)))
    per_step = C.TILT_MAX_DEG / total                 # degrees per step
    worst = cap

    try:
        for dx, dy in C.TILT_DIRECTIONS:
            T.restore_state(tw, state)
            snap = T.snapshot(tw, tw.present_ids())
            n = int(math.ceil(worst / per_step - 1e-9))
            for s in range(n):
                th = math.radians(per_step * (s + 1))
                m.opt.gravity[:] = (g * math.sin(th) * dx,
                                    g * math.sin(th) * dy,
                                    -g * math.cos(th))
                mujoco.mj_step(m, d)
                if (s + 1) % C.TILT_CHECK_EVERY == 0 or s == n - 1:
                    mujoco.mj_kinematics(m, d)
                    disp, tilt = max_displacement(tw, snap)
                    if disp > C.COLLAPSE_DISPLACEMENT or tilt > C.COLLAPSE_TILT_DEG:
                        worst = min(worst, per_step * (s + 1))
                        break
    finally:
        m.opt.gravity[:] = g0
        T.restore_state(tw, state)
    return worst


def level_for(margin: float, drop: float, high_below: float | None = None,
              medium_drop: float | None = None) -> str:
    """Risk level from the margin left and the margin this removal cost."""
    high_below = C.RISK_HIGH_BELOW_DEG if high_below is None else high_below
    medium_drop = C.RISK_MEDIUM_DROP_DEG if medium_drop is None else medium_drop
    if margin < high_below:
        return "high"
    if drop >= medium_drop:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# One block
# ---------------------------------------------------------------------------

@dataclass
class BlockRisk:
    outcome: str              # stable | collapse, from the removal alone
    max_displacement: float   # metres, from the removal alone
    max_tilt_deg: float
    tilt_margin_deg: float    # margin left afterwards; 0 if it collapsed
    margin_drop_deg: float    # base margin minus margin left
    risk_level: str           # low | medium | high

    def as_dict(self) -> dict:
        return asdict(self)


def assess_block(tw: T.Tower, index: int, state, base_margin: float) -> BlockRisk:
    """Remove one block from `state`, then tilt-test whatever is left.

    `state` must be the tower as it stands, with `tw.gap_ids` describing its
    existing gaps; `base_margin` is `tilt_margin(tw, state)`, measured once per
    tower. The tower is put back exactly as `state` on return.
    """
    T.restore_state(tw, state)
    removed = delete_block(tw, index)
    if removed.outcome == "collapse":
        margin = 0.0
    else:
        # From here on the removed block is just another gap.
        tw.gap_ids.append(index)
        try:
            margin = tilt_margin(tw, T.save_state(tw))
        finally:
            tw.gap_ids.remove(index)
    T.restore_state(tw, state)
    drop = base_margin - margin
    return BlockRisk(removed.outcome, removed.max_displacement,
                     removed.max_tilt_deg, margin, drop, level_for(margin, drop))


def risk_map(index_map: np.ndarray, levels: dict[int, str]) -> np.ndarray:
    """Pixel -> 0 background or gap, 1 low, 2 medium, 3 high.

    `index_map` is a segmentation image of 1-based block indices (the
    `*_seg.png` files); `levels` maps 0-based block index to a risk level.
    """
    out = np.zeros(index_map.shape, dtype=np.uint8)
    for index, level in levels.items():
        out[index_map == index + 1] = RISK_CODES[level]
    return out
