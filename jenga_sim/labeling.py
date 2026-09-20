"""Turn "what happened after we removed a block" into a label.

Label definitions (also in the README):
  collapse -- any OTHER block moved more than COLLAPSE_DISPLACEMENT, or
              tilted more than COLLAPSE_TILT_DEG, relative to just before
              the removal.
  stuck    -- pull mode only: the block did not come out within the force
              and time budget.
  stable   -- the block came out and the rest of the tower stayed put.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from . import config as C


def quat_angle_between(q1, q2) -> float:
    """Smallest rotation angle (degrees) taking q1 to q2. MuJoCo quats are wxyz."""
    dot = abs(float(np.dot(np.asarray(q1), np.asarray(q2))))
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def max_displacement(tw, before: dict) -> tuple[float, float]:
    """Largest position change and largest tilt change across the given blocks.

    `before` is a snapshot dict from tower.snapshot(); it should already
    exclude the block being removed.
    """
    max_disp = 0.0
    max_tilt = 0.0
    for index, (pos0, quat0) in before.items():
        d = float(np.linalg.norm(tw.pos(index) - pos0))
        max_disp = max(max_disp, d)
        max_tilt = max(max_tilt, quat_angle_between(quat0, tw.quat(index)))
    return max_disp, max_tilt


@dataclass
class Outcome:
    outcome: str                 # stable | collapse | stuck
    max_displacement: float      # metres
    max_tilt_deg: float
    peak_force: float            # newtons, 0.0 for delete mode
    extract_time: float          # seconds, 0.0 for delete mode
    pull_direction: str = ""    # end+ | end- | side; empty for delete

    def as_dict(self) -> dict:
        return asdict(self)


def classify(max_disp: float, max_tilt: float, extracted: bool,
             peak_force: float = 0.0, extract_time: float = 0.0,
             pull_direction: str = "") -> Outcome:
    """Apply the thresholds. `extracted` is False only when a pull failed."""
    if not extracted:
        label = "stuck"
    elif max_disp > C.COLLAPSE_DISPLACEMENT or max_tilt > C.COLLAPSE_TILT_DEG:
        label = "collapse"
    else:
        label = "stable"
    return Outcome(label, max_disp, max_tilt, peak_force, extract_time, pull_direction)
