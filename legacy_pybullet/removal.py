"""The two ways of taking a block out, and the numbers we record while doing it.

delete -- the block vanishes instantly. Cheap, and the cleanest signal of
          "was this block load-bearing?".
pull   -- the block is dragged out along its long axis by a force-limited
          constraint, like a finger would. Slower, but it also tells us how
          HARD the block was to pull, which is what "tempting" will mean later.
"""

from __future__ import annotations

import math
import random
import time

import pybullet as p

from . import config as C
from . import tower as T
from .labeling import Outcome, classify, max_displacement


def long_axis_world(cli: int, body: int) -> tuple[float, float, float]:
    """Unit vector along the block's long axis (local +x) in world coords."""
    _, orn = p.getBasePositionAndOrientation(body, physicsClientId=cli)
    rot = p.getMatrixFromQuaternion(orn)
    return (rot[0], rot[3], rot[6])


# ---------------------------------------------------------------------------
# Delete mode
# ---------------------------------------------------------------------------

def delete_block(cli: int, tower: T.Tower, body: int,
                 observe: float | None = None,
                 step_delay: float = 0.0) -> Outcome:
    observe = C.OBSERVE_SECONDS if observe is None else observe
    others = [b for b in tower.present_ids() if b != body]
    before = T.snapshot(cli, others)

    T.park(cli, body, index=len(tower.gap_ids) + 1)
    T.simulate(cli, observe, step_delay)

    disp, tilt = max_displacement(cli, before)
    return classify(disp, tilt, extracted=True)


# ---------------------------------------------------------------------------
# Pull mode
# ---------------------------------------------------------------------------

def pull_block(cli: int, tower: T.Tower, body: int,
               rng: random.Random | None = None,
               observe: float | None = None,
               step_delay: float = 0.0) -> Outcome:
    """Drag the block out along its long axis at constant speed.

    A JOINT_FIXED constraint to the world acts as the "finger". We move its
    anchor at PULL_SPEED and cap it at PULL_MAX_FORCE, so a genuinely jammed
    block simply refuses to move and gets labelled `stuck` -- exactly what a
    force-limited human finger would experience.
    """
    rng = rng or random.Random()
    observe = C.OBSERVE_SECONDS if observe is None else observe

    others = [b for b in tower.present_ids() if b != body]
    before = T.snapshot(cli, others)

    start_pos, start_orn = p.getBasePositionAndOrientation(body, physicsClientId=cli)
    axis = long_axis_world(cli, body)
    sign = rng.choice([-1.0, 1.0]) if C.PULL_RANDOMISE_END else 1.0
    direction = tuple(a * sign for a in axis)

    # PyBullet anchors a body to the world with the BODY as parent and -1 as
    # child (the reverse fails outright). parentFramePosition is then the grab
    # point in block-local coords, and childFramePosition the world anchor.
    cid = p.createConstraint(
        parentBodyUniqueId=body, parentLinkIndex=-1,
        childBodyUniqueId=-1, childLinkIndex=-1,
        jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=list(start_pos),
        parentFrameOrientation=[0, 0, 0, 1],
        childFrameOrientation=list(start_orn),
        physicsClientId=cli,
    )
    p.changeConstraint(cid, maxForce=C.PULL_MAX_FORCE, physicsClientId=cli)

    max_steps = int(C.PULL_MAX_SECONDS / C.TIME_STEP)
    need = C.PULL_DISTANCE * C.PULL_STUCK_FRACTION
    peak_force = 0.0
    extract_time = 0.0
    extracted = False

    for step in range(max_steps):
        t = (step + 1) * C.TIME_STEP
        commanded = min(t * C.PULL_SPEED, C.PULL_DISTANCE)
        pivot = [start_pos[i] + direction[i] * commanded for i in range(3)]
        p.changeConstraint(
            cid, jointChildPivot=pivot,
            jointChildFrameOrientation=list(start_orn),
            maxForce=C.PULL_MAX_FORCE,
            physicsClientId=cli,
        )
        p.stepSimulation(physicsClientId=cli)
        if step_delay:
            time.sleep(step_delay)

        state = p.getConstraintState(cid, physicsClientId=cli)
        if len(state) >= 3:
            f = math.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2)
            peak_force = max(peak_force, f)

        pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=cli)
        travelled = sum((pos[i] - start_pos[i]) * direction[i] for i in range(3))
        if travelled >= need:
            extracted = True
            extract_time = t
            break

    p.removeConstraint(cid, physicsClientId=cli)

    if extracted:
        # Block is clear of the tower; get it out of the way and watch.
        T.park(cli, body, index=len(tower.gap_ids) + 1)
        T.simulate(cli, observe, step_delay)
    else:
        # Jammed. Let go and let the tower relax before measuring, so we do
        # not mistake the puller's own disturbance for a collapse.
        extract_time = C.PULL_MAX_SECONDS
        T.simulate(cli, observe, step_delay)

    disp, tilt = max_displacement(cli, before)
    return classify(disp, tilt, extracted, peak_force, extract_time)


def remove(cli: int, tower: T.Tower, body: int, mode: str,
           rng: random.Random | None = None,
           step_delay: float = 0.0) -> Outcome:
    if mode == "delete":
        return delete_block(cli, tower, body, step_delay=step_delay)
    if mode == "pull":
        return pull_block(cli, tower, body, rng, step_delay=step_delay)
    raise ValueError(f"unknown removal mode: {mode!r}")
