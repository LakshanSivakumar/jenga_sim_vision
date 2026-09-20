"""Ways of taking a block out, and the numbers we record while doing it.

Only `delete` is used for the dataset. `pull` and `push` are kept because
they are implemented and instructive, but neither produces a usable label
mix -- see the "Why the dataset uses delete mode" section of the README.

delete -- the block vanishes instantly. Cheap, and the cleanest signal of
          "was this block load-bearing?".
pull   -- the block is dragged out horizontally by a force-limited
          "finger", like a hand would. Slower, but it also tells us how HARD
          the block was to pull, which is what "tempting" will mean later.
"""

from __future__ import annotations

import math
import random
import time

import mujoco
import numpy as np

from . import config as C
from . import tower as T
from .labeling import Outcome, classify, max_displacement

# ---------------------------------------------------------------------------
# Delete mode
# ---------------------------------------------------------------------------

def delete_block(tw: T.Tower, index: int, observe: float | None = None,
                 step_delay: float = 0.0, on_step=None) -> Outcome:
    observe = C.OBSERVE_SECONDS if observe is None else observe
    others = [i for i in tw.present_ids() if i != index]
    before = T.snapshot(tw, others)

    T.park(tw, index, slot=len(tw.gap_ids) + 1)
    T.simulate(tw, observe, step_delay, on_step)

    disp, tilt = max_displacement(tw, before)
    return classify(disp, tilt, extracted=True)


# ---------------------------------------------------------------------------
# Pull mode
# ---------------------------------------------------------------------------

def pull_directions(tw: T.Tower, index: int) -> tuple[str, ...]:
    """End pulls for every block, plus an outward side pull for outer slots."""
    if tw.slot_of[index] in (0, C.BLOCKS_PER_LAYER - 1):
        return ("end+", "end-", "side")
    return ("end+", "end-")


def pull_path(tw: T.Tower, index: int, direction: str = "end",
              rng: random.Random | None = None) -> tuple[str, np.ndarray, float]:
    """Resolve the action to a unit vector parallel to the table and travel.

    Project the long axis into XY: a settled block's pitch must never become
    a commanded lift. The perpendicular XY axis points outwards for side
    pulls, even on alternate layers (which are rotated by 90 degrees).
    """
    if index not in tw.present_ids():
        raise ValueError(f"block {index} is not present")
    if direction == "end":
        rng = rng or random.Random()
        direction = rng.choice(("end-", "end+")) if C.PULL_RANDOMISE_END else "end+"
    if direction not in pull_directions(tw, index):
        raise ValueError(f"{direction!r} is not an available pull for {tw.describe(index)}")
    axis = tw.long_axis(index)
    axis[2] = 0.0
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        raise ValueError("cannot choose a horizontal pull for a vertical block")
    axis /= length
    if direction == "side":
        axis = np.array([-axis[1], axis[0], 0.0])
        # Slots increase in world Y on even layers and world X on odd layers.
        # Local +Y therefore runs in the *opposite* slot order on odd layers.
        row_axis = 0 if tw.level_of[index] % 2 else 1
        outward = tw.slot_of[index] - (C.BLOCKS_PER_LAYER - 1) / 2
        if axis[row_axis] * outward < 0:
            axis *= -1
        return direction, axis, C.PULL_SIDE_DISTANCE
    if direction == "end-":
        axis *= -1
    return direction, axis, C.PULL_DISTANCE


def pull_block(tw: T.Tower, index: int, rng: random.Random | None = None,
               observe: float | None = None, step_delay: float = 0.0,
               on_step=None, *, direction: str = "end") -> Outcome:
    """Pull with a bounded axial force and a compliant, weight-supporting grip.

    The hand follows a straight horizontal path and supports the selected
    block's own weight, but does not prescribe its height. An attitude controller keeps
    an overhanging block from drooping. Its gains use each principal moment
    of inertia, and angular errors/velocities share the local body frame.

    Unsuccessful attempts are rolled back, preserving the dataset/playground
    convention that a `stuck` trial leaves the initial tower available. The
    force limit itself is enforced during every simulation step.
    """
    rng = rng or random.Random()
    observe = C.OBSERVE_SECONDS if observe is None else observe
    d, m = tw.data, tw.model
    action, direction, distance = pull_path(tw, index, direction, rng)
    original_dt = float(m.opt.timestep)
    substeps = max(1, math.ceil(original_dt / C.PULL_TIME_STEP))
    dt = original_dt / substeps
    others = [i for i in tw.present_ids() if i != index]
    before = T.snapshot(tw, others)
    pre_pull = T.save_state(tw)
    start_pos, start_quat = tw.pos(index), tw.quat(index)
    bid, va = tw.body_id[index], tw.vadr[index]
    gid = tw.geom_id[index]
    other_geoms = {tw.geom_id[i] for i in others}
    mass = m.body_mass[bid]

    # Explicit feedback must respect the actual model timestep. Roll inertia
    # is much smaller than pitch/yaw inertia for a long thin Jenga block;
    # using one scalar gain for all axes can make roll violently unstable.
    omega = min(C.PULL_OMEGA, 0.3 / dt)
    ang_omega = min(C.PULL_ANG_OMEGA, 0.15 / dt)
    stiffness, damping = mass * omega**2, 2 * mass * omega
    inertial_rotation = np.empty(9)
    mujoco.mju_quat2Mat(inertial_rotation, m.body_iquat[bid])
    ri = inertial_rotation.reshape(3, 3)
    inertia = (ri * m.body_inertia[bid]) @ ri.T
    angular_error = np.empty(3)
    original_wrench = d.xfrc_applied[bid].copy()
    support = -mass * m.opt.gravity
    warning_counts = d.warning.number.copy()
    bad = [int(mujoco.mjtWarning.mjWARN_BADQPOS),
           int(mujoco.mjtWarning.mjWARN_BADQVEL),
           int(mujoco.mjtWarning.mjWARN_BADQACC)]

    need = distance * C.PULL_STUCK_FRACTION
    peak_force = 0.0
    extract_time = C.PULL_MAX_SECONDS
    extracted = False
    try:
        # Resolve moving contacts more finely, without changing the normal
        # tower/delete timestep or the viewer callback/realtime pacing rate.
        m.opt.timestep = dt
        for s in range(int(C.PULL_MAX_SECONDS / dt)):
            t = (s + 1) * dt
            offset = tw.pos(index) - start_pos
            # Keep moving until clear: friction can make a neighbour follow,
            # so stopping the hand at the nominal clearance distance can leave
            # a movable block falsely "stuck" in its last contact.
            target = t * C.PULL_SPEED
            # Guide the block along the selected line in XY, with no height
            # target. Clamp the combined horizontal force, including guidance.
            error = direction * target - offset
            error[2] = 0.0
            velocity = d.qvel[va:va + 3].copy()
            velocity[2] = 0.0
            horizontal_force = stiffness * error - damping * velocity
            horizontal_force *= min(1.0, C.PULL_MAX_FORCE /
                                    max(np.linalg.norm(horizontal_force), 1e-15))
            force = float(np.dot(horizontal_force, direction))

            # mju_subQuat and free-joint angular qvel are both body-local.
            # xfrc_applied, in contrast, takes force AND torque in world space.
            mujoco.mju_subQuat(angular_error, start_quat, tw.quat(index))
            torque = inertia @ (ang_omega**2 * angular_error
                                - 2 * ang_omega * d.qvel[va + 3:va + 6])
            torque *= min(1.0, C.PULL_MAX_TORQUE / max(np.linalg.norm(torque), 1e-15))
            rotation = d.xmat[bid].reshape(3, 3)
            d.xfrc_applied[bid, :3] = original_wrench[:3] + horizontal_force + support
            d.xfrc_applied[bid, 3:] = original_wrench[3:] + rotation @ torque
            peak_force = max(peak_force, abs(force))
            T.step(tw)

            # MuJoCo can automatically reset after numerical divergence.
            # Never mistake that reset/teleport for a successful extraction.
            if (np.any(d.warning.number[bad] > warning_counts[bad])
                    or not np.isfinite(d.qpos).all()
                    or not np.isfinite(d.qvel).all()):
                T.restore_state(tw, pre_pull)
                raise FloatingPointError("pull simulation became numerically unstable")
            reached = float(np.dot(tw.pos(index) - start_pos, direction)) >= need
            if reached:
                # Distance alone is insufficient when a neighbour follows
                # the moving block. Do not teleport it out of a live contact.
                reached = not any(
                    con.efc_address >= 0 and (
                        (con.geom1 == gid and con.geom2 in other_geoms)
                        or (con.geom2 == gid and con.geom1 in other_geoms))
                    for con in d.contact[:d.ncon]
                )
            if (s + 1) % substeps == 0 or reached:
                if on_step is not None:
                    on_step()
                if step_delay:
                    time.sleep(step_delay)
            if reached:
                extracted = True
                extract_time = t
                break
    finally:
        # Also release the grip if a viewer callback raises or is interrupted.
        d.xfrc_applied[bid] = original_wrench
        m.opt.timestep = original_dt

    if not extracted:
        T.restore_state(tw, pre_pull)
        return classify(0.0, 0.0, False, peak_force, extract_time,
                        pull_direction=action)

    T.park(tw, index, slot=len(tw.gap_ids) + 1)
    T.simulate(tw, observe, step_delay, on_step)
    disp, tilt = max_displacement(tw, before)
    return classify(disp, tilt, True, peak_force, extract_time,
                    pull_direction=action)


# ---------------------------------------------------------------------------
# Push mode
# ---------------------------------------------------------------------------

def _finger_force(tw: T.Tower, index: int) -> float:
    """Normal force the fingertip is putting into the block, in newtons."""
    d, m = tw.data, tw.model
    tips, gid = set(tw.finger_geoms), tw.geom_id[index]
    wrench = np.zeros(6)
    total = 0.0
    for c in range(d.ncon):
        con = d.contact[c]
        pair = (con.geom1, con.geom2)
        if gid in pair and (pair[0] in tips or pair[1] in tips):
            mujoco.mj_contactForce(m, d, c, wrench)
            total += abs(float(wrench[0]))     # contact-frame x is the normal
    return total


def push_block(tw: T.Tower, index: int, rng: random.Random | None = None,
               observe: float | None = None, step_delay: float = 0.0,
               on_step=None, *, direction: str = "end") -> Outcome:
    """Slide the block out by pushing on its far end face with a fingertip.

    This is how the move is really made, and the difference matters. A grip
    can apply force in any direction, so whenever the block resisted, the hand
    ended up hauling the tower along with it -- measured at 50-67 N straight
    up against a tower weighing 8.9 N. A fingertip is UNILATERAL: it can only
    push along its contact normal, and there is nothing attaching it to the
    block, so it physically cannot drag the tower.

    Nothing supports the block's weight either. It stays held up by its own
    neighbours the whole way, exactly as in the real game, so there is no
    drooping overhang to lever the tower apart.

    The finger stalls rather than forcing: when the contact force reaches
    PUSH_MAX_FORCE it stops advancing until the block yields. If it never
    yields inside the time budget the trial is `stuck`.
    """
    rng = rng or random.Random()
    observe = C.OBSERVE_SECONDS if observe is None else observe
    d, m = tw.data, tw.model
    action, direction, _distance = pull_path(tw, index, direction, rng)

    others = [i for i in tw.present_ids() if i != index]
    before = T.snapshot(tw, others)
    pre_push = T.save_state(tw)

    start_pos = tw.pos(index)
    mid = tw.finger_mocap
    # Start just off the face at the back of the block, i.e. behind it
    # relative to the way it will travel. The finger is turned to match the
    # block so a flat pad meets the face squarely.
    finger0 = start_pos - direction * T.finger_reach()
    d.mocap_pos[mid] = finger0
    d.mocap_quat[mid] = tw.quat(index)
    mujoco.mj_forward(m, d)

    need = C.BLOCK_LENGTH * C.PUSH_GRASP_FRACTION
    dt = float(m.opt.timestep)
    warning_counts = d.warning.number.copy()
    bad = [int(mujoco.mjtWarning.mjWARN_BADQPOS),
           int(mujoco.mjtWarning.mjWARN_BADQVEL),
           int(mujoco.mjtWarning.mjWARN_BADQACC)]

    advanced = 0.0
    peak_force = 0.0
    extract_time = C.PUSH_MAX_SECONDS
    extracted = False
    try:
        for s in range(int(C.PUSH_MAX_SECONDS / dt)):
            # Stall instead of shoving: a finger that is already pressing as
            # hard as it can simply stops making progress.
            if _finger_force(tw, index) < C.PUSH_MAX_FORCE:
                advanced += C.PUSH_SPEED * dt
            d.mocap_pos[mid] = finger0 + direction * advanced

            T.step(tw)
            peak_force = max(peak_force, _finger_force(tw, index))

            if (np.any(d.warning.number[bad] > warning_counts[bad])
                    or not np.isfinite(d.qpos).all()
                    or not np.isfinite(d.qvel).all()):
                T.restore_state(tw, pre_push)
                raise FloatingPointError("push simulation became numerically unstable")

            if on_step is not None:
                on_step()
            if step_delay:
                time.sleep(step_delay)

            if float(np.dot(tw.pos(index) - start_pos, direction)) >= need:
                extracted = True
                extract_time = (s + 1) * dt
                break
    finally:
        T.stow_finger(tw)
        mujoco.mj_forward(m, d)

    if not extracted:
        T.restore_state(tw, pre_push)
        return classify(0.0, 0.0, False, peak_force, extract_time,
                        pull_direction=action)

    # Enough of it is sticking out to take hold of, so the other hand lifts it
    # clear. The rest of the removal is instantaneous, like delete mode.
    T.park(tw, index, slot=len(tw.gap_ids) + 1)
    T.simulate(tw, observe, step_delay, on_step)
    disp, tilt = max_displacement(tw, before)
    return classify(disp, tilt, True, peak_force, extract_time,
                    pull_direction=action)


def remove(tw: T.Tower, index: int, mode: str,
           rng: random.Random | None = None, step_delay: float = 0.0,
           on_step=None, *, direction: str = "end") -> Outcome:
    if mode == "delete":
        return delete_block(tw, index, step_delay=step_delay, on_step=on_step)
    if mode == "pull":
        return pull_block(tw, index, rng, step_delay=step_delay, on_step=on_step,
                          direction=direction)
    if mode == "push":
        return push_block(tw, index, rng, step_delay=step_delay, on_step=on_step,
                          direction=direction)
    raise ValueError(f"unknown removal mode: {mode!r}")
