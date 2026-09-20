"""Build, jitter, settle and pre-gap a Jenga tower in MuJoCo.

Blocks are identified by BLOCK INDEX (0..53) everywhere in this project.
MuJoCo body / geom ids are an implementation detail kept inside the Tower.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import config as C


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _block_xml(index: int, level: int, slot: int, rng: random.Random) -> str:
    """One free-floating block body, jittered."""
    half = [C.BLOCK_LENGTH / 2, C.BLOCK_WIDTH / 2, C.BLOCK_HEIGHT / 2]
    if C.JITTER_SIZE_FRAC > 0:
        half = [h * (1.0 + rng.uniform(-1, 1) * C.JITTER_SIZE_FRAC) for h in half]

    yaw = 0.0 if level % 2 == 0 else math.pi / 2
    z = C.BLOCK_HEIGHT / 2 + level * (C.BLOCK_HEIGHT + C.BUILD_LAYER_GAP)
    offset = (slot - 1) * C.BLOCK_WIDTH
    x, y = (0.0, offset) if level % 2 == 0 else (offset, 0.0)

    x += rng.uniform(-1, 1) * C.JITTER_POS
    y += rng.uniform(-1, 1) * C.JITTER_POS
    yaw += math.radians(rng.uniform(-1, 1) * C.JITTER_YAW_DEG)
    qw, qz = math.cos(yaw / 2), math.sin(yaw / 2)

    # Vary brightness, not hue: jittering each channel independently turns
    # some blocks olive. Real blocks differ in shade, not colour.
    shade = 1.0 + rng.uniform(-1, 1) * C.BLOCK_COLOUR_JITTER
    rgba = " ".join(f"{min(1.0, max(0.0, c * shade)):.4f}" for c in C.BLOCK_COLOUR)

    return (
        f'    <body name="block{index}" pos="{x:.6f} {y:.6f} {z:.6f}" '
        f'quat="{qw:.8f} 0 0 {qz:.8f}">\n'
        f'      <freejoint/>\n'
        f'      <geom name="g{index}" size="{half[0]:.6f} {half[1]:.6f} {half[2]:.6f}" '
        f'rgba="{rgba} 1"/>\n'
        f'    </body>'
    )


def _finger_geoms_xml() -> str:
    """The pushing surface. See FINGER_SHAPE in config for why shape matters."""
    common = 'contype="1" conaffinity="1" rgba="0.85 0.45 0.45 1"'
    if C.FINGER_SHAPE == "pad":
        h = C.FINGER_PAD_HALF
        return (f'      <geom name="fingertip" type="box" '
                f'size="{h[0]} {h[1]} {h[2]}" {common}/>')
    if C.FINGER_SHAPE == "dual":
        o = C.FINGER_DUAL_OFFSET
        return (f'      <geom name="fingertip" type="sphere" '
                f'size="{C.FINGER_RADIUS}" pos="0 {o} 0" {common}/>\n'
                f'      <geom name="fingertip2" type="sphere" '
                f'size="{C.FINGER_RADIUS}" pos="0 {-o} 0" {common}/>')
    return (f'      <geom name="fingertip" type="sphere" '
            f'size="{C.FINGER_RADIUS}" {common}/>')


def finger_reach() -> float:
    """Distance from the block centre to the finger body origin at first touch."""
    if C.FINGER_SHAPE == "pad":
        nose = C.FINGER_PAD_HALF[0]
    else:
        nose = C.FINGER_RADIUS
    return C.BLOCK_LENGTH / 2 + nose + C.FINGER_GAP


def build_xml(seed: int = 0) -> str:
    """Generate the MJCF model: independent free blocks on a table."""
    rng = random.Random(seed)
    bodies = []
    index = 0
    for level in range(C.NUM_LAYERS):
        for slot in range(C.BLOCKS_PER_LAYER):
            bodies.append(_block_xml(index, level, slot, rng))
            index += 1

    lx, ly, lz = C.LIGHT_POS
    return f"""<mujoco model="jenga">
  <compiler angle="radian"/>
  <option timestep="{C.TIME_STEP}" gravity="0 0 {C.GRAVITY}"
          integrator="{C.INTEGRATOR}" cone="{C.CONE}"
          iterations="{C.SOLVER_ITERATIONS}" ls_iterations="{C.LS_ITERATIONS}"/>
  <visual>
    <global offwidth="{max(C.IMAGE_SIZE, 640)}" offheight="{max(C.IMAGE_SIZE, 480)}" fovy="{C.CAMERA_FOV}"/>
    <!-- Mostly ambient light: MuJoCo's default point light throws a strong
         bright-to-dark gradient across the floor, and we want a plain,
         evenly lit background for the dataset. -->
    <headlight ambient="0.55 0.55 0.55" diffuse="0.45 0.45 0.45" specular="0 0 0"/>
    <quality shadowsize="4096"/>
  </visual>
  <default>
    <geom type="box" density="{C.WOOD_DENSITY}" condim="3"
          friction="{C.LATERAL_FRICTION} {C.TORSIONAL_FRICTION} {C.ROLLING_FRICTION}"
          solref="{C.SOLREF[0]} {C.SOLREF[1]}"
          solimp="{' '.join(str(v) for v in C.SOLIMP)}"
          margin="{C.CONTACT_MARGIN}"/>
  </default>
  <worldbody>
    <!-- directional, so the floor is lit evenly instead of hot-spotted -->
    <light directional="true" pos="{lx} {ly} {lz}" dir="-0.35 -0.25 -1"
           diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05" castshadow="true"/>
    <geom name="ground" type="plane" size="{C.GROUND_SIZE} {C.GROUND_SIZE} 0.05"
          rgba="{C.GROUND_COLOUR[0]} {C.GROUND_COLOUR[1]} {C.GROUND_COLOUR[2]} 1"
          friction="{C.LATERAL_FRICTION} {C.TORSIONAL_FRICTION} {C.ROLLING_FRICTION}"/>
    <!-- Fingertip used by push mode. Kinematic (mocap), so it moves exactly
         where we put it; collisions are switched on only while pushing. -->
    <body name="finger" mocap="true" pos="0 0 -10">
      <!-- Compiled WITH collisions on. MuJoCo prunes geom pairs that can
           never collide when the model is compiled, so a geom built with
           contype=0 can never be switched on later -- setting the flag at
           runtime silently does nothing. Disabling at runtime does work,
           which is what park() relies on. The finger is therefore stowed by
           moving it away, not by turning it off. -->
{_finger_geoms_xml()}
    </body>
{chr(10).join(bodies)}
  </worldbody>
</mujoco>"""


# ---------------------------------------------------------------------------
# Tower
# ---------------------------------------------------------------------------

@dataclass
class Tower:
    model: mujoco.MjModel
    data: mujoco.MjData
    body_id: list[int] = field(default_factory=list)
    geom_id: list[int] = field(default_factory=list)
    qadr: list[int] = field(default_factory=list)   # qpos address per block
    vadr: list[int] = field(default_factory=list)   # qvel address per block
    level_of: list[int] = field(default_factory=list)
    slot_of: list[int] = field(default_factory=list)
    gap_ids: list[int] = field(default_factory=list)
    finger_mocap: int = 0
    finger_geom: int = -1
    finger_geoms: list[int] = field(default_factory=list)
    seed: int = 0
    n_gaps: int = 0

    @property
    def n_blocks(self) -> int:
        return len(self.body_id)

    def present_ids(self) -> list[int]:
        """Blocks still standing (parked ones excluded)."""
        gone = set(self.gap_ids)
        return [i for i in range(self.n_blocks) if i not in gone]

    def candidate_ids(self) -> list[int]:
        top = C.NUM_LAYERS - 1 - C.EXCLUDE_TOP_LAYERS
        return [i for i in self.present_ids() if self.level_of[i] <= top]

    def describe(self, index: int) -> str:
        return f"L{self.level_of[index]:02d}P{self.slot_of[index]}"

    def find(self, level: int, slot: int) -> int | None:
        for i in range(self.n_blocks):
            if self.level_of[i] == level and self.slot_of[i] == slot:
                return i
        return None

    # -- poses --------------------------------------------------------------
    def pos(self, index: int) -> np.ndarray:
        return self.data.xpos[self.body_id[index]].copy()

    def quat(self, index: int) -> np.ndarray:
        return self.data.xquat[self.body_id[index]].copy()

    def long_axis(self, index: int) -> np.ndarray:
        """Unit vector along the block's long axis (local +x) in world coords."""
        return self.data.xmat[self.body_id[index]].reshape(3, 3)[:, 0].copy()


def build_tower(seed: int = 0) -> Tower:
    """Compile a fresh model and return an unsettled tower."""
    model = mujoco.MjModel.from_xml_string(build_xml(seed))
    data = mujoco.MjData(model)
    tw = Tower(model=model, data=data, seed=seed)

    n = C.NUM_LAYERS * C.BLOCKS_PER_LAYER
    for i in range(n):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"block{i}")
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"g{i}")
        jid = model.body_jntadr[bid]
        tw.body_id.append(bid)
        tw.geom_id.append(gid)
        tw.qadr.append(model.jnt_qposadr[jid])
        tw.vadr.append(model.jnt_dofadr[jid])
        tw.level_of.append(i // C.BLOCKS_PER_LAYER)
        tw.slot_of.append(i % C.BLOCKS_PER_LAYER)

    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger")
    tw.finger_mocap = int(model.body_mocapid[finger])
    tw.finger_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip")
    tw.finger_geoms = [g for g in
                       (tw.finger_geom,
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip2"))
                       if g >= 0]

    mujoco.mj_forward(model, data)
    return tw


# ---------------------------------------------------------------------------
# Stepping, settling, measurement
# ---------------------------------------------------------------------------

def step(tw: Tower, n: int = 1) -> None:
    """Step the simulation, leaving body poses in sync with the state.

    mj_step computes kinematics and THEN integrates qpos, so when it returns
    data.xpos still describes the pose from before the last integration. Every
    measurement here reads xpos, so we resync once at the end -- otherwise any
    later mj_forward (parking a block, say) looks like the tower twitched.
    """
    for _ in range(n):
        mujoco.mj_step(tw.model, tw.data)
    mujoco.mj_kinematics(tw.model, tw.data)


def simulate(tw: Tower, seconds: float, step_delay: float = 0.0,
             on_step=None) -> None:
    """Step for a fixed duration. step_delay > 0 slows it to watchable speed."""
    for _ in range(int(seconds / C.TIME_STEP)):
        mujoco.mj_step(tw.model, tw.data)
        if on_step is not None:
            on_step()
        if step_delay:
            time.sleep(step_delay)
    mujoco.mj_kinematics(tw.model, tw.data)


def max_velocity(tw: Tower, blocks) -> tuple[float, float]:
    lin = ang = 0.0
    for i in blocks:
        v = tw.data.qvel[tw.vadr[i]:tw.vadr[i] + 6]
        lin = max(lin, float(np.linalg.norm(v[:3])))
        ang = max(ang, float(np.linalg.norm(v[3:])))
    return lin, ang


def settle(tw: Tower, blocks=None, max_seconds: float | None = None
           ) -> tuple[bool, float]:
    """Step until the tower stops moving (or we run out of patience).

    Returns (converged, simulated_seconds). See SETTLE_* in config for why
    this measures net displacement per window rather than velocity.
    """
    from .labeling import max_displacement

    blocks = tw.present_ids() if blocks is None else blocks
    max_seconds = C.SETTLE_MAX_SECONDS if max_seconds is None else max_seconds
    max_windows = max(1, int(max_seconds / C.TIME_STEP) // C.SETTLE_WINDOW_STEPS)

    prev = snapshot(tw, blocks)
    quiet = 0
    for w in range(max_windows):
        step(tw, C.SETTLE_WINDOW_STEPS)
        disp, tilt = max_displacement(tw, prev)
        prev = snapshot(tw, blocks)

        if disp < C.SETTLE_POS_TOL and tilt < C.SETTLE_ANG_TOL:
            quiet += 1
            if quiet >= C.SETTLE_STABLE_CHECKS:
                return True, (w + 1) * C.SETTLE_WINDOW_STEPS * C.TIME_STEP
        else:
            quiet = 0

    return False, max_windows * C.SETTLE_WINDOW_STEPS * C.TIME_STEP


def snapshot(tw: Tower, blocks) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Record pose of each block so we can measure how far it moves later."""
    return {i: (tw.pos(i), tw.quat(i)) for i in blocks}


# ---------------------------------------------------------------------------
# State save / restore
# ---------------------------------------------------------------------------
# For a world of free bodies, qpos + qvel IS the state. Copying two small
# arrays is far cheaper than PyBullet's saveState/restoreState.

def save_state(tw: Tower) -> tuple[np.ndarray, np.ndarray]:
    return tw.data.qpos.copy(), tw.data.qvel.copy()


def restore_state(tw: Tower, state: tuple[np.ndarray, np.ndarray]) -> None:
    qpos, qvel = state
    tw.data.qpos[:] = qpos
    tw.data.qvel[:] = qvel
    tw.data.qacc[:] = 0
    tw.data.qacc_warmstart[:] = 0
    tw.data.time = 0.0

    # Put every block back in play, then re-park the ones that were already
    # gaps in this tower. Parking lives in the MODEL (collision flags), not in
    # qpos/qvel, so a restore does not undo it -- and a gap block restored to
    # its spot under the floor WITH collisions back on gets fired up through
    # the tower by the ground plane.
    unpark_all(tw)
    for i in tw.gap_ids:
        _park_flags(tw, i)
    stow_finger(tw)

    mujoco.mj_forward(tw.model, tw.data)


def save_state_npz(tw: Tower, path) -> None:
    np.savez_compressed(path, qpos=tw.data.qpos, qvel=tw.data.qvel,
                        seed=tw.seed, gaps=np.array(tw.gap_ids, dtype=np.int32))


def load_state_npz(tw: Tower, path) -> None:
    with np.load(path) as z:
        tw.gap_ids = z["gaps"].tolist()
        tw.n_gaps = len(tw.gap_ids)
        restore_state(tw, (z["qpos"], z["qvel"]))


# ---------------------------------------------------------------------------
# Removing blocks by parking them below the floor
# ---------------------------------------------------------------------------
# MuJoCo's model is fixed at compile time, so a body cannot be deleted at
# runtime. Disable its collisions and hold it below the floor until restore.

def park(tw: Tower, index: int, slot: int = 0) -> None:
    """Take a block out of play: hide it, freeze it, stop it colliding.

    Collisions off and gravity compensated, so it simply sits where we put it
    instead of accelerating away forever.
    """
    _park_flags(tw, index)
    gid, bid = tw.geom_id[index], tw.body_id[index]
    a = tw.qadr[index]
    x, y, z = C.PARK_POSITION
    tw.data.qpos[a:a + 3] = (x + slot * C.PARK_SPACING, y, z)
    tw.data.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
    tw.data.qvel[tw.vadr[index]:tw.vadr[index] + 6] = 0.0
    mujoco.mj_forward(tw.model, tw.data)


def _park_flags(tw: Tower, index: int) -> None:
    """Switch a block out of the physics: no collisions, no gravity.

    Gravity is cancelled with a held force rather than model.body_gravcomp,
    because MuJoCo only computes gravity compensation when some body has it
    set at COMPILE time -- setting the field at runtime silently does nothing.
    """
    tw.model.geom_contype[tw.geom_id[index]] = 0
    tw.model.geom_conaffinity[tw.geom_id[index]] = 0
    bid = tw.body_id[index]
    tw.data.xfrc_applied[bid, 2] = tw.model.body_mass[bid] * abs(C.GRAVITY)


def stow_finger(tw: Tower) -> None:
    """Put the fingertip away, well below the floor and far from everything.

    Moved rather than switched off, because a disabled geom cannot be
    re-enabled (see the note in build_xml). It cannot touch the ground plane
    down there either: MuJoCo skips contacts between two bodies that both
    have no degrees of freedom, and the finger is kinematic.
    """
    tw.data.mocap_pos[tw.finger_mocap] = (0.0, 0.0, -10.0)


def unpark_all(tw: Tower) -> None:
    """Put every block back in play. Model flags are not part of the saved
    state, so a restore has to undo them explicitly."""
    for i in range(tw.n_blocks):
        tw.model.geom_contype[tw.geom_id[i]] = 1
        tw.model.geom_conaffinity[tw.geom_id[i]] = 1
        tw.data.xfrc_applied[tw.body_id[i]] = 0.0


def make_gaps(tw: Tower, rng: random.Random, n_gaps: int) -> int:
    """Pre-remove n_gaps blocks that the tower survives, to vary the towers."""
    from .labeling import max_displacement

    made = 0
    for _ in range(n_gaps):
        for _attempt in range(C.GAP_ATTEMPTS):
            candidates = tw.candidate_ids()
            if not candidates:
                tw.n_gaps = made
                return made
            victim = rng.choice(candidates)

            state = save_state(tw)
            others = [i for i in tw.present_ids() if i != victim]
            before = snapshot(tw, others)

            park(tw, victim, slot=len(tw.gap_ids))
            settle(tw, others, max_seconds=C.OBSERVE_SECONDS)
            disp, tilt = max_displacement(tw, before)

            if disp < C.COLLAPSE_DISPLACEMENT and tilt < C.COLLAPSE_TILT_DEG:
                tw.gap_ids.append(victim)
                made += 1
                break
            restore_state(tw, state)
        else:
            break  # every attempt collapsed it; stop adding gaps

    tw.n_gaps = made
    return made
