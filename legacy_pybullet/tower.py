"""Build, jitter, settle and pre-gap a Jenga tower.

Everything here works in simulation world units (see config.SCALE).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import pybullet as p

from . import config as C


# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------

def connect(gui: bool = False) -> int:
    """Open a PyBullet client and apply our physics settings."""
    cli = p.connect(p.GUI if gui else p.DIRECT)
    configure_physics(cli)
    return cli


def configure_physics(cli: int) -> None:
    p.setGravity(0, 0, C.GRAVITY, physicsClientId=cli)
    p.setPhysicsEngineParameter(
        fixedTimeStep=C.TIME_STEP,
        numSubSteps=C.NUM_SUBSTEPS,
        numSolverIterations=C.SOLVER_ITERATIONS,
        # Tight contact tolerance: the default (0.02) is ~13% of our block
        # thickness and makes the stack buzz.
        contactBreakingThreshold=C.CONTACT_BREAKING_THRESHOLD,
        # Penetration recovery handled separately from the velocity solve, so
        # slightly-overlapping blocks ease apart instead of popping.
        useSplitImpulse=1 if C.USE_SPLIT_IMPULSE else 0,
        splitImpulsePenetrationThreshold=C.SPLIT_IMPULSE_PENETRATION_THRESHOLD,
        erp=C.ERP,
        contactERP=C.CONTACT_ERP,
        # Bodies must never fall asleep: a sleeping neighbour would not react
        # when we pull a block out from under it.
        deterministicOverlappingPairs=1,
        physicsClientId=cli,
    )
    p.setPhysicsEngineParameter(enableConeFriction=0, physicsClientId=cli)


# ---------------------------------------------------------------------------
# Tower
# ---------------------------------------------------------------------------

@dataclass
class Tower:
    cli: int
    plane_id: int
    block_ids: list[int] = field(default_factory=list)
    level_of: dict[int, int] = field(default_factory=dict)
    slot_of: dict[int, int] = field(default_factory=dict)
    half_extents: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    colour_of: dict[int, list] = field(default_factory=dict)
    seed: int = 0
    n_gaps: int = 0
    gap_ids: list[int] = field(default_factory=list)

    @property
    def num_layers(self) -> int:
        return max(self.level_of.values()) + 1 if self.level_of else 0

    def present_ids(self) -> list[int]:
        """Blocks still standing in the tower (parked ones excluded)."""
        return [b for b in self.block_ids if b not in self.gap_ids]

    def candidate_ids(self) -> list[int]:
        """Blocks that are legal to attempt to remove."""
        top = self.num_layers - 1
        limit = top - C.EXCLUDE_TOP_LAYERS
        return [b for b in self.present_ids() if self.level_of[b] <= limit]

    def describe(self, body: int) -> str:
        return f"L{self.level_of[body]:02d}P{self.slot_of[body]}"

    def find(self, level: int, slot: int) -> int | None:
        for b in self.block_ids:
            if self.level_of[b] == level and self.slot_of[b] == slot:
                return b
        return None


def _layer_yaw(level: int) -> float:
    """Layers alternate 90 degrees."""
    return 0.0 if level % 2 == 0 else math.pi / 2.0


def build_tower(cli: int, seed: int = 0, jitter: bool = True) -> Tower:
    """Create the ground plane and a full 18-layer tower. Does not settle it."""
    rng = random.Random(seed)
    p.resetSimulation(physicsClientId=cli)
    configure_physics(cli)

    # A wide, thin box rather than GEOM_PLANE: PyBullet draws planes with a
    # fixed checkerboard texture, and the dataset wants a plain background.
    half = [C.GROUND_SIZE, C.GROUND_SIZE, C.GROUND_SIZE * 0.05]
    plane_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cli)
    plane_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                    rgbaColor=list(C.GROUND_COLOUR) + [1.0],
                                    physicsClientId=cli)
    plane_id = p.createMultiBody(0, plane_shape, plane_vis,
                                 basePosition=[0, 0, -half[2]],
                                 physicsClientId=cli)
    p.changeDynamics(
        plane_id, -1,
        lateralFriction=C.LATERAL_FRICTION,
        restitution=C.RESTITUTION,
        physicsClientId=cli,
    )

    tower = Tower(cli=cli, plane_id=plane_id, seed=seed)

    for level in range(C.NUM_LAYERS):
        yaw = _layer_yaw(level)
        z = C.BLOCK_HEIGHT / 2.0 + level * (C.BLOCK_HEIGHT + C.BUILD_LAYER_GAP)

        for slot in range(C.BLOCKS_PER_LAYER):
            offset = (slot - 1) * C.BLOCK_WIDTH

            # A layer's blocks sit side by side across its long axis.
            if level % 2 == 0:
                x, y = 0.0, offset
            else:
                x, y = offset, 0.0

            half = [C.BLOCK_LENGTH / 2.0, C.BLOCK_WIDTH / 2.0, C.BLOCK_HEIGHT / 2.0]
            if jitter and C.JITTER_SIZE_FRAC > 0:
                half = [h * (1.0 + rng.uniform(-1, 1) * C.JITTER_SIZE_FRAC) for h in half]

            block_yaw = yaw
            if jitter:
                x += rng.uniform(-1, 1) * C.JITTER_POS
                y += rng.uniform(-1, 1) * C.JITTER_POS
                block_yaw += math.radians(rng.uniform(-1, 1) * C.JITTER_YAW_DEG)

            # Vary brightness, not hue: jittering each channel independently
            # turns some blocks olive. Real Jenga blocks differ in shade, not
            # colour.
            shade = 1.0 + rng.uniform(-1, 1) * C.BLOCK_COLOUR_JITTER
            colour = [min(1.0, max(0.0, c * shade)) for c in C.BLOCK_COLOUR] + [1.0]

            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cli)
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half, rgbaColor=colour, physicsClientId=cli
            )
            body = p.createMultiBody(
                baseMass=C.BLOCK_MASS,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, z],
                baseOrientation=p.getQuaternionFromEuler([0, 0, block_yaw]),
                physicsClientId=cli,
            )
            p.changeDynamics(
                body, -1,
                lateralFriction=C.LATERAL_FRICTION,
                spinningFriction=C.SPINNING_FRICTION,
                rollingFriction=C.ROLLING_FRICTION,
                restitution=C.RESTITUTION,
                linearDamping=C.LINEAR_DAMPING,
                angularDamping=C.ANGULAR_DAMPING,
                # Never sleep -- a dozing block would ignore a neighbour leaving.
                activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
                physicsClientId=cli,
            )

            tower.block_ids.append(body)
            tower.level_of[body] = level
            tower.slot_of[body] = slot
            tower.half_extents[body] = tuple(half)
            tower.colour_of[body] = colour

    return tower


# ---------------------------------------------------------------------------
# Settling and motion measurement
# ---------------------------------------------------------------------------

def max_velocity(cli: int, bodies) -> tuple[float, float]:
    lin = ang = 0.0
    for b in bodies:
        v, w = p.getBaseVelocity(b, physicsClientId=cli)
        lin = max(lin, math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2))
        ang = max(ang, math.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2))
    return lin, ang


def settle(cli: int, bodies, max_seconds: float | None = None) -> tuple[bool, float]:
    """Step until everything is quiet (or we run out of patience).

    Returns (converged, simulated_seconds).
    """
    max_seconds = C.SETTLE_MAX_SECONDS if max_seconds is None else max_seconds
    max_steps = int(max_seconds / C.TIME_STEP)
    quiet = 0

    for step in range(max_steps):
        p.stepSimulation(physicsClientId=cli)
        if step % C.SETTLE_CHECK_EVERY == 0:
            lin, ang = max_velocity(cli, bodies)
            if lin < C.SETTLE_LINEAR_VEL and ang < C.SETTLE_ANGULAR_VEL:
                quiet += 1
                if quiet >= C.SETTLE_STABLE_CHECKS:
                    return True, (step + 1) * C.TIME_STEP
            else:
                quiet = 0

    return False, max_seconds


def snapshot(cli: int, bodies) -> dict[int, tuple]:
    """Record pose of each body so we can measure how far it moves later."""
    return {b: p.getBasePositionAndOrientation(b, physicsClientId=cli) for b in bodies}


def simulate(cli: int, seconds: float, step_delay: float = 0.0) -> None:
    """Step for a fixed duration. step_delay > 0 slows it to watchable speed."""
    for _ in range(int(seconds / C.TIME_STEP)):
        p.stepSimulation(physicsClientId=cli)
        if step_delay:
            time.sleep(step_delay)


# ---------------------------------------------------------------------------
# Removing blocks by parking them far away
# ---------------------------------------------------------------------------
# We never call removeBody: saveBullet/restoreState can only restore a world
# whose set of bodies is unchanged. Teleporting the block a kilometre away is
# functionally identical and keeps every trial restorable from one state file.

def park(cli: int, body: int, index: int = 0) -> None:
    x, y, z = C.PARK_POSITION
    p.resetBasePositionAndOrientation(
        body, [x + index * 10.0, y, z], [0, 0, 0, 1], physicsClientId=cli
    )
    p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0], physicsClientId=cli)


def make_gaps(cli: int, tower: Tower, rng: random.Random, n_gaps: int) -> int:
    """Pre-remove n_gaps blocks that the tower survives, to vary the towers.

    Returns the number of gaps actually made (a tower may refuse some).
    """
    from .labeling import max_displacement  # local import avoids a cycle

    made = 0
    for _ in range(n_gaps):
        for _attempt in range(C.GAP_ATTEMPTS):
            candidates = tower.candidate_ids()
            if not candidates:
                return made
            victim = rng.choice(candidates)

            state = p.saveState(physicsClientId=cli)
            others = [b for b in tower.present_ids() if b != victim]
            before = snapshot(cli, others)

            park(cli, victim, index=len(tower.gap_ids))
            settle(cli, others, max_seconds=C.OBSERVE_SECONDS)
            disp, tilt = max_displacement(cli, before)

            if disp < C.COLLAPSE_DISPLACEMENT and tilt < C.COLLAPSE_TILT_DEG:
                tower.gap_ids.append(victim)
                p.removeState(state, physicsClientId=cli)
                made += 1
                break
            p.restoreState(stateId=state, physicsClientId=cli)
            p.removeState(state, physicsClientId=cli)
        else:
            # Every attempt at this gap collapsed the tower; stop adding gaps.
            break

    tower.n_gaps = made
    return made
