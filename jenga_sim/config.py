"""All tunable numbers live here.

Everything is in real SI units: metres, kilograms, seconds, newtons. A block
is genuinely 15 x 25 x 75 mm and genuinely weighs 17 grams, so a force of
2 N here is 2 N on a real table.

(The earlier PyBullet version had to model a 10x-oversized tower because
Bullet's contact tolerances are absolute distances of a few mm -- a large
fraction of a real block's 15 mm thickness. MuJoCo parameterises contact
softness as a TIME constant (solref) rather than a distance, so real scale
works directly. See README "Why MuJoCo".)
"""

# ---------------------------------------------------------------------------
# Block + tower geometry (real Jenga)
# ---------------------------------------------------------------------------
# Block LOCAL axes are always: x = long, y = wide, z = thick.
# Layers alternate by yawing the whole layer 90 degrees about z.
BLOCK_LENGTH = 0.075   # 7.5 cm, local x
BLOCK_WIDTH = 0.025    # 2.5 cm, local y
BLOCK_HEIGHT = 0.015   # 1.5 cm, local z

BLOCKS_PER_LAYER = 3
NUM_LAYERS = 18        # 18 * 3 = 54 blocks

WOOD_DENSITY = 600.0   # kg/m^3 -> a block masses ~16.9 g, weighs ~0.166 N
BLOCK_MASS = WOOD_DENSITY * BLOCK_LENGTH * BLOCK_WIDTH * BLOCK_HEIGHT
BLOCK_WEIGHT = BLOCK_MASS * 9.81

# Vertical gap left between layers when building, so blocks never start
# interpenetrating. The tower settles this out in the first few steps.
BUILD_LAYER_GAP = 0.0002

# ---------------------------------------------------------------------------
# Per-block randomisation (real towers are not perfect)
# ---------------------------------------------------------------------------
JITTER_POS = 0.001       # +/- 1 mm in x and y
JITTER_YAW_DEG = 1.0     # +/- 1 degree about the vertical axis
# +/- fraction of each half-extent. THIS ONE MATTERS MORE THAN IT LOOKS, and
# it is a genuine trade-off -- measured over 6 seeds:
#     jitter   towers standing   blocks carrying no load
#      0.0%         6/6                    1%
#      0.2%         5/6                   19%
#      0.4%         1/6                   23%
#      1.0%         0/6                   32%
# Small size differences create uneven contact loads and loose blocks. Side
# contacts and the remaining supports still determine whether a pull is safe.
# Larger variation can make the initial tower unstable.
JITTER_SIZE_FRAC = 0.002

# ---------------------------------------------------------------------------
# Physics engine (MuJoCo)
# ---------------------------------------------------------------------------
TIME_STEP = 0.002
INTEGRATOR = "implicitfast"
CONE = "pyramidal"        # "elliptic" is more accurate but would not settle here
SOLVER_ITERATIONS = 100
LS_ITERATIONS = 50

GRAVITY = -9.81
LATERAL_FRICTION = 0.45
TORSIONAL_FRICTION = 0.005
ROLLING_FRICTION = 0.0001

# Contact softness. solref = (time constant, damping ratio). This is what lets
# us work at real Jenga scale: it is a TIME, not a distance, so it does not
# care how big the blocks are.
#
# 4x the timestep, not the 2x you might reach for first. Measured over 5 seeds:
#   solref    towers standing   worst drift   settling   position buzz
#   2x dt          5/5            0.679 cm     0.06 mm      11.9 um
#   4x dt          5/5            0.251 cm     0.24 mm       3.5 um
# Softer contacts converge better here, so the tower is both steadier and
# visibly calmer, and it costs only a fifth of a millimetre of extra settling.
SOLREF = (4 * TIME_STEP, 1.0)
SOLIMP = (0.95, 0.99, 0.0001)
# Contact margin. Sharp box corners dragging across a neighbour's face can
# snap between edge and face contacts; a small margin smooths that over.
CONTACT_MARGIN = 0.0

# ---------------------------------------------------------------------------
# Settling
# ---------------------------------------------------------------------------
# "Settled" is measured as NET pose change over a short window, not as
# instantaneous velocity. A block resting on an imperfect contact can buzz at
# 1+ rad/s while going nowhere, so a velocity threshold never fires; net
# motion over a window drops cleanly from ~14000 um to ~30 um once the tower
# is actually at rest.
SETTLE_MAX_SECONDS = 2.0
SETTLE_WINDOW_STEPS = 25      # 0.05 s at the default timestep
SETTLE_POS_TOL = 1e-4         # 0.1 mm net movement per window
SETTLE_ANG_TOL = 0.25         # degrees net rotation per window
SETTLE_STABLE_CHECKS = 3      # consecutive quiet windows needed

# A freshly built tower must not drift more than this while standing untouched.
BUILD_REJECT_DRIFT = 0.003    # 3 mm
BUILD_VERIFY_SECONDS = 0.75

# ---------------------------------------------------------------------------
# Mid-game gaps (pre-removed blocks, so towers are not all pristine)
# ---------------------------------------------------------------------------
# Pristine towers essentially never collapse from one removal (measured on the
# PyBullet build: 204/204 stable), so gaps are what create collapse labels.
GAPS_MIN = 4
GAPS_MAX = 14
GAP_ATTEMPTS = 6              # retries if a "safe" removal collapses it
STACK_REMOVED_ON_TOP = False  # real Jenga rule; off for v0.1

EXCLUDE_TOP_LAYERS = 1        # topmost layer is not a legal candidate

# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------
OBSERVE_SECONDS = 0.5         # how long to watch after the block is gone
# Where removed blocks go. Just under the floor, which is opaque from every
# camera angle we use, so they never appear in a rendered image. They are also
# frozen there (collisions off, gravity compensated) -- an earlier version
# parked them 100 m away and let them fall, which after ten seconds put a body
# 500 m below the scene. That blows up the renderer's bounding box, destroys
# shadow-map resolution and makes the whole tower shimmer.
PARK_POSITION = (0.0, 0.0, -0.25)
PARK_SPACING = 0.1

# Pull mode
PULL_SPEED = 0.04             # m/s, about how fast a careful hand moves
PULL_DISTANCE = BLOCK_LENGTH * 1.15
PULL_SIDE_DISTANCE = BLOCK_WIDTH * 1.15  # outer block only needs to clear its width
# Actual axial force cap, enforced throughout the pull (about 4.14 N).
PULL_MAX_FORCE = 25.0 * BLOCK_WEIGHT
PULL_MAX_SECONDS = 4.0
PULL_STUCK_FRACTION = 0.92    # must travel this fraction of PULL_DISTANCE
PULL_RANDOMISE_END = True
PULL_TIME_STEP = 0.0005       # substep sliding contacts; normal/delete dt unchanged

# Compliant grip bandwidths. Gains are derived from the selected block's
# actual mass and full inertia tensor in removal.py. These bandwidths keep
# explicit feedback stable at the default timestep; larger model timesteps
# are capped at runtime. Supporting ONLY the block's own weight avoids both
# gravitational droop and a fixed-height constraint jacking up the tower.
PULL_OMEGA = 0.3 / TIME_STEP
PULL_ANG_OMEGA = 0.15 / TIME_STEP
PULL_MAX_TORQUE = 6.0 * BLOCK_WEIGHT * BLOCK_LENGTH / 2  # N.m

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
COLLAPSE_DISPLACEMENT = 0.01  # 1 cm: any other block moving further = collapse
COLLAPSE_TILT_DEG = 10.0

# ---------------------------------------------------------------------------
# Camera / rendering
# ---------------------------------------------------------------------------
IMAGE_SIZE = 512
CAMERA_DISTANCE = 0.45
CAMERA_AZIMUTH = 48.0
CAMERA_ELEVATION = -22.0
CAMERA_TARGET_Z_FRAC = 0.45   # fraction of tower height to look at
CAMERA_FOV = 50.0

RANDOMISE_CAMERA = False      # turn on later for sim-to-real
CAMERA_AZIMUTH_RANGE = (0.0, 360.0)
CAMERA_ELEVATION_RANGE = (-35.0, -10.0)
CAMERA_DISTANCE_RANGE = (0.38, 0.58)

RANDOMISE_LIGHTING = False
LIGHT_POS = (0.35, 0.25, 0.75)

BLOCK_COLOUR = (0.78, 0.60, 0.34)   # plain wood
BLOCK_COLOUR_JITTER = 0.12          # +/- brightness, not hue
BACKGROUND_COLOUR = (0.92, 0.92, 0.94)
GROUND_COLOUR = (0.86, 0.85, 0.82)
GROUND_SIZE = 1.0

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
DEFAULT_TOWERS = 20
CANDIDATES_PER_TOWER = 8      # sample this many blocks per tower; None = all
DEFAULT_SEED = 0

# ---------------------------------------------------------------------------
# Push mode -- the way people actually take a middle block out
# ---------------------------------------------------------------------------
# A fingertip presses on the block's exposed end face and slides it through the
# tower. This is unilateral: a finger can only PUSH. It cannot grip, so it can
# never drag the tower along with the block, which is the failure mode that
# made every grip-and-pull variant destroy the tower.
# Fingertip shape. A single sphere makes POINT contact on a flat face: any
# offset between the contact normal and the block's centre of mass is a torque
# (F x r), so the block yaws, and in an outer slot that pivots its inner edge
# into the middle block and drags the whole layer sideways.
#   "sphere" -- one 5 mm ball (point contact)
#   "pad"    -- a flat plate, face-to-face, so a twist meets a restoring couple
#   "dual"   -- two balls spread across the face width, same idea
FINGER_SHAPE = "pad"
FINGER_RADIUS = 0.005          # 5 mm fingertip
FINGER_PAD_HALF = (0.002, BLOCK_WIDTH * 0.32, BLOCK_HEIGHT * 0.32)
FINGER_DUAL_OFFSET = BLOCK_WIDTH / 3.0
FINGER_GAP = 0.0005            # start just clear of the face
PUSH_SPEED = 0.02              # m/s -- deliberately slower than a grab
PUSH_MAX_FORCE = 25.0 * BLOCK_WEIGHT   # finger stalls above this
PUSH_MAX_SECONDS = 6.0
# Once this much of the block sticks out the far side you could take hold of
# it with the other hand, which is how the move really ends. Past that point
# the finger's job is done, so we treat it as extracted.
PUSH_GRASP_FRACTION = 0.55
