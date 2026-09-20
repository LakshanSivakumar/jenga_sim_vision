"""All tunable numbers live here.

Read the SCALE note first -- it explains why every length in this file looks
10x bigger than a real Jenga block.
"""

# ---------------------------------------------------------------------------
# World scale
# ---------------------------------------------------------------------------
# A real Jenga block is 15 x 25 x 75 mm. Bullet's contact-handling tolerances
# (collision margins, penetration slop, contact-breaking distance) are absolute
# distances of roughly 1-4 mm. At real Jenga size those tolerances are a large
# fraction of the block's 15 mm thickness, so stacks visibly sink, jitter and
# eventually explode.
#
# Fix: simulate a tower that is SCALE times bigger in every dimension, i.e. we
# work in decimetres instead of metres. Mass scales with volume (SCALE**3) so
# the density stays a real wood density and the world stays self-consistent --
# it is simply a giant Jenga tower made of real wood.
#
# One caveat to be aware of when reading timings: gravity is NOT scaled, so
# this is Froude-similar rather than fully similar. Events take sqrt(SCALE)
# ~= 3.2x longer here than for a real desk-sized tower. That is why the settle
# and observation windows below look generous.
SCALE = 10.0


def m(metres: float) -> float:
    """Real-world metres -> simulation world units."""
    return metres * SCALE


def cm(centimetres: float) -> float:
    """Real-world centimetres -> simulation world units."""
    return centimetres / 100.0 * SCALE


# ---------------------------------------------------------------------------
# Block + tower geometry
# ---------------------------------------------------------------------------
# Real block: 1.5 cm thick (stacking axis), 2.5 cm wide, 7.5 cm long.
# In the sim the block's LOCAL axes are always: x = long, y = width, z = thick.
# Layers alternate by yawing the whole layer 90 degrees about z.
BLOCK_LENGTH = cm(7.5)   # local x
BLOCK_WIDTH = cm(2.5)    # local y
BLOCK_HEIGHT = cm(1.5)   # local z

BLOCKS_PER_LAYER = 3
NUM_LAYERS = 18          # 18 * 3 = 54 blocks

WOOD_DENSITY = 600.0     # kg/m^3
# Mass of one block at simulation scale (giant tower, real wood density).
BLOCK_MASS = WOOD_DENSITY * (BLOCK_LENGTH * BLOCK_WIDTH * BLOCK_HEIGHT)

# Vertical gap left between layers when building, so blocks never start
# interpenetrating. The tower settles this out in the first few steps.
BUILD_LAYER_GAP = cm(0.02)

# ---------------------------------------------------------------------------
# Per-block randomisation (real towers are not perfect)
# ---------------------------------------------------------------------------
JITTER_POS = cm(0.1)       # +/- 1 mm in x and y
JITTER_YAW_DEG = 1.0       # +/- 1 degree about the vertical axis
JITTER_SIZE_FRAC = 0.0     # +/- fraction of each half-extent; 0 disables

# ---------------------------------------------------------------------------
# Physics engine
# ---------------------------------------------------------------------------
TIME_STEP = 1.0 / 240.0
NUM_SUBSTEPS = 1              # 1 is enough at this scale; 2 is ~3x slower
SOLVER_ITERATIONS = 100
GRAVITY = -9.81

LATERAL_FRICTION = 0.45
SPINNING_FRICTION = 0.005     # tiny; stops blocks pivoting on a corner forever
ROLLING_FRICTION = 0.0
RESTITUTION = 0.0             # wood on wood does not bounce

LINEAR_DAMPING = 0.04
ANGULAR_DAMPING = 0.04

# Contacts closer than this are treated as touching. Default is 0.02 which is
# 13% of our block thickness -- far too loose and a major source of jitter.
CONTACT_BREAKING_THRESHOLD = cm(0.02)

# Split impulse separates penetration-recovery from the velocity solve, so
# blocks that sink slightly into each other push apart smoothly instead of
# popping. This is the single biggest stability win for tall stacks.
USE_SPLIT_IMPULSE = True
SPLIT_IMPULSE_PENETRATION_THRESHOLD = -cm(0.05)

ERP = 0.2
CONTACT_ERP = 0.2

# ---------------------------------------------------------------------------
# Settling
# ---------------------------------------------------------------------------
SETTLE_MAX_SECONDS = 6.0
SETTLE_LINEAR_VEL = m(0.002)      # world units / s
SETTLE_ANGULAR_VEL = 0.02         # rad / s
SETTLE_STABLE_CHECKS = 5          # consecutive quiet checks needed
SETTLE_CHECK_EVERY = 12           # steps between checks

# A freshly built tower must not drift more than this while standing untouched.
BUILD_REJECT_DRIFT = cm(0.3)
BUILD_VERIFY_SECONDS = 2.0        # quick self-stand check during generation

# ---------------------------------------------------------------------------
# Mid-game gaps (pre-removed blocks, so towers are not all pristine)
# ---------------------------------------------------------------------------
# Pristine towers essentially never collapse from one removal (measured:
# 204/204 stable), so gaps are what create collapse labels at all.
GAPS_MIN = 4
GAPS_MAX = 14
GAP_ATTEMPTS = 6                  # retries if a "safe" removal collapses it
STACK_REMOVED_ON_TOP = False      # real Jenga rule; off for v0.1

# Topmost layer is not a legal candidate in real Jenga.
EXCLUDE_TOP_LAYERS = 1

# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------
# How long to watch the tower after the block is gone. Measured: 1.5 s gives
# identical labels to 3.0 s across 60 trials, at half the cost.
OBSERVE_SECONDS = 1.5
PARK_POSITION = (1000.0, 0.0, 0.0)  # where removed blocks are teleported to

# Pull mode
PULL_SPEED = m(0.015)             # world units per second
PULL_DISTANCE = BLOCK_LENGTH * 1.15
# Newtons the puller can apply before it gives up and calls the block stuck.
# Measured on real towers: peak pull force runs ~1000-5200 N (6-31x a block's
# own weight), median ~1900 N. 18x sits in the upper quartile, so genuinely
# jammed blocks get labelled `stuck` and ordinary ones come out.
PULL_MAX_FORCE = 18.0 * BLOCK_MASS * 9.81
PULL_MAX_SECONDS = 12.0
PULL_STUCK_FRACTION = 0.92        # must travel this fraction of PULL_DISTANCE
PULL_RANDOMISE_END = True

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
COLLAPSE_DISPLACEMENT = cm(1.0)   # any other block moves more than this
COLLAPSE_TILT_DEG = 10.0          # ...or tilts more than this

# ---------------------------------------------------------------------------
# Camera / rendering
# ---------------------------------------------------------------------------
IMAGE_SIZE = 512
CAMERA_DISTANCE = m(0.45)
CAMERA_YAW = 48.0
CAMERA_PITCH = -22.0
CAMERA_TARGET_Z_FRAC = 0.45       # fraction of tower height to look at
CAMERA_FOV = 50.0
CAMERA_NEAR = 0.05
CAMERA_FAR = m(5.0)

RANDOMISE_CAMERA = False          # turn on later for sim-to-real
CAMERA_YAW_RANGE = (0.0, 360.0)
CAMERA_PITCH_RANGE = (-35.0, -10.0)
CAMERA_DISTANCE_RANGE = (m(0.38), m(0.58))

RANDOMISE_LIGHTING = False
LIGHT_DIRECTION = (0.6, 0.35, 1.0)

BLOCK_COLOUR = (0.78, 0.60, 0.34)   # plain wood
BLOCK_COLOUR_JITTER = 0.12   # +/- brightness, not hue
BACKGROUND_COLOUR = (0.92, 0.92, 0.94)
GROUND_COLOUR = (0.86, 0.85, 0.82)
# The ground is a big thin box, not GEOM_PLANE: PyBullet renders a plane with
# a hard-coded checkerboard texture, and we want a plain background.
GROUND_SIZE = m(2.0)

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
DEFAULT_TOWERS = 20
CANDIDATES_PER_TOWER = 8          # sample this many blocks per tower; None = all
DEFAULT_SEED = 0
