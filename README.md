# Jenga Risk Vision — Physics Simulator (v0.2)

Generates training data for a computer-vision system that looks at a photo of
a Jenga tower, finds every block, and gives each one a risk level for removal:
**low**, **medium** or **high**.

There is no real dataset of per-block removal outcomes, so we make one: build
a tower in MuJoCo, photograph it from several angles, then remove every block
in turn and measure what happens to the tower.

This is the simulator and data generator only. No models yet — but see
[the recommended pipeline](#recommended-model-pipeline), because the data is
shaped around it.

---

## Setup

Python 3.11. Full command reference is in [USAGE.md](USAGE.md).

### macOS / Linux

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

### Windows (PowerShell)

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Everywhere below, Windows users substitute `.venv\Scripts\python.exe` for
`.venv/bin/python`. That is the only difference -- except that `play.py` is
actually *simpler* on Windows, because it does not need the `mjpython` step.

### One macOS wrinkle: `mjpython`

MuJoCo's interactive viewer **must** be launched with `mjpython` on macOS (a
normal `python` cannot own the GUI event loop there). `mjpython` ships with the
`mujoco` wheel. Windows and Linux do not need any of this.

If your venv was made by `uv`, the first run fails with:

```
Library not loaded: @rpath/libpython3.11.dylib
```

`mjpython` is a signed app bundle that `dlopen()`s the venv's Python, which
needs a shared `libpython`. uv's standalone CPython ships that dylib but does
not put it on the rpath, and `DYLD_LIBRARY_PATH` does not help because macOS
strips `DYLD_*` for signed binaries. One of the paths `mjpython` *does* search
is `<venv>/bin/../`, so copying the dylib there fixes it:

```bash
./tools/fix_mjpython_uv.sh .venv
```

Only `play.py` needs this. `generate.py`, `inspect_data.py` and `pytest` all
render offscreen and run under plain `python`.

---

## The three things you can run

```bash
.venv/bin/mjpython play.py                      # interactive playground (macOS)
.venv/bin/python generate.py --towers 20        # headless dataset
.venv/bin/python inspect_data.py data/          # eyeball the labels
.venv/bin/python -m pytest -q                   # physics sanity tests
```

On Windows: `.venv\Scripts\python.exe play.py` for the playground, and
`.venv\Scripts\python.exe` in place of `.venv/bin/python` for the rest.

### `play.py` — interactive playground

| Control | Does |
|---|---|
| `W` / `S` | move the selection up / down a level |
| `A` / `D` | move the selection across the layer |
| `P` | pull the selected block out slowly (force-limited) |
| `X` | delete the selected block instantly |
| `R` | reset to the saved tower |
| `N` | build a new random tower |
| `C` | save a snapshot PNG to `snapshots/` |
| `Q` | quit |
| mouse | left-drag orbits, right-drag pans, scroll zooms; double-click a block then ctrl+left-drag to shove it |
| `F1` | MuJoCo's own help overlay |

The selected block turns green. Outcomes print to the terminal with
displacement, tilt and (for pulls) peak force. Keys only register while the
MuJoCo window has focus.

Flags: `--seed N`, `--gaps N`, `--slow 0` (run removals at full speed instead
of watchable speed).

### `generate.py` — headless dataset

```bash
python generate.py --towers 20 --out data/my_run
```

Every tower is photographed `--views` times (default 4) and **every block in
it** gets a risk label. Prints the risk-level breakdown, the survivors' tilt
margins, and how often the structural rule agrees with the physics. See
[USAGE.md](USAGE.md) for every flag.

Budget roughly a minute of wall time per tower with the default 5 workers:
each surviving block gets a tilt test, and that is where the time goes.

### `inspect_data.py` — sanity-check by eye

```bash
python inspect_data.py data/my_run                         # whole towers
python inspect_data.py data/my_run --blocks --balanced     # single blocks
```

Towers mode colours every block green / amber / red by risk — what the
finished model should produce from a photo. Blocks mode highlights one block
per panel, which is better for checking individual labels.

---

## Why MuJoCo, and why real scale

This project started on PyBullet and was ported. The reason is worth knowing,
because it shaped everything else.

PyBullet's contact tolerances (collision margin, penetration slop,
contact-breaking distance) are **absolute distances** of a few millimetres. A
real Jenga block is only 15 mm thick, so those tolerances are a large fraction
of the block and stacks sink, buzz and eventually explode. The workaround was
to simulate a tower 10x oversized.

MuJoCo parameterises contact softness as a **time constant** (`solref`)
rather than a distance, so it does not care how big the blocks are. Real
15 mm blocks work directly, and everything in `config.py` is in honest SI
units: a block is 16.9 g, the tower is 27 cm tall, and a pull force of 6 N is
6 N.

Measured head to head on the same 54-block tower, both machines idle:

| | wall-time per **real** second of physics | drift over 10 s |
|---|---|---|
| PyBullet, 10x scale | 3.47 s | 0.059 cm |
| MuJoCo, real scale | 0.89 s | 0.030 cm |

About 3.9x faster and 2x more accurate — but note *why*. Raw solver throughput
is comparable (219 vs 278 steps/s). The win comes from dropping the 10x hack:
an oversized world is only Froude-similar, so PyBullet needed `sqrt(10)` ≈ 3.2x
more *simulated* seconds per real second of physics. The engine is not
magically faster; it just let us stop lying about scale.

---

## Risk levels

| Level | Means |
|---|---|
| **high** | removing the block collapses the tower, or leaves it falling at the slightest touch (`tilt_margin_deg < RISK_HIGH_BELOW_DEG`, default 1°) |
| **medium** | the tower survives, but is measurably more fragile than before (`margin_drop_deg ≥ RISK_MEDIUM_DROP_DEG`, default 0.3°) |
| **low** | the tower survives, about as robust as it was |

Fragility is measured with a **tilt test**: the table is tilted slowly towards
each of the tower's four faces until the tower gives way, and the weakest
direction is its tilt margin. It is measured twice:

- `base_tilt_deg` (in `towers.csv`) — the tower as it stands, before anything
  is removed.
- `tilt_margin_deg` (in `labels.csv`) — what is left after removing this
  block. **0 means the removal collapsed the tower.**

Their difference is `margin_drop_deg`: how much fragility **this block's**
removal added. That drop is what separates medium from low.

Train on the continuous columns and bin at the end — the thresholds can then
move without regenerating anything, and `inspect_data.py --medium-drop X`
previews a different split.

"Collapse" itself means any other block moved more than
`COLLAPSE_DISPLACEMENT` (1 cm) or tilted more than `COLLAPSE_TILT_DEG` (10°).

### Why "medium" is defined like this

Three definitions were tried, and measured, before this one.

**Probability of collapse — does not work.** Each block was removed 8 times
under invisible perturbations (±20% friction on every block, a small random
nudge): across 60 blocks, **every one came out at exactly 0% or 100%**.
Removing a block either collapses the tower or it does not, and small
uncertainties never change which. That leaves two classes, not three.

**Displacement bands — too thin.** Displacement is sharply bimodal (median
1.5 mm, then straight to 250 mm+), with only 9% of removals in between, and
"shifted a bit" is not really "risky".

**Absolute tilt margin — describes the tower, not the block.** An early probe
showed survivors' margins spread over 4–8°, which looked like a good third
class. But that spread was *between towers*: different towers start with
different margins. Within a tower, **85% of surviving removals change the
margin by less than 0.3°**. An absolute threshold therefore labelled every
block in an already-wobbly tower "medium" — on the first two towers generated,
77% medium and 1% low — which says nothing about which block to take.

**Margin drop — what is used.** Measuring the change a removal causes, relative
to the tower's own starting margin, is a property of the block. It is a subtle
signal — most removals cost nothing, a tail costs up to a degree — and the
0.3° threshold is twice the tilt test's resolution (~0.15°), so it sits above
measurement noise.

Validated on 20 towers, 897 blocks:

| | |
|---|---|
| split | **low 59.8% · medium 15.4% · high 24.9%** |
| towers containing all three levels | 18 of 20 |
| survivors' margin drop | median 0.00°, p75 0.30°, p90 0.60°, p95 0.75° |
| structural rule agrees with collapse | 99.1% |

Medium has a clear physical meaning. **Edge blocks are medium 27% of the time,
middle blocks 3%.** Taking an edge from a full layer leaves two blocks offset to
one side, so that layer's support shifts off-centre and the tower tips more
easily that way; taking the middle leaves the layer symmetric. That is the
real-Jenga instinct "take the middle ones first", recovered from physics.

### The structural rule

One line predicts collapse-on-removal with **98.0% accuracy** on 4,155
simulated removals, using nothing but which blocks are present:

> A removal collapses the tower if it leaves its layer **empty, or standing on
> a single edge block**.

| left in the layer after removal | collapse |
|---|---|
| nothing, or one edge block | 100% |
| middle only, two blocks, or both edges | 2–5% |

This is the real-Jenga rule, and it is why the data is shaped the way it is:
**high risk is mostly a structural fact about the tower**, visible in which
slots are filled. `jenga_sim.risk.structural_rule` implements it, and both
`generate.py` and `inspect_data.py` report how often the physics agrees with
it — a free check that nothing has broken.

The physics earns its keep on the medium/low boundary, which depends on gaps
across several layers, and on the ~2% of collapses the rule misses.

### Recommended model pipeline

```
photo ──► CNN finds the blocks ──► presence grid (18 × 3) ──► risk model ──► low / medium / high per block
```

- **Stage 1 is the computer vision**: detect and segment the blocks, and work
  out which slots are filled. The simulator provides unlimited labelled images
  and masks (`images/`, `masks/`), from several randomised angles per tower.
  Pretrained detectors (e.g. YOLO-seg) are strong here.
- **Stage 2 reads the grid** (`towers.csv` → `grid`) and predicts each block's
  risk or tilt margin. Because a grid from a real photo is exactly the same
  kind of object as one from the simulator, this stage has **no sim-to-real
  gap at all**.
- **Train an end-to-end CNN (photo → risk) as the comparison.** If it gets
  close to the structural rule on high risk, it has learned the grid; if not,
  that is the argument for splitting.
- **Photograph a real tower** 30–50 times and hand-label the filled slots.
  Detectors trained only on simulated images rarely transfer perfectly, and
  that small real set is how you will know.

---

## Removal modes

The dataset uses **`delete` followed by the tilt test** for every block.
`pull` and `push` remain in the code and in `play.py`, but no longer feed the
dataset — see [below](#why-the-dataset-uses-delete-mode).

**`delete`** — the block is teleported away instantly, then the tower is
watched for `OBSERVE_SECONDS`. Cheap, and the cleanest measure of "was this
block load-bearing?".

**`pull`** — a compliant grip pulls along the block's initial long axis.
An axial spring and damper follow a moving target, with force clamped to
`PULL_MAX_FORCE` at every step. The grip supports the selected block's own
weight and uses bounded torque to keep its initial orientation, while leaving
height and lateral translation free. It does not hold up the tower with a
fixed-height constraint. Rotational feedback uses each block's full inertia
tensor and the correct local/world coordinate transforms. Pulls use 0.5 ms
physics substeps to resolve sliding contacts; ordinary simulation and delete
mode retain the 2 ms timestep. Viewer redraws and slow-motion delays keep
their original cadence.

`peak_force` records the force actually applied. When the force saturates it
is a lower bound on what a faster extraction might require. A successful
extraction must reach the configured distance and clear contact with the
other blocks, then is followed by `OBSERVE_SECONDS` of observation. An unsuccessful
attempt is labelled `stuck` and rolled back to the initial tower. That rollback
is a dataset/playground convention: a real unsuccessful tug can disturb a
tower. Numerical divergence during pulling raises an error rather than emitting a label.

Blocks are never deleted from the model. Removal disables their collisions
and parks them below the floor with gravity cancelled. Restoring the initial
state puts candidate blocks back while keeping pre-existing gaps parked.

---

## Output format

```
data/my_run/
  images/tower_0003_v0.png          photo, view 0 (the fixed 3/4 reference shot)
  images/tower_0003_v1.png          ...more views: randomised camera, light, colours
  masks/tower_0003_v0_seg.png       pixel -> block index 1-54, 0 = none
  risk/tower_0003_v0_risk.png       pixel -> 0 none, 1 low, 2 medium, 3 high
  states/tower_0003.npz             physics state (qpos, qvel, seed, gaps)
  towers.csv                        one row per tower, including the grid
  labels.csv                        one row per block, every block in every tower
  views.csv                         one row per image
  visibility.csv                    pixels of each block in each image
```

**`towers.csv`** — `tower_id, seed, num_blocks, num_gaps, grid,
base_tilt_deg, settle_seconds`. `grid` is 54 characters, `1` = block present,
`0` = gap; character *i* is block index *i* = `level * 3 + slot`, bottom layer
first. `base_tilt_deg` is the tower's own tilt margin before anything is
removed.

**`labels.csv`** — `tower_id, block_id, level, position_in_level, legal,
outcome, max_displacement, max_tilt_deg, tilt_margin_deg, risk_level, seed`.
`block_id` is 1-based (it matches the seg maps); the grid is 0-based, so
`grid[block_id - 1]`. `legal` is false for the top layer, which real Jenga
does not allow taking from — those blocks are still labelled, because they
still appear in the photo.

**`views.csv`** — `tower_id, view, camera_params, look_params`: enough to
reproduce each shot.

**`visibility.csv`** — `tower_id, view, block_id, pixels`. A block with 0
pixels in a view is completely hidden there; drop those block/view pairs when
training anything that looks at pixels.

A single block's mask is simply `seg == block_id`
(`jenga_sim.dataset.load_block_mask` does it), so per-block mask files are not
written — at 54 blocks and several views that would be hundreds of files per
tower carrying nothing the seg map does not.

Every image of a tower shares the same labels: the risk of a block does not
depend on where the camera is. That is what makes extra views nearly free
training data — physics is the expensive part, rendering is cheap.

---

## Config knobs worth tweaking

Everything lives in `jenga_sim/config.py`. The three that matter most:

1. **`JITTER_SIZE_FRAC`** (default 0.002) — how much blocks vary in size. This
   is the least obvious knob and the most important, because it is a genuine
   trade-off. Measured over 6 seeds:

   | jitter | towers standing | blocks carrying no load |
   |---|---|---|
   | 0.0% | 6/6 | 1% |
   | 0.2% | 5/6 | 19% |
   | 0.4% | 1/6 | 23% |
   | 1.0% | 0/6 | 32% |

   Tiny size differences produce uneven contact loads and loose blocks.
   They do not guarantee a safe extraction: side contacts and support geometry
   also matter. Too much variation and the towers stop standing.

2. **`GAPS_MIN` / `GAPS_MAX`** (default 4–14) — how many blocks are removed
   before the trial starts. Pristine towers are almost never destroyed by a
   single removal (measured on the PyBullet build: 204 out of 204 `stable`),
   because taking one block from a full 3-block layer still leaves two to
   carry the load. Raise the range for more collapses.

3. **`RISK_MEDIUM_DROP_DEG` / `RISK_HIGH_BELOW_DEG`** (default 0.3° / 1°) —
   how much margin a removal must cost to count as medium, and how little
   margin left counts as high. These only affect the `risk_level` column; the
   margins are always stored, so you can re-bin later without regenerating.
   Preview a split with `inspect_data.py --medium-drop 0.5`.

Also worth knowing: `VIEWS_PER_TOWER` sets how many photos each tower gets
(cheap — only rendering); `TILT_RAMP_SECONDS` trades tilt-test accuracy
against generation time; `COLLAPSE_DISPLACEMENT` / `COLLAPSE_TILT_DEG` define
what counts as a collapse; `OBSERVE_SECONDS` is how long the tower is watched
after a removal.

---

## Physics settings, and what actually mattered

The tower has to stand still when untouched — that is the acceptance test
everything else depends on. What mattered, in order:

- **Working at real scale on a time-parameterised contact model.** See
  "Why MuJoCo" above.
- **`solref` set to 4x the timestep.** Stiff enough that blocks do not
  sink into each other, soft enough that the solver converges.
- **Measuring "settled" as net pose change over a window, not as velocity.**
  A block resting on an imperfect contact buzzes at over 1 rad/s while going
  nowhere, so a velocity threshold never fires at all. Net motion over a
  0.05 s window drops cleanly from ~14,000 µm to ~30 µm once the tower is
  genuinely at rest.

Measured, 8 seeds, 10 simulated seconds untouched, at the default
`JITTER_SIZE_FRAC = 0.002`: worst drift **0.251 cm**, worst tilt **2.87°**.
With identical blocks (`JITTER_SIZE_FRAC = 0`) it is **0.033 cm** and 0.24°.
That is the cost of realistic blocks, and it is why `BUILD_REJECT_DRIFT`
exists: `generate.py` verifies every tower stands on its own before using it,
and throws away the ones that do not.

---

## Why the dataset uses delete mode

`delete` is the only mode that produces a usable label mix. `pull` and `push`
are implemented and work mechanically, but both destroy the tower on almost
every trial, so their labels carry very little information. This is the most
important empirical result in the project, so the evidence is recorded here.

Measured on 48 blocks across 6 towers, with `delete` run on the **same**
blocks as a control:

| mode | stable | collapse |
|---|---|---|
| delete | 67% | 33% |
| push (fingertip) | 6% | 94% |

The telling number is not the rate but the pairing: push collapsed the tower
on **29 blocks that delete removed cleanly**, and succeeded on **zero** blocks
that delete could not. Extraction never reveals a block that is safer than
delete says -- it only destroys ones that were not.

### What was tried

Six extraction mechanisms, in order:

1. 3D spring-damper grip (`xfrc_applied`) -- block droops 39 degrees, NaN
2. Axis-only force -- worse, 85% collapse
3. Force plus an attitude controller -- diverged to NaN
4. Rigid weld to a mocap handle, force budget applied after the fact -- 0% stable
5. Weld whose handle tracks the block vertically -- 0% stable
6. Compliant weld (solref swept) -- 0-6% stable

Then a fingertip that **pushes** on the end face, which is how the move is
really made. A grip can pull in any direction; a finger can only push, so it
cannot drag the tower. This was the right mechanism and did help -- median
disturbance fell from 41 cm to 11 cm -- but not enough.

### What the diagnostics ruled out

- **Friction is not the whole story.** Swept 0.45 down to 0.05 against the
  grip: no effect at all. That sweep was later invalidated (the grip was
  applying 50-67 N vertically and swamping everything). Re-run against push,
  friction 0.20 did roughly double the stable rate -- but only 1 tower in 3
  still stands at that friction, so it trades label quality for usable towers.
- **The grip was jacking the tower into the air.** Decomposing the weld force
  showed it was 100% vertical, 50-67 N, against a tower weighing 8.94 N.
  Holding a block's height rigidly means any settling onto it meets unbounded
  resistance. This is why `peak_force` was independent of friction.
- **Contact-point yaw is not the cause.** A single sphere makes point contact,
  so an off-centre normal should twist the block into its neighbours. Tested
  against a flat pad and two spread contacts: outer slots stayed at exactly
  0% for all three shapes, 26 trials each. The hypothesis was wrong.
- **Contact margin, elliptic friction cone and a 5-term solimp all made
  extraction worse**, despite margin and elliptic both improving static drift
  (0.049 and 0.052 cm against 0.110 cm baseline).

### The one real signal

Outer blocks fail at 0% while middle blocks reach 30-40%. That ordering is
correct Jenga -- the middle block of a layer is the one you can take -- and it
is the only structure any extraction mode has produced. Something about
sliding a block out of an outer slot disturbs the layer in a way that
instantaneous removal does not, and that remains unexplained.

---

## Known limitations

- **Medium vs low is the soft boundary.** High is clean — a structural rule
  gets 98% of it. Medium rests on a margin drop of a fraction of a degree,
  measured with ~0.15° resolution. It is real, but it is small. Expect any
  model to separate high from the rest much more easily than medium from low,
  and treat medium as the class to scrutinise first if the labels look wrong.
- **The tower still colours the medium label a little.** Sturdier towers have
  more medium blocks (r = +0.62 between `base_tilt_deg` and a tower's medium
  share), because a tower with more margin has more to lose. That is far
  better than the absolute threshold, which made every block in a fragile
  tower medium, but it is not zero. `base_tilt_deg` itself is a useful
  tower-level risk a finished system could report alongside the blocks.
- **Slow creep, not collapse, after about 2 seconds.** Every real collapse
  happens within 0.25 s of a removal (the collapse rate is flat from 0.25 s to
  2 s). Past that, towers creep steadily at 0.1–0.17 cm/s without
  accelerating, and a long enough watch mislabels that as collapse — at 8 s,
  52% of removals would read "collapse". That is why `OBSERVE_SECONDS` is 0.5.
  The creep itself is unexplained.
- **The tilt test measures tipping, not every way to fail.** It tilts gravity,
  which is equivalent to tilting the table. A careless hand, a knock on one
  block, or the next player's move are not modelled directly — tilt is a
  standard stand-in for "how much disturbance can this take".
- **The tilt ramp reads slightly high.** The tower is already moving before it
  has travelled the 1 cm that counts as failure, and the table keeps tilting
  meanwhile. The bias is the same for every block, so rankings are fine;
  absolute angles are a little generous. A slower `TILT_RAMP_SECONDS` reduces
  it at the cost of generation time.
- **The label mix depends heavily on `GAPS_MAX`, `JITTER_SIZE_FRAC` and the risk
  thresholds**, not on anything intrinsic to Jenga. Do not read the base rates
  as physical fact.
- **Blocks are identical apart from a 0.2% size jitter.** Real blocks are
  warped, chipped and vary in friction. That realism would change which blocks
  are loose.
- **A fraction of towers will not stand** at the default size jitter and get
  rejected during generation. That costs throughput, not correctness.
- **Simulated images are not photographs.** Randomised views help, but plain
  colours and perfect edges are still a gap. See the pipeline section for why
  stage 2 does not care, and why stage 1 needs some real photos.
- **`play.py` cannot swap the model live**, so pressing `N` closes the viewer
  and reopens it with a new tower. Its controls print to the terminal because
  MuJoCo's passive viewer has no simple custom-text API.

---

## Ideas for next

- **Train the stage-2 baseline now** — grid → risk, on `towers.csv` and
  `labels.csv` alone. The structural rule is the bar to beat on high risk; the
  interesting question is how well anything predicts medium vs low.
- **Photograph a real tower** and hand-label filled slots, to measure how well
  a sim-trained detector transfers.
- Wood-grain textures and chipped edges on the blocks, to narrow the look gap.
- Per-block friction variation, and warping rather than uniform scaling — the
  size-jitter result suggests block imperfection drives much of the loose-block
  behaviour.
- Implement `STACK_REMOVED_ON_TOP` so towers evolve like real games, rather
  than only losing blocks.
- Show the selected block's risk live in `play.py`.
- Record the tilt margin in each of the four directions, not only the weakest:
  "risky if you knock it left" is more useful than a single number.
