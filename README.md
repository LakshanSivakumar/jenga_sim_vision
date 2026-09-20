# Jenga Risk Vision — Physics Simulator (v0.1)

Generates training data for a computer-vision model that predicts, per Jenga
block, how risky it is to remove. There is no real dataset with per-block
removal outcomes, so we make one: build a tower in MuJoCo, photograph it, then
try removing each block and record what happened.

This is v0.1 — the simulator and data generator only. No models yet.

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
python generate.py --towers 20 --mode delete --candidates 8 --out data/
python generate.py --towers 20 --mode pull   --candidates 8 --out data/
```

`--candidates all` tries every legal block instead of a sample. Prints a label
breakdown and average time per trial at the end, and names the config knob to
turn if a label is missing.

### `inspect_data.py` — sanity-check by eye

```bash
python inspect_data.py data/ --balanced --n 12
```

Writes `data/sample_grid.png`: sample towers with the candidate block tinted
and outlined — green `stable`, red `collapse`, yellow `stuck` — plus
per-outcome statistics in the terminal.

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

## Label definitions

| Label | Means |
|---|---|
| `collapse` | any **other** block moved more than `COLLAPSE_DISPLACEMENT` (default 1 cm) **or** tilted more than `COLLAPSE_TILT_DEG` (default 10°), measured against its pose immediately before the removal |
| `stuck` | pull mode only: the block did not reach the extraction distance within the time budget while respecting `PULL_MAX_FORCE` |
| `stable` | the block came out and nothing else moved past those thresholds |

Continuous values recorded alongside the label:

- `max_displacement` — furthest any other block moved, in metres
- `max_tilt_deg` — largest orientation change of any other block
- `peak_force` — highest applied axial pulling force, in newtons (excluding
  weight support and grip torque). **This is the
  "ease of pulling" measure**, and the basis for "tempting" later: a block
  with low `peak_force` but high collapse risk is a trap.
- `extract_time` — seconds to pull the block clear

---

## Removal modes

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
data/
  images/tower_0003.png                RGB before the removal, 512x512
  masks/tower_0003_seg.png             segmentation, pixel value = block index
  masks/tower_0003_block_17.png        binary mask of the candidate block
  states/tower_0003.npz                saved sim state (qpos + qvel)
  labels.csv                           one row per (tower, block) trial
  towers.csv                           one row per tower
```

`labels.csv`: `tower_id, block_id, level, position_in_level, mode, outcome,
max_displacement, max_tilt_deg, peak_force, extract_time, mask_pixels, seed,
camera_params`

`towers.csv`: `tower_id, seed, num_blocks, num_gaps, settled, settle_seconds,
camera_params`

In `*_seg.png` the pixel value is the **block index** (1–54, 0 = background),
not the raw MuJoCo geom id — geom ids are an implementation detail, block
indices are stable. `mask_pixels` is 0 when the candidate block is completely
hidden from the camera; those rows are unlearnable from the image alone, so
filter them out when training.

Saved files contain block poses, velocities, the seed and gap indices.
Restoration resets solver warm-start data and reapplies the saved gaps'
collision and gravity flags so one removal trial cannot contaminate the next.
Reproduction requires the same simulator version and configuration.

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

3. **`PULL_MAX_FORCE`** (default 25x block weight ≈ 4.14 N) — the actual
   axial force cap. Lower it for gentler attempts and potentially more `stuck`
   outcomes. `PULL_MAX_TORQUE` separately caps the attitude-control torque.

Also worth knowing: `COLLAPSE_DISPLACEMENT` / `COLLAPSE_TILT_DEG` define the
labels themselves; `TIME_STEP` trades speed against stability;
`OBSERVE_SECONDS` directly controls how long generation takes;
`RANDOMISE_CAMERA` and `RANDOMISE_LIGHTING` are off by default and exist for
sim-to-real later.

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

- **`extract_time` remains close to distance ÷ speed** for loose blocks, but
  increases when friction causes the compliant grip to lag. A `stuck` row
  records the time budget. Use force and outcome together when assessing risk.
- **`stuck` cannot occur in delete mode** — it is a pull-only label.
- **The label mix depends heavily on `GAPS_MAX` and `JITTER_SIZE_FRAC`**, not
  on anything intrinsic to Jenga. Do not read the base rates as physical fact.
- **A fraction of towers will not stand** at the default size jitter and get
  rejected during generation. That costs throughput, not correctness.
- **Pull mode is slower than delete mode**, because several seconds of
  physical motion are integrated with finer contact substeps.
- **The grip is an idealised hand**, not a simulated pair of fingers. It
  supports the block's own weight at its centre and applies bounded torque;
  finger contact geometry and tactile probing are not modelled.
- **Pull and delete labels can differ.** Pulling transmits friction and can
  move neighbouring blocks; deletion cannot. `collapse` includes sliding past
  the displacement threshold, even when the tower remains upright.
- **No on-screen control overlay in `play.py`.** MuJoCo's passive viewer has no
  simple custom-text API, so the controls print to the terminal and live status
  goes there too. MuJoCo's own overlay is on `F1`.
- **`play.py` cannot swap the model live**, so pressing `N` closes the viewer
  and reopens it with a new tower.
- **No sim-to-real work yet.** Camera and lighting randomisation are
  implemented but off.

---

## Ideas for v0.2

- Turn on `RANDOMISE_CAMERA` / `RANDOMISE_LIGHTING`, add wood textures, then
  test on real photographs.
- Per-block friction variation, and warping rather than uniform scaling — the
  size-jitter result suggests block imperfection drives most of the
  interesting behaviour.
- Implement `STACK_REMOVED_ON_TOP` so towers evolve like real games.
- Record the full force–time curve per pull, not just the peak. The shape
  (static-friction spike vs sustained drag) should separate "wedged" from
  "heavy" blocks.
- Define "tempting" concretely: low `peak_force` percentile AND `collapse`
  outcome, then measure how common such blocks actually are.
- Hand-crafted features (blocks above, gaps in the layer, load carried,
  centre-of-mass offset) for a Random Forest baseline before going to a CNN.
  Note that `load carried` is directly measurable in sim but not from a photo —
  which is exactly the thing the vision model has to infer.
- Parallelise generation across processes; each tower is independent.
