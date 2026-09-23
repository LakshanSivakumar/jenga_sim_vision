# Terminal guide

Every command assumes you are in the project root.

Nothing here needs the venv "activated" -- each command calls the venv's own
interpreter directly, which is harder to get wrong. If you prefer activating,
use `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\Activate.ps1`
(Windows) and then plain `python`.

**Two paths differ by platform**, and that is the only difference in the whole
guide. Everywhere below you see `.venv/bin/python`, Windows users type
`.venv\Scripts\python.exe`:

| | macOS / Linux | Windows |
|---|---|---|
| run a script | `.venv/bin/python X.py` | `.venv\Scripts\python.exe X.py` |
| the viewer | `.venv/bin/mjpython play.py` | `.venv\Scripts\python.exe play.py` |

---

## 0. One-time setup

Only needed on a fresh clone or a new machine. Python 3.11 is what this was
built and tested against.

### macOS / Linux

    uv venv --python 3.11 .venv
    uv pip install --python .venv/bin/python -r requirements.txt

macOS only, and only for `play.py`:

    ./tools/fix_mjpython_uv.sh .venv

MuJoCo's viewer must be launched with `mjpython` on macOS. `mjpython` is a
signed app bundle that loads your venv's Python and needs a shared
`libpython`. uv's Python ships that library but does not put it on the search
path, and `DYLD_LIBRARY_PATH` cannot fix it because macOS strips `DYLD_*` for
signed binaries. The script copies the library where `mjpython` already looks.

### Windows (PowerShell)

If you do not have `uv`:

    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

Then:

    uv venv --python 3.11 .venv
    uv pip install --python .venv\Scripts\python.exe -r requirements.txt

There is no `mjpython` step on Windows -- the MuJoCo viewer runs under plain
`python`, which makes `play.py` simpler there than on a Mac.

If you would rather not use `uv`, install Python 3.11 from python.org and:

    py -3.11 -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt

### Check it worked (either platform)

    .venv/bin/python -c "import mujoco; print(mujoco.__version__)"

Windows:

    .venv\Scripts\python.exe -c "import mujoco; print(mujoco.__version__)"

Expect `3.13.0` or newer.

---

## 1. Run the tests

    .venv/bin/python -m pytest -q

Takes about 3 minutes -- most of it is simulating towers for real. The
headline test is `test_untouched_tower_stands_for_ten_seconds`.

Run one test, with print output visible:

    .venv/bin/python -m pytest tests/test_stability.py -k untouched -s

Just the risk labelling (grid, tilt test, risk levels), about 40 seconds:

    .venv/bin/python -m pytest tests/test_risk.py -q

Two tests in `tests/test_pull_directions.py` currently fail. They assert that
side-pulling blocks 48 and 50 leaves the tower standing, which the physics
does not do; pull mode is not used for the dataset.

---

## 2. Play with a tower (interactive)

macOS -- note `mjpython`, not `python`:

    .venv/bin/mjpython play.py

Windows / Linux -- plain python is correct:

    .venv\Scripts\python.exe play.py

| key | does |
|---|---|
| `W` / `S` | move selection up / down a level |
| `A` / `D` | move selection across the layer |
| `P` / `O` | pull horizontally out of either end |
| `B` | pull an outer block sideways |
| `X` | delete the block instantly |
| `R` | reset to the saved tower |
| `N` | new random tower |
| `C` | snapshot PNG into `snapshots/` |
| `Q` | quit |

Mouse: left-drag orbits, right-drag pans, scroll zooms. Double-click a block
then ctrl+left-drag to shove it around. `F1` shows MuJoCo's own overlay.
Keys only register while the MuJoCo window has focus.

Options:

    .venv/bin/mjpython play.py --seed 7 --gaps 10 --slow 0

- `--seed N` pick a specific tower (same seed = same tower, always)
- `--gaps N` how many blocks are pre-removed (default: random 4-14)
- `--slow 0` run removals at full speed instead of watchable speed

---

## 3. Generate a dataset

    .venv/bin/python generate.py --towers 20 --out data/my_run

For every tower: build it, knock random gaps in it, photograph it from several
angles, then remove **every** block in turn and measure its risk. Budget about
a minute of wall time per tower with the default workers.

| flag | default | meaning |
|---|---|---|
| `--towers N` | 20 | how many towers to build |
| `--views N` | 4 | photos per tower; view 0 is the fixed reference shot, the rest randomise camera, lighting and colours |
| `--workers N` | half your cores | parallel processes; `1` = serial |
| `--seed N` | 0 | changes every tower in the run |
| `--out DIR` | `data/` | where to write |
| `--resume` | off | finish an interrupted run in `--out`, or add more towers to a finished one |

A big overnight run, kept awake with `caffeinate` so the Mac cannot sleep
until it finishes (macOS only; the lid must stay open):

    caffeinate -ims .venv/bin/python generate.py --towers 300 --views 6 --out data/big_run

Each tower is saved to `<out>/parts/` the moment it finishes, and the CSVs are
rebuilt from those files at the end. If a run is interrupted -- a crash, a
restart, a closed lid -- nothing already finished is lost. Pick up where it
stopped with the **same command plus `--resume`**:

    caffeinate -ims .venv/bin/python generate.py --towers 300 --views 6 --out data/big_run --resume

`--resume` refuses to run if `--seed` or `--views` differ from the original, so
two different runs can never be mixed into one dataset. Raising `--towers` with
`--resume` extends a finished run. Without `--resume`, `generate.py` refuses to
write into a folder that already holds a run.

Reproducibility: a tower is fully determined by its seed, so `--workers` only
changes speed; the CSVs come out identical whatever it is set to. Changing
`--views` adds or removes photos but never changes the labels or the earlier
views.

Debugging a crash -- one tower, serial, so the traceback is readable:

    .venv/bin/python generate.py --towers 1 --views 1 --workers 1 --out /tmp/scratch_run

The summary at the end prints the low / medium / high split, how much margin
surviving removals cost, and how often the structural rule agrees with the
physics. That
last number should sit around 98%; if it drops sharply, something in the
physics has changed.

---

## 4. Look at the labels

Whole towers, every block coloured green / amber / red by risk:

    .venv/bin/python inspect_data.py data/my_run

One highlighted block per panel, evenly across the three levels:

    .venv/bin/python inspect_data.py data/my_run --blocks --balanced --n 30 --cols 6

Writes `data/my_run/sample_grid.png` and prints the label statistics.

| flag | meaning |
|---|---|
| `--blocks` | one block per panel instead of whole towers |
| `--balanced` | blocks mode: sample evenly across low / medium / high |
| `--legal-only` | blocks mode: skip top-layer blocks, which are not legal moves |
| `--view N` | which photo of each tower to show (default: a random one) |
| `--n N`, `--cols N` | panels, and how many per row |
| `--medium-drop X` | preview a different medium/low threshold: degrees of margin a removal must cost |
| `--high-below X` | preview a different high/medium threshold |
| `--out FILE` | write the PNG somewhere else |

The two threshold flags re-bin from the stored margins on the fly, so you can
try a split before committing to it -- nothing is regenerated.

---

## 5. Read the data yourself

What a run produces:

    data/my_run/
      images/tower_0003_v0.png        photo, view 0
      masks/tower_0003_v0_seg.png     pixel -> block index 1-54, 0 = none
      risk/tower_0003_v0_risk.png     pixel -> 0 none, 1 low, 2 medium, 3 high
      states/tower_0003.npz           physics state
      towers.csv                      one row per tower, incl. presence grid
      labels.csv                      one row per block
      views.csv                       one row per image
      visibility.csv                  pixels of each block in each image

The risk level split:

    .venv/bin/python -c "import pandas as pd; d=pd.read_csv('data/my_run/labels.csv'); print(d.risk_level.value_counts())"

In Python -- everything the two model stages need:

```python
import pandas as pd
from jenga_sim.dataset import load_block_mask

towers = pd.read_csv("data/my_run/towers.csv", dtype={"grid": str})
labels = pd.read_csv("data/my_run/labels.csv")

# Stage 2 (grid -> risk): a tower's grid, plus every block's label
grid = towers.set_index("tower_id").grid[3]      # '1' present / '0' gap, 54 chars
blocks = labels[labels.tower_id == 3]             # block_id is 1-based: grid[block_id - 1]

# Stage 1 (photo -> blocks): one block's pixels in one photo
mask = load_block_mask("data/my_run", tower_id=3, view=0, block_id=17)
```

Two things to remember when training:

- **Drop hidden block/view pairs** -- `visibility.csv` rows with `pixels == 0`.
  The block is invisible in that photo, so nothing can be learned from it.
- **Train on the continuous columns, bin at the end.** `tilt_margin_deg` and
  `margin_drop_deg` carry more information than the three levels, and let you
  move the thresholds without retraining.

---

## 6. Change the physics

Everything tunable is in `jenga_sim/config.py`, in real SI units. A block is
16.9 g, the tower is 27 cm tall, forces are newtons.

    open jenga_sim/config.py          # macOS
    notepad jenga_sim\config.py       # Windows

The ones worth touching, with their measured trade-offs written next to them:

- `RISK_MEDIUM_DROP_DEG` / `RISK_HIGH_BELOW_DEG` -- how much margin a removal
  must cost to be medium, and how little margin left is high. Preview first
  with `inspect_data.py --medium-drop`.
- `JITTER_SIZE_FRAC` -- how much blocks vary in size. Controls how many blocks
  are load-free, and how many towers stay standing. Read the table first.
- `GAPS_MIN` / `GAPS_MAX` -- blocks pre-removed before trials. This is what
  produces high-risk labels at all; pristine towers almost never fall.
- `VIEWS_PER_TOWER` -- photos per tower. Cheap: only rendering.
- `TILT_RAMP_SECONDS` -- slower is more accurate and slower to generate.

After changing anything physical, re-run the tests before generating data:

    .venv/bin/python -m pytest -q -k "stands or sink or contact_forces"

---

## 7. Use it as a library

    .venv/bin/python

```python
from jenga_sim import config as C, tower as T, removal as R

tw = T.build_tower(seed=1)          # compile a tower
T.settle(tw)                        # let it come to rest
state = T.save_state(tw)            # cheap snapshot (qpos + qvel)

out = R.delete_block(tw, tw.find(level=8, slot=1))
print(out.outcome, out.max_displacement)

T.restore_state(tw, state)          # put it back, exactly

from jenga_sim import risk
base = risk.tilt_margin(tw, state)                # the tower's own margin
r = risk.assess_block(tw, tw.find(level=8, slot=1), state, base)
print(r.risk_level, r.tilt_margin_deg, r.margin_drop_deg)
print(risk.grid_string(tw))                       # the presence grid
```

`R.push_block` exists and works mechanically but is not exposed on the
command line, because its labels are 94% collapse. It is kept for reference.

---

## Troubleshooting

**`play.py` exits with `Library not loaded: @rpath/libpython3.11.dylib`**
Run `./tools/fix_mjpython_uv.sh .venv`. See section 0.

**`launch_passive requires mjpython`**
You ran `python play.py`. Use `.venv/bin/mjpython play.py`.

**`--workers` crashes on Windows, or spawns endless processes**
It should not: `generate.py` sets the `spawn` start method explicitly and is
guarded by `if __name__ == "__main__"`, which is exactly what Windows needs.
If you copy the generation code into a notebook or a script without that
guard, you will get runaway processes -- keep the guard.

**Generation is slow**
Check `--workers`. Physics is single-threaded CPU work and the GPU is not used
for it at all, so worker count is the only real lever. Try `--workers 4` as
well as the default -- on Apple Silicon, keeping every worker on a performance
core can beat using more of them.

**A run prints `[reject] tower N would not stand`**
Normal. Towers that drift on their own are thrown away rather than used. If
almost every tower is rejected, `JITTER_SIZE_FRAC` is too high.

**A risk level is missing, or the split looks lopsided**
The summary at the end of a run tells you which knob to turn. For medium vs
low, set `RISK_MEDIUM_DROP_DEG` to a percentile of the margin drop printed in
the summary -- and preview it with `inspect_data.py --medium-drop` before
regenerating anything.

**`inspect_data.py` says there is no labels.csv, or complains about columns**
It reads the current output format only. Folders written before the risk
levels were added (per-block `mask` files, a `mode` column) will not load --
regenerate them.
