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

Takes about 2.5 minutes -- most of it is simulating towers for real. The
headline test is `test_untouched_tower_stands_for_ten_seconds`.

Run one test, with print output visible:

    .venv/bin/python -m pytest tests/test_stability.py -k untouched -s

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

The normal run. Delete mode is the default and the only mode whose labels are
trustworthy -- see "Why the dataset uses delete mode" in the README.

    .venv/bin/python generate.py --towers 20 --out data/my_run

All options:

| flag | default | meaning |
|---|---|---|
| `--towers N` | 20 | how many towers to build |
| `--mode` | `delete` | `delete` or `pull` |
| `--candidates N` | 8 | blocks tried per tower; `all` for every legal block |
| `--workers N` | half your cores | parallel processes; `1` = serial |
| `--seed N` | 0 | changes every tower in the run |
| `--out DIR` | `data/` | where to write |
| `--pull-direction` | `end` | `end`, `side` or `all`; only with `--mode pull` |

A bigger run, every block, all cores:

    .venv/bin/python generate.py --towers 100 --candidates all --workers 8 --out data/big_run

Reproducibility: a tower is fully determined by its seed, so `--workers` only
changes speed. The CSVs come out byte-identical whether you use 1 worker or 8.
Different `--seed` gives different towers.

Debugging a crash -- run serially so the traceback is readable:

    .venv/bin/python generate.py --towers 2 --candidates 2 --workers 1 --out /tmp/scratch_run

---

## 4. Look at the labels

    .venv/bin/python inspect_data.py data/my_run --balanced

Writes `data/my_run/sample_grid.png`: sample towers with the candidate block
tinted and outlined, green `stable`, red `collapse`, yellow `stuck`. Prints
per-outcome statistics to the terminal too.

| flag | meaning |
|---|---|
| `--balanced` | sample evenly across outcomes, so rare labels still appear |
| `--n N` | how many samples in the grid (default 12) |
| `--cols N` | grid width (default 4) |
| `--out FILE` | write the PNG somewhere else |

---

## 5. Read the data yourself

    .venv/bin/python -c "import pandas as pd; d=pd.read_csv('data/my_run/labels.csv'); print(d['outcome'].value_counts()); print(d.head())"

What a run produces:

    data/my_run/
      images/tower_0003.png             RGB before the removal, 512x512
      masks/tower_0003_seg.png          segmentation, pixel value = block index
      masks/tower_0003_block_17.png     binary mask of the candidate block
      states/tower_0003.npz             qpos + qvel, the full sim state
      labels.csv                        one row per trial
      towers.csv                        one row per tower
      sample_grid.png                   written by inspect_data.py

When training, drop rows where `mask_pixels == 0` -- the candidate block is
completely hidden from the camera in those, so they cannot be learned from the
image.

---

## 6. Change the physics

Everything tunable is in `jenga_sim/config.py`, in real SI units. A block is
16.9 g, the tower is 27 cm tall, forces are newtons.

    open jenga_sim/config.py          # macOS
    notepad jenga_sim\config.py       # Windows

The three worth touching, with their measured trade-offs written next to them:

- `JITTER_SIZE_FRAC` -- how much blocks vary in size. Controls how many blocks
  are load-free, and how many towers stay standing. Read the table first.
- `GAPS_MIN` / `GAPS_MAX` -- blocks pre-removed before trials. This is what
  produces collapse labels at all; pristine towers almost never fall.
- `OBSERVE_SECONDS` -- how long the tower is watched after a removal.
  Directly sets how long generation takes.

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

**A label class is missing**
The summary at the end of a run tells you which knob to turn. `stuck` can only
ever appear in pull mode.
