#!/usr/bin/env python3
"""Interactive Jenga playground.

    mjpython play.py          (macOS -- MuJoCo needs mjpython for the viewer)
    python play.py            (Linux / Windows)
    mjpython play.py --tower 3          (open tower 3 of data/big_run)

Pick a block with the keys, then pull it, delete it, tilt-test it, or shove
the tower around with the mouse. Outcomes are printed to this terminal.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

from jenga_sim import camera, config as C, removal, risk, tower as T

HIGHLIGHT = (0.15, 0.95, 0.30, 1.0)
TILT_HOLD_SECONDS = 1.5      # keep the table tilted this long after a side gives way
WHY = {"high": "it collapses, or leaves the tower about to fall",
       "medium": f"the tower survives but takes {C.RISK_MEDIUM_DROP_DEG}+ deg less tilt",
       "low": "the tower is about as stable as before"}

CONTROLS = """
------------------------------------------------------------------
  JENGA PLAYGROUND
------------------------------------------------------------------
  W / S      move the selection up / down a level
  A / D      move the selection across the layer
             the selected block turns green

  P / O      pull horizontally through opposite ends of the block
  B          pull an outer block sideways, away from its row
  X          delete the selected block instantly
  T          tilt test: tilt the table each way until the tower gives
             way, then give the risk level of the block you removed
  R          reset to the saved tower
  N          build a new random tower
  C          save a snapshot PNG into snapshots/
  Q / Esc    quit

  Mouse      left-drag orbits, right-drag pans, scroll zooms.
             Double-click a block then ctrl+left-drag to shove it.
  F1         MuJoCo's own help overlay

  Keys register while the MuJoCo window has focus.
------------------------------------------------------------------
"""


class Playground:
    def __init__(self, seed: int, gaps: int | None, slow: float,
                 state_file: Path | None = None, imperfect: bool = False):
        self.gaps = gaps
        self.imperfect = imperfect
        self.state_file = state_file
        self.slow = slow
        self.selected = None
        self.level = 8
        self.slot = 1
        self.snap_n = 0
        self.pending: list[str] = []
        self.next_seed: int | None = None
        self.frame_seconds = 1.0 / 60.0
        self.steps_per_frame = max(1, round(self.frame_seconds / C.TIME_STEP))
        self.build(seed)

    # -- tower ------------------------------------------------------------
    def build(self, seed: int):
        self.labels = {}
        if self.state_file is not None:
            self.load(self.state_file)
        else:
            print(f"\n[build] tower seed={seed} ...", flush=True)
            self.seed = seed
            self.tw = T.build_tower(seed=seed, imperfect=self.imperfect)
            converged, secs = T.settle(self.tw)

            rng = random.Random(seed)
            n = self.gaps if self.gaps is not None else rng.randint(C.GAPS_MIN, C.GAPS_MAX)
            made = T.make_gaps(self.tw, rng, n) if n else 0
            print(f"[build] settled={converged} in {secs:.2f}s sim | gaps={made} | "
                  f"{len(self.tw.present_ids())}/{self.tw.n_blocks} blocks standing",
                  flush=True)

        self.base_rgba = self.tw.model.geom_rgba.copy()
        self.saved = T.save_state(self.tw)
        self.saved_gaps = list(self.tw.gap_ids)
        self.base_margin = None
        self.selected = None

    def load(self, path: Path):
        """A tower exactly as generate.py saved it, gaps and all.

        Its tilt test and risk levels then reproduce labels.csv to the
        hundredth of a degree, so the dataset's labels are shown alongside.
        """
        with np.load(path) as z:
            self.seed = int(z["seed"])
        self.tw = T.build_tower(seed=self.seed)
        T.load_state_npz(self.tw, path)
        tower_id = int(path.stem.split("_")[1])
        labels = path.parent.parent / "labels.csv"
        if labels.exists():
            rows = pd.read_csv(labels).query("tower_id == @tower_id")
            self.labels = {int(r.block_id) - 1: (r.risk_level, r.tilt_margin_deg)
                           for r in rows.itertuples()}
        print(f"\n[build] tower {tower_id} from {path.parent.parent} | "
              f"gaps={len(self.tw.gap_ids)} | "
              f"{len(self.tw.present_ids())}/{self.tw.n_blocks} blocks standing",
              flush=True)

    # -- selection --------------------------------------------------------
    def select(self, level: int, slot: int):
        block = self.tw.find(level, slot)
        if block == self.selected:
            return
        if self.selected is not None:
            gid = self.tw.geom_id[self.selected]
            self.tw.model.geom_rgba[gid] = self.base_rgba[gid]
        self.selected = block
        if block is None:
            return
        self.tw.model.geom_rgba[self.tw.geom_id[block]] = HIGHLIGHT
        state = "GONE" if block in self.tw.gap_ids else "ready"
        print(f"[select] {self.tw.describe(block)} ({state}) | standing "
              f"{len(self.tw.present_ids())}/{self.tw.n_blocks}", flush=True)

    def _unhighlight(self):
        if self.selected is not None:
            gid = self.tw.geom_id[self.selected]
            self.tw.model.geom_rgba[gid] = self.base_rgba[gid]

    # -- actions ----------------------------------------------------------
    def do_remove(self, mode: str, viewer, direction: str = "end+"):
        block = self.selected
        if block is None:
            print("[!] no block at that level/position")
            return
        if block in self.tw.gap_ids:
            print(f"[!] {self.tw.describe(block)} is already gone")
            return
        if mode == "pull" and direction not in removal.pull_directions(self.tw, block):
            print("[!] Side pulls are available for outer blocks; use P or O for the centre.")
            return

        name = self.tw.describe(block)
        print(f"\n[{mode}] {name} ...", flush=True)
        if mode == "pull":
            print(f"  direction       : {direction} (horizontal)")
        self._unhighlight()
        t0 = time.time()
        # Throttle redraws during the removal for the same reason.
        counter = {"n": 0}

        def on_step():
            counter["n"] += 1
            if counter["n"] % self.steps_per_frame == 0:
                viewer.sync()

        out = removal.remove(self.tw, block, mode, random.Random(),
                             step_delay=self.slow, on_step=on_step, direction=direction)
        if out.outcome != "stuck":
            self.tw.gap_ids.append(block)
        self.selected = None

        print(f"  outcome         : {out.outcome.upper()}")
        print(f"  max displacement: {out.max_displacement * 100:.3f} cm")
        print(f"  max tilt        : {out.max_tilt_deg:.2f} deg")
        if mode == "pull":
            print(f"  peak pull force : {out.peak_force:.2f} N  "
                  f"({out.peak_force / C.BLOCK_WEIGHT:.0f}x the block's own weight; "
                  f"budget {C.PULL_MAX_FORCE:.2f} N)")
            print(f"  extract time    : {out.extract_time:.2f} s")
        print(f"  [{time.time() - t0:.1f}s wall]", flush=True)

    def do_tilt(self, viewer):
        """Run the dataset's tilt test where you can watch it.

        It starts from the saved tower with your removals replayed, which is
        how generate.py labels a block, rather than from the live tower: a
        tower left running in the viewer keeps settling, and over 20 s that
        alone can move the margin by a couple of degrees.
        """
        tw = self.tw
        removed = [b for b in tw.gap_ids if b not in self.saved_gaps]
        self._unhighlight()
        self.selected = None
        if self.base_margin is None:
            print("\n[tilt] measuring the untouched tower first ...", flush=True)
            tw.gap_ids = list(self.saved_gaps)
            self.base_margin = risk.tilt_margin(tw, self.saved)

        # The saved tower, then each of your removals as a clean delete.
        tw.gap_ids = list(self.saved_gaps)
        T.restore_state(tw, self.saved)
        for b in removed:
            out = removal.delete_block(tw, b)
            tw.gap_ids.append(b)
            if out.outcome == "collapse":
                viewer.sync()
                print(f"\n[tilt] removing {tw.describe(b)} collapses the tower, so "
                      f"there is nothing left to tilt.\n  risk of {tw.describe(b)}: "
                      f"HIGH ({WHY['high']})\n  press R to reset", flush=True)
                self._say(viewer, "Collapsed", f"{tw.describe(b)}: HIGH")
                return
        state = T.save_state(tw)
        what = (", ".join(tw.describe(b) for b in removed) + " removed") if removed \
            else "the untouched tower"
        print(f"\n[tilt] {what}: tilting the table each way ...", flush=True)

        tick = [time.time()]

        def show(side, angle):
            self._draw_gravity(viewer, side, angle)
            self._say(viewer, f"Tilt test, side {side + 1} of 4", f"{angle:5.2f} deg")
            viewer.sync()
            lag = self.slow * self.steps_per_frame - (time.time() - tick[0])
            if lag > 0:
                time.sleep(lag)
            tick[0] = time.time()

        steps = [0]

        def on_step(side, angle):
            steps[0] += 1
            if steps[0] % self.steps_per_frame == 0 and viewer.is_running():
                show(side, angle)

        def on_side(side, angle, failed):
            if not failed:
                print(f"  side {side + 1}: still standing at {angle:.2f} deg"
                      + ("" if angle >= C.TILT_MAX_DEG else
                         " (no need to go further: a weaker side is already known)"))
                return
            print(f"  side {side + 1}: gave way at {angle:.2f} deg", flush=True)
            # Keep the table tilted a moment so you can see what went.
            for _ in range(round(TILT_HOLD_SECONDS / self.frame_seconds)):
                if not viewer.is_running():
                    break
                T.step(tw, self.steps_per_frame)
                show(side, angle)

        margin = risk.tilt_margin(tw, state, on_step=on_step, on_side=on_side)
        self._draw_gravity(viewer)
        drop = self.base_margin - margin
        print(f"  tilt margin     : {margin:.2f} deg (its weakest side)")
        if not removed:
            self._say(viewer, "Tilt margin", f"{margin:.2f} deg")
            return
        print(f"  untouched tower : {self.base_margin:.2f} deg\n"
              f"  drop            : {drop:.2f} deg")
        if len(removed) > 1:
            print("  risk levels are for one removal at a time: press R, then "
                  "remove a single block", flush=True)
            self._say(viewer, "Tilt margin", f"{margin:.2f} deg (drop {drop:.2f})")
            return
        b = removed[0]
        level = risk.level_for(margin, drop)
        print(f"  risk of {tw.describe(b)}: {level.upper()} ({WHY[level]})")
        if b in self.labels:
            print(f"  dataset says    : {self.labels[b][0].upper()} "
                  f"({self.labels[b][1]:.2f} deg left)")
        print("  press R to reset", flush=True)
        self._say(viewer, f"{tw.describe(b)}: {level.upper()}",
                  f"drop {drop:.2f} deg, {margin:.2f} left")

    @staticmethod
    def _say(viewer, left: str, right: str):
        """Text in the top-left corner of the window, until the next key."""
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                          mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right))

    @staticmethod
    def _draw_gravity(viewer, side: int | None = None, angle: float = 0.0):
        """Two arrows beside the tower: grey is straight down, red is where
        gravity points now. Tilting gravity is how the test tilts the table."""
        scn = viewer.user_scn
        scn.ngeom = 0
        if side is None:
            return
        dx, dy = C.TILT_DIRECTIONS[side]
        th = math.radians(angle)
        top = np.array([0.10, -0.09, 0.27])     # to the right in the opening view
        for rgba, down in (((0.55, 0.55, 0.55, 0.7), (0.0, 0.0, -1.0)),
                           ((0.85, 0.15, 0.15, 1.0),
                            (math.sin(th) * dx, math.sin(th) * dy, -math.cos(th)))):
            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                                np.zeros(3), np.zeros(9), np.array(rgba, dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.003,
                                 top, top + 0.16 * np.array(down))
            scn.ngeom += 1

    def do_reset(self):
        self._unhighlight()
        self.tw.gap_ids = list(self.saved_gaps)
        T.restore_state(self.tw, self.saved)
        self.selected = None
        print(f"[reset] tower restored | "
              f"{len(self.tw.present_ids())}/{self.tw.n_blocks} standing", flush=True)

    def do_snapshot(self):
        out = Path("snapshots")
        out.mkdir(exist_ok=True)
        self._unhighlight()
        rgb, _seg = camera.render(self.tw, camera.default_camera())
        path = out / f"snap_{self.snap_n:03d}.png"
        camera.save_rgb(path, rgb)
        self.snap_n += 1
        print(f"[snap] wrote {path}", flush=True)

    # -- main loop --------------------------------------------------------
    def key_callback(self, keycode: int):
        """Runs on the viewer's thread, so just queue the work."""
        self.pending.append(chr(keycode).lower() if 32 < keycode < 127 else "")

    def run(self) -> int | None:
        """Returns the seed of the next tower to build, or None to quit."""
        print(CONTROLS)
        with mujoco.viewer.launch_passive(
                self.tw.model, self.tw.data,
                key_callback=self.key_callback,
                show_left_ui=False, show_right_ui=False) as viewer:

            tower_h = C.NUM_LAYERS * C.BLOCK_HEIGHT
            viewer.cam.lookat[:] = (0, 0, tower_h * C.CAMERA_TARGET_Z_FRAC)
            viewer.cam.distance = C.CAMERA_DISTANCE
            viewer.cam.azimuth = C.CAMERA_AZIMUTH
            viewer.cam.elevation = C.CAMERA_ELEVATION

            while viewer.is_running():
                tick = time.time()

                while self.pending:
                    key = self.pending.pop(0)
                    viewer.clear_texts()
                    if key == "q":
                        return None
                    elif key == "w":
                        self.level = min(C.NUM_LAYERS - 1, self.level + 1)
                    elif key == "s":
                        self.level = max(0, self.level - 1)
                    elif key == "d":
                        self.slot = min(C.BLOCKS_PER_LAYER - 1, self.slot + 1)
                    elif key == "a":
                        self.slot = max(0, self.slot - 1)
                    elif key == "p":
                        self.do_remove("pull", viewer, "end+")
                    elif key == "o":
                        self.do_remove("pull", viewer, "end-")
                    elif key == "b":
                        self.do_remove("pull", viewer, "side")
                    elif key == "x":
                        self.do_remove("delete", viewer)
                    elif key == "t":
                        self.do_tilt(viewer)
                    elif key == "r":
                        self.do_reset()
                    elif key == "c":
                        self.do_snapshot()
                    elif key == "n":
                        # A new tower is a new compiled model, and the viewer
                        # is bound to this one, so hand a seed back to main().
                        return random.randrange(1 << 30)

                self.select(self.level, self.slot)

                # Step a whole frame's worth of physics per redraw. Stepping
                # once per sync means a 500 Hz Python loop, which the
                # interpreter cannot hold -- the sim falls behind real time and
                # the viewer visibly stutters. Batching to ~60 FPS fixes it.
                for _ in range(self.steps_per_frame):
                    mujoco.mj_step(self.tw.model, self.tw.data)
                mujoco.mj_kinematics(self.tw.model, self.tw.data)
                viewer.sync()

                lag = self.frame_seconds - (time.time() - tick)
                if lag > 0:
                    time.sleep(lag)
        return None


def main():
    ap = argparse.ArgumentParser(description="Interactive Jenga playground")
    ap.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    ap.add_argument("--gaps", type=int, default=None,
                    help="pre-removed blocks (default: random in config range)")
    ap.add_argument("--slow", type=float, default=C.TIME_STEP,
                    help="sleep per physics step during a removal or tilt test, so "
                         "you can watch it happen (0 = as fast as possible)")
    ap.add_argument("--imperfect", action="store_true",
                    help="build slightly messy towers: a small lean, shifted layers, "
                         "a few blocks sticking out")
    ap.add_argument("--tower", type=int, default=None,
                    help="open this tower from a generate.py run instead of building one")
    ap.add_argument("--data", default="data/big_run",
                    help="the generate.py output folder --tower reads from")
    args = ap.parse_args()

    state_file = None
    if args.tower is not None:
        state_file = Path(args.data) / "states" / f"tower_{args.tower:04d}.npz"
        if not state_file.exists():
            sys.exit(f"no {state_file} -- is --data a generate.py output folder?")

    seed = args.seed
    while seed is not None:
        seed = Playground(seed, args.gaps, args.slow, state_file, args.imperfect).run()
        state_file = None          # N builds a fresh random tower
    print("bye")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "mjpython" in str(exc).lower() and sys.platform == "darwin":
            sys.exit("\nOn macOS the MuJoCo viewer must be launched with "
                     "mjpython:\n\n    .venv/bin/mjpython play.py\n")
        raise
