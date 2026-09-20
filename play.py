#!/usr/bin/env python3
"""Interactive Jenga playground.

    mjpython play.py          (macOS -- MuJoCo needs mjpython for the viewer)
    python play.py            (Linux / Windows)

Pick a block with the keys, then pull it, delete it, or shove the tower
around with the mouse. Outcomes are printed to this terminal.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from jenga_sim import camera, config as C, removal, tower as T

HIGHLIGHT = (0.15, 0.95, 0.30, 1.0)

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
    def __init__(self, seed: int, gaps: int | None, slow: float):
        self.gaps = gaps
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
        print(f"\n[build] tower seed={seed} ...", flush=True)
        self.seed = seed
        self.tw = T.build_tower(seed=seed)
        converged, secs = T.settle(self.tw)

        rng = random.Random(seed)
        n = self.gaps if self.gaps is not None else rng.randint(C.GAPS_MIN, C.GAPS_MAX)
        made = T.make_gaps(self.tw, rng, n) if n else 0

        self.base_rgba = self.tw.model.geom_rgba.copy()
        self.saved = T.save_state(self.tw)
        self.saved_gaps = list(self.tw.gap_ids)
        self.selected = None
        print(f"[build] settled={converged} in {secs:.2f}s sim | gaps={made} | "
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
                    help="sleep per physics step during a removal, so you can "
                         "watch it happen (0 = as fast as possible)")
    args = ap.parse_args()

    seed = args.seed
    while seed is not None:
        seed = Playground(seed, args.gaps, args.slow).run()
    print("bye")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "mjpython" in str(exc).lower() and sys.platform == "darwin":
            sys.exit("\nOn macOS the MuJoCo viewer must be launched with "
                     "mjpython:\n\n    .venv/bin/mjpython play.py\n")
        raise
