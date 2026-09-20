#!/usr/bin/env python3
"""Interactive Jenga playground.

    python play.py

Pick a block with the sliders, then pull it, delete it, or just shove the
tower around with the mouse. Outcomes are printed to this terminal.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import pybullet as p

from jenga_sim import config as C
from jenga_sim import camera, removal, tower as T

HIGHLIGHT = [0.15, 0.95, 0.30, 1.0]

HUD_LINES = [
    "W/S level   A/D position",
    "P  pull selected block",
    "X  delete selected block",
    "R  reset tower",
    "N  new random tower",
    "C  snapshot png",
    "Q  quit",
]

CONTROLS = """
------------------------------------------------------------------
  JENGA PLAYGROUND
------------------------------------------------------------------
  Sliders (top-left panel)   choose Level (0 = bottom) and Position (0/1/2)
  W / S                      move selection up / down a level
  A / D                      move selection across the layer
                             the selected block turns green
  Buttons (same panel)       same actions as the keys below
  Mouse drag                 grab and shove blocks directly

  P    pull the selected block out slowly (force-limited constraint)
  X    delete the selected block instantly
  R    reset to the saved tower
  N    build a new random tower
  C    save a snapshot PNG into snapshots/
  Q    quit

  Keys only register while the PyBullet window has focus.
------------------------------------------------------------------
"""


class Playground:
    def __init__(self, seed: int, gaps: int | None, slow: float):
        self.cli = T.connect(gui=True)
        self.gaps = gaps
        self.slow = slow
        self.selected = None
        self.snap_n = 0
        self.level = 8
        self.slot = 1
        self._slider_level = None
        self._slider_slot = None

        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.cli)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self.cli)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self.cli)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self.cli)

        self.new_tower(seed)

    # -- widgets ----------------------------------------------------------
    def _make_widgets(self):
        """Sliders, buttons, and a STATIC on-screen control list.

        The text must never be updated after creation: in this PyBullet build,
        replacing (or removing) a debug text item silently invalidates the
        user debug PARAMETER handles, and the sliders stop reading a few
        frames later. So live status goes to the terminal instead.
        """
        self.s_level = p.addUserDebugParameter("Level", 0, C.NUM_LAYERS - 1, 8,
                                               physicsClientId=self.cli)
        self.s_slot = p.addUserDebugParameter("Position", 0, C.BLOCKS_PER_LAYER - 1, 1,
                                              physicsClientId=self.cli)
        self.b_pull = p.addUserDebugParameter("PULL block", 1, 0, 0, physicsClientId=self.cli)
        self.b_del = p.addUserDebugParameter("DELETE block", 1, 0, 0, physicsClientId=self.cli)
        self.b_reset = p.addUserDebugParameter("RESET tower", 1, 0, 0, physicsClientId=self.cli)
        self.b_new = p.addUserDebugParameter("NEW random tower", 1, 0, 0, physicsClientId=self.cli)
        self.b_snap = p.addUserDebugParameter("SNAPSHOT png", 1, 0, 0, physicsClientId=self.cli)

        top = C.NUM_LAYERS * C.BLOCK_HEIGHT
        span = C.BLOCK_LENGTH
        for i, line in enumerate(HUD_LINES):
            p.addUserDebugText(
                line, [span, -span, top * 1.15 - i * C.BLOCK_HEIGHT * 1.6],
                textColorRGB=[0.1, 0.1, 0.1], textSize=1.2,
                physicsClientId=self.cli)
        self._sync_buttons()

    def _read_param(self, handle, fallback):
        """Read a slider/button, tolerating PyBullet's rare read race.

        With ~50 bodies on screen the GUI render thread occasionally loses a
        read (about 1 in 10,000). It recovers on the next frame, so reusing
        the previous value is all that is needed -- but letting the exception
        escape would kill the session.
        """
        try:
            return p.readUserDebugParameter(handle, physicsClientId=self.cli)
        except p.error:
            return fallback

    def _sync_buttons(self):
        self.prev = {k: p.readUserDebugParameter(h, physicsClientId=self.cli)
                     for k, h in (("pull", self.b_pull), ("del", self.b_del),
                                  ("reset", self.b_reset), ("new", self.b_new),
                                  ("snap", self.b_snap))}

    def _button(self, key, handle) -> bool:
        v = self._read_param(handle, self.prev[key])
        hit = v != self.prev[key]
        self.prev[key] = v
        return hit

    # -- tower ------------------------------------------------------------
    def new_tower(self, seed: int):
        print(f"\n[build] tower seed={seed} ...", flush=True)
        self.seed = seed
        self.tower = T.build_tower(self.cli, seed=seed)
        conv, secs = T.settle(self.cli, self.tower.block_ids)

        rng = random.Random(seed)
        n = self.gaps if self.gaps is not None else rng.randint(C.GAPS_MIN, C.GAPS_MAX)
        made = T.make_gaps(self.cli, self.tower, rng, n) if n else 0

        print(f"[build] settled={conv} in {secs:.2f}s sim | gaps={made} | "
              f"{len(self.tower.present_ids())}/{len(self.tower.block_ids)} blocks standing")
        self.selected = None
        self.saved_state = p.saveState(physicsClientId=self.cli)
        self._original_gaps = list(self.tower.gap_ids)

        # build_tower calls resetSimulation, which destroys every debug widget.
        self._make_widgets()

        tower_h = C.NUM_LAYERS * C.BLOCK_HEIGHT
        p.resetDebugVisualizerCamera(
            cameraDistance=C.CAMERA_DISTANCE, cameraYaw=C.CAMERA_YAW,
            cameraPitch=C.CAMERA_PITCH,
            cameraTargetPosition=[0, 0, tower_h * C.CAMERA_TARGET_Z_FRAC],
            physicsClientId=self.cli)

    # -- selection --------------------------------------------------------
    def _unhighlight(self):
        if self.selected is not None and self.selected in self.tower.colour_of:
            p.changeVisualShape(self.selected, -1,
                                rgbaColor=self.tower.colour_of[self.selected],
                                physicsClientId=self.cli)

    def select(self, level: int, slot: int):
        body = self.tower.find(level, slot)
        if body == self.selected:
            return
        self._unhighlight()
        self.selected = body
        if body is None:
            return
        p.changeVisualShape(body, -1, rgbaColor=HIGHLIGHT, physicsClientId=self.cli)
        state = "GONE" if body in self.tower.gap_ids else "ready"
        print(f"[select] {self.tower.describe(body)} ({state}) | standing "
              f"{len(self.tower.present_ids())}/{len(self.tower.block_ids)}", flush=True)

    def _usable(self):
        if self.selected is None:
            print("[!] no block at that level/position")
            return None
        if self.selected in self.tower.gap_ids:
            print(f"[!] {self.tower.describe(self.selected)} is already gone")
            return None
        return self.selected

    # -- actions ----------------------------------------------------------
    def do_remove(self, mode: str):
        body = self._usable()
        if body is None:
            return
        name = self.tower.describe(body)
        print(f"\n[{mode}] {name} ...", flush=True)
        self._unhighlight()
        t0 = time.time()
        out = removal.remove(self.cli, self.tower, body, mode,
                             random.Random(), step_delay=self.slow)
        self.tower.gap_ids.append(body)
        self.selected = None
        print(f"  outcome         : {out.outcome.upper()}")
        print(f"  max displacement: {out.max_displacement:.4f} world units "
              f"({out.max_displacement / C.SCALE * 100:.3f} cm real-equivalent)")
        print(f"  max tilt        : {out.max_tilt_deg:.2f} deg")
        if mode == "pull":
            print(f"  peak pull force : {out.peak_force:.0f} N "
                  f"(budget {C.PULL_MAX_FORCE:.0f} N, block weighs "
                  f"{C.BLOCK_MASS * 9.81:.0f} N)")
            print(f"  extract time    : {out.extract_time:.2f} s")
        print(f"  [{time.time() - t0:.1f}s wall]", flush=True)

    def do_reset(self):
        p.restoreState(stateId=self.saved_state, physicsClientId=self.cli)
        self._unhighlight()
        self.tower.gap_ids = list(self._original_gaps)
        self.selected = None
        print(f"[reset] tower restored | "
              f"{len(self.tower.present_ids())}/{len(self.tower.block_ids)} standing",
              flush=True)

    def do_snapshot(self):
        out = Path("snapshots")
        out.mkdir(exist_ok=True)
        rgb, _seg = camera.render(self.cli, camera.default_camera())
        path = out / f"snap_{self.snap_n:03d}.png"
        camera.save_rgb(path, rgb)
        self.snap_n += 1
        print(f"[snap] wrote {path}", flush=True)

    # -- main loop --------------------------------------------------------
    def run(self):
        print(CONTROLS)
        while True:
            if not p.getConnectionInfo(self.cli).get("isConnected"):
                break

            # Sliders and keys both drive the selection: whichever moved last
            # wins, so the slider does not fight the WASD keys.
            sl = self._read_param(self.s_level, None)
            if sl is not None:
                sl = int(round(sl))
                if sl != self._slider_level:
                    self._slider_level, self.level = sl, sl
            ss = self._read_param(self.s_slot, None)
            if ss is not None:
                ss = int(round(ss))
                if ss != self._slider_slot:
                    self._slider_slot, self.slot = ss, ss

            keys = p.getKeyboardEvents(physicsClientId=self.cli)

            def pressed(ch):
                k = ord(ch)
                return k in keys and keys[k] & p.KEY_WAS_TRIGGERED

            if pressed('w'):
                self.level = min(C.NUM_LAYERS - 1, self.level + 1)
            elif pressed('s'):
                self.level = max(0, self.level - 1)
            elif pressed('d'):
                self.slot = min(C.BLOCKS_PER_LAYER - 1, self.slot + 1)
            elif pressed('a'):
                self.slot = max(0, self.slot - 1)

            self.select(self.level, self.slot)

            if pressed('q'):
                break
            if pressed('p') or self._button("pull", self.b_pull):
                self.do_remove("pull")
            elif pressed('x') or self._button("del", self.b_del):
                self.do_remove("delete")
            elif pressed('r') or self._button("reset", self.b_reset):
                self.do_reset()
            elif pressed('n') or self._button("new", self.b_new):
                self.new_tower(random.randrange(1 << 30))
            elif pressed('c') or self._button("snap", self.b_snap):
                self.do_snapshot()

            p.stepSimulation(physicsClientId=self.cli)
            time.sleep(C.TIME_STEP)

        try:
            p.disconnect(self.cli)
        except p.error:
            pass
        print("bye")


def main():
    ap = argparse.ArgumentParser(description="Interactive Jenga playground")
    ap.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    ap.add_argument("--gaps", type=int, default=None,
                    help="pre-removed blocks (default: random in config range)")
    ap.add_argument("--slow", type=float, default=C.TIME_STEP,
                    help="sleep per step during a removal, so you can watch "
                         "(0 = as fast as possible)")
    args = ap.parse_args()
    Playground(args.seed, args.gaps, args.slow).run()


if __name__ == "__main__":
    main()
