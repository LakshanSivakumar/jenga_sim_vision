"""Writing towers, views and per-block risk labels to disk.

Layout produced:
    data/
      images/tower_0003_v0.png          RGB photo, view 0 (the fixed 3/4 shot)
      images/tower_0003_v1.png          ... more views, randomised camera/light
      masks/tower_0003_v0_seg.png       pixel -> block index (1-54), 0 = none
      risk/tower_0003_v0_risk.png       pixel -> 0 none, 1 low, 2 medium, 3 high
      states/tower_0003.npz             the tower's physics state (qpos + qvel)
      towers.csv                        one row per tower, incl. presence grid
      labels.csv                        one row per block in every tower
      views.csv                         one row per rendered image
      visibility.csv                    pixels of each block in each view

A single block's mask is `seg == block_id`, so per-block mask files are not
written: at 54 blocks and several views per tower they would be hundreds of
files per tower carrying nothing the seg map does not.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

TOWER_FIELDS = [
    "tower_id", "seed", "num_blocks", "num_gaps", "grid", "base_tilt_deg",
    "settle_seconds",
]

LABEL_FIELDS = [
    "tower_id", "block_id", "level", "position_in_level", "legal",
    "outcome", "max_displacement", "max_tilt_deg",
    "tilt_margin_deg", "margin_drop_deg", "risk_level", "seed",
]

VIEW_FIELDS = ["tower_id", "view", "camera_params", "look_params"]

VISIBILITY_FIELDS = ["tower_id", "view", "block_id", "pixels"]


class DatasetWriter:
    def __init__(self, root: str | Path, write_csv: bool = True):
        """`write_csv=False` gives a paths-only view of the dataset.

        Worker processes need to know where to save images and masks, but must
        not open (and truncate) the CSVs -- only the parent writes those.
        """
        self.root = Path(root)
        self.images = self.root / "images"
        self.masks = self.root / "masks"
        self.risk = self.root / "risk"
        self.states = self.root / "states"
        for d in (self.images, self.masks, self.risk, self.states):
            d.mkdir(parents=True, exist_ok=True)

        self.write_csv = write_csv
        if not write_csv:
            return

        self._files = {}
        self._writers = {}
        for name, fields in (("towers", TOWER_FIELDS), ("labels", LABEL_FIELDS),
                             ("views", VIEW_FIELDS), ("visibility", VISIBILITY_FIELDS)):
            f = (self.root / f"{name}.csv").open("w", newline="")
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            self._files[name] = f
            self._writers[name] = w

    # -- paths ------------------------------------------------------------
    @staticmethod
    def tower_name(tower_id: int) -> str:
        return f"tower_{tower_id:04d}"

    def image_path(self, tower_id: int, view: int = 0) -> Path:
        return self.images / f"{self.tower_name(tower_id)}_v{view}.png"

    def seg_path(self, tower_id: int, view: int = 0) -> Path:
        return self.masks / f"{self.tower_name(tower_id)}_v{view}_seg.png"

    def risk_path(self, tower_id: int, view: int = 0) -> Path:
        return self.risk / f"{self.tower_name(tower_id)}_v{view}_risk.png"

    def state_path(self, tower_id: int) -> Path:
        # qpos + qvel is the complete state of a world of free bodies, so a
        # small .npz replaces any engine-specific state blob.
        return self.states / f"{self.tower_name(tower_id)}.npz"

    # -- rows -------------------------------------------------------------
    def write(self, table: str, **row) -> None:
        self._writers[table].writerow(row)

    def close(self) -> None:
        if not self.write_csv:
            return
        for f in self._files.values():
            f.close()


# ---------------------------------------------------------------------------
# Reading helpers, for training code and inspect_data.py
# ---------------------------------------------------------------------------

def load_block_mask(root: str | Path, tower_id: int, view: int, block_id: int) -> np.ndarray:
    """Boolean mask of one block (block_id is 1-based, as in labels.csv)."""
    name = DatasetWriter.tower_name(tower_id)
    seg = np.array(Image.open(Path(root) / "masks" / f"{name}_v{view}_seg.png"))
    return seg == block_id
