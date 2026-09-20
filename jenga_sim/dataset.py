"""Writing trials to disk.

Layout produced:
    data/
      images/tower_0003.png              RGB just before the removal
      masks/tower_0003_seg.png           full segmentation (pixel -> block index)
      masks/tower_0003_block_17.png      binary mask of the candidate block
      states/tower_0003.npz              saved sim state (qpos+qvel)
      labels.csv                         one row per (tower, block) trial
      towers.csv                         one row per tower
"""

from __future__ import annotations

import csv
from pathlib import Path

LABEL_FIELDS = [
    "tower_id", "block_id", "level", "position_in_level", "mode", "outcome",
    "max_displacement", "max_tilt_deg", "peak_force", "extract_time",
    "mask_pixels", "seed", "camera_params", "pull_direction",
]

TOWER_FIELDS = [
    "tower_id", "seed", "num_blocks", "num_gaps", "settled",
    "settle_seconds", "camera_params",
]


class DatasetWriter:
    def __init__(self, root: str | Path, write_csv: bool = True):
        """`write_csv=False` gives a paths-only view of the dataset.

        Worker processes need to know where to save images and masks, but must
        not open (and truncate) the CSVs -- only the parent writes those.
        """
        self.root = Path(root)
        self.images = self.root / "images"
        self.masks = self.root / "masks"
        self.states = self.root / "states"
        for d in (self.images, self.masks, self.states):
            d.mkdir(parents=True, exist_ok=True)

        self.write_csv = write_csv
        if not write_csv:
            return

        self._labels_path = self.root / "labels.csv"
        self._towers_path = self.root / "towers.csv"
        self._labels = self._labels_path.open("w", newline="")
        self._towers = self._towers_path.open("w", newline="")
        self.labels_csv = csv.DictWriter(self._labels, fieldnames=LABEL_FIELDS)
        self.towers_csv = csv.DictWriter(self._towers, fieldnames=TOWER_FIELDS)
        self.labels_csv.writeheader()
        self.towers_csv.writeheader()

    # -- paths ------------------------------------------------------------
    def tower_name(self, tower_id: int) -> str:
        return f"tower_{tower_id:04d}"

    def image_path(self, tower_id: int) -> Path:
        return self.images / f"{self.tower_name(tower_id)}.png"

    def seg_path(self, tower_id: int) -> Path:
        return self.masks / f"{self.tower_name(tower_id)}_seg.png"

    def block_mask_path(self, tower_id: int, block_index: int) -> Path:
        return self.masks / f"{self.tower_name(tower_id)}_block_{block_index:02d}.png"

    def state_path(self, tower_id: int) -> Path:
        # qpos + qvel is the complete state of a world of free bodies, so a
        # small .npz replaces the engine-specific state blob.
        return self.states / f"{self.tower_name(tower_id)}.npz"

    # -- rows -------------------------------------------------------------
    def write_tower(self, **row) -> None:
        self.towers_csv.writerow(row)
        self._towers.flush()

    def write_trial(self, **row) -> None:
        self.labels_csv.writerow(row)
        self._labels.flush()

    def close(self) -> None:
        if not self.write_csv:
            return
        self._labels.close()
        self._towers.close()
