"""The proposal's pipeline figure: photo in -> slot layout -> risk out.

    .venv/bin/python tools/pipeline_figure.py data/v2_train --tower 11 --out docs/pipeline_figure.png

Everything comes from the dataset: the photo is the tower's view, the slot
layout is its presence grid from towers.csv, and the risk panel tints each
block in the photo by its label from labels.csv.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inspect_data import edges, tint  # noqa: E402

COLOUR = {"low": "#0ca30c", "medium": "#fab219", "high": "#d03b3b"}
WOOD, GAP_FILL, PANEL, INK, ACCENT = "#c89b62", "#eef0ec", "#eef0ec", "#222222", "#1f4e79"
LAYERS, SLOTS = 18, 3


def rgb(hex_colour):
    return tuple(int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))


def crop_box(root, tower, view, aspect=0.68, pad=0.1):
    """Box around the tower (blocks plus a little floor), `aspect` = width / height."""
    seg = np.asarray(Image.open(root / "masks" / f"tower_{tower:04d}_v{view}_seg.png"))
    ys, xs = np.nonzero(seg)
    h = (ys.max() - ys.min()) * (1 + 2 * pad)
    cy, cx = (ys.max() + ys.min()) / 2 + 0.05 * h, (xs.max() + xs.min()) / 2
    w = h * aspect
    top, left = int(max(0, cy - h / 2)), int(max(0, cx - w / 2))
    return top, min(seg.shape[0], int(top + h)), left, min(seg.shape[1], int(left + w))


def risk_photo(root, tower, view, blocks):
    name = f"tower_{tower:04d}_v{view}"
    img = np.asarray(Image.open(root / "images" / f"{name}.png").convert("RGB"),
                     dtype=np.float32) / 255
    index_map = np.asarray(Image.open(root / "masks" / f"{name}_seg.png"))
    img = img.copy()
    for block_id, level in zip(blocks.block_id, blocks.risk_level):
        tint(img, index_map == block_id, rgb(COLOUR[level]), 0.65)
    img[edges(index_map) & (index_map > 0)] *= 0.45
    return np.clip(img, 0, 1), img.shape


def draw_grid(ax, grid):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.04, 0.0), 0.92, 1.0, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=PANEL, ec="none"))
    w, h = 0.24, 0.036
    xs = [0.11, 0.38, 0.65]
    for level in range(LAYERS):
        y = 0.02 + (level / LAYERS) * 0.96 + 0.006       # bottom layer at the bottom
        for slot in range(SLOTS):
            present = grid[level * SLOTS + slot] == "1"
            ax.add_patch(Rectangle((xs[slot], y), w, h, fc=WOOD if present else GAP_FILL,
                                   ec="none" if present else "#555555",
                                   ls="--", lw=1.2, joinstyle="round"))


def arrow(fig, x0, x1, y, label):
    fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=22, lw=2.5, color=ACCENT))
    fig.text((x0 + x1) / 2, y - 0.1, label, ha="center", va="center", fontsize=15,
             fontweight="bold", color=ACCENT, linespacing=1.6)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("--tower", type=int, default=11)
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("docs/pipeline_figure.png"))
    args = ap.parse_args()

    towers = pd.read_csv(args.data / "towers.csv", dtype={"grid": str}).set_index("tower_id")
    labels = pd.read_csv(args.data / "labels.csv")
    blocks = labels[labels.tower_id == args.tower]
    grid = towers.loc[args.tower, "grid"]
    photo = np.asarray(Image.open(args.data / "images" / f"tower_{args.tower:04d}_v{args.view}.png"))
    risk, _ = risk_photo(args.data, args.tower, args.view, blocks)
    t, b, l, r = crop_box(args.data, args.tower, args.view)
    photo, risk = photo[t:b, l:r], risk[t:b, l:r]
    counts = blocks.risk_level.value_counts()

    fig = plt.figure(figsize=(20, 10.4), dpi=110)
    panels = [(0.03, "PHOTO IN"), (0.385, "SLOT LAYOUT"), (0.70, "RISK OUT")]
    for (x, title), content in zip(panels, (photo, grid, risk)):
        width = 0.16 if title == "SLOT LAYOUT" else 0.27
        ax = fig.add_axes([x + (0.27 - width) / 2 if title == "SLOT LAYOUT" else x,
                           0.21, width, 0.76])
        if title == "SLOT LAYOUT":
            draw_grid(ax, content)
        else:
            ax.imshow(content)
            ax.axis("off")
        fig.text(x + 0.135, 0.15, title, ha="center", fontsize=17, fontweight="bold", color=INK)

    arrow(fig, 0.33, 0.42, 0.62, "Stage 1\nfind blocks")
    arrow(fig, 0.61, 0.68, 0.62, "Stage 2\nrate risk")

    for i, level in enumerate(("low", "medium", "high")):
        x = 0.285 + i * 0.15
        fig.patches.append(FancyBboxPatch((x, 0.045), 0.022, 0.042, transform=fig.transFigure,
                                          boxstyle="round,pad=0,rounding_size=0.004",
                                          fc=COLOUR[level], ec="none"))
        fig.text(x + 0.03, 0.066, f"{counts.get(level, 0)} {level}", va="center",
                 fontsize=16, color=INK)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor="white")
    print(f"tower {args.tower}: {dict(counts)}, {int(towers.loc[args.tower, 'num_gaps'])} gaps "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
