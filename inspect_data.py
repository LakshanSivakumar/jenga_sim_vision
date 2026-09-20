#!/usr/bin/env python3
"""Eyeball the labels.

    python inspect_data.py data/

Draws a grid of sample towers with the candidate block outlined and coloured
by outcome: green = stable, red = collapse, yellow = stuck. If the labels are
wrong, this is usually where you notice.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

OUTCOME_COLOUR = {
    "stable": (0.15, 0.80, 0.30),
    "collapse": (0.90, 0.20, 0.20),
    "stuck": (0.95, 0.78, 0.10),
}


def overlay(image: np.ndarray, mask: np.ndarray, colour, alpha=0.45):
    """Tint the masked block and draw a bright outline around it."""
    out = image.astype(np.float32) / 255.0
    m = mask > 127
    if m.any():
        rgb = np.array(colour, dtype=np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * rgb

        # 1px outline: pixels in the mask with a neighbour outside it.
        pad = np.pad(m, 1, mode="constant")
        edge = m & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] &
                     pad[1:-1, :-2] & pad[1:-1, 2:])
        out[edge] = rgb
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Sample grid of labelled trials")
    ap.add_argument("data", nargs="?", default="data/")
    ap.add_argument("--n", type=int, default=12, help="samples in the grid")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--out", default=None, help="output PNG (default <data>/sample_grid.png)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balanced", action="store_true",
                    help="sample evenly across outcomes instead of at random")
    args = ap.parse_args()

    root = Path(args.data)
    df = pd.read_csv(root / "labels.csv")
    if df.empty:
        raise SystemExit("labels.csv is empty")

    print(f"{len(df)} trials in {root}")
    print(df["outcome"].value_counts().to_string())
    for col in ("max_displacement", "peak_force"):
        if df[col].notna().any():
            print(f"\n{col} by outcome:")
            print(df.groupby("outcome")[col].describe()[["mean", "50%", "max"]].to_string())

    rng = random.Random(args.seed)
    rows = df.to_dict("records")
    if args.balanced:
        picked, by = [], {}
        for r in rows:
            by.setdefault(r["outcome"], []).append(r)
        for v in by.values():
            rng.shuffle(v)
        i = 0
        while len(picked) < args.n and any(by.values()):
            for k in list(by):
                if by[k] and len(picked) < args.n:
                    picked.append(by[k].pop())
            i += 1
        rows = picked
    else:
        rng.shuffle(rows)
        rows = rows[:args.n]

    cols = args.cols
    nrows = (len(rows) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(3.1 * cols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, r in zip(axes, rows):
        name = f"tower_{int(r['tower_id']):04d}"
        img_p = root / "images" / f"{name}.png"
        mask_p = root / "masks" / f"{name}_block_{int(r['block_id']):02d}.png"
        if not img_p.exists():
            ax.axis("off")
            continue
        img = np.array(Image.open(img_p).convert("RGB"))
        colour = OUTCOME_COLOUR.get(r["outcome"], (0.5, 0.5, 0.5))
        if mask_p.exists():
            img = overlay(img, np.array(Image.open(mask_p).convert("L")), colour)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(colour); spine.set_linewidth(3.5)

        bits = [f"{r['outcome']}", f"L{int(r['level'])}P{int(r['position_in_level'])}"]
        if r["mode"] == "pull":
            if r.get("pull_direction"):
                bits.append(str(r["pull_direction"]))
            bits.append(f"{r['peak_force']:.2f}N")
        bits.append(f"d={r['max_displacement']:.3f}")
        if int(r.get("mask_pixels", 1)) == 0:
            bits.append("HIDDEN")
        ax.set_title("  ".join(bits), fontsize=9)

    for ax in axes[len(rows):]:
        ax.axis("off")

    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=11,
                          markerfacecolor=c, markeredgecolor="none", label=k)
               for k, c in OUTCOME_COLOUR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Jenga removal outcomes -- {root}", fontsize=13)
    fig.tight_layout(rect=[0, 0.045, 1, 0.97])

    out = Path(args.out) if args.out else root / "sample_grid.png"
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
