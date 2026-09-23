#!/usr/bin/env python3
"""Eyeball the risk labels.

    python inspect_data.py data/my_run                # whole towers, coloured
    python inspect_data.py data/my_run --blocks       # one block per panel

Towers mode (the default) colours EVERY block in a tower by its risk --
green low, amber medium, red high -- which is what the finished model is
meant to produce from a photo. Blocks mode highlights a single block per
panel, which is the better way to check individual labels.

--medium-drop / --high-below re-bin the risk levels from the stored margins
on the fly, so you can preview other thresholds without regenerating anything.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jenga_sim import config as C                          # noqa: E402
from jenga_sim.risk import RISK_LEVELS, level_for, structural_rule  # noqa: E402

COLOUR = {
    "low": (0.18, 0.78, 0.32),
    "medium": (0.98, 0.70, 0.10),
    "high": (0.90, 0.18, 0.18),
}


def load(root: Path, high_below: float | None, medium_drop: float | None):
    labels = pd.read_csv(root / "labels.csv")
    towers = pd.read_csv(root / "towers.csv", dtype={"grid": str})
    views = pd.read_csv(root / "views.csv")
    if high_below is not None or medium_drop is not None:
        labels["risk_level"] = [level_for(m, d, high_below, medium_drop)
                                for m, d in zip(labels.tilt_margin_deg,
                                                labels.margin_drop_deg)]
    return labels, towers, views


def report(labels: pd.DataFrame, towers: pd.DataFrame, root: Path) -> None:
    n = len(labels)
    print(f"{len(towers)} towers, {n} labelled blocks in {root}\n")
    print("risk level:")
    for level in RISK_LEVELS:
        k = int((labels.risk_level == level).sum())
        print(f"   {level:<7} {k:6d}  ({100 * k / max(n, 1):5.1f}%)  {'#' * int(50 * k / max(n, 1))}")

    legal = labels[labels.legal]
    if len(legal) != n:
        print(f"\nlegal moves only (top layer excluded): {len(legal)} blocks")
        for level in RISK_LEVELS:
            k = int((legal.risk_level == level).sum())
            print(f"   {level:<7} {k:6d}  ({100 * k / max(len(legal), 1):5.1f}%)")

    surv = labels[labels.outcome == "stable"].margin_drop_deg
    if len(surv):
        q = surv.quantile([.5, .75, .9, .95])
        print("\nmargin drop of survivors (deg): "
              + "  ".join(f"p{int(p * 100)}={v:.2f}" for p, v in q.items()))
        base = towers.base_tilt_deg
        print(f"towers' own tilt margin (deg): median {base.median():.1f}, "
              f"range {base.min():.1f}-{base.max():.1f}")

    grids = dict(zip(towers.tower_id, towers.grid))
    rule = [structural_rule(grids[t], b - 1) for t, b in zip(labels.tower_id, labels.block_id)]
    agree = np.mean(np.array(rule) == (labels.outcome == "collapse").to_numpy())
    print(f"structural rule agrees with collapse labels: {100 * agree:.1f}%")

    vis_path = root / "visibility.csv"
    if vis_path.exists():
        vis = pd.read_csv(vis_path)
        print(f"block-views with the block fully hidden: {100 * (vis.pixels == 0).mean():.1f}%")


def edges(index_map: np.ndarray) -> np.ndarray:
    """Pixels on a boundary between two different blocks (or block/background)."""
    pad = np.pad(index_map, 1, mode="edge")
    return ((pad[1:-1, 1:-1] != pad[:-2, 1:-1]) | (pad[1:-1, 1:-1] != pad[2:, 1:-1]) |
            (pad[1:-1, 1:-1] != pad[1:-1, :-2]) | (pad[1:-1, 1:-1] != pad[1:-1, 2:]))


def tint(img: np.ndarray, mask: np.ndarray, colour, alpha: float) -> None:
    img[mask] = (1 - alpha) * img[mask] + alpha * np.asarray(colour)


def tower_panel(root: Path, tower_id: int, view: int, blocks: pd.DataFrame) -> np.ndarray:
    name = f"tower_{tower_id:04d}_v{view}"
    img = np.asarray(Image.open(root / "images" / f"{name}.png").convert("RGB"),
                     dtype=np.float32) / 255
    index_map = np.asarray(Image.open(root / "masks" / f"{name}_seg.png"))
    img = img.copy()
    for block_id, level in zip(blocks.block_id, blocks.risk_level):
        tint(img, index_map == block_id, COLOUR[level], 0.55)
    img[edges(index_map) & (index_map > 0)] *= 0.35
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def block_panel(root: Path, row, view: int) -> np.ndarray:
    name = f"tower_{int(row.tower_id):04d}_v{view}"
    img = np.asarray(Image.open(root / "images" / f"{name}.png").convert("RGB"),
                     dtype=np.float32) / 255
    mask = np.asarray(Image.open(root / "masks" / f"{name}_seg.png")) == row.block_id
    img = img.copy()
    colour = COLOUR[row.risk_level]
    tint(img, mask, colour, 0.5)
    img[edges(mask.astype(np.uint8)) & mask] = colour
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Sample grid of risk-labelled towers or blocks")
    ap.add_argument("data", nargs="?", default="data/")
    ap.add_argument("--blocks", action="store_true",
                    help="one highlighted block per panel instead of whole towers")
    ap.add_argument("--n", type=int, default=12, help="panels in the grid")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--view", type=int, default=None,
                    help="which photo of each tower to show (default: random)")
    ap.add_argument("--balanced", action="store_true",
                    help="blocks mode: sample evenly across risk levels")
    ap.add_argument("--legal-only", action="store_true",
                    help="blocks mode: skip top-layer blocks, which are not legal moves")
    ap.add_argument("--high-below", type=float, default=None,
                    help=f"re-bin: high if tilt margin below this (default {C.RISK_HIGH_BELOW_DEG})")
    ap.add_argument("--medium-drop", type=float, default=None,
                    help=f"re-bin: medium if the removal cost this many degrees of "
                         f"margin (default {C.RISK_MEDIUM_DROP_DEG})")
    ap.add_argument("--out", default=None, help="output PNG (default <data>/sample_grid.png)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data)
    if not (root / "labels.csv").exists():
        raise SystemExit(f"no labels.csv in {root} -- is this a generate.py output folder?")
    labels, towers, views = load(root, args.high_below, args.medium_drop)
    if labels.empty:
        raise SystemExit("labels.csv is empty")
    report(labels, towers, root)

    rng = random.Random(args.seed)
    n_views = views.groupby("tower_id").view.max().to_dict()

    def pick_view(tower_id: int) -> int:
        if args.view is not None:
            return min(args.view, n_views[tower_id])
        return rng.randint(0, n_views[tower_id])

    panels, titles, borders = [], [], []
    if not args.blocks:
        ids = sorted(labels.tower_id.unique())
        rng.shuffle(ids)
        for t in ids[:args.n]:
            blocks = labels[labels.tower_id == t]
            v = pick_view(t)
            panels.append(tower_panel(root, t, v, blocks))
            c = blocks.risk_level.value_counts()
            titles.append(f"tower {t} · v{v}   L{c.get('low', 0)} "
                          f"M{c.get('medium', 0)} H{c.get('high', 0)}")
            borders.append((0.3, 0.3, 0.3))
    else:
        pool = labels[labels.legal] if args.legal_only else labels
        rows = list(pool.itertuples())
        if args.balanced:
            by = {lv: [r for r in rows if r.risk_level == lv] for lv in RISK_LEVELS}
            for v in by.values():
                rng.shuffle(v)
            rows = []
            while len(rows) < args.n and any(by.values()):
                for lv in RISK_LEVELS:
                    if by[lv] and len(rows) < args.n:
                        rows.append(by[lv].pop())
        else:
            rng.shuffle(rows)
            rows = rows[:args.n]
        for r in rows:
            panels.append(block_panel(root, r, pick_view(int(r.tower_id))))
            detail = ("collapses" if r.outcome == "collapse"
                      else f"drop {r.margin_drop_deg:.2f}°")
            titles.append(f"{r.risk_level}  L{r.level}P{r.position_in_level}  {detail}")
            borders.append(COLOUR[r.risk_level])

    cols = max(1, min(args.cols, len(panels)))
    nrows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(3.1 * cols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, title, border in zip(axes, panels, titles, borders):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(border); spine.set_linewidth(3)
        ax.set_title(title, fontsize=9)
    for ax in axes[len(panels):]:
        ax.axis("off")

    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=11,
                          markerfacecolor=COLOUR[k], markeredgecolor="none", label=k)
               for k in RISK_LEVELS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    what = "blocks" if args.blocks else "towers"
    fig.suptitle(f"Jenga block risk ({what}) -- {root}", fontsize=13)
    fig.tight_layout(rect=[0, 0.045, 1, 0.97])

    out = Path(args.out) if args.out else root / "sample_grid.png"
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
