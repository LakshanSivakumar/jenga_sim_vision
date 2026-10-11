#!/usr/bin/env python3
"""Exploratory charts for a generated dataset, as PNGs for slides and reports.

    python tools/eda_figures.py data/big_run --out docs/eda

Risk levels are a state (safe -> dangerous), so they wear fixed status colours,
and every chart labels them in text too: green and red are the classic
colour-blind confusion pair, so colour never carries the meaning alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MPath
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jenga_sim.risk import layer_left   # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e7e3"
STATUS = {"low": "#0ca30c", "medium": "#fab219", "high": "#d03b3b"}
LEVELS = ("low", "medium", "high")
DPI = 220

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 11, "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


# ---------------------------------------------------------------------------
# Chart scaffolding
# ---------------------------------------------------------------------------

def figure(title: str, subtitle: str, size=(8.0, 4.6)):
    fig = plt.figure(figsize=size, dpi=DPI)
    fig.text(0.04, 0.955, title, fontsize=15, fontweight="bold", va="top")
    fig.text(0.04, 0.885, subtitle, fontsize=10.5, color=MUTED, va="top")
    return fig


def tidy(ax, value_axis="y"):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    getattr(ax, f"{value_axis}axis").grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def legend(fig, x=0.04, y=0.80, levels=LEVELS):
    fig.legend(handles=[Patch(facecolor=STATUS[lv], label=lv) for lv in levels],
               loc="center left", bbox_to_anchor=(x - 0.008, y), ncol=len(levels), frameon=False,
               handlelength=1.1, handleheight=1.1, columnspacing=1.8, fontsize=10.5)


def rounded_bar(ax, x, base, length, width, colour, vertical=True, radius_px=8):
    """One bar: rounded corners at the data end, square at the baseline.

    Built as a single path so no seam shows where two shapes would meet. The
    radius is set in pixels and converted per axis, so corners stay circular
    whatever the axes' aspect ratio.
    """
    if length <= 0:
        return
    ax.figure.canvas.draw()
    to_px = ax.transData.transform
    px_x = abs(to_px((1, 0))[0] - to_px((0, 0))[0])
    px_y = abs(to_px((0, 1))[1] - to_px((0, 0))[1])
    k = 0.5523                                   # cubic Bezier quarter-circle constant
    C, L4 = MPath.CURVE4, MPath.LINETO
    if vertical:
        x0, x1, y0, y1 = x - width / 2, x + width / 2, base, base + length
        rx, ry = min(radius_px / px_x, width / 2), min(radius_px / px_y, length)
        verts = [(x0, y0), (x1, y0), (x1, y1 - ry),
                 (x1, y1 - ry + k * ry), (x1 - rx + k * rx, y1), (x1 - rx, y1),
                 (x0 + rx, y1),
                 (x0 + rx - k * rx, y1), (x0, y1 - ry + k * ry), (x0, y1 - ry),
                 (x0, y0)]
    else:
        y0, y1, x0, x1 = x - width / 2, x + width / 2, base, base + length
        rx, ry = min(radius_px / px_x, length), min(radius_px / px_y, width / 2)
        verts = [(x0, y1), (x0, y0), (x1 - rx, y0),
                 (x1 - rx + k * rx, y0), (x1, y0 + ry - k * ry), (x1, y0 + ry),
                 (x1, y1 - ry),
                 (x1, y1 - ry + k * ry), (x1 - rx + k * rx, y1), (x1 - rx, y1),
                 (x0, y1)]
    codes = [MPath.MOVETO, L4, L4, C, C, C, L4, C, C, C, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=colour, edgecolor="none"))


def save(fig, path):
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("wrote", path)


def share(series) -> dict:
    s = series.value_counts(normalize=True) * 100
    return {lv: float(s.get(lv, 0.0)) for lv in LEVELS}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def class_balance(L, out):
    n = len(L)
    sh = share(L.risk_level)
    counts = L.risk_level.value_counts()
    fig = figure(f"How the {n:,} blocks are labelled",
                 f"Risk of removing each block, across all {L.tower_id.nunique()} simulated towers")
    ax = fig.add_axes([0.08, 0.12, 0.88, 0.64])
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 70)
    tidy(ax)
    ax.set_yticks([0, 20, 40, 60]); ax.set_yticklabels(["0%", "20%", "40%", "60%"])
    ax.set_xticks(range(3)); ax.set_xticklabels([lv.capitalize() for lv in LEVELS], fontsize=12, color=INK)
    for i, lv in enumerate(LEVELS):
        rounded_bar(ax, i, 0, sh[lv], 0.42, STATUS[lv])
        ax.text(i, sh[lv] + 2.2, f"{sh[lv]:.1f}%", ha="center", fontsize=13, fontweight="bold")
        ax.text(i, sh[lv] + 7.4, f"{counts.get(lv, 0):,} blocks", ha="center", fontsize=10, color=MUTED)
    save(fig, out / "class_balance.png")


def by_position(L, out):
    rows = [("Edge blocks", L[L.position_in_level != 1]), ("Middle blocks", L[L.position_in_level == 1])]
    fig = figure("Middle blocks are either safe or deadly",
                 "Share of each risk level by position in the layer. Edges are where medium risk lives.",
                 size=(8.0, 3.6))
    legend(fig, y=0.74)
    ax = fig.add_axes([0.20, 0.10, 0.76, 0.52])
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, 1.6)
    tidy(ax, "x")
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks([1, 0]); ax.set_yticklabels([f"{name}\n{len(d):,} blocks" for name, d in rows], color=INK)
    for y, (_, d) in zip((1, 0), rows):
        sh, left = share(d.risk_level), 0.0
        for lv in LEVELS:
            ax.barh(y, sh[lv], left=left, height=0.5, color=STATUS[lv], edgecolor=SURFACE, linewidth=1.2)
            if sh[lv] >= 9:
                ax.text(left + sh[lv] / 2, y, f"{sh[lv]:.0f}%", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=INK if lv == "medium" else "white")
            left += sh[lv]
    save(fig, out / "risk_by_position.png")


def by_height(L, out):
    layers = range(18)
    med = [share(L[L.level == lv].risk_level)["medium"] for lv in layers]
    high = [share(L[L.level == lv].risk_level)["high"] for lv in layers]
    fig = figure("Lower blocks are more often medium risk",
                 "Share of blocks at each layer, bottom layer at the bottom. High risk barely changes with height.",
                 size=(8.0, 6.0))
    for k, (vals, lv, title) in enumerate(((med, "medium", "Medium risk"), (high, "high", "High risk"))):
        ax = fig.add_axes([0.10 + k * 0.46, 0.08, 0.40, 0.70])
        ax.set_xlim(0, 40); ax.set_ylim(-0.7, 17.7)
        tidy(ax, "x")
        ax.set_xticks([0, 10, 20, 30, 40]); ax.set_xticklabels(["0%", "10%", "20%", "30%", "40%"])
        ax.set_yticks([0, 5, 10, 15, 17])
        ax.set_yticklabels(["layer 1\n(bottom)", "6", "11", "16", "18 (top)"] if k == 0 else [])
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=8)
        for y, v in zip(layers, vals):
            rounded_bar(ax, y, 0, v, 0.62, STATUS[lv], vertical=False, radius_px=5)
        for y in (0, 15 if lv == "medium" else 17):
            ax.text(vals[y] + 0.8, y, f"{vals[y]:.0f}%", va="center", fontsize=10, fontweight="bold")
    save(fig, out / "risk_by_height.png")


def gaps_vs_high(L, T, out):
    t = (L.groupby("tower_id").risk_level.apply(lambda s: 100 * (s == "high").mean())
           .rename("high").to_frame().join(T.set_index("tower_id").num_gaps))
    m = t.groupby("num_gaps").high.mean()
    r = np.corrcoef(t.num_gaps, t.high)[0, 1]
    fig = figure("Gappier towers have more dangerous blocks",
                 f"Average share of high-risk blocks, by how many blocks were already missing (r = {r:.2f}, {len(t)} towers)")
    ax = fig.add_axes([0.08, 0.13, 0.88, 0.66])
    ax.set_xlim(m.index.min() - 0.7, m.index.max() + 0.7); ax.set_ylim(0, 55)
    tidy(ax)
    ax.set_yticks([0, 10, 20, 30, 40, 50]); ax.set_yticklabels([f"{v}%" for v in (0, 10, 20, 30, 40, 50)])
    ax.set_xticks(m.index); ax.set_xlabel("blocks already missing from the tower")
    for g, v in m.items():
        rounded_bar(ax, g, 0, v, 0.55, STATUS["high"])
    for g in (m.index.min(), m.index.max()):
        ax.text(g, m[g] + 1.5, f"{m[g]:.0f}%", ha="center", fontsize=11, fontweight="bold")
    save(fig, out / "gaps_vs_high.png")


def margin_drop(L, out, threshold=0.3, step=0.075):
    s = L[L.outcome == "stable"].margin_drop_deg
    lo, hi = -0.75, 1.95
    shown = s[(s >= lo - 1e-9) & (s < hi)]
    edges = np.round(np.arange(lo, hi + step / 2, step), 4)       # 0.3 is an edge
    counts, _ = np.histogram(shown.round(6), bins=edges)
    fig = figure("How much does one removal weaken the tower?",
                 f"Drop in tilt margin for the {len(s):,} removals the tower survived. "
                 f"{len(s) - len(shown)} outliers beyond {lo}° / {hi}° not shown.")
    legend(fig, y=0.80, levels=("low", "medium"))
    ax = fig.add_axes([0.10, 0.14, 0.86, 0.58])
    ax.set_xlim(lo - 0.05, hi + 0.05); ax.set_ylim(0, counts.max() * 1.18)
    tidy(ax)
    ax.set_xlabel("drop in tilt margin (degrees)   ·   negative = the removal made the tower steadier")
    for left, c in zip(edges[:-1], counts):
        if c:
            rounded_bar(ax, left + step / 2, 0, c, step * 0.82,
                        STATUS["medium" if left >= threshold - 1e-9 else "low"], radius_px=3)
    ax.axvline(threshold, color=INK, linewidth=0.9)
    ax.text(threshold + 0.03, counts.max() * 1.1, f"medium from {threshold}°", fontsize=10, va="center")
    save(fig, out / "margin_drop.png")


def baseline(L, T, out):
    """Same split and key as notebooks/DAP_IEDA_Jenga.ipynb section 8, so the numbers agree."""
    grids = dict(zip(T.tower_id, T.grid))
    pattern = {(): "none", (0,): "edge", (2,): "edge", (1,): "middle",
               (0, 2): "two edges", (0, 1): "edge+middle", (1, 2): "edge+middle"}
    L = L.copy()
    L["left"] = [pattern[layer_left(grids[t], b - 1)] for t, b in zip(L.tower_id, L.block_id)]
    L["band"] = pd.cut(L.level, [-1, 5, 11, 17], labels=["low", "mid", "top"]).astype(str)
    L["key"] = L.left + "|" + L.band + "|" + (L.position_in_level != 1).astype(str)
    ids = T.tower_id.sample(frac=1, random_state=0).to_numpy()
    test = set(ids[: max(1, len(ids) // 5)])
    tr, te = L[~L.tower_id.isin(test)], L[L.tower_id.isin(test)]
    table = tr.groupby("key").risk_level.agg(lambda x: x.value_counts().idxmax())
    pred = te.key.map(table).fillna(tr.risk_level.value_counts().idxmax())

    def f1(c):
        tp = ((te.risk_level == c) & (pred == c)).sum()
        fp = ((te.risk_level != c) & (pred == c)).sum()
        fn = ((te.risk_level == c) & (pred != c)).sum()
        return 2 * tp / max(2 * tp + fp + fn, 1)

    scores = {c: f1(c) for c in LEVELS}
    acc = 100 * (pred == te.risk_level).mean()
    fig = figure("The baseline to beat: medium risk is the hard part",
                 f"A lookup table on local structure, tested on {len(test)} held-out towers. "
                 f"Accuracy {acc:.1f}%, macro-F1 {np.mean(list(scores.values())):.2f}.")
    ax = fig.add_axes([0.08, 0.12, 0.88, 0.66])
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 1.15)
    tidy(ax)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("F1 score (1 = perfect)")
    ax.set_xticks(range(3)); ax.set_xticklabels([lv.capitalize() for lv in LEVELS], fontsize=12, color=INK)
    for i, lv in enumerate(LEVELS):
        rounded_bar(ax, i, 0, scores[lv], 0.42, STATUS[lv])
        ax.text(i, scores[lv] + 0.035, f"{scores[lv]:.2f}", ha="center", fontsize=13, fontweight="bold")
    save(fig, out / "baseline_f1.png")


def six_views(root, tower, out):
    names = sorted((root / "images").glob(f"tower_{tower:04d}_v*.png"))
    fig = figure(f"Six photos of the same tower",
                 "Every tower is photographed from 6 angles with randomised lighting and colours. "
                 "All six share one set of labels.", size=(12.0, 3.9))
    w = 0.92 / len(names)
    for k, p in enumerate(names):
        ax = fig.add_axes([0.04 + k * w, 0.06, w * 0.96, 0.72])
        ax.imshow(np.asarray(Image.open(p).convert("RGB"))[24:488, 64:448]); ax.axis("off")
        ax.set_title("view 0 · reference" if k == 0 else f"view {k}", fontsize=10, color=MUTED, pad=4)
    save(fig, out / "six_views.png")


def main():
    ap = argparse.ArgumentParser(description="EDA charts for a generated dataset")
    ap.add_argument("data", nargs="?", default="data/big_run")
    ap.add_argument("--out", default="docs/eda")
    ap.add_argument("--tower", type=int, default=3, help="tower shown in the six-views strip")
    args = ap.parse_args()
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    L = pd.read_csv(root / "labels.csv")
    T = pd.read_csv(root / "towers.csv", dtype={"grid": str})
    class_balance(L, out)
    by_position(L, out)
    by_height(L, out)
    gaps_vs_high(L, T, out)
    margin_drop(L, out)
    baseline(L, T, out)
    six_views(root, args.tower, out)


if __name__ == "__main__":
    main()
