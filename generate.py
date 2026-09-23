#!/usr/bin/env python3
"""Headless dataset generator.

    python generate.py --towers 20 --out data/my_run

For each tower: build it, let it settle, knock some random gaps in it, then
photograph it from several angles. Then, for EVERY block still standing,
remove it and measure the risk -- whether the tower collapses, and if not,
how far the table can be tilted before it does. See jenga_sim/risk.py.

Towers are independent, so they are generated in parallel across processes.
Physics here is single-threaded CPU work -- the GPU is only used to draw the
images -- so worker count is the main throughput lever.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from jenga_sim import camera, config as C, risk, tower as T
from jenga_sim.dataset import DatasetWriter
from jenga_sim.labeling import max_displacement


def default_workers() -> int:
    """Half the cores, which leaves room for the OS and the efficiency cores.

    On a 4+6 machine like an M4 this picks 5: the four performance cores plus
    one, which is about where the returns flatten off.
    """
    return max(1, (os.cpu_count() or 2) // 2)


def build_one(seed: int, rng: random.Random):
    """Build, settle and gap one tower. Returns (tower, ok, settle_seconds).

    `ok` is False when the tower could not stand on its own; the caller
    should throw it away.
    """
    tw = T.build_tower(seed=seed)
    converged, secs = T.settle(tw)
    if not converged:
        return tw, False, secs

    # Must be able to stand untouched without drifting.
    before = T.snapshot(tw, tw.present_ids())
    T.simulate(tw, C.BUILD_VERIFY_SECONDS)
    drift, _tilt = max_displacement(tw, before)
    if drift > C.BUILD_REJECT_DRIFT:
        return tw, False, secs

    T.make_gaps(tw, rng, rng.randint(C.GAPS_MIN, C.GAPS_MAX))
    return tw, True, secs


def is_legal(level: int) -> bool:
    """Real-Jenga rule: the top layer(s) cannot be taken from."""
    return level <= C.NUM_LAYERS - 1 - C.EXCLUDE_TOP_LAYERS


# ---------------------------------------------------------------------------
# One tower's worth of work
# ---------------------------------------------------------------------------
# Runs in a worker process, so everything it touches must be picklable: plain
# values in, plain dicts out. It writes its own image files (per-tower files,
# no contention) but never the CSVs -- the parent writes those so rows stay
# in a fixed order.

def run_tower(job: dict) -> dict:
    i, seed, n_views = job["tower_id"], job["seed"], job["views"]
    rng = random.Random(seed)
    result = {"tower_id": i, "seed": seed, "ok": False, "tower": None,
              "labels": [], "views": [], "visibility": [], "seconds": 0.0}
    t0 = time.time()

    tw, ok, settle_secs = build_one(seed, rng)
    if not ok:
        return result

    paths = DatasetWriter(job["out"], write_csv=False)
    T.save_state_npz(tw, paths.state_path(i))
    state = T.save_state(tw)
    grid = risk.grid_string(tw)
    present = tw.present_ids()

    # Photos of the tower as it stands, before anything is taken out.
    shots = camera.render_views(tw, seed, n_views)
    index_maps = {}
    for shot in shots:
        v = shot["view"]
        index_maps[v] = camera.block_index_map(tw, shot["seg"])
        camera.save_rgb(paths.image_path(i, v), shot["rgb"])
        camera.save_seg(paths.seg_path(i, v), index_maps[v])
        result["views"].append(dict(
            tower_id=i, view=v,
            camera_params=camera.camera_params_string(shot["cam"]),
            look_params=camera.look_params_string(shot["look"])))
        pixels = camera.block_pixels(tw, shot["seg"])
        for b in present:
            result["visibility"].append(dict(tower_id=i, view=v,
                                             block_id=b + 1, pixels=pixels[b]))

    base_tilt = risk.tilt_margin(tw, state)

    # Every block still standing gets a risk label.
    levels = {}
    for b in present:
        r = risk.assess_block(tw, b, state, base_tilt)
        levels[b] = r.risk_level
        result["labels"].append(dict(
            tower_id=i, block_id=b + 1,
            level=tw.level_of[b], position_in_level=tw.slot_of[b],
            legal=is_legal(tw.level_of[b]),
            outcome=r.outcome,
            max_displacement=round(r.max_displacement, 5),
            max_tilt_deg=round(r.max_tilt_deg, 3),
            tilt_margin_deg=round(r.tilt_margin_deg, 3),
            margin_drop_deg=round(r.margin_drop_deg, 3),
            risk_level=r.risk_level, seed=seed))

    for v, index_map in index_maps.items():
        camera.save_risk_map(paths.risk_path(i, v), risk.risk_map(index_map, levels))

    result["ok"] = True
    result["tower"] = dict(
        tower_id=i, seed=seed, num_blocks=len(tw.present_ids()),
        num_gaps=tw.n_gaps, grid=grid, base_tilt_deg=round(base_tilt, 3),
        settle_seconds=round(settle_secs, 3))
    result["seconds"] = time.time() - t0
    camera.release_renderers()
    return result


# ---------------------------------------------------------------------------
# Saving as we go
# ---------------------------------------------------------------------------
# Each finished tower is written to its own small file in <out>/parts/ the
# moment it arrives, and the CSVs are rebuilt from those files at the end.
# A long run that is interrupted -- a crash, a sleep, a closed lid -- then
# keeps every tower it finished, and `--resume` carries on from there.

def part_path(out, tower_id: int) -> Path:
    return Path(out) / "parts" / f"tower_{tower_id:04d}.json"


def _plain(obj):
    """Let json write numpy numbers."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def save_part(out, res: dict) -> None:
    path = part_path(out, res["tower_id"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, default=_plain))
    tmp.replace(path)          # atomic: a half-written file never looks finished


def load_parts(out) -> dict[int, dict]:
    return {r["tower_id"]: r for r in
            (json.loads(p.read_text()) for p in sorted((Path(out) / "parts").glob("tower_*.json")))}


def main():
    ap = argparse.ArgumentParser(description="Generate Jenga per-block risk data")
    ap.add_argument("--towers", type=int, default=C.DEFAULT_TOWERS)
    ap.add_argument("--views", type=int, default=C.VIEWS_PER_TOWER,
                    help="photos per tower; view 0 is the fixed reference shot")
    ap.add_argument("--out", default="data/")
    ap.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="parallel worker processes; 1 runs in this process, "
                         "which is what you want when debugging")
    ap.add_argument("--resume", action="store_true",
                    help="finish an interrupted run in --out instead of starting over")
    args = ap.parse_args()
    if args.views < 1:
        ap.error("--views must be at least 1")

    out = Path(args.out)
    run_file = out / "run.json"
    config = {"seed": args.seed, "views": args.views}
    done = {}
    if (out / "parts").exists() and any((out / "parts").glob("tower_*.json")):
        if not args.resume:
            ap.error(f"{out} already holds towers from an earlier run. Add --resume to "
                     f"finish that run, or choose a new --out.")
        earlier = json.loads(run_file.read_text()) if run_file.exists() else None
        if earlier is not None and earlier != config:
            ap.error(f"--resume must use the same settings as the run it continues: "
                     f"that run used {earlier}, this one asks for {config}")
        done = load_parts(out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    run_file.write_text(json.dumps(config))

    jobs = [dict(tower_id=i, seed=args.seed * 100000 + i, views=args.views, out=args.out)
            for i in range(args.towers) if i not in done]
    workers = max(1, min(args.workers, len(jobs) or 1))
    counts = Counter(row["risk_level"] for r in done.values() if r["ok"] for row in r["labels"])
    if done:
        print(f"resuming: {len(done)} towers already finished, {len(jobs)} to go")
    t_start = time.time()

    bar = tqdm(total=len(jobs), desc="towers", unit="tower")

    def absorb(res: dict) -> None:
        save_part(out, res)
        done[res["tower_id"]] = res
        if not res["ok"]:
            bar.write(f"[reject] tower {res['tower_id']} "
                      f"(seed {res['seed']}) would not stand")
        else:
            counts.update(row["risk_level"] for row in res["labels"])
        bar.update(1)
        bar.set_postfix(**{k: counts[k] for k in risk.RISK_LEVELS if counts[k]})

    if jobs and workers == 1:
        for job in jobs:
            absorb(run_tower(job))
    elif jobs:
        # "spawn", not "fork": a forked child inherits the parent's OpenGL
        # context, which is not valid across fork and crashes the renderer.
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for res in pool.imap_unordered(run_tower, jobs):
                absorb(res)
    bar.close()

    # Rebuild the CSVs from every finished tower, in tower order, so they are
    # identical whatever --workers was and however many times it resumed.
    results = [done[t] for t in sorted(done) if done[t]["ok"]]
    rejected = [t for t in sorted(done) if not done[t]["ok"]]
    writer = DatasetWriter(out)
    for res in results:
        writer.write("towers", **res["tower"])
        for row in res["labels"]:
            writer.write("labels", **row)
        for row in res["views"]:
            writer.write("views", **row)
        for row in res["visibility"]:
            writer.write("visibility", **row)
    writer.close()

    summarise(args, results, rejected, counts, workers, time.time() - t_start)


def summarise(args, results, rejected, counts, workers, elapsed) -> None:
    labels = [row for res in results for row in res["labels"]]
    total = len(labels)
    print("\n" + "=" * 62)
    print(f"  towers built     : {len(results)} of {args.towers}"
          f"  ({len(rejected)} rejected, would not stand)")
    print(f"  blocks labelled  : {total}   views per tower: {args.views}"
          f"   images: {len(results) * args.views}")
    print(f"  workers          : {workers}")
    print("-" * 62)
    for level in risk.RISK_LEVELS:
        n = counts[level]
        pct = 100.0 * n / total if total else 0.0
        print(f"  {level:<9}: {n:6d}  ({pct:5.1f}%)")
    print("-" * 62)
    if labels:
        drops = np.array([r["margin_drop_deg"] for r in labels
                          if r["outcome"] == "stable"])
        if drops.size:
            q = np.percentile(drops, [50, 75, 90, 95])
            print("  margin drop of survivors (deg): "
                  + "  ".join(f"p{p}={v:.2f}" for p, v in zip((50, 75, 90, 95), q)))

        # The structural rule is a free sanity check on the physics: it
        # predicted collapse 98% of the time on earlier data.
        grids = {res["tower_id"]: res["tower"]["grid"] for res in results}
        hits = sum((r["outcome"] == "collapse") ==
                   risk.structural_rule(grids[r["tower_id"]], r["block_id"] - 1)
                   for r in labels)
        print(f"  structural rule agrees with collapse labels: {100 * hits / total:.1f}%")
    if elapsed > 0 and results:
        print(f"  throughput       : {60 * len(results) / elapsed:.1f} towers/min"
              f"  ({elapsed:.0f} s total)")
    print(f"  output           : {args.out}")
    print("=" * 62)

    missing = [k for k in risk.RISK_LEVELS if counts[k] == 0]
    if missing and total:
        print(f"\n  NOTE: no {', '.join(missing)} labels in this run.")
        if "high" in missing:
            print("        Raise GAPS_MAX -- pristine towers almost never collapse.")
        if "medium" in missing or "low" in missing:
            print("        Move RISK_MEDIUM_DROP_DEG to a percentile of the margin"
                  " drop printed above.")


if __name__ == "__main__":
    main()
