#!/usr/bin/env python3
"""Headless dataset generator.

    python generate.py --towers 20 --mode pull --out data/

Builds towers, saves an image + masks + sim state for each, then for every
candidate block restores the state, removes the block, and records what
happened.

Towers are independent, so they are generated in parallel across processes.
Physics here is single-threaded CPU work -- the GPU is only used to draw the
one image per tower -- so worker count is the main throughput lever.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import time
from collections import Counter

from tqdm import tqdm

from jenga_sim import camera, config as C, removal, tower as T
from jenga_sim.dataset import DatasetWriter
from jenga_sim.labeling import max_displacement


def default_workers() -> int:
    """Half the cores, which leaves room for the OS and the efficiency cores.

    On a 4+6 machine like an M4 this picks 5: the four performance cores plus
    one, which is about where the returns flatten off.
    """
    return 4


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


# ---------------------------------------------------------------------------
# One tower's worth of work
# ---------------------------------------------------------------------------
# This runs in a worker process, so everything it touches must be picklable:
# it takes plain values in and returns plain dicts out. It writes its own
# images, masks and states (those are per-tower files, no contention) but
# never the CSVs -- the parent writes those so the rows stay ordered.

def run_tower(job: dict) -> dict:
    i = job["tower_id"]
    seed = job["seed"]
    mode = job["mode"]
    pull_direction = job["pull_direction"]
    per_tower = job["per_tower"]

    rng = random.Random(seed)
    result = {"tower_id": i, "seed": seed, "ok": False,
              "tower_row": None, "trials": [], "times": []}

    tw, ok, settle_secs = build_one(seed, rng)
    if not ok:
        return result

    paths = DatasetWriter(job["out"], write_csv=False)
    cam = camera.default_camera(rng)
    cam_str = camera.camera_params_string(cam)

    # Everything below is measured from THIS state.
    T.save_state_npz(tw, paths.state_path(i))
    mem_state = T.save_state(tw)

    rgb, seg = camera.render(tw, cam, rng=rng)
    camera.save_rgb(paths.image_path(i), rgb)
    camera.save_seg(paths.seg_path(i), camera.block_index_map(tw, seg))

    result["ok"] = True
    result["tower_row"] = dict(
        tower_id=i, seed=seed, num_blocks=len(tw.present_ids()),
        num_gaps=tw.n_gaps, settled=True,
        settle_seconds=round(settle_secs, 3), camera_params=cam_str,
    )

    candidates = tw.candidate_ids()
    if mode == "pull" and pull_direction == "side":
        candidates = [b for b in candidates if "side" in removal.pull_directions(tw, b)]
    rng.shuffle(candidates)
    if per_tower is not None:
        candidates = candidates[:per_tower]

    for block in candidates:
        block_index = block + 1
        mask_px = camera.save_binary_mask(
            paths.block_mask_path(i, block_index), tw, seg, block)

        if mode == "pull" and pull_direction == "all":
            directions = removal.pull_directions(tw, block)
        else:
            directions = (pull_direction,)

        for direction in directions:
            T.restore_state(tw, mem_state)
            t0 = time.time()
            out = removal.remove(tw, block, mode, rng, direction=direction)
            result["times"].append(time.time() - t0)
            result["trials"].append(dict(
                tower_id=i, block_id=block_index,
                level=tw.level_of[block], position_in_level=tw.slot_of[block],
                mode=mode, outcome=out.outcome, pull_direction=out.pull_direction,
                max_displacement=round(out.max_displacement, 5),
                max_tilt_deg=round(out.max_tilt_deg, 3),
                peak_force=round(out.peak_force, 2),
                extract_time=round(out.extract_time, 3),
                mask_pixels=mask_px, seed=seed, camera_params=cam_str,
            ))

    camera.release_renderers()
    return result


def main():
    ap = argparse.ArgumentParser(description="Generate Jenga removal data")
    ap.add_argument("--towers", type=int, default=C.DEFAULT_TOWERS)
    ap.add_argument("--mode", choices=["delete", "pull"], default="delete")
    ap.add_argument("--pull-direction", choices=["end", "side", "all"], default="end",
                    help="end: one random end; side: outer blocks only; "
                         "all: both ends plus outward side when available")
    ap.add_argument("--out", default="data/")
    ap.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    ap.add_argument("--candidates", default=str(C.CANDIDATES_PER_TOWER),
                    help="blocks to try per tower, or 'all'")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="parallel worker processes; 1 runs in this process, "
                         "which is what you want when debugging")
    args = ap.parse_args()
    if args.mode != "pull" and args.pull_direction != "end":
        ap.error("--pull-direction applies only to --mode pull")

    per_tower = None if args.candidates == "all" else int(args.candidates)
    workers = max(1, min(args.workers, args.towers))

    writer = DatasetWriter(args.out)
    counts = Counter()
    rejected = []
    trial_times = []
    tower_rows = []
    trial_rows = []
    t_start = time.time()

    jobs = [dict(tower_id=i, seed=args.seed * 100000 + i, mode=args.mode,
                 pull_direction=args.pull_direction, out=args.out,
                 per_tower=per_tower)
            for i in range(args.towers)]

    bar = tqdm(total=len(jobs), desc="towers", unit="tower")

    def absorb(res: dict) -> None:
        if not res["ok"]:
            rejected.append(res["tower_id"])
            bar.write(f"[reject] tower {res['tower_id']} "
                      f"(seed {res['seed']}) would not stand")
        else:
            tower_rows.append(res["tower_row"])
            trial_rows.extend(res["trials"])
            trial_times.extend(res["times"])
            for row in res["trials"]:
                counts[row["outcome"]] += 1
        bar.update(1)
        bar.set_postfix(**{k: counts[k] for k in ("stable", "collapse", "stuck")
                           if counts[k]})

    if workers == 1:
        for job in jobs:
            absorb(run_tower(job))
    else:
        # "spawn", not "fork": a forked child inherits the parent's OpenGL
        # context, which is not valid across fork and crashes the renderer.
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for res in pool.imap_unordered(run_tower, jobs):
                absorb(res)
    bar.close()

    # Workers finish out of order; sort so the CSVs are byte-identical
    # whatever --workers was set to.
    tower_rows.sort(key=lambda r: r["tower_id"])
    trial_rows.sort(key=lambda r: (r["tower_id"], r["block_id"],
                                   r.get("pull_direction") or ""))
    for row in tower_rows:
        writer.write_tower(**row)
    for row in trial_rows:
        writer.write_trial(**row)
    writer.close()
    camera.release_renderers()

    total = sum(counts.values())
    elapsed = time.time() - t_start
    print("\n" + "=" * 58)
    print(f"  towers requested : {args.towers}")
    print(f"  towers rejected  : {len(rejected)}")
    print(f"  trials written   : {total}   mode={args.mode}")
    print(f"  workers          : {workers}")
    print("-" * 58)
    for label in ("stable", "collapse", "stuck"):
        n = counts[label]
        pct = 100.0 * n / total if total else 0.0
        print(f"  {label:<9}: {n:5d}  ({pct:5.1f}%)")
    print("-" * 58)
    if trial_times:
        # Only removal trials are timed, not tower building or rendering, so
        # this is CPU time per trial -- not a share of the wall clock.
        print(f"  avg time / trial : {sum(trial_times)/len(trial_times):.2f} s of CPU")
    if elapsed > 0:
        done = args.towers - len(rejected)
        print(f"  throughput       : {60*done/elapsed:.1f} towers/min, "
              f"{total/elapsed:.1f} trials/s")
    print(f"  total wall time  : {elapsed:.0f} s")
    print(f"  output           : {args.out}")
    print("=" * 58)

    missing = [k for k in ("stable", "collapse", "stuck") if counts[k] == 0]
    if missing:
        print(f"\n  NOTE: no {', '.join(missing)} labels in this run.")
        if "collapse" in missing:
            print("        Raise GAPS_MAX in config.py -- pristine towers almost")
            print("        never collapse from a single removal.")
        if "stable" in missing:
            print("        Check untouched controls, contact loads and grip behaviour")
            print("        before changing tower geometry or label thresholds.")
        if "stuck" in missing:
            print("        Lower PULL_MAX_FORCE in config.py (delete mode can")
            print("        never produce `stuck` -- it is a pull-only label).")


if __name__ == "__main__":
    main()
