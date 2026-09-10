#!/usr/bin/env python3
"""Tabulate temporal aggregation across configs, and test whether it improves brand results.

Three questions, and the third is the one that decides whether this ships:

    1. INDEX SIZE   how many vectors does one appearance collapse to?
    2. STRUCTURE    how long are the tracks, and how many are single-frame noise candidates?
    3. BRAND LEVEL  does tracking find the same brands with fewer rows, and does `--min-frames`
                    remove the absent-brand controls faster than it removes real brands?

(3) matters because (1) and (2) are guaranteed by construction -- any grouping reduces rows and
produces a length distribution. Only (3) says the grouping is right. It is measured by running
the same image-query identification used in 12_logo_pool over the per-frame detections and then
over the tracks, and comparing which target brands survive against which known-absent controls
survive.

    python3 eval/tools/score_temporal.py --experiment eval/experiments/15_temporal/nba \\
                                         --brands nba
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "box_gt"))

from aggregate_temporal import load, representatives, track  # noqa: E402


def brand_hits(vectors: np.ndarray, refs: np.ndarray, ref_brand: List[str],
               names: List[str], floor: float) -> Dict[str, int]:
    """Rows assigned to each of `names` by nearest reference, above `floor`.

    On the GPU when there is one. This is 83k crops against 69k references at 768-d, roughly
    9 TFLOP per call and a dozen calls per title -- numpy on CPU turns a scoring pass into
    half an hour of matrix multiply, which is not the thing being measured.
    """
    if len(vectors) == 0:
        return {n: 0 for n in names}

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        torch, device = None, "cpu"

    if torch is not None and device == "cuda":
        ref_t = torch.from_numpy(np.ascontiguousarray(refs)).to(device)
        best_all, arg_all = [], []
        for start in range(0, len(vectors), 8192):
            chunk = torch.from_numpy(
                np.ascontiguousarray(vectors[start:start + 8192])).to(device)
            block = chunk @ ref_t.T
            values, indices = block.max(dim=1)
            best_all.append(values.cpu().numpy())
            arg_all.append(indices.cpu().numpy())
        best = np.concatenate(best_all)
        arg = np.concatenate(arg_all)
    else:
        best = np.zeros(len(vectors), np.float32)
        arg = np.zeros(len(vectors), np.int64)
        for start in range(0, len(vectors), 2048):
            block = vectors[start:start + 2048] @ refs.T
            best[start:start + 2048] = block.max(axis=1)
            arg[start:start + 2048] = block.argmax(axis=1)

    winner = np.array([ref_brand[i] for i in arg])
    ok = best >= floor
    return {n: int(((winner == n) & ok).sum()) for n in names}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--brands", choices=["nba", "mitchells"])
    parser.add_argument("--pool", default="eval/experiments/12_logo_pool/pool.npz")
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--cos", type=float, default=0.7)
    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--keep", type=int, default=2,
                        help="vectors per track, matching aggregate_temporal")
    parser.add_argument("--floor", type=float, default=0.55)
    args = parser.parse_args()

    runs = sorted(n for n in os.listdir(args.experiment)
                  if os.path.exists(os.path.join(args.experiment, n, "out.jsonl")))
    if not runs:
        print(f"no runs under {args.experiment}", file=sys.stderr)
        return 1

    pool = refs = ref_brand = targets = controls = mapping = None
    if args.brands and os.path.exists(args.pool):
        sys.path.insert(0, os.path.join(HERE, "..", "box_gt"))
        from score_brands import BRAND_LISTS, CONTROLS
        from score_brands_image import map_to_pool

        pool = np.load(args.pool, allow_pickle=True)
        refs, ref_brand = pool["vectors"], list(pool["brands"])
        targets = BRAND_LISTS[args.brands]
        mapping = map_to_pool(targets, sorted(set(ref_brand)))
        controls = map_to_pool(CONTROLS, sorted(set(ref_brand)))

    head = (f"{'config':16}{'dets':>8}{'tracks':>8}{'rows saved':>12}"
            f"{'median len':>12}{'single':>9}{'mark-sec':>11}")
    print(f"\n{args.experiment}\n{head}\n{'-' * len(head)}")

    results = {}
    for name in runs:
        run_dir = os.path.join(args.experiment, name)
        sources = load(os.path.join(run_dir, "out.jsonl"))
        dets = [d for v in sources.values() for d in v]
        tracks = []
        for source, detections in sources.items():
            for tr in track(detections, args.iou, args.cos, args.max_gap):
                tr["source"] = source
                tracks.append(tr)
        lengths = np.array([t["frames"] for t in tracks]) if tracks else np.array([0])
        singles = int((lengths == 1).sum())
        duration = lengths.sum() / args.fps
        print(f"{name:16}{len(dets):>8}{len(tracks):>8}"
              f"{f'{100 * (1 - len(tracks) / max(1, len(dets))):.0f}%':>12}"
              f"{int(np.median(lengths)):>12}"
              f"{f'{100 * singles / max(1, len(tracks)):.0f}%':>9}"
              f"{f'{duration:.0f}s':>11}")
        results[name] = {"dets": dets, "tracks": tracks}

    if targets is None:
        return 0

    # Does tracking keep the brands and drop the controls?
    print(f"\nbrand level, image query at cos>={args.floor} "
          f"({len(targets)} targets, {sum(1 for c in controls)} controls in pool)")
    head = (f"{'config':16}{'per-frame rows':>16}{'targets':>9}{'ctrl':>6}"
            f"{'tracks':>9}{'targets':>9}{'ctrl':>6}{'min>=3':>9}{'targets':>9}{'ctrl':>6}")
    print(f"{head}\n{'-' * len(head)}")
    for name, data in results.items():
        row = f"{name:16}"
        det_vecs = np.array([d["vector"] for d in data["dets"] if d["vector"] is not None],
                            dtype=np.float32)
        row += f"{len(det_vecs):>16}"
        hits = brand_hits(det_vecs, refs, ref_brand,
                          [mapping[t] for t in targets if t in mapping]
                          + [controls[c] for c in controls], args.floor)
        tgt = sum(1 for t in targets if t in mapping and hits.get(mapping[t], 0))
        ctl = sum(1 for c in controls if hits.get(controls[c], 0))
        row += f"{f'{tgt}/{len(mapping)}':>9}{f'{ctl}/{len(controls)}':>6}"

        for min_frames in (1, 3):
            keep = [t for t in data["tracks"] if t["frames"] >= min_frames]
            members = [m for t in keep for m in representatives(t["members"], args.keep)]
            vecs = np.array([m["vector"] for m in members if m["vector"] is not None],
                            dtype=np.float32)
            hits = brand_hits(vecs, refs, ref_brand,
                              [mapping[t] for t in targets if t in mapping]
                              + [controls[c] for c in controls], args.floor)
            tgt = sum(1 for t in targets if t in mapping and hits.get(mapping[t], 0))
            ctl = sum(1 for c in controls if hits.get(controls[c], 0))
            row += f"{len(vecs):>9}{f'{tgt}/{len(mapping)}':>9}{f'{ctl}/{len(controls)}':>6}"
        print(row)
    print("\ntargets = brands present in the title AND in the pool; ctrl = brands known absent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
