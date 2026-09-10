#!/usr/bin/env python3
"""Collapse per-frame detections into tracks: one record per mark-appearance, with a time range.

Why the pipeline needs this
--------------------------
The tagger is a `FrameModel` and correctly knows nothing about time: it emits one vector per
detection per sampled frame. For a *search* index that is fine. For a brand coverage report --
"which brands appeared, and for how long" -- it is the wrong shape, and it is wasteful in a way
that got worse with every step of this work:

    a courtside hoarding visible for 30s at 2 fps is 60 near-identical crops, 60 near-identical
    768-d vectors and 60 index rows, describing ONE appearance.

Three things fall out of grouping them, and they are the reason this is worth doing before any
further detector work:

    INDEX SIZE   one vector per appearance instead of per frame. Step 1 multiplied crops ~7x
                 and step 2 doubled them again, so this is where that is paid back.

    DURATION     a coverage report sells visibility, and visibility is a time range. Per-frame
                 counts only approximate it, and they approximate it *badly* when detection is
                 intermittent -- which 11_titles showed it is.

    PRECISION    for free, and it is the cheap answer to the single-crop noise that made the
                 title measurements hard to read. A box that recurs in the same screen position
                 across forty frames is a real mark; one that appears in a single frame and
                 retrieves weakly is a distractor getting lucky. `--min-frames` is that filter.

How tracks are formed
---------------------
Greedy nearest-match per frame, gated on BOTH geometry and appearance:

    IoU        the box must overlap its predecessor. A static hoarding barely moves within a
               shot, and across a camera cut it jumps -- so IoU gating does implicit shot
               segmentation, and no shot detector is needed for a first cut. (`model-shot`
               would do it properly, and would help on a slow pan where IoU decays smoothly.)

    cosine     the crop must still look like the same thing. Geometry alone merges two different
               logos that happen to occupy the same screen position across a cut, which on
               broadcast sport -- where the camera returns to the same framing repeatedly -- is
               not a hypothetical.

A track tolerates `--max-gap` missed frames before it is closed, because detection is
intermittent: the mark is still there, the detector blinked. That is exactly the case per-frame
counting gets wrong and a track gets right.

    python3 eval/tools/aggregate_temporal.py --run eval/experiments/15_temporal/nba/D_tiled
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "box_gt"))


def load(path: str):
    """Detections grouped by source, each sorted by frame index."""
    rows: Dict[str, List[Dict]] = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") != "tag":
                continue
            data = record["data"]
            vector = data.get("vector")
            frame_info = data.get("frame_info") or {}
            box = data["additional_info"].get("box") or frame_info.get("box")
            if box is None:
                continue
            rows.setdefault(data["source_media"], []).append({
                "frame": frame_info.get("frame_idx", 0),
                "time": data.get("start_time", 0),
                "box": (box["x1"], box["y1"], box["x2"], box["y2"]),
                "score": data["additional_info"].get("score", 0.0),
                "prompt": data["additional_info"].get("prompt"),
                "vector": np.asarray(vector, dtype=np.float32) if vector else None,
                "tag": data["tag"],
            })
    for source in rows:
        rows[source].sort(key=lambda r: r["frame"])
    return rows


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def track(detections: List[Dict], iou_thr: float, cos_thr: float, max_gap: int) -> List[Dict]:
    """Greedy per-frame association into tracks.

    `max_gap` counts SAMPLED frames, not source frame indices. The tagger stamps
    `frame_info.frame_idx` from the source video, so at fps 1 against 25 fps material
    consecutive samples are 25 apart -- measuring the gap in those units closes every track
    after one frame and silently returns one track per detection.
    """
    open_tracks: List[Dict] = []
    closed: List[Dict] = []

    by_frame: Dict[int, List[Dict]] = {}
    for det in detections:
        by_frame.setdefault(det["frame"], []).append(det)

    # Source frame index -> position in the sampled sequence.
    ordinal = {raw: i for i, raw in enumerate(sorted(by_frame))}
    by_frame = {ordinal[raw]: dets for raw, dets in by_frame.items()}

    for frame in sorted(by_frame):
        # Retire tracks that have been unmatched for too long.
        still_open = []
        for tr in open_tracks:
            if frame - tr["last_frame"] > max_gap:
                closed.append(tr)
            else:
                still_open.append(tr)
        open_tracks = still_open

        # Highest score first, so a confident detection claims its track before a weak one.
        claimed = set()
        for det in sorted(by_frame[frame], key=lambda d: -d["score"]):
            best, best_score = None, 0.0
            for i, tr in enumerate(open_tracks):
                if i in claimed or tr["last_frame"] == frame:
                    continue
                geometry = iou(det["box"], tr["last_box"])
                if geometry < iou_thr:
                    continue
                appearance = 1.0
                # Compare against the track's most recent member, not a fixed representative:
                # a mark drifts in scale and blur across a shot, and the nearest-in-time
                # appearance is the fair comparison.
                previous = tr["members"][-1]["vector"]
                if det["vector"] is not None and previous is not None:
                    appearance = float(det["vector"] @ previous)
                    if appearance < cos_thr:
                        continue
                combined = geometry + appearance
                if combined > best_score:
                    best, best_score = i, combined
            if best is None:
                open_tracks.append({
                    "first_frame": frame, "last_frame": frame,
                    "first_time": det["time"], "last_time": det["time"],
                    "last_box": det["box"], "frames": 1,
                    "prompt": det["prompt"], "tag": det["tag"],
                    "members": [det],
                })
            else:
                claimed.add(best)
                tr = open_tracks[best]
                tr["last_frame"], tr["last_time"] = frame, det["time"]
                tr["last_box"] = det["box"]
                tr["frames"] += 1
                tr["members"].append(det)
    return closed + open_tracks


def representatives(members: List[Dict], keep: int) -> List[Dict]:
    """Up to `keep` members of a track, spread evenly across its span.

    Which members survive is not a detail, and it is measured (eval/experiments/15_temporal).
    Keeping ONE per track loses a brand on one of two titles -- Amazon in Mitchells, which has
    two identifiable crops in the whole film, both absorbed into tracks whose loudest member
    retrieved as something else. The original rule here kept the highest DETECTOR score, and
    detector confidence is simply not retrievability.

    Six single-representative rules were compared -- highest score, first, last, middle, biggest
    box, medoid. Only that one Amazon case discriminates between them at all, and among six
    rules it is close to a coin flip which ones land on the right frame; the principled
    candidate (biggest box, since retrieval falls off with crop pixels) LOST. So no
    single-representative rule is defensible on the evidence.

    Keeping TWO costs 8-17 points of row reduction and has no measured recall loss on either
    title or either backend, which is why `keep` defaults to 2. Spread rather than top-N by
    score because top-N can be adjacent frames of the same instant, and the point of a second
    vector is appearance variation.

    Averaging the members was considered and rejected without measuring: a mean blends a sharp
    crop with motion-blurred ones and lands between them, retrieving worse than the sharp one
    alone.
    """
    if keep >= len(members):
        return members
    indices = np.linspace(0, len(members) - 1, keep).round().astype(int)
    return [members[i] for i in dict.fromkeys(indices.tolist())]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="directory holding out.jsonl")
    # Swept in eval/experiments/15_temporal. `iou` anywhere in 0.05-0.70 and `cos` anywhere in
    # 0.00-0.92 keep every findable brand, so neither is delicate -- and `cos` is close to
    # inert, changing 3% of tracks and no brands, so the appearance gate is cheap insurance
    # rather than the load-bearing part it was assumed to be.
    parser.add_argument("--iou", type=float, default=0.3,
                        help="box overlap needed to continue a track")
    parser.add_argument("--cos", type=float, default=0.7,
                        help="crop similarity needed to continue a track")
    # The one sensitive knob: 16 sampled frames of tolerance starts fusing distinct marks and
    # costs a brand. 4 and 8 are both safe; 0 disables tracking entirely (every track is one
    # frame), which is the useful null case.
    parser.add_argument("--max-gap", type=int, default=4,
                        help="sampled frames a track may go unmatched before closing")
    parser.add_argument("--keep", type=int, default=2,
                        help="vectors kept per track, spread across its span (see "
                             "`representatives`); 1 loses a brand, 2 loses none measured")
    # Off by default. It was added as a noise filter on the theory that single-frame tracks are
    # distractors, and that theory is wrong under image-query identification: the absent-brand
    # controls score 0 at every setting, so there is no noise to remove and raising this only
    # costs real brands. It was needed for TEXT queries, where controls did fire.
    parser.add_argument("--min-frames", type=int, default=1,
                        help="drop tracks shorter than this; measured to cost recall, not buy "
                             "precision")
    parser.add_argument("--fps", type=float, default=2.0, help="sampling rate, for durations")
    args = parser.parse_args()

    sources = load(os.path.join(args.run, "out.jsonl"))
    total_dets = sum(len(v) for v in sources.values())

    all_tracks, kept = [], []
    for source, detections in sources.items():
        for tr in track(detections, args.iou, args.cos, args.max_gap):
            tr["source"] = os.path.basename(source)
            all_tracks.append(tr)
            if tr["frames"] >= args.min_frames:
                kept.append(tr)

    lengths = np.array([t["frames"] for t in all_tracks]) if all_tracks else np.array([0])
    rows = sum(len(representatives(t["members"], args.keep)) for t in kept)
    print(f"{total_dets} detections over {len(sources)} sources "
          f"-> {len(all_tracks)} tracks -> {len(kept)} kept (min-frames {args.min_frames})")
    print(f"index rows: {total_dets} -> {rows} at keep={args.keep}  "
          f"({100 * (1 - rows / max(1, total_dets)):.0f}% fewer vectors)")
    print(f"track length: median {int(np.median(lengths))} frames, "
          f"p90 {int(np.quantile(lengths, 0.9))}, max {int(lengths.max())}")
    singles = int((lengths == 1).sum())
    print(f"single-frame tracks: {singles}/{len(all_tracks)} "
          f"({100 * singles / max(1, len(all_tracks)):.0f}%)")
    dur = np.array([t["frames"] / args.fps for t in kept]) if kept else np.array([0.0])
    # Summed across marks, not screen time: many marks are visible at once, so this exceeds
    # the video duration and is "mark-seconds". Screen time per BRAND needs the identification
    # step, not this file.
    print(f"appearance duration: median {np.median(dur):.1f}s, "
          f"{dur.sum():.0f}s summed across marks")

    # Two files, because the two consumers want different shapes and conflating them is what
    # lost a brand in the first place. `appearances` is one row per track with a time range --
    # the coverage report. `vectors` is what goes in the index, `keep` rows per track.
    appearances = os.path.join(args.run, "appearances.jsonl")
    with open(appearances, "w") as handle:
        for tr in kept:
            handle.write(json.dumps({
                "source": tr["source"], "tag": tr["tag"], "prompt": tr["prompt"],
                "frames": tr["frames"],
                "start_time": tr["first_time"], "end_time": tr["last_time"],
                "duration_s": round(tr["frames"] / args.fps, 2),
                "box": {k: round(v, 4) for k, v in
                        zip(("x1", "y1", "x2", "y2"), tr["last_box"])},
            }) + "\n")

    vectors = os.path.join(args.run, "track_vectors.jsonl")
    with open(vectors, "w") as handle:
        for i, tr in enumerate(kept):
            for member in representatives(tr["members"], args.keep):
                handle.write(json.dumps({
                    "track": i, "source": tr["source"], "tag": tr["tag"],
                    "prompt": member["prompt"], "score": member["score"],
                    "start_time": member["time"], "frames": tr["frames"],
                    "box": {k: round(v, 4) for k, v in
                            zip(("x1", "y1", "x2", "y2"), member["box"])},
                    "vector": member["vector"].tolist()
                    if member["vector"] is not None else None,
                }) + "\n")
    print(f"wrote {appearances} ({len(kept)} appearances)")
    print(f"wrote {vectors} ({rows} vectors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
