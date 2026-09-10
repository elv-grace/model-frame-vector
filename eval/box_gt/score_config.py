#!/usr/bin/env python3
"""Score production tagger runs AT THEIR OWN OPERATING POINT, and slice by box shape.

Why this exists beside score_boxes.py
-------------------------------------
`score_boxes.py` answers "which detector is best": it sweeps the confidence threshold per
backend and reports each at its own optimum, because detector scores are not comparable
across models. That is the right instrument for choosing a model and the wrong one for
checking a config change -- a run that has ALREADY been gated by the tagger gets swept a
second time here, and the sweep reports a sub-threshold optimum that the config does not
deliver.

Two consequences of that mismatch are worth stating, because both are load-bearing:

1. `score_boxes.py`'s `coverage` column never applies a score gate at all -- `coverage_recall`
   takes an *IoU* threshold, not a confidence one -- so it scores whatever floor the run file
   happens to have been written at. The README's headline coverage figures (gdino 0.61,
   yoloe26 0.25) are that number, and they describe an ungated detector rather than the
   shipping config. At the shipped `conf` of 0.15, gdino's is 0.33.

2. Nothing in the existing scoring reads box SHAPE. Two thirds of ground-truth marks are small,
   which is measured and documented -- but marks are also often wide, and a courtside hoarding
   or a ribbon board is a 5:1-to-9:1 wordmark. Aspect ratio turns out to separate the backends
   far more sharply than size does, and it was invisible.

So this script fixes the operating point (whatever the run emitted, that is what is scored) and
reports coverage by aspect band as well as by size band.

    python3 eval/box_gt/score_config.py
    python3 eval/box_gt/score_config.py --runs 10_config_ab --iou 0.5

Three recall columns, because they fail differently
---------------------------------------------------
    coverage      the mark was hit at IoU 0.5. The strict localisation number.
    containment   some box covers >=90% of the mark's area, at any size.
    usable        some box covers >=90% of the mark AND is no more than 4x its area.

Coverage and containment diverge on exactly YOLOE's wide-mark failure: on a 261x29 px hoarding
it emits a roughly square box four times too large, scoring IoU 0.24 and containment 1.00. It is
not missing the board, it is boxing it loosely -- and downstream that is worse than a miss, since
the crop gets embedded and a vector of "mostly LED board, 20% wordmark" takes an index slot
without ever retrieving.

But containment alone is **density-sensitive and will mislead you**. A prompt-free backend
emitting 133 boxes per frame scores containment 0.96 while its coverage is 0.125: with that many
boxes, nearly every small mark falls inside *something* -- a jersey, a hat, a chair. Nothing was
localised. `usable` is the column to compare across sources at different box rates, and
`usable / 1k boxes` is what makes a 25-box/frame detector and a 133-box/frame one commensurable.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import paths  # noqa: E402

BRAND_TAGS = {"brand", "logo", "letter logo", "car logo", "emblem", "label"}

# Aspect bands. The split at 3.0 is where a mark stops being a badge and starts being a
# hoarding: below it the two backends behave alike, above it they diverge by 4x.
ASPECT_BANDS = [(0.0, 1.5, "<1.5 compact"), (1.5, 3.0, "1.5-3"),
                (3.0, 5.0, "3-5 wide"), (5.0, 1e9, ">=5 banner")]
SIZE_BANDS = [(0, 16, "<16px"), (16, 24, "16-24"), (24, 32, "24-32"),
              (32, 48, "32-48"), (48, 1e9, "48+")]


def png_size(path: str) -> Tuple[int, int]:
    """Width and height from the IHDR chunk, so scoring needs no image library."""
    with open(path, "rb") as handle:
        head = handle.read(26)
    return struct.unpack(">II", head[16:24])


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def box_area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def contained(gt, det, frac: float) -> bool:
    """Does `det` cover at least `frac` of `gt`'s area, whatever its own size?"""
    x1, y1 = max(gt[0], det[0]), max(gt[1], det[1])
    x2, y2 = min(gt[2], det[2]), min(gt[3], det[3])
    if x2 <= x1 or y2 <= y1:
        return False
    area = box_area(gt)
    return area > 0 and (x2 - x1) * (y2 - y1) / area >= frac


def is_usable(gt, det, frac: float, bloat: float) -> bool:
    """Contains the mark, and is not so much larger than it that the crop stops being of it."""
    return contained(gt, det, frac) and box_area(det) <= bloat * box_area(gt)


def load_gt() -> Tuple[List[Dict], int]:
    with open(paths.BOX_LABELS) as handle:
        frames = json.load(handle)["frames"]
    marks, done = [], 0
    for name, frame in sorted(frames.items()):
        if not frame.get("done"):
            continue
        done += 1
        image = os.path.join(paths.FRAMESET, "frames", f"{name}.png")
        width, height = png_size(image) if os.path.exists(image) else (1920, 1080)
        for box in frame["boxes"]:
            if box["cls"] != "brand":
                continue
            geom = (box["x1"], box["y1"], box["x2"], box["y2"])
            pixel_w, pixel_h = (geom[2] - geom[0]) * width, (geom[3] - geom[1]) * height
            if pixel_w <= 0 or pixel_h <= 0:
                continue
            marks.append({"frame": name, "box": geom,
                          "aspect": pixel_w / pixel_h, "short": min(pixel_w, pixel_h)})
    return marks, done


def load_run(path: str) -> Dict[str, List[Tuple]]:
    """Brand boxes per frame, as the tagger actually emitted them -- already gated, deduped
    and truncated to max_detections, which is the whole point.

    Assumes ONE record per detection. That holds for every run under eval/, which are made
    without `output_tags`, and the strip step in run_config_ab.sh enforces it by dropping
    vector-less records before it removes the vectors.

    It matters because `output_tags` emits a second, vector-less FrameTag per detection sharing
    the same `additional_info`, so after vectors are stripped the twin is byte-identical to its
    parent and every box here would be counted twice. There is no field left to tell them apart:
    `additional_info.kind` used to do it and no longer exists.
    """
    out = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # The runtime interleaves `progress` records with `tag` records on one stream.
            if record.get("type") != "tag":
                continue
            data = record["data"]
            if data["tag"] not in BRAND_TAGS:
                continue
            box = data["frame_info"]["box"]
            frame = os.path.splitext(os.path.basename(data["source_media"]))[0]
            out[frame].append((box["x1"], box["y1"], box["x2"], box["y2"]))
    return out


def band_of(value: float, bands) -> str:
    for low, high, name in bands:
        if low <= value < high:
            return name
    return bands[-1][2]


def score(marks, dets, iou_thr: float, contain_frac: float, bloat: float) -> Dict:
    hit = {"coverage": [], "containment": [], "usable": []}
    per_band = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for mark in marks:
        rows = dets.get(mark["frame"], [])
        covered = any(iou(mark["box"], d) >= iou_thr for d in rows)
        hit["coverage"].append(covered)
        hit["containment"].append(any(contained(mark["box"], d, contain_frac) for d in rows))
        hit["usable"].append(any(is_usable(mark["box"], d, contain_frac, bloat) for d in rows))
        for kind, bands, value in (("aspect", ASPECT_BANDS, mark["aspect"]),
                                   ("size", SIZE_BANDS, mark["short"])):
            cell = per_band[kind][band_of(value, bands)]
            cell[0] += covered
            cell[1] += 1
    total = len(marks) or 1
    boxes = sum(len(v) for v in dets.values())
    return {
        "coverage": sum(hit["coverage"]) / total,
        "containment": sum(hit["containment"]) / total,
        "usable": sum(hit["usable"]) / total,
        "boxes": boxes,
        # Marks made usable per 1000 boxes emitted: the only way to compare a 25-box/frame
        # detector against a 133-box/frame one without rewarding sheer volume.
        "usable_per_1k": sum(hit["usable"]) / boxes * 1000 if boxes else float("nan"),
        "bands": {k: {b: (c[0] / c[1], c[1]) for b, c in v.items()} for k, v in per_band.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="10_config_ab",
                        help="experiment name or path holding runs/<config>/out.jsonl")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--containment", type=float, default=0.9,
                        help="fraction of a mark's area some box must cover")
    parser.add_argument("--bloat", type=float, default=4.0,
                        help="for `usable`, the largest a box may be relative to the mark")
    args = parser.parse_args()

    marks, frames = load_gt()
    print(f"{len(marks)} ground-truth brand marks over {frames} frames "
          f"| IoU {args.iou} | containment {args.containment}")

    results = {}
    for path in sorted(glob.glob(paths.runs_glob(args.runs))):
        name = os.path.basename(os.path.dirname(path))
        wall_path = os.path.join(os.path.dirname(path), "wall_seconds")
        wall = float(open(wall_path).read()) if os.path.exists(wall_path) else float("nan")
        row = score(marks, load_run(path), args.iou, args.containment, args.bloat)
        row["ms_per_frame"] = wall * 1000 / frames
        results[name] = row

    if not results:
        print(f"no runs under {paths.runs_glob(args.runs)}", file=sys.stderr)
        return 1

    head = (f"{'config':24}{'coverage':>10}{'contain':>9}{'usable':>8}"
            f"{'boxes/fr':>10}{'usable/1k':>11}{'wall/frame*':>13}")
    print(f"\n{head}\n{'-' * len(head)}")
    for name, row in results.items():
        print(f"{name:24}{row['coverage']:10.3f}{row['containment']:9.3f}{row['usable']:8.3f}"
              f"{row['boxes'] / frames:10.1f}{row['usable_per_1k']:11.1f}"
              f"{row['ms_per_frame']:11.0f}ms")
    # Stated with an asterisk because at 25 frames it is mostly container start and weight
    # load, which is why the CHEAPER backend posts the larger number here. For throughput,
    # run against video (eval/experiments/10_config_ab/video/) where the load amortises.
    print("\n* wall clock / frame including container start and model load -- "
          "NOT throughput at n=25; see the video runs.")

    for kind, bands in (("aspect", ASPECT_BANDS), ("size", SIZE_BANDS)):
        order = [b[2] for b in bands]
        print(f"\ncoverage by {kind} band")
        counts = next(iter(results.values()))["bands"][kind]
        print(f"{'config':24}" + "".join(f"{b:>15}" for b in order))
        print(f"{'(n marks)':24}" + "".join(f"{counts.get(b, (0, 0))[1]:>15}" for b in order))
        for name, row in results.items():
            cells = "".join(f"{row['bands'][kind].get(b, (0, 0))[0]:15.2f}" for b in order)
            print(f"{name:24}{cells}")

    out = os.path.join(paths.experiment(args.runs), "scores_config.json")
    with open(out, "w") as handle:
        json.dump(results, handle, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
