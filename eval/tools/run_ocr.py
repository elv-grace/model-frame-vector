#!/usr/bin/env python3
"""Read the words on the screen, so a brand text query has something to match exactly.

Why an OCR channel at all
-------------------------
Text queries against the crop vectors have two failure modes, and both are measured. The rank of
the first CORRECT crop for each brand -- correct meaning its nearest reference in the 12_logo_pool
pool is that brand at cosine >= 0.55, which was verified by eye there:

    brand              mark              `fast` rank   `coverage` rank
    NBA                letters                     3               134
    Tissot             wordmark                    5                 4
    AT&T               letters + globe            12                13
    Nike               swoosh, no text            15             1,732
    American Express   dense box, tiny type    5,674            11,240
    Amazon Prime       wordmark, small         5,269            31,496

**Wordmark-shaped.** The brands sort by how legible their lettering is, and in the same order on
both paths. SigLIP's text tower is reading letterforms through the image encoder; given a shape,
or letters too small to resolve, it has nothing to match and guesses. The modality gap leaves it
no headroom either -- a correct text->image match sits near cosine 0.05-0.3 against 0.5-0.9 for
image->image.

**Rank-fragile.** Rank is a position in a list whose length the DETECTOR sets. State Farm moves
23 -> 881 between the two paths with the same query and the same embedder, purely because
`coverage` emits 6.5x the crops; its percentile only moves 0.26% -> 1.52%. So buying detection
recall directly degrades text-query usability, which is a trade no parameter can settle.

An OCR string has neither problem: it matches rather than ranks, so it does not decay as the crop
count grows, and legibility is a text recogniser's job rather than a 768-d bottleneck's. It also
attacks what `12_logo_pool` found to be the real binding constraint -- 16 of 29 named brands have
no reference image at all, and a brand with no reference can still be found if its name is
readable.

What it cannot do is help a mark with no text in it. Nike stays unreachable this way.

Two modes, because they cost very different things
--------------------------------------------------
    --mode frame   one OCR pass per frame, at full resolution. Cost is per FRAME, so it is
                   identical for the `fast` path and the `coverage` path and does not care how
                   many crops the detector emitted. Also yields text boxes, which are candidate
                   brand proposals in their own right -- see score_ocr.py --boxes.
    --mode crop    one OCR pass per crop, from a run's saved boxes, re-cut from the source frame
                   at the tagger's `crop_padding`. Cost is per CROP, which on the coverage path
                   is ~59 per frame. Its one advantage is magnification: a 22 px mark is 22 px in
                   the frame and fills the crop.

Frame mode is the one that can ship. Crop mode is measured to answer whether attaching a string
to each crop needs its own pass, or whether frame-mode strings can be assigned to the crops they
overlap for free.

    PYTHONPATH=~/.cache/ocr-deps CUDA_VISIBLE_DEVICES=3 python3 eval/tools/run_ocr.py \\
        --frames test-files/frames/nba --out eval/experiments/17_ocr/nba/frame_mag1

easyocr is not a dependency of this package and is installed out of tree for the experiment
(`pip install --target ~/.cache/ocr-deps easyocr scikit-image python-bidi shapely pyclipper
ninja lazy_loader imageio tifffile networkx`), hence the PYTHONPATH. Weights land in --models on
first use.

Output: ocr.jsonl, one record per frame (or per crop). Frame-mode boxes are in NORMALIZED frame
coordinates, so they can be compared against detections and against box ground truth without
carrying image sizes around.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Dict, List

import numpy as np

MIN_CROP_PIXELS = 16
CROP_PADDING = 0.06


def quad_to_box(quad) -> List[float]:
    """easyocr returns a four-point polygon; the rest of this repo speaks axis-aligned boxes."""
    points = np.asarray(quad, dtype=np.float32)
    return [float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max())]


def frame_paths(directory: str) -> List[str]:
    out: List[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        out.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(out)


def read_run_boxes(run_dir: str) -> Dict[str, List[Dict]]:
    """Detections per source frame, from a run's out.jsonl. Same reader as sweep_padding.py."""
    by_source: Dict[str, List[Dict]] = {}
    with open(os.path.join(run_dir, "out.jsonl")) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") != "tag":
                continue
            data = record["data"]
            frame_info = data.get("frame_info") or {}
            box = data["additional_info"].get("box") or frame_info.get("box")
            if box is None:
                continue
            source = os.path.splitext(os.path.basename(data["source_media"]))[0]
            by_source.setdefault(source, []).append({
                "box": [box["x1"], box["y1"], box["x2"], box["y2"]],
                "score": data["additional_info"].get("score", 0.0),
                "prompt": data["additional_info"].get("prompt"),
            })
    return by_source


def cut(image, box, padding: float):
    """BaseDetector._crop, in PIL terms, so crop-mode OCR sees what the embedder sees."""
    width, height = image.size
    x1, y1, x2, y2 = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    if min(x2 - x1, y2 - y1) < MIN_CROP_PIXELS:
        return None
    pad_w, pad_h = (x2 - x1) * padding, (y2 - y1) * padding
    cx1, cy1 = max(0, int(round(x1 - pad_w))), max(0, int(round(y1 - pad_h)))
    cx2, cy2 = min(width, int(round(x2 + pad_w))), min(height, int(round(y2 + pad_h)))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return image.crop((cx1, cy1, cx2, cy2))


def build_reader(models: str, gpu: bool):
    import easyocr

    return easyocr.Reader(["en"], gpu=gpu, model_storage_directory=models, verbose=False)


def run_frame_mode(reader, files: List[str], handle, mag: float) -> Dict:
    from PIL import Image

    regions, start = 0, time.time()
    for path in files:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        out = []
        for quad, text, conf in reader.readtext(np.asarray(image), mag_ratio=mag):
            x1, y1, x2, y2 = quad_to_box(quad)
            out.append({"text": text, "conf": round(float(conf), 4),
                        "box": [x1 / width, y1 / height, x2 / width, y2 / height]})
        regions += len(out)
        handle.write(json.dumps({
            "frame": os.path.splitext(os.path.basename(path))[0],
            "size": [width, height], "regions": out}) + "\n")
    elapsed = time.time() - start
    return {"mode": "frame", "frames": len(files), "regions": regions,
            "seconds": round(elapsed, 1),
            "per_frame_ms": round(elapsed / max(1, len(files)) * 1000, 1)}


def run_crop_mode(reader, run_dir: str, frames_dir: str, handle, mag: float,
                  limit: int, padding: float) -> Dict:
    from PIL import Image

    by_source = read_run_boxes(run_dir)
    total = sum(len(v) for v in by_source.values())
    if limit and total > limit:
        # Highest-scoring first, so a capped pass is the crops most likely to matter rather
        # than whichever frames sort first. The cap is reported, not hidden.
        flat = sorted(((s, r) for s, rows in by_source.items() for r in rows),
                      key=lambda pair: -pair[1]["score"])[:limit]
        by_source = {}
        for source, row in flat:
            by_source.setdefault(source, []).append(row)

    crops, regions, start = 0, 0, time.time()
    for source, rows in by_source.items():
        path = None
        for ext in (".jpg", ".png", ".jpeg"):
            candidate = os.path.join(frames_dir, source + ext)
            if os.path.exists(candidate):
                path = candidate
                break
        if path is None:
            continue
        image = Image.open(path).convert("RGB")
        for row in rows:
            piece = cut(image, row["box"], padding)
            if piece is None:
                continue
            found = reader.readtext(np.asarray(piece), mag_ratio=mag)
            crops += 1
            regions += len(found)
            handle.write(json.dumps({
                "frame": source, "box": row["box"], "score": row["score"],
                "prompt": row["prompt"],
                # Crop-mode boxes are relative to the CROP and not comparable to frame
                # coordinates, so only the strings are kept.
                "regions": [{"text": t, "conf": round(float(c), 4)} for _, t, c in found],
            }) + "\n")
    elapsed = time.time() - start
    return {"mode": "crop", "run": run_dir, "crops": crops, "of_total": total,
            "regions": regions, "seconds": round(elapsed, 1),
            "per_crop_ms": round(elapsed / max(1, crops) * 1000, 2)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["frame", "crop"], default="frame")
    parser.add_argument("--frames", required=True, help="directory holding the source frames")
    parser.add_argument("--out", required=True, help="directory to write ocr.jsonl into")
    parser.add_argument("--run", help="crop mode: the run directory whose boxes to re-cut")
    parser.add_argument("--models", default=os.path.expanduser("~/.cache/ocr-models"))
    parser.add_argument("--mag", type=float, default=1.0,
                        help="easyocr mag_ratio: upsample before text DETECTION, not recognition")
    parser.add_argument("--limit", type=int, default=0, help="crop mode: cap crops")
    parser.add_argument("--padding", type=float, default=CROP_PADDING)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.mode == "crop" and not args.run:
        parser.error("--mode crop needs --run")

    os.makedirs(args.out, exist_ok=True)
    reader = build_reader(args.models, gpu=not args.cpu)

    path = os.path.join(args.out, "ocr.jsonl")
    with open(path, "w") as handle:
        if args.mode == "frame":
            files = frame_paths(args.frames)
            print(f"{len(files)} frames -> {path}")
            stats = run_frame_mode(reader, files, handle, args.mag)
        else:
            stats = run_crop_mode(reader, args.run, args.frames, handle, args.mag,
                                  args.limit, args.padding)
    stats["mag_ratio"] = args.mag
    with open(os.path.join(args.out, "stats.json"), "w") as handle:
        json.dump(stats, handle, indent=1)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
