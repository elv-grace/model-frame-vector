#!/usr/bin/env python3
"""Does `crop_padding` change what a crop retrieves? Re-crop and re-embed from saved boxes.

Why this cannot be measured with `usable`
-----------------------------------------
`crop_padding` does not move boxes. It only decides how much context is included when the crop
is cut, so every box-geometry metric -- coverage, containment, usable -- is completely blind to
it. The only way it shows up is in what the resulting vector retrieves.

Why it needs re-measuring now
-----------------------------
The main README measures padding by embedding the 446 ground-truth boxes at four paddings and
asking whether a query built at one padding still retrieves the same object from an index built
at another. That is a self-consistency measurement: it establishes that a padding MISMATCH costs
about 8% of top-1, which is why the query side pads to match the index.

It does not ask which padding is best in absolute terms, against real references. And the box
distribution has since changed twice -- the confidence gate came down (step 1) and tiling was
added (step 2), both of which admit smaller and looser boxes than the ones that measurement was
made on.

Why re-embed rather than re-run
-------------------------------
Detection is the expensive half and padding does not affect it, so the boxes are reused: read a
run's boxes, cut each crop from the source frame again at each padding, embed with the same
SigLIP 2 checkpoint and settings the tagger uses, and score brand recall against the reference
pool. One detection run, N embedding passes, and the detections are held identical across the
sweep by construction rather than by hoping two runs agree.

    python3 eval/tools/sweep_padding.py \\
        --run eval/experiments/11_titles/nba/B2_step1_conf07 \\
        --frames test-files/frames/nba --brands nba
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "box_gt"))

MODEL = "google/siglip2-base-patch16-naflex"
MAX_NUM_PATCHES = 256
MIN_CROP_PIXELS = 16


def read_boxes(path: str) -> List[Dict]:
    rows = []
    with open(path) as handle:
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
            rows.append({
                "source": os.path.splitext(os.path.basename(data["source_media"]))[0],
                "box": (box["x1"], box["y1"], box["x2"], box["y2"]),
                "score": data["additional_info"].get("score", 0.0),
            })
    return rows


def crop(image, box, padding: float):
    """Exactly BaseDetector._crop, in PIL terms: pad by a fraction of the box's own size."""
    width, height = image.size
    x1, y1, x2, y2 = (box[0] * width, box[1] * height, box[2] * width, box[3] * height)
    if min(x2 - x1, y2 - y1) < MIN_CROP_PIXELS:
        return None
    pad_w, pad_h = (x2 - x1) * padding, (y2 - y1) * padding
    cx1, cy1 = max(0, int(round(x1 - pad_w))), max(0, int(round(y1 - pad_h)))
    cx2, cy2 = min(width, int(round(x2 + pad_w))), min(height, int(round(y2 + pad_h)))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return image.crop((cx1, cy1, cx2, cy2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--frames", required=True, help="directory holding the source frames")
    parser.add_argument("--brands", required=True, choices=["nba", "mitchells"])
    parser.add_argument("--pool", default="eval/experiments/12_logo_pool/pool.npz")
    parser.add_argument("--paddings", default="0.0,0.06,0.15,0.30,0.50")
    parser.add_argument("--floor", type=float, default=0.55)
    parser.add_argument("--limit", type=int, default=0, help="cap boxes, for a quick pass")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    from score_brands import BRAND_LISTS, CONTROLS
    from score_brands_image import map_to_pool

    pool = np.load(args.pool, allow_pickle=True)
    refs, ref_brand = pool["vectors"], list(pool["brands"])
    targets = BRAND_LISTS[args.brands]
    mapping = map_to_pool(targets, sorted(set(ref_brand)))
    controls = map_to_pool(CONTROLS, sorted(set(ref_brand)))

    boxes = read_boxes(os.path.join(args.run, "out.jsonl"))
    if args.limit:
        # Highest-scoring first, so a capped pass is the boxes most likely to matter rather
        # than whichever frames sort first.
        boxes = sorted(boxes, key=lambda r: -r["score"])[:args.limit]
    by_source: Dict[str, List[Dict]] = {}
    for row in boxes:
        by_source.setdefault(row["source"], []).append(row)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    ref_t = torch.from_numpy(np.ascontiguousarray(refs)).to(device)

    paddings = [float(v) for v in args.paddings.split(",")]
    print(f"{len(boxes)} boxes over {len(by_source)} frames | paddings {paddings}")
    print(f"\n{'padding':>9}{'crops':>8}{'targets':>9}{'ctrl':>7}{'mean best cos':>15}")
    print("-" * 48)

    for padding in paddings:
        vectors, kept = [], 0
        pending = []

        def flush():
            nonlocal vectors
            if not pending:
                return
            inputs = processor(images=pending, max_num_patches=MAX_NUM_PATCHES,
                               return_tensors="pt").to(device)
            with torch.no_grad():
                pooled = model.get_image_features(**inputs).pooler_output.float()
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
            vectors.append(pooled)
            pending.clear()

        for source, rows in by_source.items():
            path = None
            for ext in (".jpg", ".png", ".jpeg"):
                candidate = os.path.join(args.frames, source + ext)
                if os.path.exists(candidate):
                    path = candidate
                    break
            if path is None:
                continue
            image = Image.open(path).convert("RGB")
            for row in rows:
                piece = crop(image, row["box"], padding)
                if piece is None:
                    continue
                pending.append(piece)
                kept += 1
                if len(pending) >= args.batch:
                    flush()
        flush()

        if not vectors:
            print(f"{padding:9.2f}{0:8d}{'-':>9}{'-':>7}{'-':>15}")
            continue
        allv = torch.cat(vectors)
        best_all, arg_all = [], []
        for start in range(0, len(allv), 8192):
            block = allv[start:start + 8192] @ ref_t.T
            values, indices = block.max(dim=1)
            best_all.append(values.cpu().numpy())
            arg_all.append(indices.cpu().numpy())
        best = np.concatenate(best_all)
        arg = np.concatenate(arg_all)
        winner = np.array([ref_brand[i] for i in arg])
        ok = best >= args.floor
        per_brand = {t: int(((winner == mapping[t]) & ok).sum())
                     for t in targets if t in mapping}
        found = sum(1 for v in per_brand.values() if v)
        ctl = sum(1 for c in controls if ((winner == controls[c]) & ok).any())
        print(f"{padding:9.2f}{kept:8d}{f'{found}/{len(mapping)}':>9}"
              f"{f'{ctl}/{len(controls)}':>7}{best.mean():15.4f}"
              f"   {', '.join(f'{t}={n}' for t, n in per_brand.items() if n)}")

    print("\n`mean best cos` is over ALL crops, most of which are not target brands -- read it "
          "as\na drift indicator, not a quality score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
