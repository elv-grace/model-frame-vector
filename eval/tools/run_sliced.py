#!/usr/bin/env python3
"""Sliced (tiled) inference for the brand detector, plus an optional OWLv2 union.

The problem being attacked
--------------------------
Brand is a small-object problem: two thirds of ground-truth marks are under 32 px, and even the
shipped config only reaches 0.16 coverage below 16 px and 0.46 at 16-24 px. Resolution is the
obvious lever and it is *unavailable* for the backend that ships, because Grounding DINO cannot
be scaled -- experiment 06 measured brand AP collapsing 0.205 -> 0.001 past its native 800,
taking person down with it. That is a DETR-family property: the decoder's learned reference
points are tuned to the training resolution.

Slicing gets around it by changing what is in the frame rather than the input size. A 1920x1080
frame handed to Grounding DINO is downscaled to 800 on the shortest edge -- 0.74x -- so a 22 px
mark arrives as 16 px. Cut into 640-wide tiles and each tile is UPscaled to 800, i.e. 1.25x, and
the same mark arrives near 27 px. Roughly 1.7x more pixels on the mark, with the model still at
exactly the resolution it was trained for.

    full frame  ->  1 pass at 0.74x     large marks, hoardings, context
    tiles       ->  N passes at 1.25x   small marks
    merge       ->  class-agnostic NMS at the production iou

The full-frame pass is kept rather than replaced: a mark wider than a tile is cut by every tile
that touches it, and courtside hoardings are exactly that shape.

Why this is a tool and not the container
----------------------------------------
It is a measurement first. The Grounding DINO decode below is copied verbatim from
`general_detection/detector.py` -- phrase spans from the fast tokenizer's character offsets,
boxes read straight off `pred_boxes`, no `post_process_grounded_object_detection` -- and
`--tiles 1x1` is asserted against the shipped run before any tiled number is believed. Whatever
wins gets wired into detector.py afterwards.

    python3 eval/tools/run_sliced.py --tiles 1x1  --out eval/experiments/13_sliced/runs/S0_full
    python3 eval/tools/run_sliced.py --tiles 3x2  --out eval/experiments/13_sliced/runs/S1_3x2
    python3 eval/tools/run_sliced.py --tiles 3x2 --owlv2 --out eval/experiments/13_sliced/runs/S3_union
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Tuple

import numpy as np

# Production values, from general_detection/detector.py, config.py and prompts.py. Named here
# so drift between this tool and the tagger is a visible diff rather than a silent one.
GDINO_WEIGHTS = "IDEA-Research/grounding-dino-base"
GDINO_IMGSZ = 800
GDINO_CONF = 0.07
OWLV2_WEIGHTS = "google/owlv2-base-patch16-ensemble"
OWLV2_CONF = 0.0666
PROMPTS = ["logo", "letter logo", "brand", "car logo"]
NMS_IOU = 0.6
MAX_DETECTIONS = 100
MIN_CROP_PIXELS = 16


def iou_matrix(boxes: np.ndarray) -> np.ndarray:
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = area[:, None] + area[None, :] - inter
    return np.where(union > 0, inter / np.clip(union, 1e-9, None), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy class-agnostic NMS, returning kept indices in descending score order.

    Class-agnostic is correct here rather than a shortcut: all four brand prompts share the
    parent `brand`, so the tagger's per-parent synonym pass already compares them against each
    other. Tiling adds a second reason -- the same mark seen in two overlapping tiles must
    collapse to one box, and it may well win under a different prompt in each.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)
    order = np.argsort(-scores)
    ious = iou_matrix(boxes[order])
    keep, dead = [], np.zeros(len(order), dtype=bool)
    for i in range(len(order)):
        if dead[i]:
            continue
        keep.append(order[i])
        dead |= ious[i] > threshold
        dead[i] = True
    return np.array(keep, dtype=int)


def tile_boxes(width: int, height: int, cols: int, rows: int,
               overlap: float) -> List[Tuple[int, int, int, int]]:
    """Tile rectangles covering the frame, each overlapping its neighbours by `overlap`.

    Overlap is what stops a mark straddling a seam from being halved in both tiles; without it
    a seam is a blind spot as wide as the mark.
    """
    if cols <= 1 and rows <= 1:
        return []
    step_x, step_y = width / cols, height / rows
    pad_x, pad_y = step_x * overlap, step_y * overlap
    out = []
    for row in range(rows):
        for col in range(cols):
            x1 = max(0, int(round(col * step_x - pad_x)))
            y1 = max(0, int(round(row * step_y - pad_y)))
            x2 = min(width, int(round((col + 1) * step_x + pad_x)))
            y2 = min(height, int(round((row + 1) * step_y + pad_y)))
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2, y2))
    return out


class GroundingDino:
    """Verbatim copy of the production decode path in general_detection/detector.py.

    `post_process_grounded_object_detection` is deliberately unused: it decodes labels above a
    separate `text_threshold` (default 0.25), so every box below that keeps its geometry and
    loses its label, and its labels come from BERT token spans, which yields wordpiece
    fragments. Attribution here is from the fast tokenizer's character offsets instead.
    """

    name = "gdino"

    def __init__(self, device: str):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(GDINO_WEIGHTS)
        self.model = (AutoModelForZeroShotObjectDetection
                      .from_pretrained(GDINO_WEIGHTS).to(device).eval())
        self.processor.image_processor.size = {
            "shortest_edge": GDINO_IMGSZ,
            "longest_edge": int(round(GDINO_IMGSZ * 1333 / 800)),
        }
        self.text, self.spans = self._phrase_spans(PROMPTS)

    def _phrase_spans(self, prompts: List[str]):
        text = ". ".join(p.lower() for p in prompts) + "."
        char_spans, cursor = [], 0
        for phrase in prompts:
            start = text.index(phrase.lower(), cursor)
            char_spans.append((start, start + len(phrase)))
            cursor = start + len(phrase)
        encoded = self.processor.tokenizer(text, return_offsets_mapping=True,
                                           return_tensors="pt", truncation=True, max_length=512)
        offsets = encoded["offset_mapping"][0].tolist()
        spans = []
        for start, end in char_spans:
            idx = [i for i, (a, b) in enumerate(offsets) if b > a and a >= start and b <= end]
            spans.append((min(idx), max(idx) + 1))
        return text, spans

    def detect(self, image):
        torch = self._torch
        inputs = self.processor(images=[image], text=[self.text],
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        prob = outputs.logits[0].sigmoid()
        keep = (prob.max(dim=-1).values > GDINO_CONF).nonzero().flatten()
        if keep.numel() == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), []
        phrase_scores = torch.stack(
            [prob[keep, lo:hi].max(dim=-1).values for lo, hi in self.spans], dim=-1)
        best = phrase_scores.argmax(dim=-1)
        width, height = image.size
        cx, cy, bw, bh = outputs.pred_boxes[0][keep].unbind(-1)
        xyxy = torch.stack([(cx - bw / 2) * width, (cy - bh / 2) * height,
                            (cx + bw / 2) * width, (cy + bh / 2) * height], dim=-1)
        scores = phrase_scores.gather(1, best[:, None]).squeeze(1)
        return (xyxy.cpu().numpy().astype(np.float32),
                scores.cpu().numpy().astype(np.float32),
                [PROMPTS[i] for i in best.cpu().numpy()])


class Owlv2:
    """The union partner. Experiment 06 measured brand AP 0.308 at 193 ms against gdino's 0.205
    at 293 ms -- better AND cheaper -- and it was never shipped, on a coverage argument that was
    itself made with the gate switched off."""

    name = "owlv2"

    def __init__(self, device: str):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(OWLV2_WEIGHTS)
        self.model = (AutoModelForZeroShotObjectDetection
                      .from_pretrained(OWLV2_WEIGHTS).to(device).eval())

    def detect(self, image):
        torch = self._torch
        inputs = self.processor(text=[PROMPTS], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        sizes = torch.tensor([[image.size[1], image.size[0]]]).to(self.device)
        result = self.processor.post_process_grounded_object_detection(
            outputs, threshold=OWLV2_CONF, target_sizes=sizes)[0]
        return (result["boxes"].cpu().numpy().astype(np.float32),
                result["scores"].cpu().numpy().astype(np.float32),
                [PROMPTS[int(i)] for i in result["labels"].cpu().numpy()])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="eval/frameset/frames")
    parser.add_argument("--all-frames", action="store_true",
                        help="run every frame, not just the 25 with box ground truth")
    parser.add_argument("--tiles", default="1x1", help="COLSxROWS, e.g. 3x2")
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--owlv2", action="store_true")
    parser.add_argument("--no-full-frame", action="store_true",
                        help="tiles only, to measure what the full-frame pass contributes")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from PIL import Image

    cols, rows = (int(v) for v in args.tiles.lower().split("x"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    names = None
    if not args.all_frames:
        with open("eval/box_gt/box_labels.json") as handle:
            names = {n for n, f in json.load(handle)["frames"].items() if f.get("done")}

    paths = []
    for name in sorted(os.listdir(args.frames)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        if names is not None and stem not in names:
            continue
        paths.append(os.path.join(args.frames, name))

    models = [GroundingDino(device)] + ([Owlv2(device)] if args.owlv2 else [])
    print(f"{len(paths)} frames | tiles {cols}x{rows} overlap {args.overlap} "
          f"| models {[m.name for m in models]} | full-frame {not args.no_full_frame}")

    os.makedirs(args.out, exist_ok=True)
    start = time.time()
    raw_total = kept_total = 0
    passes = 0
    with open(os.path.join(args.out, "out.jsonl"), "w") as sink:
        for path in paths:
            image = Image.open(path).convert("RGB")
            width, height = image.size
            regions = ([] if args.no_full_frame else [(0, 0, width, height)])
            regions += tile_boxes(width, height, cols, rows, args.overlap)
            passes = len(regions) * len(models)

            all_boxes, all_scores, all_prompts = [], [], []
            for (rx1, ry1, rx2, ry2) in regions:
                region = image if (rx1, ry1, rx2, ry2) == (0, 0, width, height) \
                    else image.crop((rx1, ry1, rx2, ry2))
                for model in models:
                    boxes, scores, prompts = model.detect(region)
                    if len(boxes) == 0:
                        continue
                    # Tile-local pixels back into frame pixels.
                    all_boxes.append(boxes + np.array([rx1, ry1, rx1, ry1], dtype=np.float32))
                    all_scores.append(scores)
                    all_prompts.extend(prompts)

            if all_boxes:
                boxes = np.concatenate(all_boxes)
                scores = np.concatenate(all_scores)
                raw_total += len(boxes)
                order = nms(boxes, scores, NMS_IOU)[:MAX_DETECTIONS]
            else:
                boxes = scores = None
                order = []

            for i in order:
                x1, y1, x2, y2 = (float(v) for v in boxes[i])
                if min(x2 - x1, y2 - y1) < MIN_CROP_PIXELS:
                    continue
                sink.write(json.dumps({"type": "tag", "data": {
                    "start_time": 0, "end_time": 0, "source_media": path, "tag": "brand",
                    "frame_info": {"frame_idx": 0, "box": {
                        "x1": round(max(0.0, x1 / width), 4),
                        "y1": round(max(0.0, y1 / height), 4),
                        "x2": round(min(1.0, x2 / width), 4),
                        "y2": round(min(1.0, y2 / height), 4)}},
                    "additional_info": {"kind": "crop", "prompt": all_prompts[i],
                                        "score": round(float(scores[i]), 4),
                                        "detector": "+".join(m.name for m in models)},
                }}) + "\n")
                kept_total += 1

    wall = time.time() - start
    with open(os.path.join(args.out, "wall_seconds"), "w") as handle:
        handle.write(f"{wall:.1f}\n")
    with open(os.path.join(args.out, "params.json"), "w") as handle:
        json.dump({"tiles": args.tiles, "overlap": args.overlap, "owlv2": args.owlv2,
                   "full_frame": not args.no_full_frame, "passes_per_frame": passes,
                   "gdino_conf": GDINO_CONF, "prompts": PROMPTS, "nms_iou": NMS_IOU,
                   "max_detections": MAX_DETECTIONS}, handle, indent=1)
    print(f"{raw_total} raw -> {kept_total} kept ({kept_total / max(1, len(paths)):.1f}/frame), "
          f"{passes} passes/frame, {wall:.0f}s ({wall / max(1, len(paths)):.2f}s/frame)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
