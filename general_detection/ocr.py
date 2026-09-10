"""Read the words on the screen and hand them to the tagger: an exact channel beside the vectors.

Why this exists
---------------
A brand text query against the crop vectors has two failure modes, both measured in
eval/experiments/17_ocr.

1. **Wordmark-shaped.** The order is legibility, and it is the same order on both backends: SigLIP's
text tower is reading letterforms through the image encoder, and given a shape or letters too
small to resolve it has nothing to match on.

2. **Rank-fragile.** Rank is a position in a list whose length the DETECTOR sets -- an entity shifts rank 
between the two paths with the same query and the same embedder, purely because `coverage` emits more crops.
So buying detection recall costs text-query usability, which no parameter can reconcile.

A recognised string has neither problem. It matches rather than ranks, so it does not decay as
the crop count grows, and legibility is a text recogniser's job rather than a 768-d bottleneck's
(measured zero false positives).

It also attacks what 12_logo_pool identified as the actual binding constraint. A subset of named
brands have no reference image in the pool at all and cannot be retrieved by any similarity
method; some are found by their own name on screen.

What it cannot do is help a mark with no text in it or whose marks are not legible. 
The value is strongly content-dependent, which is the main reason this is off by default.

Frame-level, not crop-level, and that is measured
-------------------------------------------------
OCR per CROP was the obvious design and is worse on both axes (17_ocr).
easyocr's detector works better on a whole frame than on disconnected fragments of it.
One pass per frame is also why the cost is identical for both backends.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from loguru import logger

from general_detection.config import RuntimeConfig

# CRAFT text detection + a CRNN recogniser. Apache-2.0, and the weights are pulled on first load
# into the same mounted cache the other checkpoints use, so nothing is baked into the image.
READER_NAME = "easyocr-craft-crnn"


@dataclass(frozen=True)
class TextRegion:
    text: str
    conf: float
    box: Dict[str, float]           # normalized {x1,y1,x2,y2}
    xyxy: Sequence[float]           # the same box in source pixels


class TextReader:
    """One easyocr Reader, loaded lazily so a caller with `ocr` off never pays for it."""

    name = READER_NAME

    def __init__(self, cache_dir: str, device: Optional[str] = None) -> None:
        import easyocr

        # easyocr takes a boolean or a device string. Passing the tagger's device through keeps
        # it on the same card as the detectors; without it easyocr would default to cuda:0 and
        # silently occupy a GPU the caller did not ask for -- the same trap `model.py` documents
        # for the embedder.
        gpu: object = True
        try:
            import torch

            if device:
                gpu = device
            elif not torch.cuda.is_available():
                gpu = False
        except ImportError:  # pragma: no cover - torch is a hard dependency of this package
            pass

        storage = os.path.join(cache_dir, "easyocr")
        os.makedirs(storage, exist_ok=True)
        self._reader = easyocr.Reader(
            ["en"], gpu=gpu, model_storage_directory=storage, verbose=False
        )
        logger.info(f"ocr: {self.name} ready (weights in {storage})")

    def read(self, img: np.ndarray, cfg: RuntimeConfig) -> List[TextRegion]:
        """img: (H, W, 3) uint8 RGB.

        Filtered at the LOWER of the two gates, because the two consumers want different ones:
        a string is only worth indexing if it is legible, but a box is worth proposing whether
        or not the recogniser could read what was in it. Detection and recognition are separate
        stages inside easyocr, and CRAFT routinely localises a wordmark that the CRNN then
        garbles -- on box ground truth, ungating the boxes takes the `fast` path's usable
        coverage 0.167 -> 0.198 while a 0.2 gate on the strings keeps the control brands at 0/6.
        """
        floor = min(cfg.ocr_conf, cfg.ocr_box_conf)
        height, width = img.shape[:2]
        out: List[TextRegion] = []
        for quad, text, conf in self._reader.readtext(img, mag_ratio=cfg.ocr_mag):
            if conf < floor:
                continue
            points = np.asarray(quad, dtype=np.float32)
            # easyocr returns a four-point polygon; everything else here speaks axis-aligned.
            x1, y1 = float(points[:, 0].min()), float(points[:, 1].min())
            x2, y2 = float(points[:, 0].max()), float(points[:, 1].max())
            out.append(TextRegion(
                text=text.strip(),
                conf=round(float(conf), 4),
                box={"x1": round(max(0.0, x1 / width), 4),
                     "y1": round(max(0.0, y1 / height), 4),
                     "x2": round(min(1.0, x2 / width), 4),
                     "y2": round(min(1.0, y2 / height), 4)},
                xyxy=(x1, y1, x2, y2),
            ))
        return out


def _overlap_fraction(inner: Dict[str, float], outer: Dict[str, float]) -> float:
    """How much of `inner` lies inside `outer`. Not IoU: a wordmark is usually much smaller than
    the box that contains it, and IoU would score that pairing near zero."""
    x1, y1 = max(inner["x1"], outer["x1"]), max(inner["y1"], outer["y1"])
    x2, y2 = min(inner["x2"], outer["x2"]), min(inner["y2"], outer["y2"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    area = (inner["x2"] - inner["x1"]) * (inner["y2"] - inner["y1"])
    return ((x2 - x1) * (y2 - y1) / area) if area > 0 else 0.0


def texts_for(box: Dict[str, float], regions: Sequence[TextRegion], cfg: RuntimeConfig
              ) -> List[str]:
    """Legible strings from every region mostly inside `box`, in reading order (top, left)."""
    hits = [r for r in regions
            if r.conf >= cfg.ocr_conf and r.text.strip()
            and _overlap_fraction(r.box, box) >= cfg.ocr_attach_overlap]
    hits.sort(key=lambda r: (round(r.box["y1"], 2), r.box["x1"]))
    return [r.text.strip() for r in hits]


def _iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = ((a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
             + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter)
    return inter / union if union > 0 else 0.0


def proposals(regions: Sequence[TextRegion], existing: Sequence, img: np.ndarray,
              cfg: RuntimeConfig, label: str) -> List:
    """Text regions the detector did not already box, as brand Detections of their own.

    Worth its own path because a text detector emits WIDE boxes by construction, and that is
    exactly the shape YOLOE's COCO-prior regression head cannot produce. Measured on box ground
    truth (17_ocr), usable coverage and the `>=5` banner aspect band.
    
    Proposal boxes are gated at `ocr_box_conf`, which is 0.0, while indexed strings are gated at
    `ocr_conf`, which is 0.2. Two thresholds because the two uses differ: easyocr detects text
    and recognises it in separate stages, and CRAFT routinely localises a wordmark the CRNN then
    garbles. That box is still worth cropping -- the crop retrieves by image against the
    reference pool whatever the string said.

    Suppression is by IoU rather than by containment, and that distinction is load-bearing. A
    text region sitting INSIDE an existing box is usually the case worth keeping -- on the 261x29
    hoarding YOLOE emits a roughly square box four times too large, and the OCR box is the tight
    one. Only a near-duplicate box is dropped, at the same `nms_iou` the synonym pass uses.
    """
    from general_detection.detector import BaseDetector, Detection

    height, width = img.shape[:2]
    boxes = [d.box for d in existing]
    out: List[Detection] = []
    # Highest recognition confidence first, so the cap truncates the tail. This score is on
    # easyocr's own scale and is NOT comparable to a detector's -- it is never sorted against
    # one, because each source caps its own detections independently.
    for region in sorted(regions, key=lambda r: -r.conf):
        if len(out) >= cfg.max_detections:
            break
        if region.conf < cfg.ocr_box_conf:
            continue
        if any(_iou(region.box, box) >= cfg.nms_iou for box in boxes):
            continue
        x1, y1, x2, y2 = region.xyxy
        if min(x2 - x1, y2 - y1) < cfg.min_crop_pixels:
            continue
        if (region.box["x2"] - region.box["x1"]) * (region.box["y2"] - region.box["y1"]) \
                < cfg.min_box_size:
            continue
        crop = BaseDetector._crop(img, x1, y1, x2, y2, cfg.crop_padding)
        if crop is None:
            continue
        out.append(Detection(label=label, prompt="ocr", score=region.conf,
                             box=region.box, crop=crop, detector=READER_NAME))
        boxes.append(region.box)
    return out
