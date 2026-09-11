from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RuntimeConfig:
    """The four runtime tunables, injected per-request via `--params` in run.py.

    Everything else that used to live here is now a FIXED constant next to the code that
    consumes it, because every one of them was measured into its value and a caller changing
    one silently produces vectors that are not comparable with the rest of the index:

        embedder.py   MAX_NUM_PATCHES 256, NORMALIZE True, BATCH_SIZE 32
                      (was: max_num_patches, normalize, embed_batch_size, max_upscale)
        detector.py   PREDICT_IOU 0.7, NMS_IOU 0.6, CROP_PADDING 0.06, MIN_CROP_PIXELS 16,
                      and each backend's own imgsz/conf
                      (was: iou, nms_iou, cross_class_nms_iou, crop_padding, min_crop_pixels,
                       min_box_size, class_conf, brand_imgsz/brand_conf/person_imgsz/person_conf)
        prompts.py    DEFAULT_CLASS_PROMPTS  (was: class_prompts)

    Sliced inference (`brand_tiles`, `tile_overlap`) and the OCR channel (`ocr`, `ocr_conf`,
    `ocr_box_conf`, `ocr_mag`, `ocr_attach_overlap`) are gone entirely -- see
    eval/experiments/13_sliced and eval/experiments/17_ocr for what they bought.
    """

    # What to detect, and the switch for the detection phase. The whole frame is ALWAYS
    # embedded; setting this additionally detects and embeds each crop.
    #
    # "brand" expands to the four mark terms ("logo", "letter logo", "brand", "car logo"),
    # because the bare word is a far weaker prompt than the mark list. Any other term becomes
    # its own parent with itself as the phrasing, so ["car"] and ["person", "car"] are valid.
    detect_target: Optional[List[str]] = None

    # Which open-vocabulary backend detects. Ignored when `detect_target` is unset.
    #
    #   "coverage"  Grounding DINO @800, conf 0.07.  usable 0.469, 21.4 boxes/frame
    #   "fast"      YOLOE-26 @1280, conf 0.007.      usable 0.125,  3.5 boxes/frame
    #
    # "coverage" is the default: it finds roughly twice the marks, and five times as many on the
    # wide hoarding-shaped marks, for ~1.5x the end-to-end wall clock. "fast" is a CAP rather
    # than an optimisation -- choose it when index size or throughput is the binding constraint.
    detector: str = "coverage"  # was: brand_detector, when only brand used an open-vocab model

    # Hard cap on detections per frame, applied last, highest score first. Each survivor costs
    # one SigLIP 2 forward pass and one index row, so this is the primary cost knob.
    max_detections: int = 100

    # Additionally emit a vector-less FrameTag beside each detection's vector tag: same label,
    # same box, no `vector`. A visualization aid -- it lands as an ordinary tag track in EVIE
    # alongside the overlaid boxes. No extra inference.
    output_tags: bool = False
