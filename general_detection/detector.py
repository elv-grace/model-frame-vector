"""Detection backends, producing padded RGB crops and normalized boxes.

Three backends, two roles
-------------------------
    brand   open-vocabulary, one of two:
              "coverage"  Grounding DINO at its native 800, conf 0.07. THE DEFAULT.
              "fast"      YOLOE-26 at imgsz 1280. A third of the cost, a quarter of the marks.
    person  YOLO11, closed COCO-80. AP 0.752, mean IoU 0.89, 53 ms/frame -- the best and the
            cheapest model measured for this class.

Measured end to end through this module on box ground truth (eval/experiments/10_config_ab).

    config                                  coverage   usable   boxes/frame
    "fast"   yoloe26 @1280, conf 0.007         0.219    0.125           3.5
    "coverage" gdino @800, conf 0.07           0.469    0.469          21.4

`usable` is coverage's honest sibling: a box that contains the mark AND is no more than 4x its
area, i.e. a crop whose vector is actually about the mark.

Why resolution is per-backend and not a global knob
---------------------------------------------------
Measured, and the field splits (eval/experiments/06_resolution), 
so one global imgsz would be wrong for at least one detector whatever value it took.

Scores are not comparable across backends either -- YOLOE's text-similarity scores, YOLO11's
sigmoid class scores and Grounding DINO's query scores have different scales -- so each backend
carries its own measured threshold rather than sharing one `conf`.

Note on crop size: raising imgsz does NOT make crops bigger. Ultralytics rescales boxes back to
source coordinates and crops are taken from the original frame, so a 40px mark is 40px at any
imgsz. Measured: person crop median is flat at 53/51/52 px across imgsz 640/960/1280, while the
brand median FALLS 55 -> 41 px because higher resolution finds more small marks. Resolution buys
recall on small objects, not resolution in the crop.

ultralytics/YOLOE and YOLO11 are licensed under AGPL-3.0. Grounding DINO is Apache-2.0.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import supervision as sv
from loguru import logger

from general_detection.config import RuntimeConfig
from general_detection.prompts import CLOSED_VOCAB_LABEL, flatten


@dataclass(frozen=True)
class Detection:
    label: str          # parent term; becomes Tag.tag
    prompt: str         # the phrasing that actually fired
    score: float
    box: Dict[str, float]   # normalized {x1,y1,x2,y2}, UN-padded
    crop: np.ndarray        # (h, w, 3) uint8 RGB, padded by cfg.crop_padding
    detector: str = ""      # which backend produced it; provenance for additional_info


# Per-backend defaults, every value measured against box ground truth. RuntimeConfig may
# override them, but these are what the evaluation actually recommends.
BRAND_BACKENDS: Dict[str, Dict] = {
    "fast": {
        "kind": "yoloe",
        "weights": "yoloe-26l-seg.pt",
        # 1280 more than doubles brand AP over 640 for 1.6x the compute.
        "imgsz": 1280,
        # Held-out threshold, stable at 0.007 across all 11 leave-one-clip-out folds -- the
        # most stable in the study. Effectively ungated, which means `max_detections` rather
        # than this value is the real cost gate for this backend.
        "conf": 0.007,
    },
    "coverage": {
        "kind": "gdino",
        "weights": "IDEA-Research/grounding-dino-base",
        # Native resolution. Raising it does not trade speed for accuracy, it breaks the model.
        "imgsz": 800,
        # Use 0.05 for a one-off archival pass, with the cap raised to match.
        "conf": 0.07, # previously 0.15 to maximize F1, but coverage is more important than precision for embedding
    },
}

PERSON_BACKEND: Dict = {
    "kind": "yolo11",
    "weights": "yolo11l.pt",
    "imgsz": 640,           # best measured; higher is worse for this class
    "conf": 0.11,           # held-out threshold, range 0.101-0.119 across folds
}


@contextlib.contextmanager
def _chdir(path: str):
    """Run a block with the process CWD moved to `path`.

    Ultralytics resolves bare checkpoint names — and the MobileCLIP text encoder that
    `get_text_pe()` pulls down — relative to the CWD. The container's WORKDIR is ephemeral,
    so both would be re-downloaded on every run. Moving the CWD into the mounted cache for
    the duration of the load puts them somewhere persistent.
    """
    previous = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class BaseDetector:
    """Shared post-detection path: gate, suppress, crop.

    Subclasses implement `_raw()`, returning an sv.Detections whose `class_id` indexes into
    `self._prompts`. Everything from the confidence gate onward is identical across backends and
    lives here, so a new backend cannot accidentally acquire different cropping or suppression
    behaviour -- which would silently make its vectors incomparable with the others'.
    """

    name = "base"

    def __init__(self, imgsz: int, conf: float) -> None:
        self.imgsz = imgsz
        self.conf = conf
        self._prompts: List[str] = []
        self._labels: List[str] = []
        self._group_of_label: Dict[str, int] = {}

    def _set_terms(self, prompts: List[str], labels: List[str]) -> None:
        self._prompts, self._labels = prompts, labels
        # Stable id per parent term, used by the synonym-group NMS pass. dict.fromkeys
        # preserves first-seen order, so ids are deterministic for a given config.
        self._group_of_label = {label: i for i, label in enumerate(dict.fromkeys(labels))}

    def _raw(self, img: np.ndarray, cfg: RuntimeConfig) -> sv.Detections:
        raise NotImplementedError

    @staticmethod
    def _tiles(width: int, height: int, spec: str, overlap: float):
        """Tile rectangles covering the frame, each overlapping its neighbours by `overlap`.

        Overlap is what stops a mark straddling a seam from being halved in both tiles; without
        it a seam is a blind spot as wide as the mark.
        """
        cols, rows = (int(v) for v in spec.lower().split("x"))
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

    def _raw_tiled(self, img: np.ndarray, cfg: RuntimeConfig) -> sv.Detections:
        """Full-frame pass plus one pass per tile, merged into frame coordinates.

        Brand is a small-object problem -- two thirds of ground-truth marks are under 32 px --
        and resolution is the obvious lever that is NOT available here: Grounding DINO collapses
        past its native 800 (experiment 06), because a DETR decoder's learned reference points
        are tuned to the training resolution. Slicing buys the same thing without touching imgsz.
        A 1920x1080 frame is downscaled to 800 on the shortest edge, 0.74x, so a 22 px mark
        arrives as 16 px; a 960-wide tile is UPscaled to 800, and the same mark arrives near
        27 px, with the model still at exactly the resolution it was trained for.

        The full-frame pass is kept rather than replaced, and that is not belt-and-braces: a mark
        wider than a tile is cut by every tile touching it, and courtside hoardings are exactly
        that shape. Measured on box ground truth (eval/experiments/13_sliced), usable coverage
        by tiling, where `usable` counts a mark only if some box contains it and is no more than
        4x its area:

            tiling        coverage   usable   boxes/frame   passes/frame
            1x1 (off)        0.479    0.474          21.8              1
            2x2              0.599    0.589          42.9              5
            3x2              0.604    0.552          46.2              7
            4x3              0.589    0.495          56.4             13

        Finer is not better, and the aspect bands say why: the 3-5 aspect band goes 0.42 -> 0.68
        -> 0.63 -> 0.37 as tiles shrink, because a 669 px hoarding spans two 480 px tiles and
        every tile sees a fragment. 2x2 is where small-mark recall has arrived and wide marks
        have not yet been cut up. Coverage keeps creeping up past it; `usable` does not.
        """
        detections = [self._raw(img, cfg)]
        height, width = img.shape[:2]
        for (x1, y1, x2, y2) in self._tiles(width, height, cfg.brand_tiles, cfg.tile_overlap):
            tile = np.ascontiguousarray(img[y1:y2, x1:x2])
            dets = self._raw(tile, cfg)
            if dets is None or len(dets) == 0:
                continue
            # Tile-local pixels back into frame pixels, so everything downstream -- gate,
            # suppression, the crop taken from the ORIGINAL frame -- is unchanged.
            dets.xyxy = dets.xyxy + np.array([x1, y1, x1, y1], dtype=dets.xyxy.dtype)
            detections.append(dets)
        detections = [d for d in detections if d is not None and len(d) > 0]
        if not detections:
            return sv.Detections.empty()
        return detections[0] if len(detections) == 1 else sv.Detections.merge(detections)

    def detect(self, img: np.ndarray, cfg: RuntimeConfig) -> List[Detection]:
        """img: (H, W, 3) uint8 RGB."""
        if not self._prompts:
            raise RuntimeError("set_prompts() must be called before detect()")

        height, width = img.shape[:2]
        dets = self._raw_tiled(img, cfg)
        if dets is None or len(dets) == 0:
            return []

        dets = self._gate(dets, cfg)
        if len(dets) == 0:
            return []

        dets = self._dedupe(dets, cfg)
        return self._to_detections(dets, img, height, width, cfg)

    def _floor(self, cfg: RuntimeConfig) -> float:
        """Lowest gate any of this backend's classes uses, for its own pre-filter.

        Passing this backend's `conf` straight to the model would silently drop classes whose
        `class_conf` entry is lower than it.
        """
        overrides = [v for k, v in cfg.class_conf.items() if k in self._labels]
        return min([self.conf, *overrides]) if overrides else self.conf

    def _gate(self, dets: sv.Detections, cfg: RuntimeConfig) -> sv.Detections:
        """Per-parent confidence gate, defaulting to this backend's own measured threshold."""
        floors = np.array(
            [cfg.class_conf.get(self._labels[c], self.conf) for c in dets.class_id],
            dtype=np.float32,
        )
        return dets[dets.confidence >= floors]

    def _dedupe(self, dets: sv.Detections, cfg: RuntimeConfig) -> sv.Detections:
        """Collapse the duplicates introduced by expanding each parent into synonyms.

        supervision's NMS is per-class_id, so remapping class ids to parent-group ids makes
        this a synonym-group pass: "logo" and "letter logo" boxes on one badge collapse, while
        "logo" and "person" boxes — the mark and the player wearing it — do not.

        cfg.nms_iou is 0.6 rather than a fitted value. The constraint is that coverage must not
        fall: if suppression reduces the ground-truth marks hit by any detection, it is merging
        marks that are genuinely distinct (adjacent logos on a hoarding, a wordmark inside an
        emblem). Coverage is intact at 0.6 and starts falling below ~0.5. Maximising AP would
        have chosen 0.45, which destroys real marks to buy a better number.
        """
        # with_nms() carries `data` through, so stash the prompt-level id before class_id
        # is overwritten with the group id.
        dets.data["prompt_id"] = dets.class_id.copy()
        dets.class_id = np.array(
            [self._group_of_label[self._labels[c]] for c in dets.data["prompt_id"]],
            dtype=int,
        )
        dets = dets.with_nms(threshold=cfg.nms_iou, class_agnostic=False)

        if cfg.cross_class_nms_iou < 1.0:
            dets = dets.with_nms(threshold=cfg.cross_class_nms_iou, class_agnostic=True)
        return dets

    def _to_detections(
        self,
        dets: sv.Detections,
        img: np.ndarray,
        height: int,
        width: int,
        cfg: RuntimeConfig,
    ) -> List[Detection]:
        out: List[Detection] = []
        # Highest score first, so max_detections truncates the tail rather than an
        # arbitrary slice.
        for i in np.argsort(-dets.confidence):
            if len(out) >= cfg.max_detections:
                break

            x1, y1, x2, y2 = (float(v) for v in dets.xyxy[i])
            box = {
                "x1": round(max(0.0, x1 / width), 4),
                "y1": round(max(0.0, y1 / height), 4),
                "x2": round(min(1.0, x2 / width), 4),
                "y2": round(min(1.0, y2 / height), 4),
            }
            if self._box_area(box) < cfg.min_box_size:
                continue
            # Gate on the un-padded box: padding adds context, not detail.
            if min(x2 - x1, y2 - y1) < cfg.min_crop_pixels:
                continue

            crop = self._crop(img, x1, y1, x2, y2, cfg.crop_padding)
            if crop is None:
                continue

            prompt_id = int(dets.data["prompt_id"][i])
            out.append(
                Detection(
                    label=self._labels[prompt_id],
                    prompt=self._prompts[prompt_id],
                    score=round(float(dets.confidence[i]), 4),
                    box=box,
                    crop=crop,
                    detector=self.name,
                )
            )
        return out

    @staticmethod
    def _crop(
        img: np.ndarray, x1: float, y1: float, x2: float, y2: float, padding: float
    ) -> Optional[np.ndarray]:
        height, width = img.shape[:2]
        pad_w, pad_h = (x2 - x1) * padding, (y2 - y1) * padding
        cx1 = max(0, int(round(x1 - pad_w)))
        cy1 = max(0, int(round(y1 - pad_h)))
        cx2 = min(width, int(round(x2 + pad_w)))
        cy2 = min(height, int(round(y2 + pad_h)))
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        # A sliced view is non-contiguous; PIL.Image.fromarray needs contiguous memory.
        return np.ascontiguousarray(img[cy1:cy2, cx1:cx2])

    @staticmethod
    def _box_area(box: Dict[str, float]) -> float:
        return abs(box["x2"] - box["x1"]) * abs(box["y2"] - box["y1"])


class YoloeDetector(BaseDetector):
    """YOLOE open-vocabulary text-prompted detection. The default (`fast`) brand backend."""

    def __init__(self, weights: str, cache_dir: str, imgsz: int, conf: float,
                 device: Optional[str] = None) -> None:
        super().__init__(imgsz, conf)
        from ultralytics import YOLOE

        self.cache_dir = cache_dir
        self.device = device
        self.name = os.path.splitext(os.path.basename(weights))[0]

        logger.info(f"loading brand detector {weights} @ imgsz {imgsz} (cache={cache_dir})")
        with _chdir(cache_dir):
            self.model = YOLOE(weights)

    def set_prompts(self, class_prompts: Dict[str, List[str]]) -> None:
        """Encode the text prompts. Cheap to re-call: no-ops when the prompt set is
        unchanged, so a per-request set_config does not pay for a re-encode."""
        prompts, labels = flatten(class_prompts)
        if prompts == self._prompts and labels == self._labels:
            return

        logger.info(f"encoding {len(prompts)} text prompts across {len(set(labels))} classes")
        with _chdir(self.cache_dir):
            # get_text_pe() may download the MobileCLIP text encoder on first use.
            self.model.set_classes(prompts, self.model.get_text_pe(prompts))
        self._set_terms(prompts, labels)

    def _raw(self, img: np.ndarray, cfg: RuntimeConfig) -> sv.Detections:
        # Ultralytics interprets a numpy array as BGR; common_ml hands us RGB. (model-logo
        # does the same flip before its YOLO call.) Crops are taken from the RGB original.
        bgr = np.ascontiguousarray(img[:, :, ::-1])
        results = self.model.predict(
            bgr, imgsz=self.imgsz, conf=self._floor(cfg), iou=cfg.iou,
            device=self.device, verbose=False,
        )
        dets = sv.Detections.from_ultralytics(results[0])
        # YOLOE ships segmentation checkpoints; the masks are unused here and are large.
        dets.mask = None
        return dets


class Yolo11Detector(BaseDetector):
    """Closed-vocabulary COCO-80 detection. The person backend.

    It cannot serve `brand` and is never asked to: COCO-80 has no logo class, and its measured
    brand coverage of 0.04 confirms it does not box marks incidentally either. What it is, is the
    best person detector measured -- AP 0.752 and mean IoU 0.89 at 53 ms/frame, beating every
    open-vocabulary model at the one class COCO was built around.
    """

    def __init__(self, weights: str, cache_dir: str, imgsz: int, conf: float,
                 device: Optional[str] = None) -> None:
        super().__init__(imgsz, conf)
        from ultralytics import YOLO

        self.device = device
        self.name = os.path.splitext(os.path.basename(weights))[0]

        logger.info(f"loading person detector {weights} @ imgsz {imgsz} (cache={cache_dir})")
        with _chdir(cache_dir):
            self.model = YOLO(weights)
        # COCO name -> class id, for translating a requested parent onto what the head emits.
        self._coco_id_of = {name: i for i, name in self.model.names.items()}
        self._parent_of_coco: Dict[int, str] = {}

    def set_prompts(self, class_prompts: Dict[str, List[str]]) -> None:
        """Map each requested parent onto its COCO class id.

        Nothing is encoded here -- the vocabulary is fixed -- so this only records which of the
        80 classes to keep and which parent term to report them as.
        """
        prompts, labels = flatten(class_prompts)
        parent_of_coco: Dict[int, str] = {}
        for parent in dict.fromkeys(labels):
            coco = CLOSED_VOCAB_LABEL.get(parent)
            class_id = self._coco_id_of.get(coco) if coco else None
            if class_id is None:
                raise ValueError(f"{parent!r} is not a COCO-80 class this backend can serve")
            parent_of_coco[class_id] = parent
        self._parent_of_coco = parent_of_coco
        self._set_terms(prompts, labels)

    def _raw(self, img: np.ndarray, cfg: RuntimeConfig) -> sv.Detections:
        bgr = np.ascontiguousarray(img[:, :, ::-1])
        results = self.model.predict(
            bgr, imgsz=self.imgsz, conf=self._floor(cfg), iou=cfg.iou,
            device=self.device, verbose=False,
        )
        dets = sv.Detections.from_ultralytics(results[0])
        dets.mask = None
        if len(dets) == 0:
            return dets

        # Keep only the COCO classes asked for, then remap COCO ids to indices into
        # self._prompts, which is the contract the shared path downstream expects.
        keep = np.array([int(c) in self._parent_of_coco for c in dets.class_id], dtype=bool)
        dets = dets[keep]
        if len(dets) == 0:
            return dets
        dets.class_id = np.array(
            [self._labels.index(self._parent_of_coco[int(c)]) for c in dets.class_id],
            dtype=int,
        )
        return dets


class GroundingDinoDetector(BaseDetector):
    """Grounding DINO. The `coverage` brand backend.

    Prompt format follows the model's documented convention: queries lowercase, separated by
    ". " and terminated with a period, i.e. "logo. letter logo. car logo. emblem. brand. label."
    GroundingDinoProcessor (resolved by AutoProcessor) prepares the image-text pair.

    What is NOT used is the processor's `post_process_grounded_object_detection`, because both of
    its defects showed up as a bad model rather than a bad harness:

    1. It keeps boxes above `threshold` but decodes their labels above `text_threshold`, which
       defaults to 0.25. With a box threshold below that, every box in between keeps its geometry
       and loses its label -- 94% empty labels in the first sweep.

    2. Labels are always decoded from a BERT token span; the processor's own docstring says the
       `text_labels` argument is "NOT used". That produced wordpiece fragments like "##board" and
       spans straddling two phrases.

    Attribution is instead computed from the fast tokenizer's character offsets: each phrase's
    span is mapped to token positions, and a box takes the phrase carrying its highest
    probability -- exactly one clean phrase per box. Boxes are decoded from `pred_boxes`
    directly, which is what the evaluation measured, so production and the reported numbers
    share one code path.
    """

    def __init__(self, weights: str, cache_dir: str, imgsz: int, conf: float,
                 device: Optional[str] = None) -> None:
        super().__init__(imgsz, conf)
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self.name = os.path.basename(weights)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"loading brand detector {weights} @ imgsz {imgsz}")
        self.processor = AutoProcessor.from_pretrained(weights, cache_dir=cache_dir)
        self.model = (AutoModelForZeroShotObjectDetection
                      .from_pretrained(weights, cache_dir=cache_dir).to(self.device).eval())
        # Native resolution, and it must stay native -- see the module docstring. Set explicitly
        # rather than left implicit so that a future imgsz change is a visible edit here.
        self.processor.image_processor.size = {"shortest_edge": imgsz,
                                               "longest_edge": int(round(imgsz * 1333 / 800))}
        self._text = ""
        self._spans: List[tuple] = []

    def set_prompts(self, class_prompts: Dict[str, List[str]]) -> None:
        prompts, labels = flatten(class_prompts)
        if prompts == self._prompts and labels == self._labels:
            return
        self._text, self._spans = self._phrase_spans(prompts)
        logger.info(f"grounding text: {self._text!r}")
        self._set_terms(prompts, labels)

    def _phrase_spans(self, prompts: List[str]):
        """Character span of each prompt in the concatenated text, mapped to token indices."""
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
            if not idx:
                raise ValueError(f"prompt {text[start:end]!r} has no token span")
            spans.append((min(idx), max(idx) + 1))
        return text, spans

    def _raw(self, img: np.ndarray, cfg: RuntimeConfig) -> sv.Detections:
        from PIL import Image

        torch = self._torch
        floor = self._floor(cfg)
        inputs = self.processor(images=[Image.fromarray(img)], text=[self._text],
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        prob = outputs.logits[0].sigmoid()              # (queries, text tokens)
        keep = (prob.max(dim=-1).values > floor).nonzero().flatten()
        if keep.numel() == 0:
            return sv.Detections.empty()

        # Per-phrase score is the max probability inside that phrase's own token span; the
        # winning phrase is the argmax across phrases.
        phrase_scores = torch.stack(
            [prob[keep, lo:hi].max(dim=-1).values for lo, hi in self._spans], dim=-1)
        best = phrase_scores.argmax(dim=-1)

        height, width = img.shape[:2]
        cx, cy, bw, bh = outputs.pred_boxes[0][keep].unbind(-1)   # normalized cxcywh
        xyxy = torch.stack([(cx - bw / 2) * width, (cy - bh / 2) * height,
                            (cx + bw / 2) * width, (cy + bh / 2) * height], dim=-1)
        return sv.Detections(
            xyxy=xyxy.cpu().numpy().astype(np.float32),
            confidence=phrase_scores.gather(1, best[:, None]).squeeze(1)
                       .cpu().numpy().astype(np.float32),
            class_id=best.cpu().numpy().astype(int),
        )


def build_brand_detector(mode: str, cache_dir: str, cfg: RuntimeConfig,
                         device: Optional[str] = None) -> BaseDetector:
    """Construct the open-vocabulary detector named by `mode` ("fast" or "coverage")."""
    if mode not in BRAND_BACKENDS:
        raise ValueError(f"brand_detector must be one of {sorted(BRAND_BACKENDS)}, got {mode!r}")
    spec = BRAND_BACKENDS[mode]
    imgsz = cfg.brand_imgsz if cfg.brand_imgsz is not None else spec["imgsz"]
    conf = cfg.brand_conf if cfg.brand_conf is not None else spec["conf"]
    if spec["kind"] == "yoloe":
        return YoloeDetector(spec["weights"], cache_dir, imgsz, conf, device)
    return GroundingDinoDetector(spec["weights"], cache_dir, imgsz, conf, device)


def build_person_detector(cache_dir: str, cfg: RuntimeConfig,
                          device: Optional[str] = None) -> BaseDetector:
    imgsz = cfg.person_imgsz if cfg.person_imgsz is not None else PERSON_BACKEND["imgsz"]
    conf = cfg.person_conf if cfg.person_conf is not None else PERSON_BACKEND["conf"]
    return Yolo11Detector(PERSON_BACKEND["weights"], cache_dir, imgsz, conf, device)
