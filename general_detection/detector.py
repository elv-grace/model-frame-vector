"""Detection backends, producing padded RGB crops and normalized boxes.

Two backends, one role
----------------------
Both are open-vocabulary and text-prompted, and `RuntimeConfig.detector` picks between them:

    "coverage"  Grounding DINO at its native 800, conf 0.07. THE DEFAULT.
    "fast"      YOLOE-26 at imgsz 1280. A third of the cost, a quarter of the marks.

Measured end to end through this module on box ground truth (eval/experiments/10_config_ab):

    config                                  coverage   usable   boxes/frame
    "fast"     yoloe26 @1280, conf 0.007       0.219    0.125           3.5
    "coverage" gdino   @800,  conf 0.07        0.469    0.469          21.4

`usable` is coverage's honest sibling: a box that contains the mark AND is no more than 4x its
area, i.e. a crop whose vector is actually about the mark. The two coincide for gdino and diverge
for YOLOE, because YOLOE's misses on wide marks are often not misses at all -- on a 261x29 px
hoarding it emits a roughly square box four times too large, which is worse than a miss
downstream since the crop gets embedded and never retrieves.

The retired third backend
-------------------------
YOLO11 (closed COCO-80) used to serve `person` and beat every open-vocabulary model at it --
AP 0.752, mean IoU 0.89, 53 ms/frame. It is gone anyway: a second family of weights with its own
resolution, its own threshold and a score scale incomparable with the others' is a lot of surface
area for one class, and the open-vocabulary backends ground `person` at 0.92-0.97 class-agnostic
coverage. `git log` has it if the person numbers ever justify bringing it back.

Why imgsz and conf live per backend rather than as knobs
--------------------------------------------------------
Resolution splits the field (eval/experiments/06_resolution): YOLOE *gains* from it (brand AP
0.062 @640 -> 0.133 @1280) while Grounding DINO *collapses* above its native 800 (AP 0.001,
verified not to be a harness bug -- detections fall 2589 -> 240). One global value would be wrong
for one of them whatever it was.

Scores are not comparable either -- YOLOE's text-similarity scores and Grounding DINO's query
scores are on different scales -- so each backend carries the threshold that leave-one-clip-out
selection chose for it rather than sharing one `conf`.

Note on crop size: raising imgsz does NOT make crops bigger. Boxes are rescaled to source
coordinates and crops come from the original frame, so a 40px mark is 40px at any imgsz.
Resolution buys recall on small objects; it makes the average crop *smaller*.

ultralytics/YOLOE is licensed under AGPL-3.0. Grounding DINO is Apache-2.0.
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
from general_detection.prompts import flatten

# ---- fixed post-detection values, all measured. See RuntimeConfig for why they are not knobs.

# Ultralytics' internal NMS IoU, applied per class id (i.e. per phrasing) inside `predict`.
PREDICT_IOU = 0.7

# NMS across the phrasings of one parent ("logo" vs "letter logo"), applied after detection.
# PREDICT_IOU cannot do this: those are distinct class ids, so ultralytics never compares their
# boxes and one jersey badge survives as one box per phrasing -- N near-identical crops, N
# near-identical vectors, N stacked overlay rectangles. Grounding DINO emitted 768 overlapping
# brand-box pairs from 1,245 detections on the ground-truth frames; suppressing them lifts brand
# AP 0.166 -> 0.205 with coverage unchanged and cuts crops reaching the embedder by ~1/3.
#
# 0.6 is set by a CONSTRAINT, not by maximising a score: coverage must not fall. If suppression
# reduces the ground-truth marks hit by any detection, it is merging marks that are genuinely
# distinct -- adjacent logos on one hoarding, a wordmark inside an emblem -- and AP cannot see
# that. Coverage is intact at 0.6 and starts falling below ~0.5; maximising AP would have chosen
# 0.45. Suppression is never cross-parent: a mark on a jersey overlapping the player wearing it
# is TWO real findings, not a duplicate.
NMS_IOU = 0.6

# Fraction of the box's width/height added on each side before cropping. The REPORTED box stays
# un-padded, so the overlay draws the detection rather than the crop.
#
# This changes the vector, measurably: at 0.06 the object fills 89% of the crop's linear extent
# and the extra pixels are real image content, so two crops of the same object at different
# padding move apart. Measured on 446 ground-truth boxes, a +-0.06 mismatch between index and
# query loses 8% of top-1 retrievals and a 0.06-vs-0.25 mismatch loses 24% -- and nothing
# surfaces those as errors, the query just returns a confident wrong neighbour. Stamped into
# every tag's `additional_info.crop_padding` so a query pipeline can read it rather than assume.
CROP_PADDING = 0.06

# Drop crops whose shorter side is under this many source pixels, measured on the un-padded box.
# 16, and that is a measurement: retrieval recall@1 against native-resolution references runs
# 0.124 / 0.354 / 0.598 / 0.911 at 8 / 12 / 16 / 32 px. At 8 px the hit-vs-miss cosine gap is
# NEGATIVE -- wrong answers come back more confidently than right ones -- so no downstream
# similarity gate can filter them. Above 16 the gate costs more recall than it buys precision;
# the old 32 discarded two thirds of the marks the detector is asked to find.
#
# A floor set too LOW is recoverable (every tag carries `additional_info.upscale`, a monotone
# function of crop area at a fixed budget, so a consumer can filter). Too HIGH is not: the crop
# was never embedded, and getting it back means re-decoding the video and re-running detection.
MIN_CROP_PIXELS = 16


@dataclass(frozen=True)
class Detection:
    label: str          # parent term; becomes Tag.tag
    prompt: str         # the phrasing that actually fired
    score: float
    box: Dict[str, float]   # normalized {x1,y1,x2,y2}, UN-padded
    crop: np.ndarray        # (h, w, 3) uint8 RGB, padded by CROP_PADDING
    detector: str = ""      # which backend produced it; provenance for additional_info


# Per-backend weights, resolution and threshold. Measured together and not independently valid,
# so they live together rather than being separately overridable.
DETECTORS: Dict[str, Dict] = {
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
        # "conf": 0.15,  # was: the F1-optimal held-out threshold, range 0.142-0.162 across folds
        #
        # F1 is the wrong objective for a crop-and-embed index: a false-positive crop costs one
        # SigLIP forward pass, matches nothing and disappears, while a mark that is never cropped
        # is gone for good. Coverage against boxes/frame: 0.15 -> 0.328/10.8, 0.10 -> 0.448/24.7,
        # 0.07 -> 0.573/41.3, 0.05 -> 0.609/60.3. 0.07 rather than 0.05 because at 0.05 a dense
        # broadcast frame yields ~80 boxes and clips `max_detections` on a third of frames --
        # and truncation is by score, so it discards exactly the faint hoarding marks the lower
        # gate was opened to admit. At 0.07 that touches 5 frames in 120.
        "conf": 0.07,
    },
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

    def _raw(self, img: np.ndarray) -> sv.Detections:
        raise NotImplementedError

    def detect(self, img: np.ndarray, cfg: RuntimeConfig) -> List[Detection]:
        """img: (H, W, 3) uint8 RGB."""
        if not self._prompts:
            raise RuntimeError("set_prompts() must be called before detect()")

        height, width = img.shape[:2]
        dets = self._raw(img)
        if dets is None or len(dets) == 0:
            return []

        dets = self._dedupe(dets)
        return self._to_detections(dets, img, height, width, cfg)

    def _dedupe(self, dets: sv.Detections) -> sv.Detections:
        """Collapse the duplicates introduced by expanding each parent into synonyms.

        supervision's NMS is per-class_id, so remapping class ids to parent-group ids makes
        this a synonym-group pass: "logo" and "letter logo" boxes on one badge collapse, while
        "logo" and "person" boxes — the mark and the player wearing it — do not.
        """
        # with_nms() carries `data` through, so stash the prompt-level id before class_id
        # is overwritten with the group id.
        dets.data["prompt_id"] = dets.class_id.copy()
        dets.class_id = np.array(
            [self._group_of_label[self._labels[c]] for c in dets.data["prompt_id"]],
            dtype=int,
        )
        return dets.with_nms(threshold=NMS_IOU, class_agnostic=False)

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
            # Gate on the un-padded box: padding adds context, not detail.
            if min(x2 - x1, y2 - y1) < MIN_CROP_PIXELS:
                continue

            crop = self._crop(img, x1, y1, x2, y2, CROP_PADDING)
            if crop is None:
                continue

            prompt_id = int(dets.data["prompt_id"][i])
            out.append(
                Detection(
                    label=self._labels[prompt_id],
                    prompt=self._prompts[prompt_id],
                    score=round(float(dets.confidence[i]), 4),
                    box={
                        "x1": round(max(0.0, x1 / width), 4),
                        "y1": round(max(0.0, y1 / height), 4),
                        "x2": round(min(1.0, x2 / width), 4),
                        "y2": round(min(1.0, y2 / height), 4),
                    },
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


class YoloeDetector(BaseDetector):
    """YOLOE open-vocabulary text-prompted detection. The `fast` backend."""

    def __init__(self, weights: str, cache_dir: str, imgsz: int, conf: float,
                 device: Optional[str] = None) -> None:
        super().__init__(imgsz, conf)
        from ultralytics import YOLOE

        self.cache_dir = cache_dir
        self.device = device
        self.name = os.path.splitext(os.path.basename(weights))[0]

        logger.info(f"loading detector {weights} @ imgsz {imgsz} (cache={cache_dir})")
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

    def _raw(self, img: np.ndarray) -> sv.Detections:
        # Ultralytics interprets a numpy array as BGR; common_ml hands us RGB. (model-logo
        # does the same flip before its YOLO call.) Crops are taken from the RGB original.
        bgr = np.ascontiguousarray(img[:, :, ::-1])
        results = self.model.predict(
            bgr, imgsz=self.imgsz, conf=self.conf, iou=PREDICT_IOU,
            device=self.device, verbose=False,
        )
        dets = sv.Detections.from_ultralytics(results[0])
        # YOLOE ships segmentation checkpoints; the masks are unused here and are large.
        dets.mask = None
        return dets


class GroundingDinoDetector(BaseDetector):
    """Grounding DINO. The `coverage` backend, and the default.

    Prompt format follows the model's documented convention: queries lowercase, separated by
    ". " and terminated with a period, i.e. "logo. letter logo. brand. car logo."
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

        logger.info(f"loading detector {weights} @ imgsz {imgsz}")
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

    def _raw(self, img: np.ndarray) -> sv.Detections:
        from PIL import Image

        torch = self._torch
        inputs = self.processor(images=[Image.fromarray(img)], text=[self._text],
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        prob = outputs.logits[0].sigmoid()              # (queries, text tokens)
        keep = (prob.max(dim=-1).values > self.conf).nonzero().flatten()
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


def check_detector(mode: str) -> None:
    """Reject an unknown backend name.

    Split out of `build_detector` so the name can be validated on the path where no detector is
    built at all -- otherwise a typo in `detector` sits silently in a params blob until someone
    adds a `detect_target`, and only then fails.
    """
    if mode not in DETECTORS:
        raise ValueError(f"detector must be one of {sorted(DETECTORS)}, got {mode!r} "
                         f"(it only takes effect when detect_target is set)")


def build_detector(mode: str, cache_dir: str, device: Optional[str] = None) -> BaseDetector:
    """Construct the open-vocabulary detector named by `mode` ("fast" or "coverage")."""
    check_detector(mode)
    spec = DETECTORS[mode]
    if spec["kind"] == "yoloe":
        return YoloeDetector(spec["weights"], cache_dir, spec["imgsz"], spec["conf"], device)
    return GroundingDinoDetector(spec["weights"], cache_dir, spec["imgsz"], spec["conf"], device)
