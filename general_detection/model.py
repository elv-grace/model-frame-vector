"""SigLIP 2 frame-vector tagger, with an optional detection phase.

Two modes, one embedder, and `detect_target` is the switch
----------------------------------------------------------
Unset (the default), this is a plain frame embedder: one whole-frame SigLIP 2 vector per sampled
frame, no detector weights loaded at all.

Set, the detection phase runs first and only the detected crops are embedded -- one vector tag
per detection, each keeping its box so the video-editor overlay works.

Never both in one run. A frame vector and a crop vector are not comparable: NaFlex DOWNsamples a
1080p frame to the patch budget and UPsamples a 40px crop to it, so the two land in measurably
different regions of the space (see `min_crop_pixels` in config.py). One run is one vector space.

Two detectors, one embedder
---------------------------
`person` and `brand` are served by different models, because the measurements say they want
different ones: a closed COCO detector wins person outright (AP 0.752, mean IoU 0.89, and the
cheapest model in the study) while only open-vocabulary models can find brand marks at all.
Targets are routed by `prompts.split_by_detector`, and a detector with nothing routed to it is
never constructed -- a person-only request never pays to load the brand model.

They run sequentially, and the order does not matter. On one GPU two detectors do not overlap
usefully: they compete for the same SMs, so running them concurrently costs the same wall-clock
plus scheduling overhead. What does overlap is decode (CPU) against inference (GPU), which is the
frame pipeline's business rather than this module's.

Both detectors' crops are embedded in ONE batch. That matters: the embedder is the larger cost at
low detection counts, and batching across detectors keeps it near its throughput rather than its
latency.

An optional third channel: the words on the screen
--------------------------------------------------
With `ocr` set, one OCR pass runs per frame and its strings are stamped on the detections whose
boxes contain them, so a brand with a legible wordmark can be found by exact string match rather
than by a text-to-image cosine that has almost no headroom. Under `ocr: "propose"` the text
regions the detector did not box also become detections of their own. Off by default -- it is a
fixed ~400 ms per frame whose value is strongly content-dependent. See general_detection/ocr.py.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import torch
from dacite import from_dict
from loguru import logger

from common_ml.tagging.models.frame_based import FrameModel
from common_ml.tagging.models.tag_types import FrameTag

from general_detection.config import RuntimeConfig
from general_detection.detector import (
    Detection,
    build_brand_detector,
    build_person_detector,
)
from general_detection.embedder import Siglip2CropEmbedder
from general_detection.ocr import TextReader, proposals, texts_for
from general_detection.prompts import expand_target, split_by_detector

# The frame vector's box: the full image in normalized coords.
_WHOLE_FRAME_BOX = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}

QUERY_MODES = ["text", "image"]

# Accepted values for `ocr`. "text" annotates existing detections; "propose" also turns
# un-boxed text regions into detections of their own. See general_detection/ocr.py.
OCR_MODES = {"off", "text", "propose"}


class FrameVectorModel(FrameModel):
    """One SigLIP 2 vector per sampled frame, or -- with `detect_target` set -- one per
    detected crop."""

    def __init__(
        self,
        cfg: RuntimeConfig,
        embedder_model_id: str,
        cache_dir: str,
        embedder_revision: Optional[str] = None,
        device: Optional[str] = None,
        output_tags: Optional[bool] = None,
    ) -> None:
        self.config = cfg
        if output_tags is not None:
            self.config.output_tags = output_tags
        self.cache_dir = cache_dir
        self.device = device
        self._brand = None
        self._person = None
        self._brand_mode: Optional[str] = None
        self._reader: Optional[TextReader] = None
        # Which parent term OCR proposals are labelled with, set by _apply_targets alongside the
        # brand detector: a text region is a mark, so it belongs to whatever parent the
        # open-vocabulary side is serving.
        self._ocr_label: Optional[str] = None
        self._apply_targets(cfg)
        # Same device as the detectors. Without threading it through, an explicit device (say
        # "cuda:1") would place the detectors there and the embedder on cuda:0, which works but
        # copies every crop across devices and silently occupies a card the caller did not ask
        # for. None keeps the embedder's own auto-detection.
        self.embedder = Siglip2CropEmbedder(
            embedder_model_id,
            revision=embedder_revision,
            device=torch.device(device) if device else None,
        )

    # ---- configuration ----------------------------------------------------------

    def _resolve_prompts(self, cfg: RuntimeConfig) -> Dict[str, List[str]]:
        """`detect_target` wins when set, else explicit `class_prompts`. Empty means no
        detection: the frame-vector mode."""
        if cfg.detect_target:
            return expand_target(cfg.detect_target)
        return {k: list(v) for k, v in cfg.class_prompts.items()}

    def _apply_targets(self, cfg: RuntimeConfig) -> None:
        """Build/refresh only the detectors the current target actually needs.

        Loading is lazy and per-role, so a target that routes to one side never constructs the
        other and no target at all constructs neither. Rebuilding is confined to a real backend
        change (`brand_detector` switching between "fast" and "coverage"); a prompt-only change
        re-encodes text, which is seconds, rather than reloading weights.
        """
        # Validated first, before anything loads weights: a typo in --params should fail in
        # milliseconds rather than after Grounding DINO is on the GPU.
        if cfg.ocr not in OCR_MODES:
            raise ValueError(f"ocr must be one of {sorted(OCR_MODES)}, got {cfg.ocr!r}")

        open_vocab, closed = split_by_detector(self._resolve_prompts(cfg))

        # Rejected rather than silently ignored: with no detections there is nothing to stamp a
        # string onto, so an `ocr` run without a target would quietly pay nothing and find nothing.
        if cfg.ocr != "off" and not (open_vocab or closed):
            raise ValueError(
                f"ocr={cfg.ocr!r} needs detections to attach text to; pass detect_target"
            )

        if open_vocab:
            if self._brand is None or self._brand_mode != cfg.brand_detector:
                self._brand = build_brand_detector(cfg.brand_detector, self.cache_dir, cfg,
                                                   self.device)
                self._brand_mode = cfg.brand_detector
            self._brand.set_prompts(open_vocab)
            self._ocr_label = sorted(open_vocab)[0]
        else:
            # Released rather than kept idle: these are the large weights, and a caller that
            # narrowed its target to person should get the memory back.
            self._brand, self._brand_mode, self._ocr_label = None, None, None

        # Same lazy, per-role treatment as the detectors: `ocr` off never constructs the reader,
        # and turning it off releases it. It is not rebuilt for a mode change between "text" and
        # "propose", which share one pass.
        if cfg.ocr != "off":
            if self._reader is None:
                self._reader = TextReader(self.cache_dir, self.device)
        else:
            self._reader = None

        if closed:
            if self._person is None:
                self._person = build_person_detector(self.cache_dir, cfg, self.device)
            self._person.set_prompts(closed)
        else:
            self._person = None

        if not (open_vocab or closed):
            logger.info("no detect_target: one whole-frame vector per frame, no detectors loaded")
        else:
            logger.info(
                f"targets: open-vocab={sorted(open_vocab) or '-'} "
                f"({cfg.brand_detector if open_vocab else 'not loaded'}), "
                f"closed={sorted(closed) or '-'}, ocr={cfg.ocr}"
            )

    def set_config(self, config: dict) -> None:
        self.config = from_dict(RuntimeConfig, config)
        self._apply_targets(self.config)

    def get_config(self) -> dict:
        return asdict(self.config)

    # ---- tagging ----------------------------------------------------------------

    def tag_frame(self, img: np.ndarray) -> List[FrameTag]:
        """img: (H, W, 3) uint8 RGB. Returns the frame's vector tag, or one tag per detection
        when a target is set.

        Detection mode: `tag` is the parent term, `vector` the crop embedding, `box` the
        normalized un-padded detection box. The box is repeated in `additional_info` because
        that is the only field a vectorstore search row carries back -- `box` itself lands in
        `frame_info`, which the index does not store.

        With `output_tags` set, each detection emits a SECOND FrameTag right after its vector
        tag -- same label, same box, no vector -- for visual aid.

        With `ocr` set, a detection containing legible text also carries `additional_info.text`.
        The field is ABSENT rather than empty when nothing was read, so it costs nothing on the
        tags that have no text -- which means absence does not distinguish "OCR was off" from
        "no text in this box". The run's params record which."""
        cfg = self.config
        if self._brand is None and self._person is None:
            return [self._frame_tag(img)]

        detections: List[Detection] = []
        for detector in (self._person, self._brand):
            # Person first only because it is the cheap one, so a frame that is going to fail
            # some later guard fails sooner. Nothing depends on the order.
            if detector is not None:
                detections.extend(detector.detect(img, cfg))

        # One OCR pass per frame, after detection because "propose" needs the boxes to suppress
        # against. Its cost does not depend on how many were found.
        regions = self._reader.read(img, cfg) if self._reader is not None else []
        if regions and cfg.ocr == "propose" and self._ocr_label:
            detections.extend(proposals(regions, detections, img, cfg, self._ocr_label))

        crops = [d.crop for d in detections]
        if not crops:
            return []

        vectors, upscales = self.embedder.embed(crops, cfg)

        tags: List[FrameTag] = []
        for i, detection in enumerate(detections):
            info = {
                "prompt": detection.prompt,
                "score": detection.score,
                # dict(...) so this copy and the tag's own box cannot alias.
                "box": dict(detection.box),
                # crop_padding changes the vector, so it is provenance, not trivia:
                # vectors built at different padding are not comparable.
                "crop_padding": cfg.crop_padding,
                "upscale": upscales[i],
                # Which backend found it. With two detectors in play this is no longer
                # constant per run, so it is recorded per tag rather than per config.
                "detector": detection.detector,
            }
            if regions:
                # Only when something was read: an empty list on every tag would be noise in
                # every search row, and absence already says "no legible text in this box".
                text = texts_for(detection.box, regions, cfg)
                if text:
                    info["text"] = text
            tags.append(
                FrameTag(
                    tag=detection.label,
                    vector=vectors[i].tolist(),
                    box=detection.box,
                    additional_info={**info, **self._embedder_info()},
                )
            )
            if cfg.output_tags:
                # Same detection, same box, no vector: a visualization aid, so the box shows as
                # a tag track in EVIE.
                # AVModel._combine_adjacent skips tags that HAVE a vector, so
                # these get run-length merged into spans on top of the per-frame copies.
                tags.append(
                    FrameTag(
                        tag=detection.label,
                        box=detection.box,
                        additional_info=info,
                    )
                )

        return tags

    def _frame_tag(self, img: np.ndarray) -> FrameTag:
        """The no-detection output: one whole-frame vector, empty label, full-frame box.

        Same embedder and same knobs (`max_num_patches`, `normalize`) as a crop takes, so the
        only difference is what is handed to it -- here NaFlex downsamples a 1080p frame to the
        patch budget rather than upsampling a crop to it.
        """
        vectors, upscales = self.embedder.embed([np.ascontiguousarray(img)], self.config)
        return FrameTag(
            tag="",  # no class: this is the frame itself, not a detected entity
            vector=vectors[0].tolist(),
            # dict(...) so the tag owns its box and the module constant is never mutated
            box=dict(_WHOLE_FRAME_BOX),
            additional_info={
                "box": dict(_WHOLE_FRAME_BOX),
                "upscale": upscales[0],
                **self._embedder_info(),
            },
        )

    def _embedder_info(self) -> Dict:
        """Provenance stamped on every tag so the index can validate what it is storing and
        so a checkpoint or budget change is visible after the fact rather than silent."""
        return {
            "embedder": self.embedder.model_id,
            "revision": self.embedder.revision,
            "dim": self.embedder.dim,
            "normalize": self.config.normalize,
            "max_num_patches": self.config.max_num_patches,
            "query_modes": list(QUERY_MODES),
        }
