"""SigLIP 2 frame-vector tagger, with an optional detection phase.

What it emits
-------------
The whole frame is ALWAYS embedded: every sampled frame yields one vector tag with an empty
label and a full-frame box, which is the whole output when nothing else is asked for.

Setting `detect_target` adds a detection phase on top. Each surviving detection is cropped and
embedded too, so the frame keeps its vector and gains one vector per detection beside it -- each
carrying its box, which is what the video-editor overlay draws.

The frame and its crops go through the vision tower in ONE batch. That matters: the embedder is
the larger cost at low detection counts, and batching keeps it near its throughput rather than
its latency. NaFlex pads each image along the patch axis to the same budget and masks the
padding out, so a 1080p frame and a 40 px crop share a batch without either affecting the
other's vector.

One detector, and the frame vector is the constant
--------------------------------------------------
Every target goes to the open-vocabulary backend named by `detector` -- there is no per-term
routing and no second family of weights. With `detect_target` unset, no detector is constructed
at all, so the default path loads the embedder and nothing else.

A frame vector and a crop vector are not interchangeable, and both are emitted on purpose: a
frame is DOWNsampled to the patch budget and a crop is UPsampled to it, so they answer different
questions ("which frames look like this" against "where is this mark"). A consumer tells them
apart by the empty label and the full-frame box.
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
from general_detection.detector import CROP_PADDING, Detection, build_detector
from general_detection.embedder import MAX_NUM_PATCHES, NORMALIZE, Siglip2CropEmbedder
from general_detection.prompts import expand_target

# The frame vector's box: the full image in normalized coords.
_WHOLE_FRAME_BOX = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}

QUERY_MODES = ["text", "image"]


class FrameVectorModel(FrameModel):
    """One SigLIP 2 vector per sampled frame, plus one per detected crop when
    `detect_target` is set."""

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
        self._detector = None
        self._detector_mode: Optional[str] = None
        self._apply_targets(cfg)
        # Same device as the detector. Without threading it through, an explicit device (say
        # "cuda:1") would place the detector there and the embedder on cuda:0, which works but
        # copies every crop across devices and silently occupies a card the caller did not ask
        # for. None keeps the embedder's own auto-detection.
        self.embedder = Siglip2CropEmbedder(
            embedder_model_id,
            revision=embedder_revision,
            device=torch.device(device) if device else None,
        )

    # ---- configuration ----------------------------------------------------------

    def _apply_targets(self, cfg: RuntimeConfig) -> None:
        """Build, re-prompt or release the detector for the current target.

        Loading is lazy: no target constructs nothing, and narrowing to no target releases the
        weights rather than keeping them idle. Rebuilding is confined to a real backend change
        (`detector` switching between "fast" and "coverage"); a target-only change re-encodes
        text, which is seconds, rather than reloading weights.
        """
        if not cfg.detect_target:
            self._detector, self._detector_mode = None, None
            logger.info("no detect_target: one whole-frame vector per frame, no detector loaded")
            return

        class_prompts = expand_target(cfg.detect_target)
        if self._detector is None or self._detector_mode != cfg.detector:
            self._detector = build_detector(cfg.detector, self.cache_dir, self.device)
            self._detector_mode = cfg.detector
        self._detector.set_prompts(class_prompts)
        logger.info(f"targets: {sorted(class_prompts)} on {cfg.detector}")

    def set_config(self, config: dict) -> None:
        self.config = from_dict(RuntimeConfig, config)
        self._apply_targets(self.config)

    def get_config(self) -> dict:
        return asdict(self.config)

    # ---- tagging ----------------------------------------------------------------

    def tag_frame(self, img: np.ndarray) -> List[FrameTag]:
        """img: (H, W, 3) uint8 RGB. Always returns the frame's own vector tag first, then one
        tag per detection when a target is set.

        A detection tag's `tag` is the parent term, `vector` the crop embedding and `box` the
        normalized un-padded detection box. The box is repeated in `additional_info` because
        that is the only field a vectorstore search row carries back -- `box` itself lands in
        `frame_info`, which the index does not store.

        With `output_tags` set, each detection emits a SECOND FrameTag right after its vector
        tag -- same label, same box, no vector -- as a visualization aid. The frame tag is never
        twinned: its label is empty, so a vector-less copy would carry no information.
        """
        cfg = self.config
        detections: List[Detection] = (
            self._detector.detect(img, cfg) if self._detector is not None else []
        )

        # Frame first so its vector is always index 0, then the crops. One batch.
        vectors, upscales = self.embedder.embed(
            [np.ascontiguousarray(img), *(d.crop for d in detections)]
        )

        tags = [
            FrameTag(
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
        ]

        for i, detection in enumerate(detections, start=1):
            info = {
                "prompt": detection.prompt,
                "score": detection.score,
                # dict(...) so this copy and the tag's own box cannot alias.
                "box": dict(detection.box),
                # crop_padding changes the vector, so it is provenance, not trivia:
                # vectors built at different padding are not comparable.
                "crop_padding": CROP_PADDING,
                "upscale": upscales[i],
                # Which backend found it, so a mixed index stays attributable.
                "detector": detection.detector,
            }
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
                # a tag track in EVIE. AVModel._combine_adjacent skips tags that HAVE a vector,
                # so these get run-length merged into spans on top of the per-frame copies.
                tags.append(
                    FrameTag(
                        tag=detection.label,
                        box=detection.box,
                        additional_info=info,
                    )
                )

        return tags

    def _embedder_info(self) -> Dict:
        """Provenance stamped on every tag so the index can validate what it is storing and
        so a checkpoint change is visible after the fact rather than silent. `normalize` and
        `max_num_patches` are fixed constants now, but they still ride along: a query has to
        match on them, and an index built by an older version may not share them."""
        return {
            "embedder": self.embedder.model_id,
            "revision": self.embedder.revision,
            "dim": self.embedder.dim,
            "normalize": NORMALIZE,
            "max_num_patches": MAX_NUM_PATCHES,
            "query_modes": list(QUERY_MODES),
        }
