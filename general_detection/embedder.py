"""SigLIP 2 NaFlex embedder, loaded directly from HuggingFace transformers.

NaFlex ("native aspect ratio, flexible resolution") resizes each image to a *patch budget*
rather than to a fixed square, preserving aspect ratio to within one patch. A 16:9 frame is
not squashed, and a 20x160 banner crop becomes 96x672 here where a fixed-resolution
checkpoint (e.g. `-patch16-384`) would squash it 8x horizontally -- and text, signage and
wordmarks are precisely what a non-uniform squash destroys.

The three knobs below are FIXED rather than per-request. Each one changes the emitted vector,
so mixing values within one index silently costs retrievals: a query built at one budget
against an index built at another is not comparable, and nothing surfaces that as an error.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from transformers import Siglip2ImageProcessor, Siglip2VisionModel

# Fixed by the patch16 checkpoints; NaFlex varies the grid, not the patch size.
PATCH_SIZE = 16

# NaFlex resolution budget: each image is resized to cover at most this many 16x16 patches.
# 256 is the checkpoint's documented budget. 576/1024 buy detail for small-object queries at
# attention cost quadratic in the budget.
MAX_NUM_PATCHES = 256

# L2-normalize so cosine similarity reduces to a dot product, which is what the index expects.
# SigLIP 2 is trained with a sigmoid loss over scaled dot products of already-normalized
# embeddings, so cosine *is* its trained similarity and this is lossless.
NORMALIZE = True

# Images per forward pass through the vision tower. Throughput only -- it does not change a
# vector, because NaFlex pads each image along the patch axis and masks the padding out.
BATCH_SIZE = 32


class Siglip2CropEmbedder:
    """Embeds a list of RGB images (whole frames, crops, or both) into pooled SigLIP 2 vectors.

    The emitted dimension is read from the loaded checkpoint (`config.hidden_size`) rather
    than hardcoded, and is stamped into every tag's `additional_info.dim` so the index can
    validate it."""

    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,   # hub commit to pin; None -> default branch
        dtype: Optional[torch.dtype] = None,  # None -> auto (bf16/fp16 on GPU, fp32 on CPU)
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if dtype is None:
            if self.device.type == "cuda":
                # bf16 needs Ampere+ (compute capability >= 8.0); fall back to fp16.
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                logger.warning("cuda not available, embedding on cpu (slow)")
                dtype = torch.float32  # half precision is unstable / slow on CPU
        self.dtype = dtype

        logger.info(f"loading {model_id} (revision={revision}, dtype={dtype}, device={self.device})")
        # `revision` pins the processor and the model to the same hub commit so the whole
        # snapshot is reproducible.
        #
        # Vision tower only — no text embeddings are produced here. Loading a sub-model
        # makes transformers log the checkpoint's text-tower keys as UNEXPECTED; that
        # report is the discarded half of the checkpoint and is expected.
        self.processor = Siglip2ImageProcessor.from_pretrained(model_id, revision=revision)
        self.model = Siglip2VisionModel.from_pretrained(
            model_id, revision=revision, dtype=dtype
        ).to(self.device)
        self.model.eval()

        self.dim = int(self.model.config.hidden_size)
        logger.info(f"embedder ready: {self.dim}-d")

    def embed(self, images: List[np.ndarray]) -> Tuple[np.ndarray, List[float]]:
        """Return ((N, dim) float32 vectors, per-image NaFlex scale factors).

        The scale factor is the linear scale the processor actually applied -- below 1 for a
        whole frame (downsampled to the budget), above it for a crop (upsampled to it). It is
        reported per image so the heavily-interpolated tail can be filtered downstream without
        re-tagging.
        """
        vectors = np.zeros((len(images), self.dim), dtype=np.float32)
        upscales: List[float] = [0.0] * len(images)
        if not images:
            return vectors, upscales

        for start in range(0, len(images), BATCH_SIZE):
            batch = images[start : start + BATCH_SIZE]
            batch_vectors, batch_upscales = self._forward(batch)
            vectors[start : start + len(batch)] = batch_vectors
            upscales[start : start + len(batch)] = batch_upscales

        return vectors, upscales

    def _forward(self, batch: List[np.ndarray]) -> Tuple[np.ndarray, List[float]]:
        # Images of differing sizes batch fine at one budget: each is padded along the patch
        # axis to MAX_NUM_PATCHES and masked, which is what NaFlex is for. So a frame and a
        # 40px crop can share a batch without either affecting the other's vector.
        inputs = self.processor(
            images=[Image.fromarray(image) for image in batch],
            return_tensors="pt",
            max_num_patches=MAX_NUM_PATCHES,
        )

        # spatial_shapes is (num_patches_h, num_patches_w) per image, so the applied scale
        # is recoverable from the public output — no private helper import needed.
        spatial_shapes = inputs["spatial_shapes"]
        upscales = [
            round(float(int(spatial_shapes[i][0]) * PATCH_SIZE / batch[i].shape[0]), 3)
            for i in range(len(batch))
        ]

        # Only pixel_values is float; pixel_attention_mask and spatial_shapes are integer
        # bookkeeping and must keep their own dtypes.
        model_inputs = {k: v.to(self.device) for k, v in inputs.items()}
        model_inputs["pixel_values"] = model_inputs["pixel_values"].to(self.dtype)

        with torch.no_grad():
            # .float() before normalizing: dividing in bf16 lands ~0.1% off unit length,
            # which a cosine index reads as a real score difference.
            pooled = self.model(**model_inputs).pooler_output.float()
            if NORMALIZE:
                pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled.cpu().numpy(), upscales
