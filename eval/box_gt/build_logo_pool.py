#!/usr/bin/env python3
"""Embed the reference logo pool with the tagger's own SigLIP 2 checkpoint.

Why this exists
---------------
Identification in this pipeline is a query against a reference pool, and the title runs showed
that doing it from the brand's NAME barely works: with detection fixed, text->crop found 3 solid
brands out of 15. That is the regime the repo already documents -- a perfect text match sits near
cosine 0.05-0.3 where a good image match sits at 0.5-0.9, and a detected crop retrieves against a
clean reference pool at r@1 0.60. So the pool has to be embedded in the SAME space as the crops,
and queried by image.

model-logo has a pool, but its vectors are 2048-d ResNeXt -- a different space, unusable here.
The source images are what transfer. This re-embeds them with
`google/siglip2-base-patch16-naflex`, the checkpoint the tagger indexes crops with, so a crop
vector and a reference vector are directly comparable.

    python3 eval/box_gt/build_logo_pool.py --pool /ml/pools/logo_pool/2960brands \
                                           --out eval/experiments/12_logo_pool/pool.npz

Note on padding: indexed crops carry `crop_padding` 0.06 of context, reference images are tight.
The README measures a +-0.06 padding mismatch at roughly 8% of top-1 retrievals, so this is a
real but small handicap on every brand equally, and it is not corrected here -- padding a
reference outward would invent pixels that do not exist.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(pool_dir: str, per_brand: int) -> List[tuple]:
    """(brand, path) for every usable reference, capped per brand."""
    items = []
    for brand in sorted(os.listdir(pool_dir)):
        brand_dir = os.path.join(pool_dir, brand)
        if not os.path.isdir(brand_dir):
            continue
        found = []
        for root, _, files in os.walk(brand_dir):
            for name in sorted(files):
                if os.path.splitext(name)[1].lower() in EXTENSIONS:
                    found.append(os.path.join(root, name))
        if per_brand:
            found = found[:per_brand]
        items.extend((brand, path) for path in found)
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="/ml/pools/logo_pool/2960brands")
    parser.add_argument("--out", default="eval/experiments/12_logo_pool/pool.npz")
    parser.add_argument("--model", default="google/siglip2-base-patch16-naflex")
    parser.add_argument("--max-num-patches", type=int, default=256,
                        help="must match the tagger's, or the vectors are not comparable")
    parser.add_argument("--per-brand", type=int, default=0, help="0 = every image")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    items = find_images(args.pool, args.per_brand)
    brands = sorted({brand for brand, _ in items})
    print(f"{len(items)} reference images across {len(brands)} brands")
    if not items:
        print("nothing to embed", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    vectors, kept_brand, kept_path = [], [], []
    for start in range(0, len(items), args.batch):
        chunk = items[start:start + args.batch]
        images, meta = [], []
        for brand, path in chunk:
            try:
                images.append(Image.open(path).convert("RGB"))
                meta.append((brand, path))
            except Exception:
                # A handful of pool files are truncated or not images. Skipping one reference
                # costs almost nothing; aborting the build over it costs the whole run.
                continue
        if not images:
            continue
        inputs = processor(images=images, max_num_patches=args.max_num_patches,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            pooled = model.get_image_features(**inputs).pooler_output.float()
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        vectors.append(pooled.cpu().numpy().astype(np.float32))
        kept_brand.extend(b for b, _ in meta)
        kept_path.extend(p for _, p in meta)
        done = start + len(chunk)
        if done % (args.batch * 50) < args.batch:
            print(f"  {done}/{len(items)}", flush=True)

    matrix = np.concatenate(vectors)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, vectors=matrix,
                        brands=np.array(kept_brand), paths=np.array(kept_path))
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as handle:
        json.dump({"model": args.model, "max_num_patches": args.max_num_patches,
                   "images": int(matrix.shape[0]), "dim": int(matrix.shape[1]),
                   "brands": len(set(kept_brand)), "pool": args.pool}, handle, indent=1)
    print(f"wrote {args.out}: {matrix.shape[0]} x {matrix.shape[1]}, "
          f"{len(set(kept_brand))} brands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
