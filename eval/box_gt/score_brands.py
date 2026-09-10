#!/usr/bin/env python3
"""Per-brand retrieval recall over a title run: can each named brand be found at all?

Why this is the right question for a title
------------------------------------------
The box ground truth answers "was this mark boxed", which needs labelled boxes. A full title has
none, and hand-labelling one is weeks. But a brand coverage report does not actually ask about
boxes -- it asks *was this brand present, and for how long*. That is answerable from a run plus a
list of brands someone read off the screen, which is what exists here.

So this scores the pipeline end to end, the way it will actually be used: embed the brand name
with the SigLIP 2 text tower and score it against every crop vector in the run.

Why a probability floor does NOT work here, measured
-----------------------------------------------------
The obvious design -- embed the brand name, keep crops above a calibrated probability -- was
tried first and is worthless at this scale. Against 27,000 crops from the all-star game:

    query                        max p    n >= 0.10
    "State Farm"  (present)      0.857          317
    "Tissot"      (present)      0.845           82
    "Lufthansa"   (absent)       0.930          217
    "zzzzz nonexistent brand"    0.967          799

Every target brand "passes", and so does a string of z's, scoring higher than all of them. Two
things break it. The maximum over 27k x 4 comparisons is extreme for any query whatsoever; and
the template dominates the text vector, so every "<something> logo" query points in nearly the
same direction and retrieves the same generic logo-shaped crops. (The control that behaves is
"a giraffe in a tuxedo" at max p 0.138 -- semantically distant queries do separate. Brand names
do not separate from each other.)

So the question is asked DISCRIMINATIVELY instead, which is how model-logo already does
identification. Each crop is scored against a vocabulary of every target brand plus ~2,450
real distractor brands (model-logo's `cropped_pool_classes.json`) and assigned to its argmax.
A brand counts as found when some crop picks it out of that field by a margin. A brand absent
from the content now has ~2,450 competitors for every crop it might steal, and the run reports
how many crops the known-absent controls collect so the reader can see the floor.

Read the result as a JOINT measurement of detector and embedder: a brand can be missed because
no crop was taken of it, or because the crop does not retrieve from its own name. What the
comparison between two configs isolates cleanly is the DETECTION delta, since both share an
embedder and a vocabulary.

    python3 eval/box_gt/score_brands.py --compare eval/experiments/11_titles/nba --brands nba
    python3 eval/box_gt/score_brands.py --compare eval/experiments/11_titles/nba --brands nba \
                                        --dump 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List

import numpy as np

# Brands read off screen by a human watching each title. These are the recall targets; nothing
# here claims they are exhaustive, only that each one is definitely present somewhere.
#
# The two Sony entries are deliberately separate products rather than one "Sony": in the film the
# mark appears both on headphones and on a Handycam, at very different scales and against very
# different backgrounds, and collapsing them would hide a miss on either.
BRAND_LISTS: Dict[str, List[str]] = {
    "nba": ["State Farm", "NBC", "Peacock", "NBA", "Kia", "Nike", "Tissot", "Gatorade",
            "AT&T", "Michelob Ultra", "DraftKings", "American Express", "CeraVe",
            "Amazon Prime", "Olympics"],
    "mitchells": ["Instagram", "Twitter", "Facebook", "YouTube", "Amazon", "Reddit",
                  "Shasta Soda", "Waffle House", "A&W Restaurants", "See's Candies",
                  "Sony headphones", "Sony Handycam", "Wired magazine", "People magazine"],
}

# Brands definitely absent from both titles, scored through the identical pipeline. Whatever
# these collect is the noise floor: a target scoring like them has been guessed, not found.
CONTROLS = ["Lufthansa", "Banco Santander", "Carlsberg", "Renault", "Aeroflot", "Danone"]

# The BARE brand name, and that is measured rather than assumed. Against the all-star run,
# argmax over the full vocabulary at p >= 0.05:
#
#   template                    targets found   control false positives
#   "{}"                             3 / 15        0 / 6   (0 crops)
#   "a photo of the {} logo"         3 / 15        2 / 6  (40 crops)
#   "the {} logo"                    1 / 15        2 / 6  (10 crops)
#
# The caption templates are the usual CLIP zero-shot advice and they are wrong here. They pull
# every query toward a generic "logo photo" direction, which does not help separate one brand
# from another and does let absent brands collect crops -- Aeroflot alone takes 38.
TEMPLATE = "{}"

# Real brand names to compete against, so an absent brand faces ~2,440 rivals for any crop it
# might otherwise win by default. model-logo built this pool for exactly this purpose.
DISTRACTOR_POOL = ("/home/elv-grace/model-logo/weights/resnext/feature_pool/"
                   "cropped_pool_classes.json")


def load_vectors(path: str):
    """Crop vectors plus enough provenance to point a human at the frame."""
    vectors, meta = [], []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") != "tag":
                continue
            data = record["data"]
            vector = data.get("vector")
            if not vector:
                continue
            frame_info = data.get("frame_info") or {}
            vectors.append(vector)
            meta.append({
                "segment": os.path.splitext(os.path.basename(data["source_media"]))[0],
                "frame_idx": frame_info.get("frame_idx"),
                "start_time": data.get("start_time"),
                "score": data["additional_info"].get("score"),
                "prompt": data["additional_info"].get("prompt"),
                # additional_info.box is what a vectorstore row carries back, so prefer it --
                # but only a current tagger stamps it, and frame_info.box is always there.
                "box": data["additional_info"].get("box") or frame_info.get("box"),
            })
    if not vectors:
        return np.zeros((0, 1), dtype=np.float32), meta
    matrix = np.asarray(vectors, dtype=np.float32)
    # The tagger already L2-normalises, but a run made with normalize:false would silently
    # inflate every score here, so normalise defensively.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None), meta


class TextScorer:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        # logit_scale is stored as the LOG of the scale and must be exponentiated.
        self.scale = self.model.logit_scale.exp().float().item()
        self.bias = self.model.logit_bias.float().item()

    def embed(self, texts: List[str]) -> np.ndarray:
        torch = self._torch
        # max_length padding to 64 is the fixed-length tokenisation SigLIP was trained with.
        # Tokenised any other way the query vector moves far enough to change the ranking.
        inputs = self.processor.tokenizer(
            texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            pooled = self.model.get_text_features(**inputs).pooler_output.float()
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        return pooled.cpu().numpy()

    def probability(self, cosine: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(cosine * self.scale + self.bias)))


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def build_vocabulary(targets: List[str]) -> List[str]:
    """Targets, controls, and every pool brand that does not collide with one of them.

    Collisions matter more than they look. An exact duplicate ("Nike" in both lists) produces
    two identical text vectors and an arbitrary argmax tie, so a real hit is scored as a miss
    about half the time. A near-duplicate splits the vote instead: a Sony Handycam crop can pick
    the pool's bare "sony" over the target "Sony Handycam" and be recorded as a miss. Both are
    removed by substring containment on the normalised name.
    """
    with open(DISTRACTOR_POOL) as handle:
        pool = json.load(handle)
    reserved = {normalise(name) for name in targets + CONTROLS}
    distractors = []
    for name in pool:
        key = normalise(name)
        # "Negative" is the pool's own no-logo class, not a brand.
        if not key or key == "negative":
            continue
        if any(key == r or key in r or r in key for r in reserved):
            continue
        distractors.append(name)
    return targets + CONTROLS + distractors


def score_run(run_dir: str, targets: List[str], vocabulary: List[str], text: np.ndarray,
              scorer: TextScorer, floor: float, dump: int = 0) -> Dict:
    vectors, meta = load_vectors(os.path.join(run_dir, "out.jsonl"))
    empty = {"crops": 0, "assigned": 0,
             "brands": {b: {"crops": 0, "best": 0.0, "found": False, "top": []} for b in targets},
             "controls": {c: 0 for c in CONTROLS}}
    if len(vectors) == 0:
        return empty

    probs = scorer.probability(vectors @ text.T)
    winner = probs.argmax(axis=1)
    best = probs.max(axis=1)
    confident = best >= floor

    rows = {}
    for brand in targets:
        index = vocabulary.index(brand)
        mine = np.flatnonzero((winner == index) & confident)
        order = mine[np.argsort(-best[mine])]
        rows[brand] = {
            "crops": int(mine.size),
            "best": float(best[order[0]]) if mine.size else 0.0,
            "found": bool(mine.size),
            "top": [{"p": round(float(best[i]), 4), **meta[i]} for i in order[:dump]],
        }
    controls = {c: int(((winner == vocabulary.index(c)) & confident).sum()) for c in CONTROLS}
    return {"crops": len(vectors), "assigned": int(confident.sum()),
            "brands": rows, "controls": controls}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="a single <config>/ directory holding out.jsonl")
    parser.add_argument("--compare", help="a <title>/ directory holding several config dirs")
    parser.add_argument("--brands", required=True, choices=sorted(BRAND_LISTS))
    parser.add_argument("--floor", type=float, default=0.05,
                        help="calibrated probability a crop's winning brand must reach")
    parser.add_argument("--dump", type=int, default=0, help="print N top crops per brand")
    parser.add_argument("--model", default="google/siglip2-base-patch16-naflex")
    args = parser.parse_args()

    brands = BRAND_LISTS[args.brands]
    scorer = TextScorer(args.model)

    runs = {}
    if args.compare:
        for name in sorted(os.listdir(args.compare)):
            path = os.path.join(args.compare, name)
            if os.path.exists(os.path.join(path, "out.jsonl")):
                runs[name] = path
    elif args.run:
        runs[os.path.basename(args.run.rstrip("/"))] = args.run
    else:
        print("pass --run or --compare", file=sys.stderr)
        return 1

    vocabulary = build_vocabulary(brands)
    # Encode the vocabulary once and share it across runs, so the comparison between configs
    # differs only in the crops.
    text = np.concatenate([scorer.embed([TEMPLATE.format(v) for v in vocabulary[i:i + 256]])
                           for i in range(0, len(vocabulary), 256)])

    results = {name: score_run(path, brands, vocabulary, text, scorer, args.floor, args.dump)
               for name, path in runs.items()}

    names = list(results)
    print(f"\n{len(brands)} target brands vs {len(vocabulary) - len(brands) - len(CONTROLS)} "
          f"distractors + {len(CONTROLS)} controls | template {TEMPLATE!r} | "
          f"floor p>={args.floor}")
    print(" | ".join(f"{n}: {results[n]['crops']} crops" for n in names))

    header = f"{'brand':20}" + "".join(f"{n[:18]:>20}" for n in names)
    print(f"\n{header}\n{'-' * len(header)}")
    for brand in brands:
        cells = ""
        for name in names:
            row = results[name]["brands"][brand]
            mark = "yes" if row["found"] else " . "
            cells += f"{mark}  p={row['best']:.3f}  x{row['crops']}".rjust(20)
        print(f"{brand:20}{cells}")

    totals = ""
    for name in names:
        found = sum(1 for b in brands if results[name]["brands"][b]["found"])
        totals += f"{found} / {len(brands)}".rjust(20)
    print(f"{'-' * len(header)}\n{'FOUND':20}{totals}")

    # The floor, restated as data rather than as a claim.
    noise = ""
    for name in names:
        hit = sum(1 for c in CONTROLS if results[name]["controls"][c])
        noise += f"{hit} / {len(CONTROLS)}  ({sum(results[name]['controls'].values())})".rjust(20)
    print(f"{'CONTROLS (absent)':20}{noise}")

    if args.dump:
        for name in names:
            print(f"\n=== {name}: top crops")
            for brand in brands:
                for hit in results[name]["brands"][brand]["top"]:
                    print(f"  {brand:18} p={hit['p']:.3f} {hit['segment']} "
                          f"frame={hit['frame_idx']} prompt={hit['prompt']}")

    out = os.path.join(args.compare or args.run, "scores_brands.json")
    with open(out, "w") as handle:
        json.dump(results, handle, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
