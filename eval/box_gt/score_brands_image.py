#!/usr/bin/env python3
"""Identify title crops by IMAGE similarity against the reference pool, and attribute misses.

The text version of this question (`score_brands.py`) found 3 solid brands out of 15 on the
all-star game with detection already fixed. That is the modality gap doing what the README says
it does: a perfect text->image match lands near cosine 0.05-0.3, an image->image match at
0.5-0.9. This asks the same question in the regime that works.

Three failure modes, kept apart
-------------------------------
A brand coverage report is only trustworthy if a "not found" can be attributed, because the three
causes need completely different work:

    NO REFERENCE   the pool has no directory for this brand at all. Nothing about the pipeline
                   is being measured -- Gatorade, Peacock, Instagram, Reddit and Olympics are
                   simply absent from a 2,960-brand pool, so they cannot be retrieved by any
                   method and must not be counted against the detector.

    STALE          the pool has the brand under its OLD mark. KIA is the documented case: every
                   one of its 95 references is the pre-2021 red oval, and the broadcast shows the
                   current wordmark. Visually different marks, so retrieval correctly fails; the
                   fix is a pool refresh, not a model.

    REAL MISS      the pool holds a current reference and the crop still does not retrieve it.
                   This is the only bucket that measures the pipeline.

Reporting all three as one number is what makes a coverage report quietly wrong, so they are
separated here and the headline recall is over the ATTRIBUTABLE brands only.

    python3 eval/box_gt/score_brands_image.py --compare eval/experiments/11_titles/nba \
                                              --brands nba --pool eval/experiments/12_logo_pool/pool.npz
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_brands import BRAND_LISTS, CONTROLS, load_vectors  # noqa: E402

# Brands whose pool references are known to predate a rebrand, so a miss is a pool-freshness
# result rather than a pipeline one. KIA is verified by eye: all 95 references are the pre-2021
# red oval. The repo's own embedder experiment reported KIA at 0.00 for the same reason.
STALE = {"Kia"}


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def map_to_pool(targets: List[str], pool_brands: List[str]) -> Dict[str, str]:
    """Target brand -> pool directory, matching on the normalised name.

    Exact first, then containment, because the pool names things its own way: "tissot", "KIA",
    "ATT", "State Farm Insurance" for State Farm, a bare "sony" for both Sony products.
    """
    index: Dict[str, str] = {}
    for brand in pool_brands:
        index.setdefault(normalise(brand), brand)

    mapping = {}
    for target in targets:
        key = normalise(target)
        if key in index:
            mapping[target] = index[key]
            continue
        contained = [v for k, v in index.items() if k and (k in key or key in k)]
        if contained:
            # Shortest wins: for "Sony Handycam" the useful reference set is "sony", and a
            # longer accidental superstring would be a different company.
            mapping[target] = min(contained, key=len)
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", required=True, help="<title>/ dir holding config subdirs")
    parser.add_argument("--brands", required=True, choices=sorted(BRAND_LISTS))
    parser.add_argument("--pool", default="eval/experiments/12_logo_pool/pool.npz")
    parser.add_argument("--floor", type=float, default=0.55,
                        help="cosine a crop must reach against its best reference")
    parser.add_argument("--dump", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.pool, allow_pickle=True)
    refs, ref_brand = data["vectors"], list(data["brands"])
    pool_brands = sorted(set(ref_brand))
    targets = BRAND_LISTS[args.brands]
    mapping = map_to_pool(targets, pool_brands)
    control_map = map_to_pool(CONTROLS, pool_brands)

    print(f"pool: {refs.shape[0]} references, {len(pool_brands)} brands")
    missing = [t for t in targets if t not in mapping]
    stale = [t for t in targets if t in mapping and t in STALE]
    testable = [t for t in targets if t in mapping and t not in STALE]
    print(f"targets: {len(testable)} testable | {len(stale)} stale ({', '.join(stale) or '-'}) "
          f"| {len(missing)} absent from pool ({', '.join(missing) or '-'})")

    runs = {}
    for name in sorted(os.listdir(args.compare)):
        path = os.path.join(args.compare, name)
        if os.path.exists(os.path.join(path, "out.jsonl")):
            runs[name] = path

    results = {}
    for name, path in runs.items():
        crops, meta = load_vectors(os.path.join(path, "out.jsonl"))
        rows = {}
        if len(crops):
            # Chunked so the crops x references matrix never has to exist all at once.
            best_sim = np.zeros(len(crops), dtype=np.float32)
            best_ref = np.zeros(len(crops), dtype=np.int64)
            for start in range(0, len(crops), 2048):
                block = crops[start:start + 2048] @ refs.T
                best_sim[start:start + 2048] = block.max(axis=1)
                best_ref[start:start + 2048] = block.argmax(axis=1)
            winner = np.array([ref_brand[i] for i in best_ref])
            confident = best_sim >= args.floor
            for target in targets + CONTROLS:
                pool_name = mapping.get(target) or control_map.get(target)
                if pool_name is None:
                    rows[target] = {"crops": 0, "best": 0.0, "found": None, "top": []}
                    continue
                mine = np.flatnonzero((winner == pool_name) & confident)
                order = mine[np.argsort(-best_sim[mine])]
                rows[target] = {
                    "crops": int(mine.size),
                    "best": float(best_sim[order[0]]) if mine.size else
                            float(best_sim[winner == pool_name].max()
                                  if (winner == pool_name).any() else 0.0),
                    "found": bool(mine.size),
                    "top": [{"cos": round(float(best_sim[i]), 4), **meta[i]}
                            for i in order[:args.dump]],
                }
        results[name] = {"crops": len(crops), "brands": rows}

    names = list(results)
    print(f"\nfloor cos>={args.floor} | " +
          " | ".join(f"{n}: {results[n]['crops']} crops" for n in names))
    header = f"{'brand':20}{'pool':>22}" + "".join(f"{n[:16]:>18}" for n in names)
    print(f"\n{header}\n{'-' * len(header)}")
    for target in targets:
        pool_name = mapping.get(target)
        if pool_name is None:
            tag = "(absent from pool)"
        elif target in STALE:
            tag = f"{pool_name} (STALE)"
        else:
            tag = pool_name
        cells = ""
        for name in names:
            row = results[name]["brands"][target]
            if row["found"] is None:
                cells += "n/a".rjust(18)
            else:
                mark = "yes" if row["found"] else " . "
                cells += f"{mark} {row['best']:.2f} x{row['crops']}".rjust(18)
        print(f"{target:20}{tag:>22}{cells}")

    print("-" * len(header))
    for label, subset in (("FOUND (testable)", testable), ("of which stale", stale)):
        if not subset:
            continue
        totals = ""
        for name in names:
            found = sum(1 for b in subset if results[name]["brands"][b]["found"])
            totals += f"{found} / {len(subset)}".rjust(18)
        print(f"{label:20}{'':>22}{totals}")
    totals = ""
    for name in names:
        hit = sum(1 for c in CONTROLS
                  if results[name]["brands"].get(c, {}).get("found"))
        totals += f"{hit} / {sum(1 for c in CONTROLS if c in control_map)}".rjust(18)
    print(f"{'CONTROLS (absent)':20}{'':>22}{totals}")

    if args.dump:
        for name in names:
            print(f"\n=== {name}")
            for target in testable:
                for hit in results[name]["brands"][target]["top"]:
                    print(f"  {target:18} cos={hit['cos']:.3f} {hit['segment']} "
                          f"prompt={hit['prompt']}")

    out = os.path.join(args.compare, "scores_brands_image.json")
    with open(out, "w") as handle:
        json.dump({"mapping": mapping, "stale": sorted(stale), "absent": missing,
                   "results": results}, handle, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
