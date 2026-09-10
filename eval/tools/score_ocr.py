#!/usr/bin/env python3
"""Can a brand be found by the words on the screen? And do OCR boxes help detection?

Two questions, one script, because they share the run files.

    --mode brands   (default) match brand names against the recognised strings, and cross-tab
                    the result against reference-pool coverage, because the interesting cell is
                    "brand OCR finds that image queries cannot reach at all".
    --mode boxes    score OCR text boxes as brand PROPOSALS against box ground truth, alone and
                    unioned with a production run's boxes.

Matching is the part that can fool you, so it is mechanical and published
--------------------------------------------------------------------------
Every string is normalised to uppercase alphanumerics, on both sides. A brand's match key is its
name with a fixed list of generic product qualifiers removed (`QUALIFIERS` below) -- "Wired
magazine" becomes WIRED, "Sony headphones" becomes SONY -- and nothing else. No per-brand
hand-tuning, and the known-absent controls go through the identical function.

Four rules are reported side by side, because the difference between them is exactly the amount
of wishful thinking available here:

    strict    the key must appear. Keys of 5+ characters may match as a SUBSTRING of a longer
              string (so STATEFARM is found inside NBASHOWTIMEPRESENTEDBYSTATEFARM) or fuzzily at
              difflib ratio >= 0.85 (so STATEFAM, an OCR slip, still counts). Keys under 5
              characters must equal the whole string: ATT is a real read, but as a substring it
              also matches TRATTORIA, MATTHEW and MEETUSOUTSIDEATTHE.
    tokens    every token of the name must appear SOMEWHERE IN THE SAME FRAME, not necessarily in
              one string. This exists because of a real defect in `strict` rather than to inflate
              the number: the courtside board reads AMERICAN EXPRESS and easyocr returns it as
              two regions, so the concatenated key AMERICANEXPRESS is never emitted and a brand
              that is plainly legible scores 1 frame in 983. Requiring ALL tokens keeps it
              honest -- an absent brand needs every one of its words to turn up together.
    union     strict OR tokens. **The headline.** The two fail differently and neither dominates:
              strict tolerates an OCR slip across the whole name (STATEFAM at ratio 0.94) while
              tokens tolerates the name arriving as two regions. Their union keeps both, and
              still admits no control brand, which is the check that matters.
    lenient   ANY single token may match. This finds Michelob Ultra through ULTRA, which is
              right, and Amazon Prime through PRIMETIME, which is a television show. Shown so
              the cost of the loose rule is visible, not recommended.


    python3 eval/tools/score_ocr.py --ocr eval/experiments/17_ocr/nba/frame_mag1 --brands nba
    python3 eval/tools/score_ocr.py --mode boxes --ocr eval/experiments/17_ocr/box_gt_mag2 \\
        --run eval/experiments/10_config_ab/runs/B2_step1_conf07
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Set

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "box_gt"))
from score_brands import BRAND_LISTS, CONTROLS  # noqa: E402
from score_brands_image import map_to_pool  # noqa: E402

# Generic product/category words that appear in the brand lists only to separate two products of
# one company, or to say what kind of thing the brand is. They are never part of the mark, so a
# key built from them would be unmatchable. Consequence worth stating: "Sony headphones" and
# "Sony Handycam" both reduce to SONY, so this channel cannot tell them apart -- the same
# limitation the reference pool has, for the same reason.
QUALIFIERS = {"magazine", "restaurants", "soda", "candies", "headphones", "handycam"}

# Below this length a key is matched against the WHOLE string only. ATT, NBA, NBC and KIA are all
# real reads; as substrings they are also inside TRATTORIA, JASONBALDWIN, ENBC and BAKIAY.
SUBSTRING_FLOOR = 5
FUZZY = 0.85


def normalise(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def keys_for(name: str) -> Dict[str, List[str]]:
    """Match keys for one brand name, mechanically: no per-brand special cases anywhere."""
    # Apostrophes are stripped before splitting, not treated as separators: "See's" is one token
    # SEES, and splitting it into SEE + S made the token rule demand a bare "S" on screen.
    tokens = [t for t in re.split(r"[^A-Za-z0-9&]+", name.replace("'", "").replace("’", ""))
              if t]
    kept = [t for t in tokens if t.lower() not in QUALIFIERS] or tokens
    whole = normalise("".join(kept))
    parts = [normalise(t) for t in kept]
    parts = [p for p in dict.fromkeys(parts) if p]
    return {"strict": [whole] if whole else [], "tokens": parts}


def matches(key: str, string: str, allow_short_substring: bool = False) -> bool:
    """`allow_short_substring` lifts the length floor.

    The floor exists because a 3-4 character key is ambiguous as a substring -- ATT is inside
    TRATTORIA, FARM is inside STATEFARM. Under the `tokens` rule with two or more tokens the
    conjunction supplies that specificity instead: STATE *and* FARM both being on screen is not
    ambiguous, and without the lift "State Farm" scores 0 in 983 frames because its own reading,
    STATEFARM, fails the exact-match test on the 4-character half.
    """
    if not key or not string:
        return False
    if len(key) < SUBSTRING_FLOOR and not allow_short_substring:
        return key == string
    if key in string:
        return True
    return difflib.SequenceMatcher(None, key, string).ratio() >= FUZZY


def load_ocr(path: str) -> List[Dict]:
    rows = []
    with open(os.path.join(path, "ocr.jsonl")) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_brands(rows: List[Dict], names: List[str], conf: float) -> Dict:
    """Per name: in how many frames does a matching string appear?

    Rows are grouped by frame even in crop mode, where several rows share one. Without that the
    `tokens` rule would silently mean something stricter for crop mode -- AMERICAN and EXPRESS
    would have to land in the SAME CROP rather than the same frame -- and the two modes would
    stop being comparable.
    """
    # One normalised string set per frame, so a brand read three times in a frame counts once:
    # frames are the screen-time proxy, individual reads are not.
    grouped: Dict[str, Set[str]] = defaultdict(set)
    for row in rows:
        # Touch the frame even when it yielded no text, so it still counts in the denominator.
        strings = grouped[row["frame"]]
        for region in row["regions"]:
            if region["conf"] >= conf:
                key = normalise(region["text"])
                if key:
                    strings.add(key)
    per_frame: List[Set[str]] = list(grouped.values())

    out = {}
    for name in names:
        key_sets = keys_for(name)
        whole, parts = key_sets["strict"], key_sets["tokens"]
        row: Dict = {"keys": key_sets}
        per_rule_frames: Dict[str, Set[int]] = {}
        for rule, need_all, keys in (("strict", True, whole),
                                     ("tokens", True, parts),
                                     ("lenient", False, parts)):
            short_ok = rule == "tokens" and len(keys) > 1
            frames, examples = 0, {}
            per_rule_frames[rule] = set()
            for index, strings in enumerate(per_frame):
                # `hit` is the strings matched, `covered` the keys that found one. need_all
                # demands every key be covered in this frame; AMERICAN and EXPRESS may be two
                # separate regions, but both must be on screen.
                hit, covered = [], 0
                for key in keys:
                    found = [s for s in strings if matches(key, s, short_ok)]
                    if found:
                        covered += 1
                        hit.extend(found)
                if not covered or (need_all and covered < len(keys)):
                    continue
                frames += 1
                per_rule_frames[rule].add(index)
                for s in hit:
                    examples[s] = examples.get(s, 0) + 1
            row[rule] = {"frames": frames, "found": frames > 0,
                         "examples": sorted(examples.items(), key=lambda kv: -kv[1])[:5]}
        # The shipping rule. strict and tokens fail differently and neither dominates: strict
        # tolerates an OCR slip across the whole name (STATEFAM, ratio 0.94) while tokens
        # tolerates the name arriving as two regions (AMERICAN + EXPRESS). Their union keeps both
        # and still admits no control, which is the check that matters.
        union = per_rule_frames["strict"] | per_rule_frames["tokens"]
        row["union"] = {"frames": len(union), "found": bool(union),
                        "examples": row["strict"]["examples"] or row["tokens"]["examples"]}
        out[name] = row
    return out


def report_brands(args) -> int:
    rows = load_ocr(args.ocr)
    stats_path = os.path.join(args.ocr, "stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    targets = BRAND_LISTS[args.brands]
    result = score_brands(rows, targets + CONTROLS, args.conf)
    # Distinct source frames, which is the denominator in both modes: crop mode has many rows
    # per frame, and dividing by the crop count would make the percentages incomparable.
    total = len({row["frame"] for row in rows})

    # Which of these brands the reference pool can reach at all, so the rescue is visible.
    pool_brands: List[str] = []
    if os.path.exists(args.pool):
        import numpy as np
        pool_brands = sorted({str(b) for b in np.load(args.pool, allow_pickle=True)["brands"]})
    mapping = map_to_pool(targets, pool_brands) if pool_brands else {}

    rows_label = f"{stats.get('crops')} crops over " if stats.get("mode") == "crop" else ""
    print(f"{args.ocr}: {stats.get('mode', '?')} mode, {rows_label}{total} frames, "
          f"{stats.get('regions', '?')} regions, mag {stats.get('mag_ratio', '?')}, "
          f"{stats.get('per_frame_ms') or stats.get('per_crop_ms', '?')} ms each "
          f"| OCR conf >= {args.conf}")

    rules = ("strict", "tokens", "union", "lenient")
    head = f"{'brand':20}{'key':18}{'pool':>6}" + "".join(
        f"{r:>9}{'%':>7}" for r in rules) + "   examples"
    print(f"\n{head}\n{'-' * 120}")
    for name in targets:
        row = result[name]
        key = row["keys"]["strict"][0] if row["keys"]["strict"] else "-"
        if len(key) < SUBSTRING_FLOOR:
            key += "*"
        cells = "".join(f"{row[r]['frames']:>9}{row[r]['frames'] / total * 100:6.1f}%"
                        for r in rules)
        extra = ", ".join(f"{k}x{v}" for k, v in row["union"]["examples"][:3])
        print(f"{name:20}{key:18}{'yes' if name in mapping else 'no':>6}{cells}   {extra[:40]}")

    print("-" * 120)
    for label, group in (("FOUND", targets), ("CONTROLS (absent)", CONTROLS)):
        cells = ""
        for rule in rules:
            hits = sum(1 for n in group if result[n][rule]["found"])
            cells += f"{f'{hits}/{len(group)}':>9}{'':7}"
        print(f"{label:20}{'':18}{'':>6}{cells}")
    for c in CONTROLS:
        for r in rules:
            if result[c][r]["found"]:
                print(f"    ! {c} ({r}): {result[c][r]['examples'][:3]}")

    if mapping:
        # The cell that matters: 12_logo_pool found 16 of 29 named brands have no reference at
        # all, and called pool coverage the binding constraint. OCR needs no reference.
        no_ref = [n for n in targets if n not in mapping]
        rescued = [n for n in no_ref if result[n]["union"]["found"]]
        print(f"\nno reference in the pool: {len(no_ref)}/{len(targets)}"
              f"  ->  found by OCR anyway (union rule): {len(rescued)}/{len(no_ref)}"
              f"  {rescued}")

    out = os.path.join(args.ocr, f"scores_ocr_{args.brands}.json")
    with open(out, "w") as handle:
        json.dump(result, handle, indent=1)
    print(f"\nwrote {out}")
    return 0


def report_boxes(args) -> int:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "box_gt"))
    from score_config import ASPECT_BANDS, load_gt, load_run, score

    marks, frames = load_gt()
    rows = {r["frame"]: r for r in load_ocr(args.ocr)}
    # Only the LABELLED frames. run_ocr.py is normally pointed at the whole 100-frame frameset
    # while 25 of those carry ground truth, and leaving the rest in would not change coverage --
    # which indexes by mark -- but would inflate the denominator of boxes/frame and usable/1k.
    labelled = {mark["frame"] for mark in marks}
    ocr_boxes = {name: [tuple(r["box"]) for r in row["regions"] if r["conf"] >= args.conf]
                 for name, row in rows.items() if name in labelled}

    print(f"{len(marks)} marks / {frames} frames | OCR conf >= {args.conf}")
    head = (f"{'source':34}{'coverage':>10}{'contain':>9}{'usable':>8}"
            f"{'boxes/fr':>10}{'usable/1k':>11}")
    print(f"\n{head}\n{'-' * len(head)}")
    bands = {}

    def show(label, dets):
        row = score(marks, dets, args.iou, 0.9, 4.0)
        print(f"{label:34}{row['coverage']:10.3f}{row['containment']:9.3f}{row['usable']:8.3f}"
              f"{row['boxes'] / frames:10.1f}{row['usable_per_1k']:11.1f}")
        bands[label.strip()] = row["bands"]["aspect"]
        return row

    show("OCR boxes alone", ocr_boxes)
    if args.run:
        base = {k: list(v) for k, v in load_run(os.path.join(args.run, "out.jsonl")).items()}
        show(os.path.basename(args.run.rstrip("/")), base)
        union = {k: list(v) for k, v in base.items()}
        for name, boxes in ocr_boxes.items():
            union.setdefault(name, []).extend(boxes)
        show("  + OCR boxes", union)

    order = [b[2] for b in ASPECT_BANDS]
    print(f"\ncoverage by aspect band\n{'source':34}" + "".join(f"{b:>15}" for b in order))
    for label, cells in bands.items():
        print(f"{label:34}" + "".join(f"{cells.get(b, (0, 0))[0]:15.2f}" for b in order))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["brands", "boxes"], default="brands")
    parser.add_argument("--ocr", required=True, help="a run_ocr.py output directory")
    parser.add_argument("--brands", choices=sorted(BRAND_LISTS), help="brands mode: which list")
    parser.add_argument("--run", help="boxes mode: production run to union with")
    parser.add_argument("--conf", type=float, default=None,
                        help="minimum easyocr confidence. Defaults to the shipped gate for the "
                             "mode: 0.2 for strings (`ocr_conf`), 0.0 for proposal boxes "
                             "(`ocr_box_conf`) -- a box CRAFT found is worth cropping even when "
                             "the recogniser garbled what was in it")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--pool", default="eval/experiments/12_logo_pool/pool.npz")
    args = parser.parse_args()
    if args.conf is None:
        args.conf = 0.2 if args.mode == "brands" else 0.0

    if args.mode == "brands":
        if not args.brands:
            parser.error("--mode brands needs --brands")
        return report_brands(args)
    return report_boxes(args)


if __name__ == "__main__":
    raise SystemExit(main())
