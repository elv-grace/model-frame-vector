"""Detection targets for the tagger: the phrasings each target term expands to.

The schema
----------
Two known parents, measured into their current form against box-level ground truth (see eval/):

    brand   four MARK terms, and deliberately no object terms. `brand` means the mark itself --
            the GAP wordmark, not the hoodie; the NFL shield, not the helmet. Asked for
            `sportswear` a detector returns the garment; asked for `logo` it returns the
            wordmark on it, and the wordmark is the crop that retrieves against a logo pool.
            With 101 concrete object nouns only 1% of brand detections carried a mark-like
            label (63% were `shoe`); with the mark terms, 100% do.

            It was six terms until `emblem` and `label` were measured for MARGINAL coverage --
            coverage of all six, minus coverage without that one term -- against box ground
            truth (eval/experiments/10_config_ab):

                prompt          gdino @0.15   gdino @0.07   yoloe @0.007
                logo               +0.042        +0.167        +0.177
                letter logo        +0.068        +0.068        +0.000
                brand              +0.062        +0.052        +0.005
                car logo           +0.005        +0.005        +0.000
                emblem             +0.000        +0.000        +0.000
                label              +0.000        +0.000        +0.016

            `emblem` and `label` earn nothing on the shipping backend and `label` returns one
            box in 100 frames, so both are dropped: each term costs a detection pass' worth of
            candidate boxes and crops. `letter logo` is KEPT despite looking redundant -- it is
            Grounding DINO's second-largest contributor and 48% of its brand boxes on a full
            title. That it is worth nothing to YOLOE is the point: the useful prompt set is a
            property of the backend, not of the schema.

    person  one word, and its own phrasing. `person` alone reaches 0.92-0.97 class-agnostic
            coverage on every promptable backend -- identical to a 101-term list including 18
            role words (player, referee, commentator, fashion model). The role words bought
            nothing, so identifying *which* person stays a downstream query against the vectors.

Mark-CARRYING surfaces (`sign`, `banner`, `billboard`) were tested and rejected: they reproduce
the overshadowing failure one level up, since a banner is a surface a logo sits on, so the box
lands on the banner. `symbol` was tested as a seventh brand term and rejected too -- it costs the
leading model AP and wins the argmax on boxes `logo` already had.

One detector serves every target
--------------------------------
There is no routing any more. `person` used to go to a closed COCO-80 backend (YOLO11), which
beat every open-vocabulary model at that one class -- but keeping a second family of weights,
with its own resolution, its own threshold and its own incomparable score scale, for a single
class is not worth the surface area. Everything now goes to the open-vocabulary backend named by
`detector`, which grounds `person` perfectly well.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# The mark terms. Order is stable because it fixes class ids for a given config.
# BRAND_PROMPTS = ["logo", "letter logo", "car logo", "emblem", "brand", "label"]  # was: six
BRAND_PROMPTS: List[str] = ["logo", "letter logo", "brand", "car logo"]
PERSON_PROMPTS: List[str] = ["person"]

# The parents that expand to something other than themselves. Any term not in here is its own
# parent with itself as the single phrasing.
DEFAULT_CLASS_PROMPTS: Dict[str, List[str]] = {
    "brand": list(BRAND_PROMPTS),
    "person": list(PERSON_PROMPTS),
}


def expand_target(target: List[str]) -> Dict[str, List[str]]:
    """Turn a caller's `detect_target` list into the {parent: [phrasings]} form detectors take.

    A term naming a known parent expands to that parent's phrasings, so `brand` becomes the four
    mark terms rather than the literal word -- which matters, because the bare word `brand` is a
    far weaker prompt than the mark list and only Grounding DINO grounds it at all. Any other
    term becomes its own parent with itself as the single phrasing.
    """
    out: Dict[str, List[str]] = {}
    for term in target:
        key = term.strip()
        if not key:
            continue
        out[key] = list(DEFAULT_CLASS_PROMPTS.get(key, [key]))
    if not out:
        raise ValueError("detect_target resolved to no usable terms")
    return out


def flatten(class_prompts: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """Return (prompts, parent_label_per_prompt), index-aligned.

    The detector's class ids index into `prompts`, so the two lists must stay parallel:
    class id -> prompt -> parent label. Ordering follows `class_prompts` insertion order,
    which keeps class ids stable for a given config.
    """
    prompts: List[str] = []
    labels: List[str] = []
    for label, phrasings in class_prompts.items():
        if not phrasings:
            raise ValueError(f"class {label!r} has no prompts")
        for phrasing in phrasings:
            if phrasing in prompts:
                # One phrasing cannot belong to two parents: the detector emits a single
                # class id for it and the parent choice would be arbitrary.
                raise ValueError(f"prompt {phrasing!r} is used by more than one class")
            prompts.append(phrasing)
            labels.append(label)
    if not prompts:
        raise ValueError("class_prompts is empty")
    return prompts, labels
