# 12_logo_pool — identify crops by image, and attribute the misses

`11_titles` showed that fixing detection did not fix the product: on the all-star game, 6.5x the
crops moved end-to-end brand recall from 1 solid brand to 3 out of 15. This experiment asks the
same question in the modality that works, and then splits what is left into causes.

## Query by image, not by name

Text->crop identification was always going to be weak here, and the repo says why without
connecting it: a perfect text match sits near cosine 0.05-0.3 while an image match sits at
0.5-0.9, and a detected crop retrieves against a clean pool at r@1 0.60 -- an *image* number.

So the reference pool at `/ml/pools/logo_pool/2960brands` is re-embedded with the tagger's own
checkpoint and patch budget (`build_logo_pool.py`), which is what makes a crop vector and a
reference vector comparable. 68,933 images, 2,960 brands, capped at 25 references per brand.

| identification | all-star game | Mitchells | control false positives |
|---|---|---|---|
| text, brand name vs 2,454 classes | 3 solid / 15 | 2 solid / 14 | 2 / 6 |
| **image, nearest reference** | **7 / 7 testable** | **5 / 5 testable** | 0-1 / 3 |

Every image match was checked on a contact sheet and every one is correct. State Farm, Tissot
and AT&T match at 0.96-0.98 and are the courtside ribbon boards that started this whole
investigation.

## Presence is easy; duration is the product

Under image queries **both** configs find 7 of 7. Over 983 frames a brand only has to be caught
once, and even the weak detector manages that. What the config change actually buys is how often
it is caught -- and a coverage report sells visibility, not a yes/no.

| brand | frames found, `fast` | frames found, `coverage` @0.07 | x |
|---|---|---|---|
| NBA | 673 | 848 | 1.3 |
| State Farm | 120 | 186 | 1.6 |
| Tissot | 72 | 147 | 2.0 |
| **American Express** | 36 | **285** | **7.9** |
| AT&T | 11 | 26 | 2.4 |
| Nike | 9 | 22 | 2.4 |
| Amazon Prime | 2 | 8 | 4.0 |
| **Kia** | **0** | **28** | -- |

(of 983 frames sampled evenly across 4h16m)

The shipped config would bill American Express for eight times less screen time than it bought,
and report Kia as absent from a game it appears in for seven minutes. That is a stronger case
for the config change than any box metric made it look.

## Three failure modes, and they need different work

A "not found" is useless unless it can be attributed, so `score_brands_image.py` separates:

| status | count of 29 named | |
|---|---|---|
| testable | 12 | the only ones that measure the pipeline |
| stale reference | 1 | Kia |
| no reference at all | 16 | NBC, Peacock, Gatorade, Michelob Ultra, DraftKings, CeraVe, Olympics, Instagram, YouTube, Reddit, Shasta Soda, Waffle House, A&W, See's Candies, Wired, People |

**16 of 29 brands cannot be retrieved by any method**, because a 2,960-brand pool assembled some
years ago does not contain them. Counting those against the detector is simply wrong, and
reporting all three causes as one recall number is what would make the product quietly incorrect.
Pool coverage, not detection or identification, is now the binding constraint.

## Kia revises a claim in the main README

The README states that a rebranded brand is invisible until its pool entry is refreshed, citing
KIA at 0.00. All 95 of its references are verified by eye to be the pre-2021 red oval, and the
broadcast shows the current wordmark -- so the premise holds.

The conclusion does not. Under image queries Kia is found in 28 frames at up to 0.91, every match
correct. Every one of them hits the same reference (`115.jpg`, a 70x27 chrome oval on a dark car
body), and the reason is that both marks render the letters K, I and A in light-on-dark and
SigLIP 2 reads letterforms. The match is real but visibly weaker than a fresh reference: 0.91
against State Farm's 0.98.

So: **a wordmark rebrand degrades retrieval rather than killing it.** A symbol-only rebrand, with
no shared letterforms, would still be fatal. Worth restating in the main README, which currently
generalises from one case.

## Known limits

- The pool has a single `sony` directory, so Sony headphones and the Sony Handycam are one entry
  and cannot be told apart. Fine for brand coverage, wrong for product-level reporting.
- References are tight crops; indexed crops carry `crop_padding` 0.06 of context. The README puts
  a +-0.06 mismatch at roughly 8% of top-1 retrievals. It applies to every brand equally and is
  not corrected -- padding a reference outward would invent pixels.
- The cosine floor of 0.55 is not calibrated, it is picked to sit well above what the absent
  controls reach. With image matches landing at 0.86-0.98 there is a wide margin, but a proper
  sweep against labelled crops has not been run.
- 25 references per brand, alphabetical. A brand whose 26th image is its only current logo is
  penalised arbitrarily.

## Reproducing

```bash
python3 eval/box_gt/build_logo_pool.py --per-brand 25          # ~20 min on one GPU
python3 eval/box_gt/score_brands_image.py --compare eval/experiments/11_titles/nba --brands nba
python3 eval/box_gt/score_brands_image.py --compare eval/experiments/11_titles/mitchells --brands mitchells
```

`pool.npz` is 150 MB and gitignored; rebuild it rather than committing it.
