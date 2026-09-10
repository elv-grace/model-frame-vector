# 10_config_ab — what does the shipped config actually deliver?

Every earlier experiment ranks *detectors*. This one measures a *config*: the same production
tagger, run end to end under four `--params` settings, scored at the operating point each one
actually produces rather than at a swept optimum.

It exists because the two questions had drifted apart. `score_boxes.py` picks a threshold per
backend and reports each at its own best point, which is correct for choosing a model. Nobody
had asked what the chosen model delivers once the chosen threshold, the synonym NMS,
`max_detections` and `min_crop_pixels` are all applied — and the answer turned out to be
roughly half of what the README quotes.

## Two measurement faults, found first

**Coverage was never gated.** `score_boxes.py`'s `coverage_recall` takes an *IoU* threshold and
no confidence one, so it scores whatever floor the run file was written at — 0.05 for the
transformer backends, 0.005 for the ultralytics ones. The README's headline figures (gdino
0.61, yoloe26 0.25) are that number. The shipped gates are 0.15 and 0.007. Re-scored at the
gates that ship:

| gdino gate | brand coverage | boxes/frame |
|---|---|---|
| 0.15 (shipped) | 0.328 | 10.8 |
| 0.10 | 0.448 | 24.7 |
| 0.07 | 0.573 | 41.3 |
| 0.05 | 0.609 | 60.3 |

The 0.15 was selected by maximising F1, i.e. buying detection precision. For a crop-and-embed
index that is the wrong objective: a false-positive crop costs one SigLIP pass, retrieves
nothing, and disappears, while a mark that is never cropped is gone for good.

**Nothing read box shape.** Mark *size* is documented at length; mark *aspect* was not measured
at all, and it separates the backends far more sharply. On the 25 ground-truth marks with
aspect ratio ≥ 3 — courtside hoardings, ribbon boards, banner wordmarks — yoloe26 @1280 hits 2,
gdino @0.15 hits 7, gdino @0.05 hits 15, owlv2 hits 9.

Nor are yoloe's failures all misses. On the 261×29 px board in `NBA33min__000` it emits a box
that fully contains the mark and still scores IoU 0.24, because the box is roughly square and
four times too large. That is worse than a miss downstream: the crop is embedded anyway, and a
vector of "mostly LED board, 20% wordmark" takes an index slot and never retrieves. Hence the
`containment` column in `score_config.py`, reported beside coverage.

## Result — 25 box-GT frames, production path

| config | coverage | containment | boxes/frame |
|---|---|---|---|
| `fast` · yoloe26 @1280, conf 0.007, 6 prompts *(ships today)* | 0.219 | 0.214 | 3.5 |
| `coverage` · gdino @800, conf 0.15, 6 prompts | 0.188 | 0.208 | 5.1 |
| `coverage` · conf 0.07, 4 prompts, cap 100 | 0.469 | 0.641 | 21.4 |
| **`coverage` · conf 0.05, 4 prompts, cap 100** | **0.578** | **0.766** | 32.2 |

2.6x the coverage and 3.6x the containment. Note the second row: at their shipped gates the two
options are within noise of each other, so `coverage` mode does not currently deliver more
coverage than `fast` mode — which is the only reason it exists.

Coverage by aspect band (n = 112 / 55 / 19 / 6):

| | <1.5 | 1.5–3 | 3–5 | ≥5 |
|---|---|---|---|---|
| `fast`, shipped | 0.28 | 0.16 | 0.11 | 0.00 |
| `coverage` @0.05 | 0.54 | 0.64 | 0.53 | 1.00 |

Coverage by short side (n = 37 / 69 / 20 / 28 / 38):

| | <16px | 16–24 | 24–32 | 32–48 | 48+ |
|---|---|---|---|---|---|
| `fast`, shipped | 0.00 | 0.12 | 0.25 | 0.46 | 0.42 |
| `coverage` @0.05 | 0.16 | 0.46 | 0.80 | 0.86 | 0.87 |

The sub-16 px band stays hard, which is the genuine small-object floor already documented in
`09_min_crop`. No threshold moves it.

## The stills understate the load

Frozen frames are not a cost model. Over two minutes of live NBA broadcast at 1 fps the same
configs behave differently enough to change which one should ship:

| 120 frames @ 1 fps | boxes/frame | median | frames at cap | wall |
|---|---|---|---|---|
| `fast`, shipped (cap 30) | 6.5 | 6 | 0 / 119 | 105 s |
| `coverage` @0.07 (cap 100) | 59.1 | 59 | 5 / 120 | 137 s |
| `coverage` @0.05 (cap 100) | 79.7 | 88 | 39 / 120 | 162 s |

End to end that is 1.3–1.5x the wall clock, not the ~4x a per-frame cost model predicts, because
decode is a shared cost that dominates the cheap config. The real price is 9–12x the vectors:
a two-hour title at 1 fps goes from ~47k crops to 425k (0.07) or 576k (0.05).

`max_detections` also becomes the binding constraint at 0.05, and truncation is by score — it
removes exactly the low-confidence board marks the lower gate was opened to admit.

**So: 0.07 is the better routine default, 0.05 the better archival pass.**

```
--params '{"brand_detector": "coverage",
           "brand_conf": 0.07,
           "max_detections": 100,
           "class_prompts": {"brand": ["logo", "letter logo", "brand", "car logo"],
                             "person": ["person"]}}'
```

## Prompts: which two to drop, and which one to keep

Marginal coverage of each brand term (coverage of all six, minus coverage without it):

| prompt | gdino @0.15 | gdino @0.05 | yoloe @0.007 |
|---|---|---|---|
| `logo` | +0.042 | +0.167 | +0.177 |
| `letter logo` | +0.068 | +0.068 | +0.000 |
| `brand` | +0.062 | +0.052 | +0.005 |
| `car logo` | +0.005 | +0.005 | +0.000 |
| `emblem` | +0.000 | +0.000 | +0.000 |
| `label` | +0.000 | +0.000 | +0.016 |

`emblem` and `label` are dropped: zero marginal coverage at every gdino gate, and `label`
returns one box in 100 frames. `letter logo` is **kept** — it is gdino's second contributor and
48% of its brand boxes. It is worth nothing under yoloe, which is presumably where the instinct
to remove it came from; the right prompt set differs per backend.

## Caveats

192 marks, 25 frames, 11 clips. The banner band is 6 marks and the wide band 19 — the direction
is unambiguous and the mechanism is understood, but the exact operating point wants a larger
labelled set before it is called tuned. The ground truth also contains no animation.

## Reproducing

```bash
./eval/box_gt/run_config_ab.sh              # all four configs over the 25 box-GT frames
python3 eval/box_gt/score_config.py         # coverage, containment, aspect and size bands

# the video rungs (test-files/seg/ is gitignored; cut it first)
ffmpeg -ss 600 -t 120 -i test-files/long/NBA33min.mp4 -an -c:v libx264 -crf 20 \
       test-files/seg/NBA_2min.mp4
```

Runs are stored with their `vector` fields stripped — scoring reads boxes and scores only, and
the vectors were 97% of 334 MB.
