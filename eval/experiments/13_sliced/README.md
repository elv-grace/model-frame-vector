# 13_sliced — sliced inference, and the OWLv2 union it makes redundant

Brand is a small-object problem: two thirds of ground-truth marks are under 32 px, and the
shipped config reaches only 0.14 coverage below 16 px and 0.36 at 16-24 px. Resolution is the
obvious lever and it is not available -- experiment 06 measured Grounding DINO collapsing from
brand AP 0.205 to 0.001 past its native 800, taking person down with it, because a DETR
decoder's learned reference points are tuned to the training resolution.

Slicing buys the same thing without touching `imgsz`. A 1920x1080 frame is downscaled to 800 on
the shortest edge -- 0.74x -- so a 22 px mark arrives at the model as 16 px. A 960-wide tile is
*upscaled* to 800, and the same mark arrives near 27 px, with the model still at exactly the
resolution it was trained for.

## Result: 2x2 with the full-frame pass kept

| config | coverage | contain | usable | boxes/frame | passes/frame | s/frame |
|---|---|---|---|---|---|---|
| shipped, no tiling | 0.479 | 0.641 | 0.474 | 21.8 | 1 | 0.34 |
| **2x2 + full frame** | **0.599** | **0.781** | **0.589** | 42.9 | 5 | 1.38 |
| 3x2 + full frame | 0.604 | 0.750 | 0.552 | 46.2 | 7 | 1.74 |
| 4x3 + full frame | 0.589 | 0.750 | 0.495 | 56.4 | 13 | 3.40 |
| OWLv2 union, no tiling | 0.526 | 0.661 | 0.490 | 25.2 | 2 | 0.52 |
| OWLv2 union + 2x2 | 0.604 | 0.776 | 0.589 | 43.6 | 10 | 2.20 |
| **2x2 through the tagger** | **0.599** | **0.797** | **0.594** | 44.8 | 5 | -- |

`usable` counts a mark only when some box contains it *and* is no more than 4x its area: a crop
whose vector is about the mark rather than about the board it sits on. It is the column to read.

**+27% usable coverage for five detector passes.** Note that coverage and usable move together
here, unlike under YOLOE, because Grounding DINO's boxes are tight when it finds anything at all.

## Finer tiling is worse, and the aspect bands say why

| tiling | <1.5 compact | 1.5-3 | 3-5 wide | >=5 banner |
|---|---|---|---|---|
| none | 0.46 | 0.47 | 0.42 | 1.00 |
| 2x2 | 0.54 | 0.64 | **0.68** | 1.00 |
| 3x2 | 0.57 | 0.62 | 0.63 | 1.00 |
| 4x3 | 0.59 | 0.62 | **0.37** | 1.00 |

Small marks keep improving as tiles shrink -- the <16 px band goes 0.14 / 0.19 / 0.19 / 0.22 --
but wide marks are destroyed by the seams. A 669 px hoarding spans two 480 px tiles at 4x3, and
every tile sees a fragment that no longer matches the ground-truth box. 2x2 is where small-mark
recall has arrived and wide marks have not yet been cut up.

This is also why the full-frame pass is kept rather than replaced by tiles. It is the only pass
that sees a hoarding whole.

| tiling | <16px | 16-24 | 24-32 | 32-48 | 48+ |
|---|---|---|---|---|---|
| none | 0.14 | 0.36 | 0.55 | 0.75 | 0.79 |
| 2x2 | 0.19 | 0.48 | **0.85** | 0.86 | 0.89 |
| 3x2 | 0.19 | 0.51 | 0.75 | 0.86 | 0.92 |
| 4x3 | 0.22 | 0.51 | 0.70 | 0.86 | 0.84 |

## The OWLv2 union is redundant once you tile

This was expected to be the win. It is not. **2x2 + OWLv2 delivers exactly the same usable
coverage as 2x2 alone -- 0.589 -- for 1.6x the wall clock.** Without tiling OWLv2 does add
something (0.474 -> 0.490), but tiling adds far more and subsumes it: both are ways of getting
more looks at small marks, and once Grounding DINO gets those looks at proper resolution, a
second model at the wrong resolution has nothing left to contribute.

An earlier re-analysis of the ungated experiment-06 runs predicted gdino u OWLv2 worth +11.4
points of usable coverage. That was measured without tiling and does not survive it. Dropped.

## Shipped as opt-in, not as the default

`brand_tiles: "1x1"` (off), `tile_overlap: 0.2`. Off because the cost ratio is much worse than
the `brand_conf` change was -- 4x the detection wall clock for +27% usable, against 1.5x for
+2.6x -- so it is a deliberate choice for an archival or coverage-report pass rather than
something every caller pays. `"2x2"` is the recommended value.

```
--params '{"brand_tiles": "2x2"}'
```

Implemented in `BaseDetector._raw_tiled`, so it wraps `_raw` and every backend gets it, including
`fast`. Tile boxes are translated into frame coordinates before the gate/suppress/crop path, so
nothing downstream changes and crops are always taken from the original frame.

## Reproducing

```bash
CUDA_VISIBLE_DEVICES=3 python3 eval/tools/run_sliced.py --tiles 2x2 \
    --out eval/experiments/13_sliced/runs/S1_2x2
python3 eval/box_gt/score_config.py --runs 13_sliced
```

`run_sliced.py` runs on the host rather than in the container, and its `--tiles 1x1` path is
asserted against the shipped tagger before any tiled number is trusted: 0.479/0.474 against the
tagger's 0.469/0.469, a two-mark difference out of 192 attributable to a numpy NMS standing in
for supervision's. The Grounding DINO decode in it is copied verbatim from `detector.py`.

## Caveat

192 marks over 25 frames. The banner band is 6 marks and the wide band 19, so the aspect-band
story is directionally clear and numerically thin. Cost figures are host-side single-GPU and
exclude embedding, which grows with the box count -- 2x2 roughly doubles the crops, and each is
a SigLIP forward pass and an index row.
