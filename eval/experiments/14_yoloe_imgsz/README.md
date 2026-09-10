# 14_yoloe_imgsz — can resolution or slicing rescue the `fast` path?

Experiment 06 established that YOLOE, unlike the DETR-family backends, *gains* from resolution:
brand AP 0.062 -> 0.133 going 640 -> 1280. That sweep stopped at 1280, which became the shipped
`fast` default.

Separately, 10_config_ab found that YOLOE's limit is its **proposal set, not its score**:
ungating it completely, conf 0.007 -> 0, moves coverage only 0.203 -> 0.255. A COCO-trained
backbone proposes class-agnostic regions and MobileCLIP text embeddings re-score them, so a
region the backbone never proposed cannot be recovered by any threshold or prompt.

Those leave one question open: is that proposal set resolution-limited, or architecturally
limited? Two levers are available -- raise `imgsz`, or slice the frame (13_sliced) -- and they
attack it differently. Raising imgsz upscales the whole frame into a bigger tensor; slicing
feeds the model normal-sized tiles in which the marks are simply larger.

## Slicing beats resolution, roughly two to one

| config | coverage | contain | usable | boxes/frame |
|---|---|---|---|---|
| 1280 (shipped `fast`) | 0.214 | 0.214 | 0.125 | 3.5 |
| 1600 | 0.255 | 0.266 | 0.167 | 5.7 |
| 1920 | 0.276 | 0.276 | 0.167 | 6.2 |
| 2560 | 0.302 | 0.276 | 0.177 | 8.3 |
| 1280 + 2x2 tiles | 0.333 | 0.323 | 0.224 | 9.6 |
| **1280 + 3x2 tiles** | **0.385** | **0.417** | **0.292** | 12.8 |
| 1280 + 4x3 tiles | 0.380 | 0.380 | 0.234 | 13.4 |
| 2560 + 2x2 tiles | 0.396 | 0.427 | 0.276 | 16.2 |
| *`coverage` backend, no tiling* | *0.469* | *0.641* | *0.469* | *21.4* |
| *`coverage` backend, 2x2 tiles* | *0.599* | *0.797* | *0.594* | *44.8* |

So the proposal set **is** resolution-limited, and slicing exploits that far better than
upscaling does. `3x2` at the stock 1280 more than doubles usable coverage, 0.125 -> 0.292, and
beats 2560 (0.177) while feeding the model smaller tensors. Stacking both is not additive:
2560 + 2x2 lands at 0.276, below plain 3x2.

**The optimal tiling differs per backend**, which is worth knowing before copying a value across:

| | best tiling | usable at best | usable one step finer |
|---|---|---|---|
| `coverage` (gdino @800) | 2x2 | 0.589 | 0.552 (3x2) |
| `fast` (yoloe26 @1280) | 3x2 | 0.292 | 0.234 (4x3) |

Both turn over, for the same reason -- tile seams cut wide marks into fragments -- but YOLOE
tolerates finer tiles because it was not finding wide marks anyway, so it has less to lose and
more to gain from magnification.

## What neither lever fixes is the box shape

| config | <1.5 compact | 1.5-3 | 3-5 wide | >=5 banner |
|---|---|---|---|---|
| 1280 | 0.28 | 0.15 | 0.11 | **0.00** |
| 1600 | 0.26 | 0.29 | 0.21 | **0.00** |
| 1920 | 0.29 | 0.31 | 0.16 | **0.00** |
| 2560 | 0.31 | 0.33 | 0.21 | 0.17 |
| 1280 + 3x2 | 0.40 | 0.40 | 0.32 | 0.17 |

Banner-shaped marks reach one of six at best. This is the compact-box regression bias, not a
resolution problem: YOLOE's emitted brand boxes have median aspect 1.41 -- the COCO prior intact
-- and on the 261x29 px hoarding in `NBA33min__000` it emits a roughly square box four times too
large, IoU 0.24 with the mark fully inside it. More pixels do not make the head predict a 9:1
box, and neither do smaller tiles.

That failure is visible in the columns: `contain` tracks `coverage` while `usable` lags both.
The marks are inside a box; the box is not about the mark.

### But `usable` is partly unfair to tiling here, and the correction is worth having

`usable` demands a box contain **90%** of the mark. A tile that cleanly boxes half a hoarding
scores zero for it, even though that crop may well retrieve. Six of the wide marks are 11–39% of
frame width against a `3x2` tile's ~40% (with 0.2 overlap), so some are genuinely cut and some
are not. Relaxing the containment fraction separates the two effects — wide marks only, n=25:

| config | ≥90% of the mark | ≥50% | ≥25% |
|---|---|---|---|
| 1280 | **0.000** | 0.120 | 0.120 |
| 1280 + 2x2 | 0.000 | 0.280 | 0.360 |
| 1280 + 3x2 | 0.120 | **0.320** | 0.360 |
| 1280 + 4x3 | 0.040 | 0.280 | 0.440 |
| *`coverage`, no tiling* | *0.440* | *0.560* | *0.560* |
| *`coverage` + 2x2* | *0.600* | *0.800* | *0.800* |

**Tiling does help `fast` on wide marks** — crediting half-marks nearly triples `3x2`, 0.120 →
0.320 — and an earlier version of this file's "banner band 0.00 at every setting" understated
that by reading only the strict column.

It does not change the conclusion. Relaxing the threshold lifts every row, and the ordering is
identical at all three: `coverage` **untiled** beats `fast` **tiled** on exactly these marks at
every threshold. Over all 192 marks at ≥50%, `fast` + `3x2` reaches 0.479 against plain
`coverage`'s 0.516 — still behind, at 1.3× the wall clock.

Note the 4x3 row, which is the segmentation effect on its own: it beats 3x2 at ≥25% (0.440) and
loses to it at ≥90% (0.040). Finer tiles find more fragments of wide marks and fewer whole ones.

**Open question, not measured here:** whether a crop of half a wordmark actually retrieves
against a reference pool. That would need image-query identification over tiled title runs, and
it bounds how much credit the relaxed columns deserve. Until it is run, ≥90% is the number to
quote and ≥50% is the honest caveat beside it.

Reproduce with `python3 eval/box_gt/score_config.py --runs 14_yoloe_imgsz --containment 0.5`.

## Conclusion

`fast` is not "the same but cheaper", and no threshold, prompt, resolution or tiling setting
makes it so -- at its best it reaches usable 0.292 against the `coverage` backend's 0.469
untiled and 0.594 tiled. It remains the right choice when index size or throughput is the
binding constraint and partial brand recall is acceptable, and the wrong one whenever hoardings
matter.

Within that role it is substantially improvable, and the recommendation is slicing rather than
resolution:

```
--params '{"brand_detector": "fast", "brand_tiles": "3x2"}'
```

The `fast` default `imgsz` is left at 1280. The gain from raising it is real but smaller than
slicing's, it costs a quadratically larger tensor, and the wall clock here is dominated by
container start at n=25 so the true throughput cost is unmeasured -- and `fast` exists precisely
for callers paying for throughput.

## Reproducing

```bash
bash eval/experiments/14_yoloe_imgsz/run.sh
python3 eval/box_gt/score_config.py --runs 14_yoloe_imgsz
```

## Caveat

192 marks over 25 frames; the banner band is 6 marks and the wide band 19. Cost columns are
wall clock at n=25 and are load-dominated -- use them for ordering, not for capacity planning.
