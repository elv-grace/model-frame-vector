# 16_padding — is `crop_padding` 0.06 still right? (yes; no change shipped)

A negative result, and worth keeping as one: the sweep found no change worth making.

## Why it needed asking

`crop_padding` decides how much context is included when a crop is cut. It does **not** move
boxes, so every box-geometry metric in this repo -- coverage, containment, usable -- is
structurally blind to it. It can only be measured by what the resulting vector retrieves.

The main README's existing measurement embeds the 446 ground-truth boxes at four paddings and
asks whether a query built at one padding still retrieves the same object from an index built at
another. That is a *self-consistency* result: it establishes that a padding MISMATCH costs about
8% of top-1, which is why the query side pads to match the index. It never asked which padding
is best in absolute terms against real references.

And the box distribution has changed twice since: the confidence gate came down (`10_config_ab`)
and tiling arrived (`13_sliced`), both admitting smaller and looser boxes than that measurement
saw.

## Method: re-crop and re-embed, do not re-detect

Padding does not affect detection, and detection is the expensive half. So `sweep_padding.py`
reads a run's saved boxes, cuts each crop from the source frame again at each padding, embeds
with the same checkpoint and patch budget the tagger uses, and identifies against the
`12_logo_pool` references. One detection run, N embedding passes -- and the detections are held
identical across the sweep by construction rather than by hoping two runs agree.

Both backends, both titles, over the `11_titles` full-runtime frame samples.

## Result

Brands found, of those present in the title AND in the pool:

| padding | NBA `fast` | NBA `coverage` | Mitchells `fast` | Mitchells `coverage` |
|---|---|---|---|---|
| 0.00 | 7/8 | 8/8 | 3/5 | **4/5, and 1/3 controls** |
| **0.06 (shipped)** | 7/8 | **8/8** | 3/5 | **5/5, 0/3 controls** |
| 0.15 | 7/8 | 8/8 | -- | -- |
| 0.30 | 8/8 * | 8/8 | 3/5 | 5/5 |
| 0.50 | 8/8 * | 8/8 | -- | -- |

\* **not a real gain.** The extra brand is Kia on a *single crop*, and single-crop finds sit at
exactly the level the absent-brand controls reach elsewhere in this evaluation. Read it as noise.

**Zero padding is measurably worse.** On Mitchells `coverage` it loses Amazon and admits a
control false positive -- the only control false positive anywhere in this sweep. A crop cut
exactly to the box, with no surrounding context, is a worse retrieval key than one with a small
margin.

**More padding buys nothing reliable.** Brand recall never improves, and the crop counts are
mixed rather than better. Ratio of identifiable crops at 0.30 against 0.06, NBA `coverage`:

| brand | 0.06 | 0.30 | ratio |
|---|---|---|---|
| Tissot | 261 | 461 | 1.77 |
| NBA | 6,874 | 9,527 | 1.39 |
| Kia | 36 | 45 | 1.25 |
| AT&T | 92 | 110 | 1.20 |
| American Express | 441 | 461 | 1.05 |
| State Farm | 713 | 682 | 0.96 |
| Nike | 47 | 44 | 0.94 |
| Amazon Prime | 7 | 6 | 0.86 |

Three of eight decline and the median ratio is ~1.13, which this sample cannot separate from
nothing. An earlier reading of the `fast` arm alone -- where Tissot went 68 -> 191 and American
Express 36 -> 71 -- suggested padding was worth 2-3x the identifiable crops. Generalised across
both paths and both titles, **it is not**; that was two brands on one arm.

## Conclusion: 0.06 stays

The shipped value sits just above the point where too-tight crops start costing brands, and
below the point where added context dilutes the crop without giving anything back. Nothing is
changed in `general_detection/`.

Two things this does establish:

- **Do not set `crop_padding` to 0.** It looks like the tightest, purest crop and it measurably
  retrieves worse. The mechanism is visible in `mean best cos`, which *rises* as padding falls --
  crops get more similar to the tight pool references on average -- while brand recall falls.
  Average similarity and retrieval quality point in opposite directions here.
- **`mean best cos` is not a quality metric.** It is dominated by the ~57,000 crops that are not
  target brands at all. It declines monotonically with padding across every arm while recall
  holds or improves. It is reported as a drift indicator and should be read as nothing more.

## Reproducing

```bash
CUDA_VISIBLE_DEVICES=3 python3 eval/tools/sweep_padding.py \
    --run eval/experiments/11_titles/nba/B2_step1_conf07 \
    --frames test-files/frames/nba --brands nba
```

## Caveats

Brand recall is over pool-covered brands only -- 8 of 15 for the game, 5 of 14 for the film --
so this sweep can see at most 8 and 5 outcomes per arm, and one brand moving is the resolution
limit. That is precisely why the single-crop Kia change is discounted rather than reported as
7/8 -> 8/8.

The query side must pad to match whatever the index used; that is the older measurement in the
main README and is unaffected by anything here.
