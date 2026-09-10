# 15_temporal — one row per appearance instead of per frame

The tagger is a `FrameModel` and correctly knows nothing about time: one vector per detection per
sampled frame. For a search index that is fine. For a brand coverage report -- "which brands
appeared, and for how long" -- it is the wrong shape, and after steps 1 and 2 it is expensive:

    a courtside hoarding visible for 30s at 2 fps is 60 near-identical crops, 60 near-identical
    768-d vectors and 60 index rows, describing ONE appearance.

Steps 1 and 2 multiplied crops by roughly 7x and then 2x. This is where that is paid back.

## Setup: both backends, both titles, tiled and not

Aggregation needs CONSECUTIVE frames, which the 15s-spaced presence sample in `11_titles`
deliberately does not have -- consecutive samples there are different shots. So these are the
twelve contiguous 45s segments per title at **2 fps**, run four ways: each backend at its own
best tiling from `13_sliced` / `14_yoloe_imgsz`, plus each untiled so the step-3 gain separates
from the step-2 gain.

| arm | NBA tags | Mitchells tags | NBA wall |
|---|---|---|---|
| `coverage` + 2x2 | 83,334 | 63,626 | 2,413s |
| `coverage` plain | 54,026 | 33,364 | 929s |
| `fast` + 3x2 | 29,964 | 17,416 | 914s |
| `fast` plain | 9,311 | 4,676 | 261s |

83,334 vectors from nine minutes of content is itself the argument: extrapolated to the 4h16m
game that is ~2.4M index rows, so per-appearance grouping is a prerequisite for using tiling on
anything long rather than an optimisation.

## Result: 33-71% fewer index rows, no measured recall loss

| arm | detections | tracks | rows at keep=2 | saved |
|---|---|---|---|---|
| NBA `coverage` + 2x2 | 83,334 | 22,358 | 30,395 | 64% |
| NBA `coverage` plain | 54,026 | 11,106 | 15,513 | **71%** |
| NBA `fast` + 3x2 | 29,964 | 9,688 | 12,635 | 58% |
| NBA `fast` plain | 9,311 | 2,109 | 2,721 | **71%** |
| Mitchells `coverage` + 2x2 | 63,626 | 27,772 | 38,588 | 39% |
| Mitchells `coverage` plain | 33,364 | 14,686 | 20,468 | 39% |
| Mitchells `fast` + 3x2 | 17,416 | 8,767 | 11,682 | 33% |
| Mitchells `fast` plain | 4,676 | 2,167 | 2,931 | 37% |

Animation compresses about half as well as sport: faster cuts and a moving camera make shorter
tracks. And **tiling makes tracking less effective** (71% -> 64% on NBA coverage, 71% -> 58% on
NBA fast), because tiled runs emit more marginal, spatially-jittered boxes that associate less
cleanly across frames.

Brand-level recall, identified by image query against the `12_logo_pool` references, is
unchanged by aggregation at `keep=2` on every arm:

| arm | per-frame rows | tracks @keep=2 | controls |
|---|---|---|---|
| NBA `coverage` + 2x2 | 8/8 | **8/8** | 0/3 |
| NBA `coverage` plain | 7/8 | **7/8** | 0/3 |
| NBA `fast` + 3x2 | 7/8 | **7/8** | 0/3 |
| Mitchells `coverage` + 2x2 | 5/5 | **5/5** | 0/3 |
| Mitchells `fast` + 3x2 | 4/5 | **4/5** | 0/3 |

## Why `keep` is 2 and not 1

One representative per track was the original design, and it **loses a brand**: Amazon in
Mitchells, which has two identifiable crops in the whole film, both absorbed into tracks whose
representative retrieved as something else. The rule was "highest detector score", and detector
confidence is not retrievability.

Six single-representative rules were then compared on the `coverage` + 2x2 arms:

| rule | Mitchells | NBA |
|---|---|---|
| all members (no aggregation) | 5/5 | 8/8 |
| highest detector score | **4/5** | 8/8 |
| first member | 5/5 | 8/8 |
| last member | **4/5** | 8/8 |
| middle member | 5/5 | 8/8 |
| biggest box | **4/5** | 8/8 |
| medoid (closest to track mean) | **4/5** | 8/8 |

**Only that one Amazon case discriminates between any of them**, and among six rules it is close
to a coin flip which land on the right frame. The principled candidate -- biggest box, since the
main README measures retrieval falling off with crop pixels -- LOST, as did the medoid. So no
single-representative rule is defensible on this evidence, and picking `first` because it
happened to win on n=1 brand would be reading noise.

Keeping two costs 8-17 points of row reduction and has no measured recall loss on either title
or either backend. `keep=2` ships. Members are spread evenly across the track's span rather than
taken as top-2 by score, because top-2 can be adjacent frames of the same instant and the point
of a second vector is appearance variation.

Averaging the members was rejected without measuring: a mean blends a sharp crop with
motion-blurred ones and lands between them, retrieving worse than the sharp one alone.

## Two outputs, because conflating them is what caused the bug

    appearances.jsonl     one row per track: time range, duration, last box. The coverage report.
    track_vectors.jsonl   `keep` rows per track, with vectors. The search index.

The original single-file design served the report and quietly damaged the index. They are
different questions and they want different shapes.

## Thresholds: only `max_gap` is delicate

Swept one axis at a time on NBA `coverage` + 2x2. Over-merging looks exactly like good
compression in a row-count table, so the probe is whether DISTINCT brands survive as the
thresholds loosen:

| axis | range | tracks | brands found |
|---|---|---|---|
| `iou` | 0.05 → 0.70 | 16,234 → 33,462 | 8/8 throughout |
| `cos` | 0.00 → 0.92 | 21,712 → 31,714 | 8/8 throughout |
| `max_gap` | 0 / 2 / 4 / 8 / **16** | 83,334 / 24,440 / 22,423 / 20,927 / 19,500 | 8/8 … **7/8 at 16** |

`iou` and `cos` are both insensitive across their whole useful range. `max_gap` is the one that
matters: 16 sampled frames of tolerance (8s at 2 fps) starts fusing distinct marks and costs a
brand. 4 ships; 8 is also safe. `max_gap 0` disables tracking entirely and is the useful null.

**The appearance gate is nearly inert.** `cos 0.00` against `0.70` changes 3% of tracks and no
brands, so the stated rationale for it -- that geometry alone would fuse two different logos
occupying the same screen position across a cut -- is *not* supported here. It stays as cheap
insurance and is documented as such rather than as load-bearing. On slower pans or content with
repeated framings it may earn its place; on this material it does not.

## `min_frames` was a mistake, and ships off

It was added as a noise filter, on the theory that single-frame tracks are distractors getting
lucky. Under image-query identification that theory is wrong: the absent-brand controls score
**0 at every setting**, so there is no noise to remove, and raising it only costs real brands --
8/8 → 7/8 on NBA tiled coverage, 6/8 → 5/8 on NBA plain fast, 5/5 → 4/5 on Mitchells.

The noise it was aimed at was a **text**-query artifact, where controls did fire 2/6
(`11_titles`). Image queries had already eliminated it in `12_logo_pool`. Kept as a flag,
defaulted to 1, documented as measured-harmful.

## Two silent bugs found here, both worth knowing

**`max_gap` counted source frame indices, not sampled frames.** The tagger stamps
`frame_info.frame_idx` from the source video, so at 2 fps against 25 fps material consecutive
samples are ~12 apart. Every track closed after one frame and the tool reported "0% fewer
vectors" with one track per detection -- a believable null result rather than a crash.

**The brand-attribution matmul ran on CPU.** 83k crops against 69k references at 768-d is ~9
TFLOP per call and a dozen calls per title. It now runs on the GPU; the first attempt was going
to spend half an hour multiplying matrices.

## Reproducing

```bash
# contiguous segments, not the 15s presence sample
./eval/box_gt/sample_title.sh test-files/val/basketball_allstars.mp4 nba 12 45
bash eval/experiments/15_temporal/run.sh

python3 eval/tools/aggregate_temporal.py --run eval/experiments/15_temporal/nba/C_cov_tiled --fps 2
CUDA_VISIBLE_DEVICES=3 python3 eval/tools/score_temporal.py \
    --experiment eval/experiments/15_temporal/nba --brands nba
```

## Caveats

Brand-level recall here is over the brands that are BOTH present in the title and in the
2,960-brand pool -- 8 of 15 for the game, 5 of 14 for the film. The rest cannot be retrieved by
any method and say nothing about aggregation. The `keep=1` failure rests on a single brand, so
`keep=2` is a conservative response to thin evidence rather than a tuned optimum; what it is
conservative about is losing a brand, which is the failure a coverage report cannot absorb.

Duration figures assume the sampled frame rate is the truth. A track of 12 frames at 2 fps is
reported as 6s, which is right only if the mark was continuously visible; tracks tolerate a gap
of 4, so a reported duration can span a genuine 2s absence.
