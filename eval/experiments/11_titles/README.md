# 11_titles — the step-1 config on two whole titles

`10_config_ab` measured a config against box ground truth: 25 frames, 192 labelled marks. This
runs the same two configs over two complete titles and asks the question the product actually
asks — *which brands can be found at all* — using brand lists read off screen by a human rather
than labelled boxes.

| title | runtime | frames sampled | span |
|---|---|---|---|
| all-star game | 4h16m | 983 | 307s – 15,037s (98%) |
| The Mitchells vs. the Machines | 2h06m | 483 | 150s – 7,380s (98%) |

## Sampling: one frame every 15s, not a few long segments

The first pass used twelve 45s segments per title, and that was the wrong instrument. Twelve
segments of a four-hour game is 3.5% of it, and the 45 frames inside one segment share a camera,
a possession and a set of hoardings — nearer one look than forty-five. Brands are not evenly
distributed (a courtside board rotates, a sponsor owns one quarter), so a clustered sample
under-counts them.

One frame every 15s costs the same GPU time and sees the whole runtime. `sample_frames.sh` does
this; `sample_title.sh` still cuts segments, because dwell time needs continuity and presence
does not.

## Detection: the config change does what 10_config_ab predicted

| | crops | per frame |
|---|---|---|
| `fast`, six prompts, yoloe26 @1280 conf 0.007 cap 30 | 8,843 | 9.0 |
| `coverage`, four prompts, gdino @800 conf 0.07 cap 100 | 57,914 | 58.9 |

6.5x the crops from **four** prompts against six — the increase is backend, gate and cap, not
vocabulary. The per-prompt split is worth keeping:

| prompt | `fast` | `coverage` |
|---|---|---|
| `letter logo` | 358 | **27,736** |
| `brand` | 151 | 17,535 |
| `logo` | 7,842 | 11,401 |
| `car logo` | 331 | 1,242 |
| `label` | 143 | — |
| `emblem` | 18 | — |

`letter logo` is 48% of what Grounding DINO returns and 4% of what YOLOE returns, which is the
same asymmetry the box-GT marginal-coverage table showed (+0.068 against +0.000). Dropping it
would have cost gdino half its crops.

## Identification: where it stops working

Each crop is assigned to a brand by argmax over the target names plus ~2,440 real distractor
brands, with six brands known to be absent carried through as a live noise floor.

| | all-star game | Mitchells |
|---|---|---|
| `fast` | 1 / 15 | 3 / 14 |
| `coverage` @0.07 | **4 / 15** | **5 / 14** |
| controls that collected crops (`coverage`) | 2 / 6 (2 crops) | 1 / 6 (1 crop) |

Single-crop finds sit at exactly the control noise level and should not be counted. The
defensible finds are Tissot (27 crops), Olympics (14) and Gatorade (5) in the game, and Sony
Handycam (44) and Sony headphones (5) in the film — **three and two**.

So: detection improved by every measure and end-to-end recall went from 1 solid brand to 3 out
of 15. **The binding constraint is no longer detection.** That is consistent with what the repo
already documents without connecting it — a detected crop retrieves against a clean pool at r@1
0.60, and that is the *image*-query number, while this is text against 2,454 classes where a
perfect match sits near cosine 0.05–0.3 against 0.5–0.9 for image.

## Two measurement traps found here

**A probability floor does not work at this scale.** The first version of `score_brands.py`
embedded the brand name and kept crops above a calibrated probability. Against 27,000 crops it
reported 15/15 brands found at p≈0.9 — and `"zzzzz nonexistent brand"` scored 0.967, above every
real brand, while Lufthansa scored 0.930 in a basketball game. The maximum over 27k comparisons
is extreme for any query. Hence argmax over a large real vocabulary, plus controls.

**Caption templates hurt.** Measured under that protocol:

| template | targets found | control false positives |
|---|---|---|
| `"{}"` | 3 / 15 | **0 / 6** (0 crops) |
| `"a photo of the {} logo"` | 3 / 15 | 2 / 6 (40 crops) |
| `"the {} logo"` | 1 / 15 | 2 / 6 (10 crops) |

The templates are the standard CLIP zero-shot advice and they are wrong here: they pull every
query toward a generic "logo photo" direction, which does not separate brands and does let
absent brands collect crops — Aeroflot alone takes 38. The bare name ships.

**Vocabulary collisions matter too.** The distractor pool contains exact duplicates of NBA, Kia,
Nike, Tissot, AT&T, American Express, Twitter, Facebook and Amazon, plus vote-splitters like a
bare `sony` against the target `Sony Handycam`. An exact duplicate makes the argmax a coin flip;
a near-duplicate steals the crop. Both score real hits as misses, and both are removed by
normalised-substring matching before scoring.

## Reproducing

```bash
./eval/box_gt/sample_frames.sh test-files/val/basketball_allstars.mp4 nba 15
./eval/box_gt/run_title_ab.sh nba frames
python3 eval/box_gt/score_brands.py --compare eval/experiments/11_titles/nba --brands nba
```

Runs here keep their `vector` fields, unlike `10_config_ab` — the question is retrieval, and that
needs the embeddings. They are large (1.4 GB) and gitignored.

## Caveats

The brand lists are what a human noticed, not an exhaustive annotation, so a brand absent from
the list but present on screen is invisible to this. And a "not found" here confounds three
causes — never detected, detected but not identified, or no usable reference in the pool. Those
are separated in `12_logo_pool`, and the split matters: five of the fifteen NBA targets have no
reference in the 2,960-brand pool at all.
