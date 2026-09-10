# 17_ocr — read the words on the screen (shipped as `ocr`, off by default)

The vector index answers "what does this look like". A brand coverage report keeps asking "is
STATE FARM on screen", and that question has an exact answer that no cosine can give.

## Why the vector text channel is weak, measured properly

The premise this step started from was wrong and the measurement corrected it. The rank of the
first **correct** crop for each brand — correct meaning its nearest reference in the
`12_logo_pool` pool is that brand at cosine ≥ 0.55, which was verified by eye there:

| brand | mark | `fast` rank | `coverage` rank | `fast` %ile | `coverage` %ile |
|---|---|---|---|---|---|
| NBA | letters | **3** | 134 | 0.03% | 0.23% |
| Tissot | wordmark | 5 | **4** | 0.06% | 0.01% |
| AT&T | letters + globe | 12 | **13** | 0.14% | 0.02% |
| Nike | swoosh, no text | **15** | 1,732 | 0.17% | 2.99% |
| State Farm | wordmark | **23** | 881 | 0.26% | 1.52% |
| American Express | dense box, tiny type | 5,674 | **11,240** | 64% | 19% |
| Amazon Prime | wordmark, small | 5,269 | 31,496 | 60% | 54% |
| Kia | stylised wordmark | *no crops* | 15,489 | — | 27% |

Two things, and neither is "the coverage backend broke text search".

**Wordmark-shaped.** Brands sort by how legible their lettering is, in the *same order on both
backends*. SigLIP's text tower is reading letterforms through the image encoder; given a shape
(Nike) or letters too small to resolve (American Express), it has nothing to match on and
guesses. The modality gap leaves no headroom either — a correct text→image match sits near cosine
0.05–0.3 against 0.5–0.9 for image→image.

**Rank-fragile.** Rank is a position in a list whose length the *detector* sets. State Farm moves
23 → 881 with the same query and the same embedder, purely because `coverage` emits 6.5× the
crops; its percentile moves only 0.26% → 1.52%. So **buying detection recall directly degrades
text-query usability**, and no parameter reconciles that. Someone typing "State Farm" and scanning
50 results finds it on `fast` and not on `coverage`.

This also revises `11_titles`, which reported `coverage` beating `fast` at text identification
4/15 to 1/15. That is a *different* question — argmax against a 2,454-class vocabulary, where
having more crops of a brand is an advantage. Both results are right; neither is the ranking one.

A recognised string has neither problem. It matches rather than ranks, so it does not decay as
the crop count grows, and legibility is a text recogniser's job rather than a 768-d bottleneck's.

## Result: strong on broadcast, weak on animation, and no false positives anywhere

easyocr (CRAFT text detection + a CRNN recogniser, Apache-2.0), one pass per frame, over the same
full-runtime frame samples as `11_titles`.

| | all-star game (983 fr) | Mitchells (483 fr) |
|---|---|---|
| brands found by name on screen | **10 / 15** | 4 / 14 |
| of the brands the reference pool cannot reach at all | **5 of 7** | 2 of 9 |
| known-absent control brands collecting anything | **0 / 6** | **0 / 6** |
| ms per frame @1080p | 398 | 292 |

The control row is the one that makes the rest believable. The vector-text channel put 2 of 6
absent brands on the board; an exact string match puts none, because there is no argmax for an
absent brand to win by default.

And the second row is the point. `12_logo_pool` found that **16 of 29 named brands have no
reference image at all** and named pool coverage the binding constraint. OCR needs no reference:
NBC, Peacock, DraftKings, CeraVe and Olympics are all found by their own name on screen. Five
brands that were previously unreachable by *any* method.

Screen-time, which is what a coverage report sells, lines up with the image-query numbers where
both channels work — State Farm in 187 frames here against 186 by image query — and diverges
where they measure different things: NBA in 113 frames by name against 848 by image, because the
NBA silhouette usually carries no letters.

### What it does not find, and why

| brand | why |
|---|---|
| Nike | swoosh. No text to read; unreachable by this channel by construction. |
| Kia | the current mark fuses K-I-A into one connected geometric form. |
| Gatorade | bolt; the wordmark is rarely legible at broadcast scale. |
| Michelob Ultra | the board reads ULTRA and COURTSIDE, never MICHELOB. |
| Amazon Prime | the wordmark is present but small; only PRIMETIME (an NBC show) is read. |

## Blockiness is survivable; broken strokes are not

Asked whether OCR can box a wordmark whose letters are blocky, tested on 20 reference images per
brand from `/ml/pools/logo_pool/2960brands`:

| wordmark | letterforms | refs read |
|---|---|---|
| **IBM** | eight horizontal stripes cutting every stroke | **1 / 20** |
| FedEx | standard type | 8 / 20 |
| KIA *(pool holds the pre-2021 oval)* | standard type | 11 / 20 |
| Sony | standard type | 11 / 20 |

IBM is the clean case: the stripes sever the glyph strokes and both stages fail. That is the same
mechanism behind Kia's 0/983 in the broadcast, while the pool's older standard-letterform Kia
oval reads 11/20. **OCR reads glyphs, not logos.** Styling that keeps strokes intact and
separated — blocky, slab, condensed, outlined — mostly survives; styling that cuts them (IBM) or
fuses them (current Kia) does not. Pixel size matters more than style.

## The credits trap

Attributing the film's reads by position in the runtime, using 6450 s as the credits boundary:

| | t | read |
|---|---|---|
| **in-film** | 690, 705 | `'HOUSE` — the Waffle House sign, partially |
| **in-film** | 900 | `People`, conf 1.00 |
| **in-film** | 3735 | `Sees`, `Sees EANDIES` |
| credits | 6540 | `MOCK WiRED COVER USED With PERMISSION OF CONdE NASt` |
| credits | 6540 | `people Logo ANd COVER desiGN` |
| credits | 6585 | `SONY` |

Two consequences worth carrying downstream.

**The WIRED logo is never read** — not at either magnification, not in frame mode, not in crop
mode. The only hit is the credits line naming it. It is recorded here as *unattributable* rather
than as a miss: at one frame per 15 s a brief magazine shot may not have been sampled at all.

**End credits are a brand-name goldmine and a coverage-report trap.** The `Sony` find is the
studio card, not the headphones or the Handycam the brand list means. A report that counts credits
would bill Sony for screen time it never bought as a placement. No filter is shipped for this —
string length does not separate them, because `ALL-STAR INTERMISSION PRESENTED BY AMERICAN
EXPRESS` is a legitimate sponsor read of the same shape — so it belongs in the report layer,
where a title's credits range is known.

## OCR boxes are also proposals, and that is where `fast` gains

A text detector emits **wide** boxes by construction, which is exactly the shape
`14_yoloe_imgsz` proved YOLOE's COCO-prior regression head cannot produce at any resolution or
tiling. Scored against box ground truth (192 marks, 25 frames):

| source | coverage | usable | boxes/frame | `>=5` banner band |
|---|---|---|---|---|
| OCR boxes alone | 0.229 | 0.083 | 3.9 | 0.50 |
| `fast` | 0.219 | 0.125 | 3.5 | **0.00** |
| **`fast` + OCR** | **0.396** | **0.198** | 7.4 | **0.50** |
| `fast` + 3x2 tiles | 0.385 | 0.292 | 12.8 | 0.17 |
| `coverage` | 0.469 | 0.469 | 21.4 | 1.00 |
| `coverage` + OCR | 0.516 | 0.484 | 25.4 | 1.00 |
| `coverage` + 2x2 | 0.599 | 0.589 | 42.9 | 1.00 |
| `coverage` + 2x2 + OCR | 0.604 | 0.594 | 45.3 | 1.00 |

**+58% usable on `fast`, +3% on `coverage`, +1% once `coverage` is tiled.** Grounding DINO already
boxes banners at 1.00, so there is nothing left there to add. `propose` is a `fast`-path feature.

### Two confidence gates, not one

The boxes want a different threshold from the strings, and this was a real bug before it was a
finding. easyocr detects and recognises in separate stages, and CRAFT routinely localises a
wordmark the CRNN then garbles — that box is still worth cropping, because the crop retrieves by
*image* against the pool whatever the string said.

| proposal box gate | `fast` usable | `coverage` usable |
|---|---|---|
| ≥ 0.2, same as the strings | 0.167 | 0.484 |
| **≥ 0.0, ungated** | **0.198** | 0.484 |

So `ocr_conf` 0.2 gates indexed strings (and keeps controls at 0/6) while `ocr_box_conf` 0.0
gates proposal boxes.

## Per crop is worse and much more expensive

The obvious alternative — OCR each crop, where a 22 px mark fills the frame — loses on both axes:

| mode | brands, game | brands, film | ms/frame, `coverage` | ms/frame, `fast` |
|---|---|---|---|---|
| per frame *(path-independent)* | **10 / 15** | **4 / 14** | **398** | **398** |
| per crop, `coverage` boxes | 8 / 15 | 4 / 14 | 1,072 | — |
| per crop, `fast` boxes | 9 / 15 | 5 / 14 | — | 148 |

58.9 crops per frame on the default path at 18 ms each is 1,072 ms — comparable to the *entire*
current pipeline wall clock — and it finds less, because easyocr's detector works better on a
whole frame than on 59 disconnected fragments of it. Per-frame cost is also identical for both
backends, which per-crop is not: on `fast`, at 9.0 crops/frame, crop mode is *cheaper* than frame
mode (148 ms against 398) and finds one brand fewer on the game and one more on the film. That is
the only cell where crop mode is competitive, and it is a wash.

Crop mode does find things frame mode misses (`SEES`, `ULTRA`, `ATT` exactly, and the WIRED
credits line), so the two are complementary rather than ordered — but not at 2.7× the cost of the
better one. Rejected.

*The `coverage` crop arms are capped at the 6,000 highest-scoring crops of 57,914 and 15,759, so
their recall is a lower bound. The cost figure is not affected; it is per-crop measured and
multiplied out.*

## Magnification: 1.0

| | ms/frame | brands, game | box-GT `fast` usable |
|---|---|---|---|
| mag 1.0 | 398 | 10 / 15 | 0.198 |
| mag 2.0 | 642 | 11 / 15 | 0.198 |

The extra brand at 2.0 is Gatorade on a single frame reading `GATRADE`, which sits at exactly the
level this evaluation discounts everywhere else. 62% more cost for one single-frame read.
`ocr_mag: 2.0` is the archival rung, the same way `brand_conf: 0.05` is.

## Shipped: `ocr`, default `"off"`

```
--params '{"ocr": "text"}'                              # strings on existing detections
--params '{"brand_detector": "fast", "ocr": "propose"}' # + text regions as brand proposals
```

`"text"` adds no crops, no index rows and no change to detection — purely additive to what a run
already emits. `"propose"` additionally turns un-boxed text regions into brand detections.

**Off by default** for two reasons. It costs a fixed ~400 ms/frame — a third again on top of the
`coverage` path's end-to-end wall clock — and unlike every other knob here that cost does not
scale with how much it finds. And the value is strongly content-dependent: 10 of 15 on a
broadcast against 3 of 14 on animation.

This is also the first step in this series to **add a dependency**. easyocr is Apache-2.0, so it
adds no licence obligation beyond ultralytics', and its weights download on first use into the
same mounted cache as the others rather than being baked into the image.

## How matching is decided, because that is where you can fool yourself

Strings are normalised to uppercase alphanumerics on both sides. A brand's key is its name with a
fixed list of generic qualifiers removed (`QUALIFIERS` in `score_ocr.py`) — "Wired magazine" →
WIRED, "Sony headphones" → SONY — and nothing else. No per-brand tuning, and the controls go
through the identical function. Four rules are reported:

- `strict` — the key must appear; 5+ characters may match as a substring or fuzzily at ratio 0.85
  (so `STATEFAM` counts), shorter keys must equal the whole string (`ATT` is real, but as a
  substring it also matches `TRATTORIA` and `MATTHEW`).
- `tokens` — every token must appear somewhere in the *same frame*. This exists because of a real
  defect in `strict`: the board reads AMERICAN EXPRESS as two regions, so `AMERICANEXPRESS` is
  never emitted and a plainly legible brand scored 1 frame in 983.
- **`union` — strict OR tokens. The headline.** The two fail differently and neither dominates.
- `lenient` — any single token. Finds Michelob Ultra through ULTRA, which is right, and Amazon
  Prime through PRIMETIME, which is a television show. Shown to price the loose rule, not used.

A consequence worth stating: "Sony headphones" and "Sony Handycam" both reduce to SONY, so this
channel cannot tell them apart — the same limitation the reference pool has, for the same reason.

## Reproducing

```bash
pip install --target ~/.cache/ocr-deps easyocr scikit-image python-bidi shapely pyclipper \
    ninja lazy_loader imageio tifffile networkx

export PYTHONPATH=~/.cache/ocr-deps
CUDA_VISIBLE_DEVICES=3 python3 eval/tools/run_ocr.py \
    --frames test-files/frames/nba --out eval/experiments/17_ocr/nba/frame_mag1
python3 eval/tools/score_ocr.py --ocr eval/experiments/17_ocr/nba/frame_mag1 --brands nba

CUDA_VISIBLE_DEVICES=3 python3 eval/tools/run_ocr.py \
    --frames eval/frameset/frames --out eval/experiments/17_ocr/box_gt/frame_mag1
python3 eval/tools/score_ocr.py --mode boxes --conf 0.0 \
    --ocr eval/experiments/17_ocr/box_gt/frame_mag1 \
    --run eval/experiments/10_config_ab/runs/A_shipped_fast
```

## Caveats

- 15 and 14 brands per title, so one brand is 6–7 points. The control row carries most of the
  confidence here, not the sample size.
- The banner aspect band is 6 marks, so `fast`'s 0.00 → 0.50 is three marks.
- English only. `easyocr.Reader(["en"])` — a title with non-Latin signage needs the language list
  extended, which costs another recogniser.
- Reads are not verified against pixel ground truth. A read is counted when the string matches;
  no one confirmed by eye that the frame contains the mark. The credits attribution above is the
  one place that check was done, and it found two spurious brands out of five.
