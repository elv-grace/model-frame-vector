# model-frame-vector — SigLIP 2 frame and crop embedder

Turns video and images into search vectors with [SigLIP 2](https://huggingface.co/docs/transformers/model_doc/siglip2)
NaFlex (`google/siglip2-base-patch16-naflex`, **768-d**, L2-normalized). One container, two modes,
and `detect_target` is the switch:

| `--params` | what it embeds | tags per frame | detector weights loaded |
|---|---|---|---|
| `{}` — **the default** | the whole frame | 1 | none |
| `{"detect_target": ["brand", "person"]}` | each detected crop | one per detection | YOLO11 + Grounding DINO |

**Never both in one run.** A frame is *downscaled* to the patch budget and a crop is *upscaled* to
it, so their vectors sit in measurably different regions of the space — one run is one index.

Implements `common_ml`'s `FrameModel`, so frame extraction (ffmpeg/PyAV), image-vs-video dispatch
and tag serialization come from the tagger runtime. Model code is in [`general_detection/`](general_detection/).

> 📈 **The measurements and the decisions behind every default** live in
> [**Courtside Blindness**](https://claude.ai/code/artifact/97c7e4ad-5c65-4aa5-820e-cd7c2e88fdfe)
> (brand recall: thresholds, tiling, temporal aggregation, OCR) and
> [**Brand and Person Detector Sweep**](https://claude.ai/code/artifact/8a432b7c-8601-4d7d-bea9-d527e67141b2)
> (which detector, which prompts). Raw runs are in [`eval/`](eval/README.md).

## Run it

```bash
make build                  # or ./build.sh — no weights needed at build time
./test.sh                   # smoke test on test-files/ at fps=1; ./test.sh 5 for fps=5
```

Paths go in on **stdin**, one per line; `--output-path` is required; everything else is a JSON
`--params` object:

```bash
echo /elv/test/clip.mp4 | podman run --rm -i \
    --volume="$(pwd)/test-files:/elv/test:ro" \
    --volume="$(pwd)/test-output:/elv/tags:U" \
    --volume=detection_cache:/root/.cache \
    --device nvidia.com/gpu=0 \
    model-frame-vector \
    --output-path /elv/tags/out.jsonl \
    --params '{"fps": 1}'
```

The `/root/.cache` mount is **required** — see [Deployment](#deployment).

### The configurations worth running

```bash
--params '{}' # 1. frame vectors. The default.
--params '{"detect_target": ["brand", "person"]}' # 2. crop vectors, routine indexing
--params '{"detect_target": ["person"]}' # 3. person only; the brand model never loads
--params '{"detect_target": ["brand"], "brand_tiles": "2x2",
           "max_detections": 200, "ocr": "text"}' # 4. archival brand pass, ~2x the cost of 2
--params '{"detect_target": ["brand"], "brand_detector": "fast"}' # 5. hard ceiling on index size / GPU time
--params '{"detect_target": ["brand"], "brand_detector": "fast", "brand_tiles": "3x2", "ocr": "propose"}' # 6. improved 5's recall
```

(4) raises the cap because `2x2` roughly doubles the boxes and a dense broadcast frame already
emits ~59 at `1x1`; truncation is by score, so the cap would remove exactly the faint hoarding
marks the config exists to catch. (5) emits 3.5 boxes/frame against 21.4 and finds proportionally
fewer marks — it is a **cap, not an optimisation**; spending wall clock to improve it in place is
always worse than spending the same wall clock on the default backend.

## Output

Every tag carries a `vector` and a normalized `box`. `common_ml` then attaches
`start_time`/`end_time` (ms), `source_media` and `frame_info.{frame_idx, box}`. Vector tags are
**never run-length merged**, which is what keeps the box available for the EVIE overlay.

**Frame mode** — one tag, empty `tag`, full-frame box:

```json
{"tag": "", "vector": [...768], "box": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
 "additional_info": {"box": {...}, "upscale": 0.18, "embedder": "google/siglip2-base-patch16-naflex",
                     "revision": "b53b807d…", "dim": 768, "normalize": true,
                     "max_num_patches": 256, "query_modes": ["text", "image"]}}
```

**Detection mode** — one tag per detection, `tag` is the **parent term** (`brand`, `person`, or
whatever was asked for) and `box` is the **un-padded** detection box:

```json
{"tag": "brand", "vector": [...768], "box": {"x1": 0.41, "y1": 0.72, "x2": 0.53, "y2": 0.75},
 "additional_info": {"box": {...}, "prompt": "letter logo", "score": 0.19,
                     "detector": "grounding-dino-base", "crop_padding": 0.06, "upscale": 5.2,
                     "text": ["STATE FARM"], "embedder": "…", "dim": 768, …}}
```

### `additional_info`

| field | why it is there |
|---|---|
| `box` | the same box as `frame_info.box`, repeated because `additional_info` is the **only** field a vectorstore search row returns |
| `embedder`, `revision`, `dim`, `normalize`, `max_num_patches` | the embedding recipe. A query that does not match on all of these is not comparable to what is indexed |
| `query_modes` | which query modalities this space accepts (`text`, `image`) |
| `upscale` | the NaFlex scale actually applied, so the heavily-interpolated tail can be filtered downstream without re-tagging |
| `prompt` | *detection only* — which phrasing fired (`"ocr"` for a text-region proposal) |
| `score` | *detection only* — detector confidence. For an `"ocr"` proposal this is *recognition* confidence, **not on the same scale** |
| `detector` | *detection only* — which backend found it. Per tag, not per run: several are in play |
| `crop_padding` | *detection only* — **changes the vector.** Mixing paddings in one index costs retrievals silently |
| `text` | *detection only, `ocr` on* — words read inside this box. **Absent rather than empty**, so absence does not distinguish "OCR off" from "no text here" |

### Seeing the boxes in the video editor

`output_tags: true` emits a **second, vector-less** tag beside each detection's vector tag — same
label, same box — which lands as an ordinary tag track in EVIE alongside the overlaid boxes. No
extra inference, but the tag count more than doubles (the vector-less twins also get run-length
merged into spans, which the vector tags are not).

> The twins share their parent's `additional_info` dict, so the only difference is the presence of
> `vector`. Drop vector-less records *before* stripping vectors to store a run, or every detection
> is counted twice.

## Runtime parameters

Injected per request as a JSON `--params` object. Defaults and their provenance are documented
inline in [`general_detection/config.py`](general_detection/config.py). Frame sampling rate (`fps`)
is handled generically by the tagger runtime.

### Embedding — applies in both modes

| param | default | meaning |
|---|---|---|
| `max_num_patches` | `256` | NaFlex resolution budget (16×16 patches). `576`/`1024` buy detail at attention cost quadratic in the budget. |
| `normalize` | `true` | L2-normalize so cosine == dot product. |
| `embed_batch_size` | `32` | Crops per forward pass. |
| `max_upscale` | `null` | Cap on NaFlex upscaling. **Do not set to 1.0** — it makes crop *size* a dominant nuisance axis in the embedding. ~4.0 is the sane version. |

### Detection — only when `detect_target` is set

| param | default | meaning |
|---|---|---|
| `detect_target` | `null` | **The mode switch.** A known parent (`brand`, `person`) expands to its phrasings; anything else becomes its own parent, so `["car"]` is valid. Routed per term: `person` → YOLO11 @640, everything else → the open-vocabulary backend. A detector with nothing routed to it never loads. |
| `brand_detector` | `"coverage"` | `"coverage"` = Grounding DINO @800 conf 0.07; `"fast"` = YOLOE-26 @1280 conf 0.007 — a third of the cost and a quarter of the marks. |
| `brand_tiles` | `"1x1"` (off) | Sliced inference, `"COLSxROWS"`. Runs the detector on the whole frame *and* on each overlapping tile, so a mark that would arrive downscaled arrives at proper resolution. **`"2x2"` is the recommended value** (`"3x2"` for `fast`); finer is worse, because seams cut wide marks into fragments. |
| `tile_overlap` | `0.2` | Fraction of a tile's own size added on each side, so neighbours overlap. Without it a tile seam is a blind spot as wide as the mark sitting on it. |
| `max_detections` | `100` | Cap per frame, highest score first. **The primary cost knob** — each survivor is one SigLIP forward pass and one index row. |
| `class_prompts` | `{}` | Explicit `{parent: [phrasings]}`. Also turns detection on; `detect_target` wins if both are set. |
| `class_conf` | `{}` | Per-parent confidence override. Empty because **each backend carries its own measured threshold**. Passing this **replaces** the dict, so list every class you want gated. |
| `brand_imgsz`, `brand_conf`, `person_imgsz`, `person_conf` | `null` | Override a backend's measured defaults. |
| `iou` / `nms_iou` / `cross_class_nms_iou` | `0.7` / `0.6` / `1.0` | The three dedupe stages: ultralytics' internal per-phrasing NMS, then synonym-group NMS across one parent's phrasings, then cross-parent NMS (**disabled, and it should stay disabled** — a logo on a jersey overlapping its wearer is two real findings). |
| `min_box_size` / `min_crop_pixels` / `crop_padding` | `0.0` / `16` / `0.06` | Crop selection. `min_crop_pixels` is a **measured** floor: below ~8 px wrong answers come back *more* confidently than right ones. `crop_padding` changes the vector — hold it fixed for the life of an index. |

There is **no global `conf`**. Detector scores are not comparable across backends — YOLOE's
text-similarity scores, YOLO11's sigmoid class scores and Grounding DINO's query scores are on
different scales — so each carries the threshold that leave-one-clip-out selection chose for it:

| backend | imgsz | conf |
|---|---|---|
| Grounding DINO (`coverage`) | 800 | 0.07 |
| YOLOE-26 (`fast`) | 1280 | 0.007 |
| YOLO11 (person) | 640 | 0.11 |

`imgsz` is per detector because one value cannot be right for all three: YOLOE **gains** from
resolution, Grounding DINO **collapses** above its native 800, and YOLO11 is **best at 640**.
Raising it also does not make crops bigger — boxes are rescaled to source coordinates and crops
come from the original frame.

### OCR — an exact text channel beside the vectors

Requires `detect_target` (there is nothing to stamp a string onto otherwise, so it is rejected
rather than ignored). Off by default: the cost is a fixed ~400 ms/frame regardless of what it
finds, and the value is strongly content-dependent — 10 of 15 brands found by name on a
broadcast, 4 of 14 on animation.

| param | default | meaning |
|---|---|---|
| `ocr` | `"off"` | `"text"` runs one pass per frame and stamps the words inside each detection's box into `additional_info.text`. `"propose"` additionally turns un-boxed text regions into detections of their own — **for the `fast` path**, which cannot emit wide boxes; it adds ~1% to `coverage`, which already boxes banners. |
| `ocr_conf` | `0.2` | Minimum recognition confidence for a **string** to be indexed. Below it the output is single characters and jersey-number fragments. |
| `ocr_box_conf` | `0.0` | Minimum confidence for a text region to be a **proposal box**. Ungated on purpose: CRAFT routinely localises a wordmark the CRNN then garbles, and the box is still worth cropping because the crop retrieves by *image*. |
| `ocr_mag` | `1.0` | easyocr's `mag_ratio` — upsample before text *detection*. 2.0 costs 62% more and buys one read; the archival rung. |
| `ocr_attach_overlap` | `0.7` | Fraction of a text region that must lie inside a box for its string to attach. Containment, not IoU: a wordmark is far smaller than the box holding it. |

Three limits worth knowing: it reads **glyphs, not logos** (IBM's striped wordmark reads 1 of 20);
**end credits are a trap** for a coverage report and no shipped filter excludes them; and it is
**English only** (`easyocr.Reader(["en"])`).

### Output

| param | default | meaning |
|---|---|---|
| `output_tags` | `false` | Detection mode only. Also emit a vector-less tag beside each detection's vector tag, so it shows up in EVIE's tag tracks. [Above](#seeing-the-boxes-in-the-video-editor). |

## Deployment

**No weights are baked into the image**, and the deploy environment **must mount a persistent
volume at `/root/.cache`** or everything is re-fetched on every run. One mount covers all of it:

| downloaded | into | when |
|---|---|---|
| SigLIP 2 (~1.5 GB) | `HF_HOME=/root/.cache` | always — it is the embedder |
| YOLO11 / YOLOE / Grounding DINO + the MobileCLIP text encoder | `storage.cache_path` | only when a request sets `detect_target` |
| easyocr CRAFT + CRNN | `storage.cache_path/easyocr` | only when a request sets `ocr` |

The `hub/` cache is keyed by repo id, so one volume can also serve the sibling SigLIP 2 taggers.
See [`test.sh`](test.sh) for a working invocation (`--volume=detection_cache:/root/.cache`).

### Rebuilding after a `common-ml` change (stale layer in podman)

```bash
git submodule update --init --recursive
buildscripts/build_container.bash -t model-frame-vector:latest . -f Containerfile --no-cache
```

## Notes on the index

The vectors are **768-d**, so the space needs `embedding_size: 768`. That is under the
vectorstore's 1024 ceiling, so **no projection is involved** — the vectors are right-zero-padded
to 1024, and trailing zeros leave the dot product and both norms unchanged, so cosine is preserved
exactly. The dimension is read from the checkpoint (`config.hidden_size`), never hardcoded.

`base` rather than `so400m` partly because so400m emits 1152-d and would not fit, and partly
because its ~4–5× per-crop compute is paid once per *detection*, up to `max_detections` times per
frame. On brand-logo retrieval over a 5,953-image gallery, `base`-NaFlex ties
`siglip2-large-patch16-384` (r@1 0.926 vs 0.929, inside the ±0.026 noise band) at **5.95 ms/crop
against 36.4**.

**Escalation order** if crop quality proves short: `max_num_patches` 256 → 576, then
`siglip2-large-patch16-384`. Raising `imgsz` is *not* on this list — it finds more small marks, it
does not make crops bigger.

## Query side must use the same checkpoint

- Load the full `Siglip2Model` (this tagger only ever loads the vision tower) and use
  `Siglip2Processor`/`AutoProcessor` for text. SigLIP was trained with `padding="max_length",
  max_length=64`; a bare tokenizer call does not do that and the mismatch degrades scores
  **quietly rather than erroring**. In `transformers` 5, `get_text_features()` returns the output
  object — take `.pooler_output`.
- **Text→vector and vector→vector need separate thresholds.** Image and text embeddings do not
  share a cone on the unit sphere: cos(image, image) for related content lands ~0.5–0.9, while
  cos(image, text) for a *perfect* match lands ~0.05–0.3. One threshold cannot serve both. Prefer
  SigLIP 2's calibrated probability for text queries — `sigmoid(cos * logit_scale.exp() +
  logit_bias)`, and note `logit_scale` is stored as the **log** of the scale.
- **Query by example** goes through the vision tower, so preprocess it the way the index was
  built: same `max_num_patches`, and in detection mode the same `crop_padding` (every tag carries
  it for exactly this reason — a ±0.06 mismatch loses 8% of top-1 retrievals, 0.06-vs-0.25 loses
  24%, and nothing surfaces those as errors).
- **Do not try to close the modality gap at tag time.** Any transform on stored vectors would have
  to apply identically to text queries, and subtracting the image mean leaves every query
  dominated by the gap offset so they all collapse toward one direction.

## Tests

```bash
pip install -e .[test]
pytest tests/                 # detector and embedder stubbed: no weights, no GPU
```

`ELV_DETECTION_INTEGRATION=1` additionally runs end to end against real weights in both modes
(needs `test-files/1.mp4` and a GPU). For container smoke tests:

```bash
make test    # the default frame-vector path
IMAGE_NAME=model-frame-vector ./buildscripts/testers/test-model.sh --params '{"detect_target": ["brand", "person"]}'
IMAGE_NAME=model-frame-vector ./buildscripts/testers/test-model.sh --params '{"detect_target": ["brand"], "brand_tiles": "2x2", "ocr": "text"}'
IMAGE_NAME=model-frame-vector ./buildscripts/testers/test-model.sh --params '{"detect_target": ["brand", "person"], "output_tags": true}'
```

## License

**AGPL-3.0** — see [LICENSE](LICENSE). The container links `ultralytics` (YOLOE, YOLO11), which is
AGPL-3.0, so the combined work is AGPL-3.0 even though the default frame-vector path never loads
it. The detectors are isolated behind
[`general_detection/detector.py`](general_detection/detector.py) so they can be replaced; swapping
in an Apache-2.0 open-vocabulary detector (OWLv2, or Grounding DINO alone — either keeps the whole
stack inside HF transformers) means editing that one module.
