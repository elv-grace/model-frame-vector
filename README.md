# model-frame-vector — SigLIP 2 frame and crop embedder

Turns video and images into search vectors with [SigLIP 2](https://huggingface.co/docs/transformers/model_doc/siglip2)
NaFlex (`google/siglip2-base-patch16-naflex`, **768-d**, L2-normalized).

**Every sampled frame is embedded whole.** Pass `detect_target` and an open-vocabulary detector
runs first, so each detection is also cropped and embedded — the frame keeps its vector and gains
one per detection beside it, each carrying its bounding box.

| `--params` | vectors per frame | weights loaded |
|---|---|---|
| `{}` | 1 — the whole frame | SigLIP 2 only |
| `{"detect_target": ["brand", "person"]}` | 1 + one per detection | SigLIP 2 + Grounding DINO |

Implements `common_ml`'s `FrameModel`, so frame extraction (ffmpeg/PyAV), image-vs-video dispatch
and tag serialization come from the tagger runtime. Model code is in [`general_detection/`](general_detection/).

> 📈 **The measurements and the decisions behind every default:**
> [**Courtside Blindness**](https://claude.ai/code/artifact/97c7e4ad-5c65-4aa5-820e-cd7c2e88fdfe)
> (detector thresholds, prompt set, and what tiling and OCR bought before they were removed) and
> [**Brand and Person Detector Sweep**](https://claude.ai/code/artifact/8a432b7c-8601-4d7d-bea9-d527e67141b2)
> (which detector, which embedder). Raw runs are in [`eval/`](eval/README.md).

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

### The `/root/cache` mount is required

**No weights are baked into the image**, so without a persistent volume at `/root/.cache`
everything lands in the container's ephemeral layer and is re-fetched every run. One mount covers
all of it:

| downloaded | when |
|---|---|
| SigLIP 2 (~1.5 GB) → `HF_HOME` | always — it is the embedder |
| Grounding DINO *or* YOLOE + its MobileCLIP text encoder → `storage.cache_path` | only when a request sets `detect_target` |

The `hub/` cache is keyed by repo id, so one volume can also serve the sibling SigLIP 2 taggers.
See [`test.sh`](test.sh) for a working invocation (`--volume=detection_cache:/root/.cache`).

### Typical invocations

```bash
--params '{}'                                                       # frame vectors only
--params '{"detect_target": ["brand", "person"]}'                   # + brand and person crops
--params '{"detect_target": ["car", "dog"]}'                        # any noun; it is open-vocabulary
--params '{"detect_target": ["brand"], "detector": "fast"}'         # cap index size / GPU time
--params '{"detect_target": ["brand"], "max_detections": 200}'      # archival: let dense frames through
--params '{"detect_target": ["person"], "output_tags": true}'       # + boxes as EVIE tag tracks
```

### Rebuilding after a `common-ml` change (stale layer in podman)

```bash
git submodule update --init --recursive
buildscripts/build_container.bash -t model-frame-vector:latest . -f Containerfile --no-cache
```

## Output

Every tag carries a `vector` and a normalized `box`. `common_ml` then attaches
`start_time`/`end_time` (ms), `source_media` and `frame_info.{frame_idx, box}`. Vector tags are
**never run-length merged**, which is what keeps the box available for the EVIE overlay.

**The frame vector**, always emitted first — empty `tag`, full-frame box:

```json
{"tag": "", "vector": [...768], "box": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
 "additional_info": {"box": {...}, "upscale": 0.18, "embedder": "google/siglip2-base-patch16-naflex",
                     "revision": "b53b807d…", "dim": 768, "normalize": true,
                     "max_num_patches": 256, "query_modes": ["text", "image"]}}
```

**A detection vector**, one per crop — `tag` is the **parent term** (`brand`, `person`, or whatever
was asked for) and `box` is the **un-padded** detection box:

```json
{"tag": "brand", "vector": [...768], "box": {"x1": 0.41, "y1": 0.72, "x2": 0.53, "y2": 0.75},
 "additional_info": {"box": {...}, "prompt": "letter logo", "score": 0.19,
                     "detector": "grounding-dino-base", "crop_padding": 0.06, "upscale": 5.2,
                     "embedder": "…", "dim": 768, …}}
```

A consumer tells the two apart by the empty label and the full-frame box. They are not
interchangeable: a frame is *downsampled* to the patch budget (`upscale` < 1) and a crop is
*upsampled* to it (`upscale` > 1), so they answer different questions — "which frames look like
this" against "where is this mark".

### Why `box` is the un-padded detection

The crop handed to the embedder is 12% wider and taller than `box` (`CROP_PADDING` 0.06 on each
side), and `box` deliberately does not follow it. `box` answers *where the object is*; the padding
is a preprocessing choice about what the embedder needs to see around it, and three things depend
on keeping those separate:

- **The overlay draws this box.** `box` becomes `frame_info.box`, which EVIE multiplies by canvas
  size. A padded box would draw every rectangle 12% too large — reading as a sloppy detector
  rather than as deliberate context.
- **The un-padded box is the invertible one.** Padded → un-padded is *lossy*: `_crop` clamps to the
  frame, so a mark against the edge got less than 0.06 on that side and the padded box cannot say
  how much. Un-padded → padded is exact, from `box` plus `additional_info.crop_padding`.
- **It keeps boxes comparable when padding is not.** Ground-truth IoU has to score the object's
  extent, and `crop_padding` has changed before. If `box` moved with it, every eval number would
  shift and a mixed index would hold boxes on two conventions with nothing to tell them apart.

So the padding is recorded rather than baked in — which is exactly what lets a query pipeline
rebuild the same crop geometry instead of assuming it.

### `additional_info`

| field | why it is there |
|---|---|
| `box` | the same box as `frame_info.box`, repeated because `additional_info` is the **only** field a vectorstore search row returns |
| `embedder`, `revision`, `dim`, `normalize`, `max_num_patches` | the embedding recipe. A query that does not match on all of these is not comparable to what is indexed |
| `query_modes` | which query modalities this space accepts (`text`, `image`) |
| `upscale` | the NaFlex scale actually applied, so the heavily-interpolated tail can be filtered downstream without re-tagging |
| `prompt` | *detections only* — which phrasing fired |
| `score` | *detections only* — detector confidence |
| `detector` | *detections only* — which backend found it |
| `crop_padding` | *detections only* — **changes the vector**, so build an image query at the same padding |

### Seeing the boxes in the video editor

`output_tags: true` emits a **second, vector-less** tag beside each detection's vector tag — same
label, same box — which lands as an ordinary tag track in EVIE alongside the overlaid boxes. No
extra inference, but the tag count more than doubles (the vector-less twins also get run-length
merged into spans, which the vector tags are not). The frame tag is never twinned: its label is
empty, so a copy would carry no information.

> The twins share their parent's `additional_info` dict, so the only difference is the presence of
> `vector`. Drop vector-less records *before* stripping vectors to store a run, or every detection
> is counted twice.

### Notes on the index

The vectors are **768-d**. **No projection is involved** — vectors are right-zero-padded to the vectorstore index size
1024-d. The dimension is read from the checkpoint (`config.hidden_size`), never hardcoded.

Text→vector and vector→vector need **separate thresholds**: image and text embeddings do not
share a cone on the unit sphere, so cos(image, image) for related content lands ~0.5–0.9 while
cos(image, text) for a *perfect* match lands ~0.05–0.3. A query-by-example goes through the vision
tower, , but the query side needs the **text tower**,
so the same checkpoint is loaded separately.

## Runtime parameters

Four, injected per request as a JSON `--params` object. Frame sampling rate (`fps`) is handled
generically by the tagger runtime.

| param | default | meaning |
|---|---|---|
| `detect_target` | `null` | Turns the detection phase on. `"brand"` expands to the four mark terms (`logo`, `letter logo`, `brand`, `car logo`) because the bare word is a far weaker prompt; every other term is its own prompt, so `["car"]` and `["person", "car"]` are valid. Unset, no detector weights load at all. |
| `detector` | `"coverage"` | `"coverage"` = Grounding DINO @800 conf 0.07 (usable 0.469, 21.4 boxes/frame, 1.14 s/frame); `"fast"` = YOLOE-26 @1280 conf 0.007 (usable 0.125, 3.5 boxes/frame, 0.88 s/frame). |
| `max_detections` | `100` | Cap per frame, highest score first. **The primary cost knob** — each survivor is one SigLIP forward pass and one index row. |
| `output_tags` | `false` | Also emit a vector-less tag beside each detection's vector tag. [Above](#seeing-the-boxes-in-the-video-editor). |

`"coverage"` is the default because it finds roughly twice the marks — and five times as many on
wide hoarding-shaped marks — for only ~1.3× the wall clock. The s/frame figures above are end to
end over 120 frames of live 1080p broadcast at 1 fps (`fast` 105 s, `coverage` 137 s); the
detectors alone are **98 ms and 293 ms**, so the 3× per-frame gap between the models shows up as
1.3× in practice because video decode is a shared cost that dominates the cheap config.

`"fast"` is a **cap, not an optimisation**. It offers a ceiling: 3.5
boxes/frame against 21.4, so a two-hour title at 1 fps indexes ~25k rows instead of ~154k. Choose
it when index size or GPU time is a hard constraint and partial recall is acceptable.

### Everything else is fixed on purpose

Each of the values below changes the emitted vector or was measured into place, and a caller
changing one produces vectors that are not comparable with the rest of the index — silently, with
no error. They are constants next to the code that consumes them:

| where | fixed values |
|---|---|
| [`embedder.py`](general_detection/embedder.py) | `MAX_NUM_PATCHES` 256, `NORMALIZE` True, `BATCH_SIZE` 32 |
| [`detector.py`](general_detection/detector.py) | `PREDICT_IOU` 0.7, `NMS_IOU` 0.6, `CROP_PADDING` 0.06, `MIN_CROP_PIXELS` 16, and each backend's own `imgsz`/`conf` |
| [`prompts.py`](general_detection/prompts.py) | `DEFAULT_CLASS_PROMPTS` — what `brand` and `person` expand to |

`imgsz` and `conf` are per backend rather than shared because neither could be right for both:
YOLOE **gains** from resolution (brand AP 0.062 @640 → 0.133 @1280) while Grounding DINO
**collapses** above its native 800, and their score scales are not comparable (text-similarity
against query probabilities), so one threshold could only ever suit one of them.

Two things were removed rather than defaulted off — `git log` has them if the numbers ever justify
bringing them back:

- **Sliced inference** (`brand_tiles`, `tile_overlap`) — +27% usable coverage at `2x2` for five
  detector passes per frame. See [`eval/experiments/13_sliced/`](eval/experiments/13_sliced/).
- **The OCR channel** (`ocr`, `ocr_conf`, `ocr_box_conf`, `ocr_mag`, `ocr_attach_overlap`) — a
  fixed ~400 ms/frame regardless of what it found, and strongly content-dependent (10 of 15 brands
  read by name on a broadcast, 4 of 14 on animation). See
  [`eval/experiments/17_ocr/`](eval/experiments/17_ocr/).

The **YOLO11** person backend is also gone. It beat every open-vocabulary model at that one class
(AP 0.752, mean IoU 0.89, 53 ms/frame), but is a lot of surface area for one class — and the
open-vocabulary backends ground `person` at 0.92–0.97 class-agnostic coverage.

## Tests

```bash
pip install -e .[test]
pytest tests/                 # detector and embedder stubbed: no weights, no GPU
```

`ELV_DETECTION_INTEGRATION=1` additionally runs end to end against real weights in both modes
(needs `test-files/1.mp4` and a GPU). For container smoke tests:

```bash
make test               # frame vectors only
IMAGE_NAME=model-frame-vector ./buildscripts/testers/test-model.sh --params '{"detect_target": ["brand", "person"]}'
IMAGE_NAME=model-frame-vector ./buildscripts/testers/test-model.sh --params '{"detect_target": ["brand"], "detector": "fast"}'
IMAGE_NAME=model-frame-vector ./buildscripts/testers/test-model.sh --params '{"detect_target": ["person"], "output_tags": true}'
```

## License

**AGPL-3.0** — see [LICENSE](LICENSE). The container links `ultralytics` (YOLOE), which is
AGPL-3.0, so the combined work is AGPL-3.0 even though neither the default frame-vector path nor
the `coverage` detector loads it.

The detectors are isolated behind [`general_detection/detector.py`](general_detection/detector.py)
so they can be replaced: dropping `"fast"` for an Apache-2.0 backend (OWLv2, or Grounding DINO
alone — either keeps the whole stack inside HF transformers) means editing that one module and
takes the AGPL obligation with it.
