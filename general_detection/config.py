from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RuntimeConfig:
    """Runtime tunables for the frame-vector tagger, injected per-request via `--params`
    in run.py.

    See the README's "Runtime parameters" table for provenance: some defaults are
    inherited from sibling taggers, others are explicitly uncalibrated placeholders and
    are marked as such below."""

    # ---- what to embed ----------------------------------------------------------

    # THE MODE SWITCH. None (the default) means no detection at all: one whole-frame SigLIP 2
    # vector per sampled frame. Setting it turns on the detection phase, and then only the
    # detected crops are embedded.
    #
    # A term naming a known parent expands to that parent's phrasings, so "brand" becomes the
    # four mark terms "brand", "logo", "car logo", and "letter logo". Any other term becomes its
    # own parent with itself as the single phrasing, so ["car"] is a valid target.
    #
    # Targets are ROUTED to detectors: "person" goes to the closed COCO backend, everything else
    # to the open-vocabulary one, and a detector with nothing routed to it is never loaded. So
    # ["person"] never pays for the brand model, and ["car"] never pays for the person model.
    # With this unset, none of the detector weights are loaded at all.
    detect_target: Optional[List[str]] = None # previously None meant ["brand", "person"]

    # Which open-vocabulary backend serves non-person targets. Ignored with detection off.
    #
    #   "coverage"  Grounding DINO @800, conf 0.07.  coverage 0.469, usable 0.469, 21 boxes/frame
    #   "fast"      YOLOE-26 @1280, conf 0.007.      coverage 0.219, usable 0.125,  3.5 boxes/frame
    #
    # "coverage" is the default. Each crop is a SigLIP forward pass and an index row.
    # Choose "fast" when index size or throughput is the
    # binding constraint and partial brand recall is acceptable.
    brand_detector: str = "coverage"

    # ---- detection --------------------------------------------------------------

    # Explicit {parent: [phrasings]} mapping, for controlling phrasings per parent. Empty by
    # default; `detect_target` wins when both are set. Setting either one turns detection on.
    # `DEFAULT_CLASS_PROMPTS` is what `detect_target: ["brand", "person"]` expands to.
    # Overriding it re-encodes the text prompts (seconds); it does not reload the model.
    class_prompts: Dict[str, List[str]] = field(default_factory=dict)

    # Per-parent confidence overrides. Empty by default, because each backend now carries its
    # own measured threshold (see detector.py BRAND_BACKENDS / PERSON_BACKEND).
    # CAREFUL: passing class_conf via --params REPLACES
    # this dict, it does not merge into it, so pass every class you want gated.
    class_conf: Dict[str, float] = field(default_factory=dict)

    # Sliced inference: "COLSxROWS". "1x1" is off. The detector runs once on the whole frame
    # and once per tile, and the tile boxes are merged back in frame coordinates before the
    # normal gate/suppress/crop path -- so nothing downstream changes, including the crops,
    # which are always taken from the original frame.
    #
    # OFF by default because of tradeoff between detection wall clock and usable coverage.
    # Turn it on for an archival or coverage-report pass, where the
    # index is built once and small marks matter.
    # ("2x2" for "coverage", "3x2" for "fast")
    # See detector.py `_raw_tiled` for aspect-band evidence.
    brand_tiles: str = "1x1"

    # Fraction of a tile's own width/height added on each side, so neighbouring tiles overlap.
    # Without it a tile seam is a blind spot as wide as the mark sitting on it.
    tile_overlap: float = 0.2

    # ---- OCR channel -------------------------------------------------------------

    # Read the words on the screen and index them beside the vectors. Requires `detect_target`:
    # there is nothing to stamp a string onto without detections, so setting it with detection
    # off is an error rather than a no-op.
    #
    #   "off"      no OCR pass. The default (see eval/experiments/17_ocr).
    #   "text"     one OCR pass per frame; every detection whose box contains a text region
    #              gets `additional_info.text`. No new crops, no new index rows, no change to
    #              detection -- purely additive to what a run already emits.
    #   "propose"  additionally, text regions the detector did NOT box become brand detections
    #              of their own, with `prompt: "ocr"`.
    #
    # OFF by default for two reasons, both measured (eval/experiments/17_ocr).
    ocr: str = "off"

    # Minimum easyocr recognition confidence for a STRING to be attached to a tag. 0.2 is where
    # the reads stop being words: below it the output is single characters and fragments of
    # jersey numbers. At 0.2 all six known-absent control brands score zero across both titles.
    ocr_conf: float = 0.2

    # Minimum confidence for a text region to be used as a PROPOSAL BOX under `ocr: "propose"`.
    # Zero -- i.e. every region CRAFT localises -- because the box is worth cropping no matter the text.
    ocr_box_conf: float = 0.0

    # easyocr's `mag_ratio`: upsample the frame before text DETECTION (not recognition).
    # Worth setting for an archival pass, not for a routine one. Measured.
    ocr_mag: float = 1.0

    # Fraction of a text region that must lie inside a detection box for its string to be
    # attached to that detection. Containment rather than IoU: a wordmark is usually far smaller
    # than the box holding it, and IoU would score that pairing near zero.
    ocr_attach_overlap: float = 0.7

    # Ultralytics' internal NMS IoU, applied per class id (i.e. per prompt).
    iou: float = 0.7

    # NMS across the phrasings of one parent term ("logo" vs "letter logo"), applied after
    # detection. The `iou` stage above cannot do this: those are distinct class ids, so
    # ultralytics never compares their boxes, and one jersey badge survives as one box per
    # phrasing — N near-identical crops, N near-identical vectors, N stacked overlay rectangles.
    # Chosen by a CONSTRAINT rather than by maximising a score. Measured against box
    # ground truth, the constraint is that coverage must NOT fall.
    nms_iou: float = 0.6

    # NMS across different parent terms. Disabled (1.0) by default, and it should stay disabled
    # under this schema: the only cross-parent overlap available is brand against person, and a
    # mark on a jersey overlapping the player wearing it is TWO real findings, not a duplicate.
    # Enabling this would delete one of them.
    cross_class_nms_iou: float = 1.0

    # Hard cap per frame, applied last, highest score first. Each survivor costs one
    # SigLIP 2 forward pass, so this is the pipeline's primary cost knob. (Ultralytics'
    # own max_det default is 300, which here would mean 300 embeds per frame.)
    max_detections: int = 100

    # Per-detector input size and threshold. None means "use the backend's measured default"
    # (detector.py), which is what you want unless you are deliberately re-tuning.
    # These are per-detector because a single global value would be wrong for at least one
    # backend whatever it was set to. Measured (eval/experiments/06_resolution).
    # Resolution buys recall on small objects; it does not improve the crop an embedder receives, 
    # and it makes the average crop smaller.
    brand_imgsz: Optional[int] = None
    brand_conf: Optional[float] = None
    person_imgsz: Optional[int] = None
    person_conf: Optional[float] = None

    # ---- crop selection ---------------------------------------------------------

    # Drop detections whose normalized box area is below this. Same knob, same default, as
    # model-celeb-vector's RuntimeConfig.
    min_box_size: float = 0.0

    # Drop crops whose shorter side is under this many source pixels, measured on the
    # un-padded detection box. Every tag carries `additional_info.upscale`,
    # which at a fixed budget is a monotone function of crop area, so a downstream consumer can
    # raise the effective floor by filtering. A floor set too HIGH is not recoverable: the crop
    # was never embedded, and getting it back means re-decoding the video and re-detecting.
    min_crop_pixels: int = 16

    # Fraction of the box's width/height added on each side before cropping. The *reported*
    # box stays un-padded, so the overlay draws the detection rather than the crop.
    # Note this changes the emitted vector slightly, and it is best to compare vectors in an index with the same crop_padding.
    # Recorded in `additional_info.crop_padding`.
    crop_padding: float = 0.06

    # ---- embedding --------------------------------------------------------------

    # NaFlex resolution budget: the crop is resized (aspect ratio preserved to within one
    # patch) to cover at most this many 16x16 patches. 256 is the checkpoint's documented
    # budget. Crops are small and often extreme aspect ratios — a 20x160 banner becomes
    # 96x672 here — which is exactly what NaFlex preserves and a square resize destroys.
    max_num_patches: int = 256

    # Optional ceiling on the NaFlex upscale factor, applied by lowering the per-crop patch
    # budget so small crops stay nearer native resolution instead of being interpolated up.
    # None (the default) gives every crop the full `max_num_patches` budget.
    # When set, crops are bucketed by resulting budget before batching, since the processor
    # takes one max_num_patches per call. Padding is masked out, so a crop's vector depends
    # only on its own budget and never on which crops it was batched with.
    max_upscale: Optional[float] = None

    # L2-normalize each emitted vector so cosine similarity reduces to a dot product.
    # Lossless.
    normalize: bool = True

    # Crops per forward pass through the vision tower.
    embed_batch_size: int = 32

    # ---- output -----------------------------------------------------------------

    # Additionally emit a vector-less FrameTag (to visualize the bounding boxes in EVIE) beside each detection's 
    # vector tag: same label, same box, no `vector`, and no embedder provenance (because there is no associated vector). 
    # Off by default, which is the original one-vector-tag-per-detection output.
    output_tags: bool = False
