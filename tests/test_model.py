"""Unit tests for the tagger: the always-on frame vector, and the crops a target adds.

The detector and embedder are stubbed, so these run without downloading YOLOE, Grounding DINO
or SigLIP 2 weights and without a GPU. The end-to-end test at the bottom is opt-in.
"""
import os

import numpy as np
import pytest

pytest.importorskip("ultralytics")
pytest.importorskip("transformers")

from general_detection.config import RuntimeConfig
from general_detection.detector import (
    CROP_PADDING,
    DETECTORS,
    Detection,
    YoloeDetector,
    build_detector,
)
from general_detection.model import FrameVectorModel

_DIM = 8
_FULL_FRAME = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}


class _FakeEmbedder:
    model_id = "fake/siglip2"
    revision = None
    dim = _DIM

    def embed(self, images):
        # One row per image, and a distinct scale per row so the caller's indexing is testable:
        # the frame is index 0 (downsampled, <1) and the crops follow (upsampled, >1).
        vectors = np.zeros((len(images), _DIM), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors, [0.18 if i == 0 else 2.5 for i in range(len(images))]


class _FakeDetector:
    name = "fake-gdino"

    def __init__(self, detections=None):
        self._detections = detections

    def detect(self, img, cfg):
        if self._detections is not None:
            return self._detections
        return [
            Detection(
                label="brand",
                prompt="letter logo",
                score=0.9,
                box={"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4},
                crop=np.zeros((40, 60, 3), dtype=np.uint8),
                detector="fake-gdino",
            )
        ]


def _model(cfg=None, detections=None, detector=True):
    """Bypass __init__ so no weights load. `detector=False` is what an unset `detect_target`
    leaves behind: the embedder and nothing else."""
    model = object.__new__(FrameVectorModel)
    model.config = cfg or RuntimeConfig(detect_target=["brand"])
    model._detector = _FakeDetector(detections) if detector else None
    model._detector_mode = "coverage" if detector else None
    model.embedder = _FakeEmbedder()
    return model


def _frame_only(cfg=None):
    return _model(cfg or RuntimeConfig(), detector=False)


# ---- the frame vector is always emitted -----------------------------------------


def test_default_emits_exactly_one_whole_frame_vector():
    tags = _frame_only().tag_frame(np.zeros((1080, 1920, 3), dtype=np.uint8))

    assert len(tags) == 1
    tag = tags[0]
    assert tag.tag == ""   # the frame itself, not a detected entity
    assert tag.box == _FULL_FRAME
    assert len(tag.vector) == _DIM


def test_frame_vector_carries_the_recipe_and_no_detection_provenance():
    info = _frame_only().tag_frame(np.zeros((100, 100, 3), dtype=np.uint8))[0].additional_info

    assert info["embedder"] == "fake/siglip2"
    assert info["dim"] == _DIM
    assert info["normalize"] is True
    assert info["max_num_patches"] == 256
    assert info["box"] == _FULL_FRAME
    assert info["upscale"] == 0.18   # a frame is DOWNsampled to the budget
    # nothing detected it and nothing was cropped, so these would describe nothing
    for absent in ("prompt", "score", "detector", "crop_padding"):
        assert absent not in info


def test_the_frame_vector_survives_a_target_that_finds_nothing():
    """Detection is additive. A frame with no detections still yields its own vector, where
    the old crops-only mode emitted nothing at all."""
    tags = _model(detections=[]).tag_frame(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(tags) == 1
    assert tags[0].tag == ""


def test_a_blank_frame_still_gets_a_vector():
    assert len(_frame_only().tag_frame(np.zeros((64, 64, 3), dtype=np.uint8))) == 1


# ---- a target adds crops beside it ----------------------------------------------


def test_detection_adds_one_crop_vector_after_the_frame_vector():
    tags = _model().tag_frame(np.zeros((1080, 1920, 3), dtype=np.uint8))

    assert len(tags) == 2
    frame_tag, crop_tag = tags
    assert frame_tag.tag == "" and frame_tag.box == _FULL_FRAME
    # the parent term, not the phrasing that fired
    assert crop_tag.tag == "brand"
    assert len(crop_tag.vector) == _DIM
    # normalized box, which is what the video-editor overlay multiplies by canvas size
    assert set(crop_tag.box) == {"x1", "y1", "x2", "y2"}
    assert all(0.0 <= v <= 1.0 for v in crop_tag.box.values())


def test_the_frame_and_its_crops_are_embedded_in_one_batch():
    """The frame is index 0 of a single embed() call and the crops follow it, so each tag has
    to read its own row. A mixed-up index would silently give a crop the frame's vector."""
    calls = []
    model = _model()
    inner = model.embedder.embed
    model.embedder.embed = lambda images: (calls.append(len(images)), inner(images))[1]

    tags = model.tag_frame(np.zeros((100, 100, 3), dtype=np.uint8))

    assert calls == [2]                          # one call, frame + one crop
    assert tags[0].additional_info["upscale"] == 0.18   # the frame's row
    assert tags[1].additional_info["upscale"] == 2.5    # the crop's row


def test_crop_tag_carries_the_provenance_the_index_needs():
    info = _model().tag_frame(np.zeros((100, 100, 3), dtype=np.uint8))[1].additional_info

    assert info["prompt"] == "letter logo"  # which phrasing fired, for recall tuning
    assert info["score"] == 0.9
    assert info["dim"] == _DIM
    assert info["detector"] == "fake-gdino"
    assert info["embedder"] == "fake/siglip2"
    # crop_padding changes the vector: vectors built at different padding are not comparable
    assert info["crop_padding"] == pytest.approx(CROP_PADDING)
    # lets the heavily-interpolated tail be filtered downstream without re-tagging
    assert info["upscale"] == 2.5


def test_crop_tag_repeats_its_box_in_additional_info():
    """A vectorstore search row carries `additional_info` and nothing else, so the box has to
    ride there to survive the round trip."""
    tag = _model().tag_frame(np.zeros((100, 100, 3), dtype=np.uint8))[1]

    assert tag.additional_info["box"] == tag.box
    # a copy, so mutating one cannot move the other
    assert tag.additional_info["box"] is not tag.box


# ---- output_tags ----------------------------------------------------------------


def test_output_tags_is_off_by_default():
    tags = _model().tag_frame(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(tags) == 2
    assert all(t.vector is not None for t in tags)


def test_output_tags_adds_a_vectorless_twin_after_each_crop_tag():
    tags = _model(RuntimeConfig(detect_target=["brand"], output_tags=True)).tag_frame(
        np.zeros((100, 100, 3), dtype=np.uint8)
    )

    assert len(tags) == 3
    frame_tag, vector_tag, plain_tag = tags
    assert frame_tag.vector is not None
    assert vector_tag.vector is not None
    assert plain_tag.vector is None
    # same detection: the twin is the same label on the same box, which is what lets EVIE draw
    # it as an ordinary tag track
    assert plain_tag.tag == vector_tag.tag == "brand"
    assert plain_tag.box == vector_tag.box


def test_vectorless_twin_keeps_the_detection_provenance_but_not_the_embedder_provenance():
    plain_info = _model(RuntimeConfig(detect_target=["brand"], output_tags=True)).tag_frame(
        np.zeros((100, 100, 3), dtype=np.uint8)
    )[2].additional_info

    assert plain_info["prompt"] == "letter logo"
    assert plain_info["score"] == 0.9
    assert plain_info["detector"] == "fake-gdino"
    assert plain_info["box"] == {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4}
    # embedder/dim/max_num_patches describe a vector this tag does not carry
    assert "embedder" not in plain_info
    assert "dim" not in plain_info


def test_output_tags_leaves_the_frame_vector_untwinned():
    """The frame tag's label is empty, so a vector-less copy would carry no information."""
    tags = _model(RuntimeConfig(detect_target=["brand"], output_tags=True)).tag_frame(
        np.zeros((100, 100, 3), dtype=np.uint8)
    )
    frame_tags = [t for t in tags if t.tag == ""]
    assert len(frame_tags) == 1
    assert frame_tags[0].vector is not None


def test_output_tags_adds_nothing_when_nothing_is_detected():
    cfg = RuntimeConfig(detect_target=["brand"], output_tags=True)
    tags = _model(cfg, detections=[]).tag_frame(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(tags) == 1 and tags[0].tag == ""


@pytest.mark.parametrize("passed, expected", [(None, False), (True, True), (False, False)])
def test_constructor_output_tags_sets_the_config_for_this_run(monkeypatch, passed, expected):
    """The kwarg is a convenience for callers holding the model directly. None leaves the
    config alone; a bool sets it."""
    monkeypatch.setattr(FrameVectorModel, "_apply_targets", lambda self, cfg: None)
    monkeypatch.setattr(
        "general_detection.model.Siglip2CropEmbedder",
        lambda *args, **kwargs: _FakeEmbedder(),
    )

    model = FrameVectorModel(
        cfg=RuntimeConfig(),
        embedder_model_id="fake/siglip2",
        cache_dir="/tmp",
        output_tags=passed,
    )
    assert model.config.output_tags is expected


# ---- the runtime surface is only four params ------------------------------------


def test_runtime_config_exposes_exactly_the_four_intended_params():
    """The rest are fixed constants next to the code that consumes them, because each one
    changes the emitted vector and a mixed index costs retrievals silently."""
    assert set(RuntimeConfig().__dataclass_fields__) == {
        "detect_target", "detector", "max_detections", "output_tags",
    }


@pytest.mark.parametrize("gone", [
    "max_num_patches", "normalize", "embed_batch_size", "max_upscale",   # embedding
    "brand_tiles", "tile_overlap", "class_prompts", "class_conf",        # detection
    "brand_imgsz", "brand_conf", "person_imgsz", "person_conf",
    "iou", "nms_iou", "cross_class_nms_iou",
    "min_box_size", "min_crop_pixels", "crop_padding",                   # crop selection
    "ocr", "ocr_conf", "ocr_box_conf", "ocr_mag", "ocr_attach_overlap",  # the OCR channel
    "brand_detector",                                                    # renamed to `detector`
    "embed_whole_frame",                                                 # the frame is always embedded
])
def test_retired_params_are_gone(gone):
    """dacite ignores unknown keys, so a stale --params blob carrying one of these is silently
    dropped rather than erroring. Pin that they really are absent."""
    assert not hasattr(RuntimeConfig(), gone)


def test_detector_defaults_to_coverage_and_rejects_an_unknown_name():
    assert RuntimeConfig().detector == "coverage"
    with pytest.raises(ValueError, match="detector must be one of"):
        build_detector("yolo11", "/tmp")


def test_shipped_backend_thresholds_are_reachable():
    """A gate above a class's whole score range deletes the class instead of trimming it. An
    earlier default ({"object": 0.5}) did exactly that: measured on real content the catch-all
    never exceeded 0.105. This pins that the shipped thresholds stay inside (0, 1) so a typo
    cannot silently mute a backend."""
    for name, spec in DETECTORS.items():
        assert 0.0 < spec["conf"] < 1.0, f"{name} threshold {spec['conf']} is unreachable"
        assert spec["imgsz"] % 32 == 0 or spec["kind"] == "gdino", \
            f"{name} imgsz {spec['imgsz']} is not a multiple of 32"


def test_only_the_two_open_vocab_backends_remain():
    """YOLO11 is retired: a second family of weights with an incomparable score scale is a lot
    of surface area for one class."""
    assert set(DETECTORS) == {"fast", "coverage"}
    assert {spec["kind"] for spec in DETECTORS.values()} == {"yoloe", "gdino"}


# ---- cropping -------------------------------------------------------------------


def test_crop_padding_expands_the_crop_but_not_the_reported_box():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # 40x40 box, 10% padding -> 4px each side
    crop = YoloeDetector._crop(img, 10, 10, 50, 50, 0.1)
    assert crop.shape[:2] == (48, 48)


def test_crop_clamps_at_the_frame_edge():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = YoloeDetector._crop(img, 0, 0, 20, 20, 0.5)
    assert crop.shape[:2] == (30, 30)   # padded to -10..30, clamped to 0..30


def test_crop_is_contiguous_for_pil():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = YoloeDetector._crop(img, 10, 10, 50, 50, 0.0)
    # a sliced view is non-contiguous and PIL.Image.fromarray rejects it
    assert crop.flags["C_CONTIGUOUS"]


def test_degenerate_box_yields_no_crop():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert YoloeDetector._crop(img, 50, 50, 50, 50, 0.0) is None


# ---- dedupe and the fixed gates -------------------------------------------------


def _stub_detector(conf=0.007):
    detector = object.__new__(YoloeDetector)
    detector._prompts = ["logo", "letter logo", "person"]
    detector._labels = ["brand", "brand", "person"]
    detector._group_of_label = {"brand": 0, "person": 1}
    detector.conf = conf
    detector.imgsz = 1280
    return detector


def _overlapping_detections(sv):
    box = [10.0, 10.0, 110.0, 110.0]
    return sv.Detections(
        xyxy=np.array([box, box, box], dtype=np.float32),
        confidence=np.array([0.9, 0.8, 0.7], dtype=np.float32),
        class_id=np.array([0, 1, 2]),
    )


def test_dedupe_collapses_synonyms_but_preserves_cross_class_nesting():
    sv = pytest.importorskip("supervision")
    out = _stub_detector()._dedupe(_overlapping_detections(sv))

    # "logo" and "letter logo" fire on the same pixels and collapse to one; the overlapping
    # "person" survives, because a mark on the player wearing it is two real findings.
    labels = sorted(_stub_detector()._labels[int(i)] for i in out.data["prompt_id"])
    assert labels == ["brand", "person"]


def test_dedupe_keeps_the_highest_scoring_phrasing():
    sv = pytest.importorskip("supervision")
    detector = _stub_detector()
    out = detector._dedupe(_overlapping_detections(sv))

    kept = {detector._prompts[int(i)] for i in out.data["prompt_id"]}
    assert "logo" in kept and "letter logo" not in kept   # 0.9 beats 0.8


def test_the_backend_threshold_is_the_only_gate():
    """There is no `class_conf` and no global `conf` any more: scores are not comparable across
    backends, so each carries its own measured threshold and nothing overrides it per request."""
    detector = _stub_detector(conf=0.25)
    assert detector.conf == 0.25
    assert not hasattr(detector, "_floor")
    assert not hasattr(detector, "_gate")


def test_max_detections_truncates_by_score():
    sv = pytest.importorskip("supervision")
    detector = _stub_detector()
    dets = sv.Detections(
        xyxy=np.array([[0, 0, 60, 60], [100, 100, 160, 160], [200, 200, 260, 260]],
                      dtype=np.float32),
        confidence=np.array([0.3, 0.9, 0.6], dtype=np.float32),
        class_id=np.array([0, 0, 2]),
        data={"prompt_id": np.array([0, 0, 2])},
    )
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    out = detector._to_detections(dets, img, 400, 400, RuntimeConfig(max_detections=2))
    assert [d.score for d in out] == [0.9, 0.6]   # the 0.3 is the one dropped


def test_min_crop_pixels_drops_a_box_too_small_to_retrieve():
    """Below ~8px the hit-vs-miss cosine gap goes negative, so no downstream similarity gate
    can filter these -- they have to be dropped here."""
    sv = pytest.importorskip("supervision")
    detector = _stub_detector()
    dets = sv.Detections(
        xyxy=np.array([[0, 0, 8, 8], [100, 100, 160, 160]], dtype=np.float32),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        class_id=np.array([0, 0]),
        data={"prompt_id": np.array([0, 0])},
    )
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    out = detector._to_detections(dets, img, 400, 400, RuntimeConfig())
    assert len(out) == 1 and out[0].score == 0.8


def test_tiling_is_gone():
    """Sliced inference is removed, not defaulted off -- see eval/experiments/13_sliced."""
    assert not hasattr(YoloeDetector, "_tiles")
    assert not hasattr(YoloeDetector, "_raw_tiled")


# ---- end to end (opt-in: needs weights, network on first run, and a GPU) ----------


@pytest.mark.skipif(
    not os.getenv("ELV_DETECTION_INTEGRATION"),
    reason="set ELV_DETECTION_INTEGRATION=1 to run against real weights",
)
# Both paths: the default loads only the embedder, the target also loads the detector.
@pytest.mark.parametrize("target", [None, ["brand", "person"]])
def test_end_to_end_against_a_test_file(target):
    from common_ml.tagging.file_tagger import FileTagger

    from config import config

    test_file = os.path.join(os.path.dirname(__file__), "../test-files/1.mp4")
    model = FrameVectorModel(
        cfg=RuntimeConfig(detect_target=target),
        embedder_model_id=config["model"]["embedder"]["model_id"],
        embedder_revision=config["model"]["embedder"].get("revision"),
        cache_dir=config["storage"]["cache_path"],
    )
    tags = FileTagger.from_frame_model(model).tag(test_file)

    assert len(tags) > 0
    for tag in tags:
        assert tag.source_media == test_file
        assert tag.vector is not None
        assert len(tag.vector) == model.embedder.dim
        # unit-normalized so cosine similarity reduces to a dot product downstream
        assert abs(np.linalg.norm(tag.vector) - 1.0) < 1e-3
        # vector tags pass through per-frame; common_ml does not run-length merge them,
        # which is what keeps the box available for the overlay
        assert tag.frame_info is not None
        assert tag.frame_info.box
    # a frame vector is emitted either way
    assert any(t.tag == "" for t in tags)
    if target is None:
        assert {t.tag for t in tags} == {""}
