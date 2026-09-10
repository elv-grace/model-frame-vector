"""Unit tests for the OCR channel's pure logic.

The easyocr reader itself is never constructed: these cover the parts that decide what a text
region turns into, which is where the behaviour actually lives. Weightless and GPU-free, so
unlike tests/test_model.py they run anywhere.
"""
import numpy as np
import pytest

from general_detection.config import RuntimeConfig
from general_detection.ocr import TextRegion, _iou, _overlap_fraction, texts_for

pytest.importorskip("supervision")

from general_detection.ocr import proposals  # noqa: E402


def _region(text, conf, x1, y1, x2, y2, size=(1000, 1000)):
    height, width = size
    return TextRegion(text=text, conf=conf,
                      box={"x1": x1 / width, "y1": y1 / height,
                           "x2": x2 / width, "y2": y2 / height},
                      xyxy=(x1, y1, x2, y2))


def _detection(x1, y1, x2, y2, size=(1000, 1000)):
    from general_detection.detector import Detection

    height, width = size
    return Detection(label="brand", prompt="logo", score=0.5,
                     box={"x1": x1 / width, "y1": y1 / height,
                          "x2": x2 / width, "y2": y2 / height},
                     crop=np.zeros((4, 4, 3), dtype=np.uint8), detector="fake")


# ---- attaching strings to detections --------------------------------------------


def test_a_wordmark_inside_a_much_larger_box_still_attaches():
    """The reason attachment uses containment rather than IoU. A 200x40 wordmark inside a
    600x400 hoarding box is IoU 0.03 and is obviously the same finding."""
    box = _detection(0, 0, 600, 400).box
    region = _region("STATE FARM", 0.9, 100, 100, 300, 140)

    assert _iou(region.box, box) < 0.05
    assert texts_for(box, [region], RuntimeConfig()) == ["STATE FARM"]


def test_a_region_mostly_outside_the_box_does_not_attach():
    box = _detection(0, 0, 100, 100).box
    region = _region("ELSEWHERE", 0.9, 60, 60, 260, 100)   # 20% inside

    assert texts_for(box, [region], RuntimeConfig()) == []


def test_illegible_reads_are_not_indexed_even_though_their_boxes_survive():
    """The two gates differ on purpose: `ocr_conf` decides what is worth indexing,
    `ocr_box_conf` decides what is worth cropping."""
    box = _detection(0, 0, 600, 400).box
    garbled = _region("bAh", 0.05, 100, 100, 300, 140)

    assert texts_for(box, [garbled], RuntimeConfig()) == []


def test_strings_come_back_in_reading_order():
    box = _detection(0, 0, 600, 400).box
    regions = [_region("EXPRESS", 0.9, 300, 100, 500, 140),
               _region("SPONSOR", 0.9, 100, 300, 300, 340),
               _region("AMERICAN", 0.9, 100, 100, 280, 140)]

    assert texts_for(box, regions, RuntimeConfig()) == ["AMERICAN", "EXPRESS", "SPONSOR"]


def test_overlap_fraction_is_of_the_region_not_the_union():
    inner = {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2}
    outer = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
    assert _overlap_fraction(inner, outer) == pytest.approx(1.0)
    assert _overlap_fraction(outer, inner) == pytest.approx(0.01)


# ---- text regions as proposals ---------------------------------------------------


def test_a_near_duplicate_of_an_existing_box_is_suppressed():
    existing = [_detection(100, 100, 300, 140)]
    region = _region("STATE FARM", 0.9, 102, 101, 298, 139)
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)

    assert proposals([region], existing, img, RuntimeConfig(), "brand") == []


def test_a_tight_region_inside_a_loose_box_is_KEPT():
    """The case the whole feature exists for. On a 261x29 hoarding YOLOE emits a roughly square
    box four times too large; the OCR box is the tight one, and suppressing by containment
    rather than IoU would throw it away."""
    loose = [_detection(80, 40, 340, 300)]
    region = _region("STATE FARM", 0.9, 100, 100, 300, 140)
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)

    assert _overlap_fraction(region.box, loose[0].box) == pytest.approx(1.0)
    kept = proposals([region], loose, img, RuntimeConfig(), "brand")
    assert len(kept) == 1
    assert kept[0].box == region.box


def test_a_proposal_carries_its_provenance():
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    kept = proposals([_region("TISSOT", 0.77, 100, 100, 300, 140)], [], img,
                     RuntimeConfig(), "brand")

    assert len(kept) == 1
    assert kept[0].label == "brand"
    assert kept[0].prompt == "ocr"          # so an OCR-sourced crop is filterable downstream
    assert kept[0].detector == "easyocr-craft-crnn"
    assert kept[0].score == 0.77
    assert kept[0].crop.size > 0


def test_an_unreadable_region_still_becomes_a_proposal():
    """CRAFT localises text the CRNN then garbles. The crop retrieves by IMAGE against the
    reference pool whatever the string said, so the box is kept -- ocr_box_conf is 0.0."""
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    kept = proposals([_region("5t4te F4rm", 0.02, 100, 100, 300, 140)], [], img,
                     RuntimeConfig(), "brand")

    assert len(kept) == 1


def test_proposals_respect_min_crop_pixels_and_max_detections():
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    cfg = RuntimeConfig(min_crop_pixels=16, max_detections=2)

    thin = _region("x", 0.9, 100, 100, 300, 108)         # 8 px tall
    assert proposals([thin], [], img, cfg, "brand") == []

    many = [_region(f"w{i}", 0.9, 100 * i, 100, 100 * i + 80, 140) for i in range(5)]
    assert len(proposals(many, [], img, cfg, "brand")) == 2


def test_proposals_are_ordered_by_confidence_so_the_cap_drops_the_worst():
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    cfg = RuntimeConfig(max_detections=1)
    regions = [_region("weak", 0.1, 0, 100, 80, 140),
               _region("strong", 0.95, 200, 100, 280, 140)]

    kept = proposals(regions, [], img, cfg, "brand")
    assert [k.score for k in kept] == [0.95]
