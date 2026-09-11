"""How a `detect_target` term becomes detector prompts. One backend serves every target."""
import pytest

from general_detection.prompts import (
    BRAND_PROMPTS, DEFAULT_CLASS_PROMPTS, PERSON_PROMPTS, expand_target,
)


def test_brand_expands_to_the_mark_terms():
    """`brand` must expand, not pass through literally.

    The bare word is a far weaker prompt than the mark list -- only Grounding DINO grounds it
    at all -- so a target of ["brand"] silently becoming the literal string would be a large
    quality regression that nothing else would catch.
    """
    assert expand_target(["brand"])["brand"] == BRAND_PROMPTS
    assert expand_target(["brand", "person"]) == DEFAULT_CLASS_PROMPTS


def test_person_is_its_own_phrasing():
    """`person` alone matches a 101-term list including 18 role words, so there is nothing to
    expand it into. It now goes to the open-vocabulary backend like everything else."""
    assert expand_target(["person"]) == {"person": PERSON_PROMPTS}
    assert PERSON_PROMPTS == ["person"]


def test_unknown_target_becomes_its_own_parent():
    assert expand_target(["car"]) == {"car": ["car"]}
    assert expand_target(["person", "car"]) == {"person": ["person"], "car": ["car"]}


def test_blank_terms_are_dropped_and_an_empty_target_is_an_error():
    assert expand_target(["person", "  ", ""]) == {"person": ["person"]}
    with pytest.raises(ValueError):
        expand_target(["", "   "])


def test_there_is_no_per_term_routing_any_more():
    """`split_by_detector` partitioned targets between an open-vocab and a closed COCO backend.
    The closed one (YOLO11) is retired, so every term goes to the same detector."""
    import general_detection.prompts as prompts

    assert not hasattr(prompts, "split_by_detector")
    assert not hasattr(prompts, "CLOSED_VOCAB_PARENTS")
    assert not hasattr(prompts, "CLOSED_VOCAB_LABEL")
