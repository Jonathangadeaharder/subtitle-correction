from __future__ import annotations

import json
from pathlib import Path

import pytest

from subtitle_correction.alignment_quality import (
    classify_eval_entry,
    is_aligned_mishearing,
    is_lyric_cue,
    normalize_for_compare,
    reference_is_distinct_from_target,
    word_overlap,
)


def test_normalize_for_compare_empty() -> None:
    assert normalize_for_compare("") == ""
    assert normalize_for_compare(None) == ""  # type: ignore[arg-type]


def test_normalize_for_compare_strips_punct_and_lowercases() -> None:
    assert normalize_for_compare("Hello, World!\nNew") == "hello world new"


def test_normalize_for_compare_collapses_whitespace() -> None:
    assert normalize_for_compare("a   b\tc") == "a b c"


def test_word_overlap_full() -> None:
    assert word_overlap("hello world", "hello world") == 1.0


def test_word_overlap_partial() -> None:
    # Uses min(len(wa), len(wb)); full overlap of smaller set -> 1.0
    overlap = word_overlap("hello world", "hello world foo")
    assert overlap == 1.0


def test_word_overlap_truly_partial() -> None:
    overlap = word_overlap("hello world alpha", "hello bravo charlie")
    assert 0.0 < overlap < 1.0


def test_word_overlap_empty_returns_zero() -> None:
    assert word_overlap("", "hello") == 0.0
    assert word_overlap("hello", "") == 0.0


def test_word_overlap_no_shared_returns_zero() -> None:
    assert word_overlap("alpha", "bravo") == 0.0


def test_reference_is_distinct_none_or_empty_returns_false() -> None:
    assert reference_is_distinct_from_target(None, "x") is False
    assert reference_is_distinct_from_target("", "x") is False
    assert reference_is_distinct_from_target("   ", "x") is False


def test_reference_is_distinct_true() -> None:
    assert reference_is_distinct_from_target("hello world", "goodbye world") is True


def test_reference_is_distinct_false_when_equal() -> None:
    assert reference_is_distinct_from_target("hello world", "hello world") is False


def test_is_lyric_cue_detects_markers() -> None:
    assert is_lyric_cue("♪ song") is True
    assert is_lyric_cue("<i>italic</i>") is True
    assert is_lyric_cue("plain text") is False


def test_is_lyric_cue_handles_empty_args() -> None:
    assert is_lyric_cue("", None) is False  # type: ignore[arg-type]


def test_is_aligned_mishearing_missing_inputs_returns_false() -> None:
    assert is_aligned_mishearing("", "gt", "ref") is False
    assert is_aligned_mishearing("w", "", "ref") is False
    assert is_aligned_mishearing("w", "gt", "") is False


def test_is_aligned_mishearing_lyric_returns_false() -> None:
    assert is_aligned_mishearing("♪ song", "♪ song", "♪ song") is False


def test_is_aligned_mishearing_low_gt_ref_overlap_returns_false() -> None:
    # Ground truth and reference totally unrelated
    assert is_aligned_mishearing("alpha bravo", "banana split", "computer keyboard") is False


def test_is_aligned_mishearing_high_whisper_overlap_returns_true() -> None:
    # Whisper shares content with ground truth/reference
    assert is_aligned_mishearing("the cat sat", "the cat ran", "the cat ran") is True


def test_is_aligned_mishearing_german_whisper_english_reference() -> None:
    # Whisper is German, reference English; ground truth close to reference
    result = is_aligned_mishearing(
        "ich bin hier und das ist gut",
        "the man is here and that is good",
        "the man is here and that is good",
    )
    assert isinstance(result, bool)


def test_classify_eval_entry_no_error_returns_identity() -> None:
    assert classify_eval_entry({"has_error": False}) == "identity"


def test_classify_eval_entry_no_reference() -> None:
    assert classify_eval_entry({
        "has_error": True,
        "whisper_text": "w",
        "ground_truth": "gt",
        "reference": None,
    }) == "no_reference"


def test_classify_eval_entry_empty_reference() -> None:
    assert classify_eval_entry({
        "has_error": True,
        "whisper_text": "w",
        "ground_truth": "gt",
        "reference": "   ",
    }) == "no_reference"


def test_classify_eval_entry_lyrics() -> None:
    assert classify_eval_entry({
        "has_error": True,
        "whisper_text": "♪ song",
        "ground_truth": "♪ song",
        "reference": "♪ song",
    }) == "lyrics"


def test_classify_eval_entry_in_scope() -> None:
    assert classify_eval_entry({
        "has_error": True,
        "whisper_text": "the cat sat",
        "ground_truth": "the cat ran",
        "reference": "the cat ran",
    }) == "in_scope"


def test_classify_eval_entry_misaligned() -> None:
    # Whisper shares nothing with ground truth/reference, gt != ref
    assert classify_eval_entry({
        "has_error": True,
        "whisper_text": "alpha bravo charlie",
        "ground_truth": "banana split sundae",
        "reference": "computer keyboard mouse",
    }) == "misaligned"


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ({"has_error": False}, "identity"),
        ({"has_error": True, "whisper_text": "w", "ground_truth": "g", "reference": None}, "no_reference"),
        (
            {
                "has_error": True,
                "whisper_text": "♪",
                "ground_truth": "♪",
                "reference": "♪",
            },
            "lyrics",
        ),
    ],
)
def test_classify_eval_entry_parametrized(entry: dict, expected: str) -> None:
    assert classify_eval_entry(entry) == expected


def test_classify_eval_entry_matches_dataset_labels() -> None:
    dataset = Path(__file__).resolve().parents[1] / "evaluation" / "dataset.jsonl"
    if not dataset.exists():
        return
    mismatches = []
    with dataset.open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            stored = entry.get("eval_scope")
            if not stored:
                continue
            if classify_eval_entry(entry) != stored:
                mismatches.append(entry["id"])
    assert mismatches == []
