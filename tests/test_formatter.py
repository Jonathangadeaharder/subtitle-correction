from __future__ import annotations

import pytest

from subtitle_correction.formatter import restore_formatting


def test_restore_formatting_preserves_reference_casing() -> None:
    ref = "Hello World"
    corrected = "hello world"
    out = restore_formatting(corrected, ref)
    assert out == "Hello World"


def test_restore_formatting_preserves_punctuation() -> None:
    ref = "Hello, World!"
    corrected = "hello world"
    out = restore_formatting(corrected, ref)
    assert "Hello" in out
    assert "," in out
    assert "!" in out


def test_restore_formatting_restores_line_breaks() -> None:
    ref = "Line one\nLine two"
    corrected = "line one line two"
    out = restore_formatting(corrected, ref)
    assert "\n" in out
    assert "Line one" in out
    assert "Line two" in out


def test_restore_formatting_restores_music_symbols() -> None:
    ref = "♪ some song ♪"
    corrected = "some song"
    out = restore_formatting(corrected, ref)
    assert out.startswith("♪")
    assert out.endswith("♪")


def test_restore_formatting_empty_corrected_returns_input() -> None:
    out = restore_formatting("", "anything")
    assert out == ""


def test_restore_formatting_empty_reference_adds_music_only() -> None:
    out = restore_formatting("♪hello♪", "")
    # corrected cleaned to "hello", no ref words -> music branch
    assert "♪" in out
    assert "hello" in out


def test_restore_formatting_music_only_reference_with_empty_corrected() -> None:
    # corrected with no words after stripping music
    out = restore_formatting("♪", "♪ ref ♪")
    assert "♪" in out


def test_restore_formatting_handles_word_mismatch_lengths() -> None:
    # Corrected longer than reference -> proportional line scaling
    ref = "one two three\nfour five"
    corrected = "one two three four five six"
    out = restore_formatting(corrected, ref)
    assert "\n" in out
    assert "six" in out


def test_restore_formatting_preserves_capitalization_on_match() -> None:
    ref = "Berlin is nice"
    corrected = "berlin is nice"
    out = restore_formatting(corrected, ref)
    assert "Berlin" in out


def test_restore_formatting_no_match_keeps_word() -> None:
    ref = "alpha beta"
    corrected = "gamma delta"
    out = restore_formatting(corrected, ref)
    # No matching words; output still reconstructs lines from ref structure
    assert "gamma" in out
    assert "delta" in out


def test_restore_formatting_trailing_punc_on_last_word() -> None:
    ref = "Hello world."
    corrected = "hello world"
    out = restore_formatting(corrected, ref)
    # Last word should pick up trailing period from reference
    assert out.endswith(".")


def test_restore_formatting_leading_punctuation_preserved() -> None:
    ref = '"Hello world'
    corrected = "hello world"
    out = restore_formatting(corrected, ref)
    assert out.startswith('"')


@pytest.mark.parametrize(
    ("ref", "corrected"),
    [
        ("Hello World", "hello world"),
        ("♪ song ♪", "song"),
        ("Line1\nLine2", "line1 line2"),
        ("a", "a"),
        ("", ""),
    ],
)
def test_restore_formatting_various_inputs(ref: str, corrected: str) -> None:
    out = restore_formatting(corrected, ref)
    assert isinstance(out, str)
