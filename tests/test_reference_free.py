from __future__ import annotations

from pathlib import Path

import pytest

from subtitle_correction.reference_free import (
    _strip_model_noise,
    build_prompt,
    write_preserving_timing,
)


def test_build_prompt_marks_current_cue() -> None:
    prompt = build_prompt(["prev"], "current", ["next"])
    assert "[AKTUELL]" in prompt
    assert "current" in prompt
    assert "[VORHER]" in prompt
    assert "[NACHHER]" in prompt


@pytest.mark.parametrize(
    ("raw", "original", "expected"),
    [
        ("<think>noise</think>Korrektur: Hallo", "Helo", "Hallo"),
        ("<|channel>final<|message|>Antwort: Guten Tag<|im_end|>", "Guten", "Guten Tag"),
        ('"Guten Morgen"', "Guten Morgen", "Guten Morgen"),
    ],
)
def test_strip_model_noise(raw: str, original: str, expected: str) -> None:
    assert _strip_model_noise(raw, original) == expected


def test_write_preserving_timing_only_replaces_text(tmp_path: Path) -> None:
    src = tmp_path / "in.srt"
    out = tmp_path / "out.srt"
    src.write_text("1\n00:00:01,000 --> 00:00:02,000\nold text\n", encoding="utf-8")
    write_preserving_timing(src, out, ["new text"])
    assert "00:00:01,000 --> 00:00:02,000" in out.read_text(encoding="utf-8")
    assert "new text" in out.read_text(encoding="utf-8")


def test_write_preserving_timing_rejects_count_mismatch(tmp_path: Path) -> None:
    src = tmp_path / "in.srt"
    out = tmp_path / "out.srt"
    src.write_text("1\n00:00:01,000 --> 00:00:02,000\nold\n", encoding="utf-8")
    with pytest.raises(ValueError, match="block count"):
        write_preserving_timing(src, out, ["a", "b"])