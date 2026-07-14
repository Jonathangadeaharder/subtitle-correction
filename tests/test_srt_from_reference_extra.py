from __future__ import annotations

from pathlib import Path

import pytest

from subtitle_correction.srt_from_reference import (
    _assign_reference_to_cues,
    _self_check,
    write_preserving_timing,
    align_reference_to_srt,
    strip_markdown_prose,
    unified_diff,
)


def test_strip_markdown_prose_strips_headings() -> None:
    md = "# Title\n\nFirst line.\nSecond line."
    assert strip_markdown_prose(md) == "First line.\nSecond line."


def test_strip_markdown_prose_keeps_body_only_multiple_headings() -> None:
    md = "# A\n## B\nBody text"
    assert strip_markdown_prose(md) == "Body text"


def test_strip_markdown_prose_empty() -> None:
    assert strip_markdown_prose("") == ""
    assert strip_markdown_prose("# only heading") == ""


def test_write_preserving_timing_count_mismatch_raises(tmp_path: Path) -> None:
    src = tmp_path / "in.srt"
    out = tmp_path / "out.srt"
    src.write_text("1\n00:00:01,000 --> 00:00:02,000\nold\n", encoding="utf-8")
    with pytest.raises(ValueError, match="block count"):
        write_preserving_timing(src, out, ["a", "b"])


def test_write_preserving_timing_replaces_text(tmp_path: Path) -> None:
    src = tmp_path / "in.srt"
    out = tmp_path / "out.srt"
    src.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nold1\n\n2\n00:00:02,000 --> 00:00:03,000\nold2\n",
        encoding="utf-8",
    )
    write_preserving_timing(src, out, ["new1", "new2"])
    text = out.read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:02,000" in text
    assert "00:00:02,000 --> 00:00:03,000" in text
    assert "new1" in text
    assert "new2" in text
    assert "old1" not in text


def test_assign_reference_to_cues_basic() -> None:
    assigned = _assign_reference_to_cues(
        cue_token_owner=[0, 0, 1, 1, 2],
        whisper_norm=["a", "bee", "see", "dee", "ee"],
        ref_norm=["a", "b", "see", "dee", "extra", "gap", "ee"],
        n_cues=3,
    )
    assert assigned == [0, 0, 1, 1, 1, 1, 2]


def test_assign_reference_to_cues_replacement() -> None:
    assigned = _assign_reference_to_cues(
        cue_token_owner=[0, 1],
        whisper_norm=["gutes", "maus"],
        ref_norm=["gutes", "haus"],
        n_cues=2,
    )
    assert assigned == [0, 1]


def test_assign_reference_to_cues_insert_only() -> None:
    # All reference tokens are insertions relative to whisper
    assigned = _assign_reference_to_cues(
        cue_token_owner=[0],
        whisper_norm=["a"],
        ref_norm=["a", "b", "c"],
        n_cues=1,
    )
    assert assigned == [0, 0, 0]


def test_assign_reference_to_cues_delete_only() -> None:
    # Whisper has extra tokens deleted from reference
    assigned = _assign_reference_to_cues(
        cue_token_owner=[0, 0, 1],
        whisper_norm=["a", "extra", "b"],
        ref_norm=["a", "b"],
        n_cues=2,
    )
    assert assigned == [0, 1]


def test_assign_reference_to_cues_empty_reference() -> None:
    assigned = _assign_reference_to_cues(
        cue_token_owner=[0, 1],
        whisper_norm=["a", "b"],
        ref_norm=[],
        n_cues=2,
    )
    assert assigned == []


def test_assign_reference_monotonic_enforced() -> None:
    # Construct a case where alignment would produce non-monotonic assignment
    # to exercise the monotonic-fixup loop.
    assigned = _assign_reference_to_cues(
        cue_token_owner=[0, 1, 0],
        whisper_norm=["x", "y", "z"],
        ref_norm=["x", "y", "z"],
        n_cues=2,
    )
    # Result is always non-decreasing
    assert assigned == sorted(assigned)


def test_align_reference_to_srt_with_markdown_heading(tmp_path: Path) -> None:
    whisper = tmp_path / "w.srt"
    out = tmp_path / "out.srt"
    whisper.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nhello\n\n2\n00:00:02,000 --> 00:00:03,000\nworld\n",
        encoding="utf-8",
    )
    # Reference with a markdown heading that should be stripped
    align_reference_to_srt("# Chapter\nhello world there", whisper, out)
    text = out.read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:02,000" in text
    assert "00:00:02,000 --> 00:00:03,000" in text
    assert "hello" in text
    assert "there" in text


def test_unified_diff(tmp_path: Path) -> None:
    a = tmp_path / "a.srt"
    b = tmp_path / "b.srt"
    a.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n", encoding="utf-8")
    b.write_text("1\n00:00:01,000 --> 00:00:02,000\nworld\n", encoding="utf-8")
    diff = unified_diff(a, b)
    assert "hello" in diff
    assert "world" in diff


def test_unified_diff_accepts_strings(tmp_path: Path) -> None:
    a = tmp_path / "a.srt"
    b = tmp_path / "b.srt"
    a.write_text("x\n", encoding="utf-8")
    b.write_text("y\n", encoding="utf-8")
    assert isinstance(unified_diff(str(a), str(b)), str)


def test_self_check(capsys: pytest.CaptureFixture[str]) -> None:
    _self_check()
    captured = capsys.readouterr()
    assert "self-check OK" in captured.out


def test_main_self_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["srt_from_reference", "--self-check"])
    from subtitle_correction.srt_from_reference import _main

    _main()
    captured = capsys.readouterr()
    assert "self-check OK" in captured.out


def test_main_aligns_and_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    whisper = tmp_path / "w.srt"
    ref = tmp_path / "ref.md"
    out = tmp_path / "out.srt"
    whisper.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nhello\n\n2\n00:00:02,000 --> 00:00:03,000\nworld\n",
        encoding="utf-8",
    )
    ref.write_text("# Chapter\nhello world there", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "srt_from_reference",
            "--reference",
            str(ref),
            "--whisper-srt",
            str(whisper),
            "--out",
            str(out),
        ],
    )
    from subtitle_correction.srt_from_reference import _main

    _main()
    assert out.exists()
