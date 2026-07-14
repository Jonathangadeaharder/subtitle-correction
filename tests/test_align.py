from __future__ import annotations

from pathlib import Path

import pytest

from subtitle_correction.align import (
    _is_watermark,
    compute_alignment_score,
    detect_srt_language,
    extract_language_from_filename,
)


def _write_srt(path: Path, blocks: list[tuple[int, str, str]]) -> None:
    """blocks: list of (index, 'MM:SS,mmm --> MM:SS,mmm', text)."""
    parts = []
    for idx, timing, text in blocks:
        parts.append(f"{idx}\n{timing}\n{text}")
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def _srt_block(index: int, start_sec: float, end_sec: float, text: str) -> str:
    def fmt(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

    return f"{index}\n{fmt(start_sec)} --> {fmt(end_sec)}\n{text}"


def test_is_watermark_matches_known_patterns() -> None:
    assert _is_watermark("Untertitelung von ZDF") is True
    assert _is_watermark("Subtitle") is True
    assert _is_watermark("normal dialogue") is False


def test_extract_language_from_filename_known_codes() -> None:
    assert extract_language_from_filename(Path("Movie.2020.en.srt")) == "en"
    assert extract_language_from_filename(Path("Movie.2020.de.srt")) == "de"
    assert extract_language_from_filename(Path("Movie.2020.fr.srt")) == "fr"


def test_extract_language_from_filename_unknown_code_returns_none() -> None:
    assert extract_language_from_filename(Path("Movie.2020.xx.srt")) is None
    assert extract_language_from_filename(Path("Movie.2020.srt")) is None


def test_detect_srt_language(tmp_path: Path) -> None:
    srt = tmp_path / "in.srt"
    _write_srt(
        srt,
        [
            (1, "00:00:01,000 --> 00:00:02,000", "Hello world this is english text"),
            (2, "00:00:02,000 --> 00:00:03,000", "you are the one who has been here"),
        ],
    )
    lang = detect_srt_language(srt)
    assert isinstance(lang, str)


def test_detect_srt_language_too_short_returns_und(tmp_path: Path) -> None:
    srt = tmp_path / "short.srt"
    _write_srt(srt, [(1, "00:00:01,000 --> 00:00:02,000", "hi")])
    assert detect_srt_language(srt) == "und"


def test_compute_alignment_score_perfect_match(tmp_path: Path) -> None:
    whisper = tmp_path / "w.srt"
    aligned = tmp_path / "a.srt"
    _write_srt(
        whisper,
        [
            (1, "00:00:01,000 --> 00:00:02,000", "a"),
            (2, "00:00:02,000 --> 00:00:03,000", "b"),
        ],
    )
    _write_srt(
        aligned,
        [
            (1, "00:00:01,000 --> 00:00:02,000", "a"),
            (2, "00:00:02,000 --> 00:00:03,000", "b"),
        ],
    )
    assert compute_alignment_score(whisper, aligned) == 1.0


def test_compute_alignment_score_empty_returns_zero(tmp_path: Path) -> None:
    whisper = tmp_path / "w.srt"
    aligned = tmp_path / "a.srt"
    whisper.write_text("", encoding="utf-8")
    aligned.write_text("", encoding="utf-8")
    assert compute_alignment_score(whisper, aligned) == 0.0


def test_compute_alignment_score_no_positive_starts(tmp_path: Path) -> None:
    whisper = tmp_path / "w.srt"
    aligned = tmp_path / "a.srt"
    # start at 0 -> filtered out by `start.ordinal > 0`
    _write_srt(whisper, [(1, "00:00:00,000 --> 00:00:01,000", "x")])
    _write_srt(aligned, [(1, "00:00:01,000 --> 00:00:02,000", "y")])
    assert compute_alignment_score(whisper, aligned) == 0.0


def test_align_with_alass_missing_binary_raises(tmp_path: Path) -> None:
    from subtitle_correction.align import align_with_alass

    ref = tmp_path / "ref.srt"
    bad = tmp_path / "bad.srt"
    out = tmp_path / "out.srt"
    ref.write_text("1\n00:00:01,000 --> 00:00:02,000\nx\n", encoding="utf-8")
    bad.write_text("1\n00:00:01,000 --> 00:00:02,000\ny\n", encoding="utf-8")

    # Force which() to find nothing.
    import subtitle_correction.align as align_mod

    original_which = align_mod.shutil.which
    try:
        align_mod.shutil.which = lambda _name: None  # type: ignore[method-assign]
        with pytest.raises(FileNotFoundError, match="alass-cli not found"):
            align_with_alass(ref, bad, out)
    finally:
        align_mod.shutil.which = original_which  # type: ignore[method-assign]


def test_align_with_ffsubsync_failure_raises(tmp_path: Path) -> None:
    from subtitle_correction.align import align_with_ffsubsync

    video = tmp_path / "v.mp4"
    inp = tmp_path / "in.srt"
    out = tmp_path / "out.srt"
    video.write_text("x", encoding="utf-8")
    inp.write_text("1\n00:00:01,000 --> 00:00:02,000\nx\n", encoding="utf-8")

    with pytest.raises((RuntimeError, FileNotFoundError)):
        align_with_ffsubsync(video, inp, out)


def test_detect_srt_language_handles_exception(tmp_path: Path) -> None:
    # SRT with non-language-identifiable content that may trip langdetect
    srt = tmp_path / "x.srt"
    _write_srt(srt, [(1, "00:00:01,000 --> 00:00:02,000", "123 456 7890")])
    # Should not raise; returns a string
    assert isinstance(detect_srt_language(srt), str)
