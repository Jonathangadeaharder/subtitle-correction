from __future__ import annotations

import json
from pathlib import Path

import pytest

from subtitle_correction.hallucination_filter import (
    COMMON_HALLUCINATION_PHRASES,
    FilterStats,
    SrtBlock,
    _normalize_text,
    _overlaps_speech,
    filter_by_confidence,
    filter_by_vad,
    filter_hallucination_phrases,
    filter_hallucinations,
    filter_repetitions,
    parse_srt,
    parse_srt_with_confidence,
    run_vad,
    write_srt,
)


def _block(index: int, start: float, end: float, text: str) -> SrtBlock:
    return SrtBlock(index=index, start_sec=start, end_sec=end, text=text)


def test_srt_block_duration() -> None:
    b = _block(1, 1.0, 3.5, "x")
    assert b.duration == pytest.approx(2.5)


def test_parse_srt(tmp_path: Path) -> None:
    srt = tmp_path / "in.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,500\nhello world\n\n2\n00:00:02,500 --> 00:00:04,000\nbye\n",
        encoding="utf-8",
    )
    blocks = parse_srt(srt)
    assert len(blocks) == 2
    assert blocks[0].index == 1
    assert blocks[0].start_sec == pytest.approx(1.0)
    assert blocks[0].end_sec == pytest.approx(2.5)
    assert blocks[0].text == "hello world"
    assert blocks[1].index == 2


def test_parse_srt_with_confidence_no_json(tmp_path: Path) -> None:
    srt = tmp_path / "in.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n", encoding="utf-8")
    json_path = tmp_path / "missing.json"
    blocks = parse_srt_with_confidence(srt, json_path)
    assert len(blocks) == 1
    assert blocks[0].no_speech_prob == 0.0
    assert blocks[0].avg_logprob == 0.0
    assert blocks[0].compression_ratio == 1.0


def test_parse_srt_with_confidence_merges(tmp_path: Path) -> None:
    srt = tmp_path / "in.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nhello\n\n2\n00:00:02,000 --> 00:00:03,000\nworld\n",
        encoding="utf-8",
    )
    json_path = tmp_path / "conf.json"
    json_path.write_text(
        json.dumps(
            {
                "segments": [
                    {"no_speech_prob": 0.9, "avg_logprob": -1.5, "compression_ratio": 3.0},
                    {"no_speech_prob": 0.1, "avg_logprob": -0.3, "compression_ratio": 1.2},
                ]
            }
        ),
        encoding="utf-8",
    )
    blocks = parse_srt_with_confidence(srt, json_path)
    assert blocks[0].no_speech_prob == pytest.approx(0.9)
    assert blocks[1].avg_logprob == pytest.approx(-0.3)
    assert blocks[1].compression_ratio == pytest.approx(1.2)


def test_write_srt_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "out.srt"
    blocks = [_block(1, 1.0, 2.5, "hello"), _block(2, 2.5, 4.0, "world")]
    write_srt(blocks, out)
    text = out.read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:02,500" in text
    assert "hello" in text
    assert "world" in text


def test_normalize_text() -> None:
    assert _normalize_text("Hello, World!") == "hello world"


def test_overlaps_speech_empty_segments_returns_true() -> None:
    assert _overlaps_speech(_block(1, 0.0, 1.0, "x"), []) is True


def test_overlaps_speech_zero_duration_returns_true() -> None:
    assert _overlaps_speech(_block(1, 1.0, 1.0, "x"), [{"start": 0, "end": 2}]) is True


def test_overlaps_speech_overlap_true() -> None:
    b = _block(1, 1.0, 2.0, "x")
    segs = [{"start": 1.5, "end": 2.5}]
    assert _overlaps_speech(b, segs) is True


def test_overlaps_speech_no_overlap_false() -> None:
    b = _block(1, 1.0, 2.0, "x")
    segs = [{"start": 5.0, "end": 6.0}]
    assert _overlaps_speech(b, segs) is False


def test_filter_by_vad_keeps_overlapping() -> None:
    blocks = [_block(1, 1.0, 2.0, "hi"), _block(2, 10.0, 11.0, "no speech")]
    segs = [{"start": 0.9, "end": 2.1}]
    stats = FilterStats()
    kept = filter_by_vad(blocks, segs, stats)
    assert kept == [blocks[0]]
    assert stats.dropped_vad == 1
    assert len(stats.reasons) == 1


def test_filter_by_confidence_drops_high_no_speech() -> None:
    blocks = [
        SrtBlock(index=1, start_sec=0, end_sec=1, text="x", no_speech_prob=0.9),
        SrtBlock(index=2, start_sec=1, end_sec=2, text="y", no_speech_prob=0.1),
    ]
    stats = FilterStats()
    kept = filter_by_confidence(blocks, stats)
    assert kept == [blocks[1]]
    assert stats.dropped_confidence == 1


def test_filter_by_confidence_keeps_low_no_speech_even_with_other_reasons() -> None:
    # no_speech below threshold but bad logprob/compression -> still kept
    # (the drop only triggers when no_speech > threshold)
    blocks = [
        SrtBlock(
            index=1,
            start_sec=0,
            end_sec=1,
            text="x",
            no_speech_prob=0.1,
            avg_logprob=-2.0,
            compression_ratio=3.0,
        ),
    ]
    stats = FilterStats()
    kept = filter_by_confidence(blocks, stats)
    assert kept == [blocks[0]]
    assert stats.dropped_confidence == 0


def test_filter_repetitions_drops_runs(tmp_path: Path) -> None:
    blocks = [_block(i, i, i + 1, "same") for i in range(1, 5)]
    stats = FilterStats()
    kept = filter_repetitions(blocks, stats, repeat_count=3)
    assert kept == []
    assert stats.dropped_repetition == 4


def test_filter_repetitions_keeps_when_below_count() -> None:
    blocks = [_block(1, 0, 1, "same"), _block(2, 1, 2, "same")]
    stats = FilterStats()
    kept = filter_repetitions(blocks, stats, repeat_count=3)
    assert kept == blocks
    assert stats.dropped_repetition == 0


def test_filter_repetitions_skips_empty_normalized() -> None:
    # Normalized to empty string -> not treated as a run
    blocks = [_block(1, 0, 1, "!!!"), _block(2, 1, 2, "!!!")]
    stats = FilterStats()
    kept = filter_repetitions(blocks, stats, repeat_count=2)
    assert kept == blocks


def test_filter_hallucination_phrases_drops_known(tmp_path: Path) -> None:
    blocks = [_block(1, 0, 1, "thank you for watching"), _block(2, 1, 2, "real text")]
    stats = FilterStats()
    kept = filter_hallucination_phrases(blocks, stats, lang="en")
    assert kept == [blocks[1]]
    assert stats.dropped_hallucination_phrase == 1


def test_filter_hallucination_phrases_german() -> None:
    blocks = [_block(1, 0, 1, "vielen dank")]
    stats = FilterStats()
    kept = filter_hallucination_phrases(blocks, stats, lang="de")
    assert kept == []
    assert stats.dropped_hallucination_phrase == 1


def test_filter_hallucination_phrases_falls_back_to_en() -> None:
    blocks = [_block(1, 0, 1, "thank you")]
    stats = FilterStats()
    kept = filter_hallucination_phrases(blocks, stats, lang="zz")
    assert kept == []


def test_common_hallucination_phrases_has_en_and_de() -> None:
    assert "en" in COMMON_HALLUCINATION_PHRASES
    assert "de" in COMMON_HALLUCINATION_PHRASES


def test_run_vad_loads_and_calls(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"\x00")

    captured: dict = {}

    class _FakeModel:
        def get_speech_timestamps(self, path: str, **kwargs: object) -> list[dict]:
            captured["path"] = path
            captured["kwargs"] = kwargs
            return [{"start": 0.1, "end": 0.9}]

    import sys
    import types

    fake_mlx = types.ModuleType("mlx_audio")
    fake_vad = types.ModuleType("mlx_audio.vad")

    def _fake_load(name: str) -> _FakeModel:
        captured["model_name"] = name
        return _FakeModel()

    fake_vad.load = _fake_load
    fake_mlx.vad = fake_vad
    orig = sys.modules.get("mlx_audio")
    sys.modules["mlx_audio"] = fake_mlx
    sys.modules["mlx_audio.vad"] = fake_vad
    try:
        result = run_vad(audio, threshold=0.4)
    finally:
        if orig is not None:
            sys.modules["mlx_audio"] = orig
        else:
            del sys.modules["mlx_audio"]
            del sys.modules["mlx_audio.vad"]

    assert result == [{"start": 0.1, "end": 0.9}]
    assert captured["model_name"] == "mlx-community/silero-vad"
    assert captured["path"] == str(audio)


def test_filter_hallucinations_empty_blocks(tmp_path: Path) -> None:
    srt = tmp_path / "in.srt"
    srt.write_text("", encoding="utf-8")
    out = tmp_path / "out.srt"
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"\x00")
    stats = filter_hallucinations(srt, audio, out)
    assert stats.total == 0
    assert stats.kept == 0


def test_filter_hallucinations_full_pipeline(tmp_path: Path) -> None:
    srt = tmp_path / "in.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nthank you for watching\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nreal subtitle text\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.srt"
    audio = tmp_path / "a.wav"  # non-existent -> VAD branch skipped
    stats = filter_hallucinations(
        srt,
        audio,
        out,
        lang="en",
        enable_vad=False,
        enable_confidence=False,
    )
    assert stats.total == 2
    assert stats.kept == 1  # the "thank you" phrase dropped
    assert out.exists()
    assert "real subtitle text" in out.read_text(encoding="utf-8")


def test_filter_hallucinations_with_confidence_json(tmp_path: Path) -> None:
    srt = tmp_path / "in.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\ngood text\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nbad text\n",
        encoding="utf-8",
    )
    json_path = tmp_path / "conf.json"
    json_path.write_text(
        json.dumps(
            {
                "segments": [
                    {"no_speech_prob": 0.1, "avg_logprob": -0.2, "compression_ratio": 1.0},
                    {"no_speech_prob": 0.9, "avg_logprob": -2.0, "compression_ratio": 3.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out.srt"
    stats = filter_hallucinations(
        srt,
        tmp_path / "noaudio.wav",
        out,
        json_path=json_path,
        lang="en",
        enable_vad=False,
        enable_confidence=True,
    )
    assert stats.dropped_confidence == 1
    assert stats.kept == 1
