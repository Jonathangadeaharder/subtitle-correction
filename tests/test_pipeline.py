from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from subtitle_correction.pipeline import (
    FileMetadata,
    PipelineStep,
    detect_language_from_filename,
    extract_audio,
    process_file,
    run_alignment,
    run_correction,
    run_hallucination_filter,
    run_opensubtitles_download,
    run_whisper,
    safe_dirname,
)


# ---------------- safe_dirname ----------------


def test_safe_dirname_strips_special_chars() -> None:
    # Uses Path.stem -> extension dropped
    assert safe_dirname("Movie: A Title (2020).mp4") == "Movie_A_Title_2020"


def test_safe_dirname_collapses_underscores() -> None:
    assert safe_dirname("a___b.mp4") == "a_b"


def test_safe_dirname_truncates_to_100() -> None:
    name = "x" * 200 + ".mp4"
    assert len(safe_dirname(name)) <= 100


def test_safe_dirname_strips_leading_trailing_underscores() -> None:
    assert safe_dirname("___name___.mp4") == "name"


# ---------------- detect_language_from_filename ----------------


def test_detect_language_german_keywords() -> None:
    assert detect_language_from_filename("Movie.GERMAN.dub.mp4") == "de"
    assert detect_language_from_filename("Film.Deutsch.mp4") == "de"
    assert detect_language_from_filename("Show.germandub.mp4") == "de"


def test_detect_language_english_keywords() -> None:
    assert detect_language_from_filename("Movie.english.mp4") == "en"
    assert detect_language_from_filename("Show.Englisch.mp4") == "en"


def test_detect_language_returns_none_when_unknown() -> None:
    assert detect_language_from_filename("Movie.2020.mp4") is None


# ---------------- FileMetadata ----------------


def test_file_metadata_load_creates_default_when_missing(tmp_path: Path) -> None:
    meta = FileMetadata(tmp_path / "work", "movie.mp4")
    meta.load()
    assert meta.status == PipelineStep.PENDING
    assert meta.whisper_lang == ""
    assert meta.subtitle_lang == ""
    assert meta.alignment_score == 0.0
    assert meta.error == ""


def test_file_metadata_load_existing(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "metadata.json").write_text(
        json.dumps(
            {
                "status": PipelineStep.ALIGNED,
                "whisper_lang": "en",
                "subtitle_lang": "de",
                "alignment_score": 0.8,
                "error": "",
            }
        ),
        encoding="utf-8",
    )
    meta = FileMetadata(work, "movie.mp4")
    meta.load()
    assert meta.status == PipelineStep.ALIGNED
    assert meta.whisper_lang == "en"
    assert meta.subtitle_lang == "de"
    assert meta.alignment_score == 0.8


def test_file_metadata_setters_persist(tmp_path: Path) -> None:
    meta = FileMetadata(tmp_path / "work", "movie.mp4")
    meta.load()
    # Set status LAST (only the status setter triggers save())
    meta.whisper_lang = "en"
    meta.subtitle_lang = "de"
    meta.alignment_score = 0.5
    meta.error = "boom"
    meta.status = PipelineStep.AUDIO_EXTRACTED  # persists everything
    # Reload from disk
    meta2 = FileMetadata(tmp_path / "work", "movie.mp4")
    meta2.load()
    assert meta2.status == PipelineStep.AUDIO_EXTRACTED
    assert meta2.whisper_lang == "en"
    assert meta2.subtitle_lang == "de"
    assert meta2.alignment_score == 0.5
    assert meta2.error == "boom"


def test_is_complete_aligned(tmp_path: Path) -> None:
    meta = FileMetadata(tmp_path / "w", "m.mp4")
    meta.load()
    assert meta.is_complete() is False
    meta.status = PipelineStep.ALIGNED
    assert meta.is_complete() is True
    meta.status = PipelineStep.FAILED
    assert meta.is_complete() is False


def test_needs_step_orders(tmp_path: Path) -> None:
    meta = FileMetadata(tmp_path / "w", "m.mp4")
    meta.load()
    assert meta.needs_step(PipelineStep.AUDIO_EXTRACTED) is True
    assert meta.needs_step(PipelineStep.PENDING) is False
    meta.status = PipelineStep.WHISPER_DONE
    assert meta.needs_step(PipelineStep.SUBTITLE_DOWNLOADED) is True
    assert meta.needs_step(PipelineStep.AUDIO_EXTRACTED) is False


def test_needs_step_unknown_status(tmp_path: Path) -> None:
    meta = FileMetadata(tmp_path / "w", "m.mp4")
    meta.load()
    meta._data["status"] = "garbage"
    assert meta.needs_step(PipelineStep.AUDIO_EXTRACTED) is True


# ---------------- extract_audio ----------------


def test_extract_audio_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mp4 = tmp_path / "in.mp4"
    mp4.write_bytes(b"\x00")
    wav = tmp_path / "out.wav"

    def _fake_run(cmd, **kwargs):
        wav.write_bytes(b"\x00" * 10)

        class _R:
            returncode = 0
            stderr = ""

        return _R()

    import subtitle_correction.pipeline as pl

    monkeypatch.setattr(pl.subprocess, "run", _fake_run, raising=False)
    assert extract_audio(mp4, wav) == wav


def test_extract_audio_failure_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subtitle_correction.pipeline as pl

    def _fake_run(cmd, **kwargs):
        class _R:
            returncode = 1
            stderr = "boom"

        return _R()

    monkeypatch.setattr(pl.subprocess, "run", _fake_run, raising=False)
    with pytest.raises(RuntimeError, match="ffmpeg audio extraction failed"):
        extract_audio(tmp_path / "in.mp4", tmp_path / "out.wav")


# ---------------- run_correction ----------------


def test_run_correction_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subtitle_correction.pipeline as pl

    inp = tmp_path / "in.srt"
    ref = tmp_path / "ref.srt"
    out = tmp_path / "out.srt"
    inp.write_text("x", encoding="utf-8")
    ref.write_text("x", encoding="utf-8")

    def _fake_run(cmd, **kwargs):
        out.write_text("corrected", encoding="utf-8")

        class _R:
            returncode = 0
            stderr = ""

        return _R()

    monkeypatch.setattr(pl.subprocess, "run", _fake_run, raising=False)
    assert run_correction(inp, ref, out, fused=True) == out


def test_run_correction_failure_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subtitle_correction.pipeline as pl

    def _fake_run(cmd, **kwargs):
        class _R:
            returncode = 1
            stderr = "correction failed"

        return _R()

    monkeypatch.setattr(pl.subprocess, "run", _fake_run, raising=False)
    with pytest.raises(RuntimeError, match="Subtitle correction failed"):
        run_correction(tmp_path / "in.srt", tmp_path / "ref.srt", tmp_path / "out.srt")


def test_run_correction_adapter_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subtitle_correction.pipeline as pl

    captured: dict = {}
    inp = tmp_path / "in.srt"
    ref = tmp_path / "ref.srt"
    out = tmp_path / "out.srt"
    inp.write_text("x", encoding="utf-8")
    ref.write_text("x", encoding="utf-8")

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        out.write_text("ok", encoding="utf-8")

        class _R:
            returncode = 0
            stderr = ""

        return _R()

    monkeypatch.setattr(pl.subprocess, "run", _fake_run, raising=False)
    run_correction(inp, ref, out, fused=False)
    assert "--fused" not in captured["cmd"]


# ---------------- run_alignment ----------------


def test_run_alignment_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subtitle_correction.pipeline as pl

    whisper = tmp_path / "w.srt"
    sub = tmp_path / "s.srt"
    out = tmp_path / "o.srt"
    whisper.write_text("x", encoding="utf-8")
    sub.write_text("x", encoding="utf-8")

    def _fake_run(cmd, **kwargs):
        out.write_text("aligned", encoding="utf-8")

        class _R:
            returncode = 0
            stderr = ""

        return _R()

    monkeypatch.setattr(pl.subprocess, "run", _fake_run, raising=False)
    assert run_alignment(whisper, sub, out) == out


def test_run_alignment_failure_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subtitle_correction.pipeline as pl

    def _fake_run(cmd, **kwargs):
        class _R:
            returncode = 1
            stderr = "align failed"

        return _R()

    monkeypatch.setattr(pl.subprocess, "run", _fake_run, raising=False)
    with pytest.raises(RuntimeError, match="subtitle-align failed"):
        run_alignment(tmp_path / "w.srt", tmp_path / "s.srt", tmp_path / "o.srt")


# ---------------- run_whisper ----------------


def test_run_whisper_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"\x00")
    srt = tmp_path / "out.srt"
    jsn = tmp_path / "out.json"

    fake_mlx_audio = types.ModuleType("mlx_audio")
    fake_stt = types.ModuleType("mlx_audio.stt")
    fake_generate = types.ModuleType("mlx_audio.stt.generate")

    class _Seg:
        text = "hello"
        segments = [
            {
                "text": "hi",
                "start": 0,
                "end": 1,
                "no_speech_prob": 0.1,
                "avg_logprob": -0.2,
                "compression_ratio": 1.0,
            }
        ]

    def _fake_generate_transcription(**kwargs):
        # Write the srt so output_srt.exists() is True
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
        return _Seg()

    fake_generate.generate_transcription = _fake_generate_transcription
    fake_stt.generate = fake_generate
    fake_mlx_audio.stt = fake_stt
    monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", fake_stt)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.generate", fake_generate)

    result = run_whisper(wav, srt, output_json=jsn, language="en")
    assert result == srt
    assert jsn.exists()
    data = json.loads(jsn.read_text(encoding="utf-8"))
    assert data["text"] == "hello"
    assert data["segments"][0]["no_speech_prob"] == 0.1


def test_run_whisper_no_srt_produced_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"\x00")
    srt = tmp_path / "out.srt"

    fake_mlx_audio = types.ModuleType("mlx_audio")
    fake_stt = types.ModuleType("mlx_audio.stt")
    fake_generate = types.ModuleType("mlx_audio.stt.generate")

    def _fake_generate_transcription(**kwargs):
        return type("R", (), {"text": "", "segments": []})()

    fake_generate.generate_transcription = _fake_generate_transcription
    fake_stt.generate = fake_generate
    fake_mlx_audio.stt = fake_stt
    monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", fake_stt)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.generate", fake_generate)

    with pytest.raises(FileNotFoundError, match="Whisper did not produce SRT"):
        run_whisper(wav, srt)


def test_run_whisper_moves_globbed_srt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"\x00")
    srt = tmp_path / "out.srt"
    # Simulate whisper writing a differently-named srt in the same dir
    globbed = tmp_path / "out_extra.srt"
    globbed.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")

    fake_mlx_audio = types.ModuleType("mlx_audio")
    fake_stt = types.ModuleType("mlx_audio.stt")
    fake_generate = types.ModuleType("mlx_audio.stt.generate")

    def _fake_generate_transcription(**kwargs):
        return type("R", (), {"text": "", "segments": []})()

    fake_generate.generate_transcription = _fake_generate_transcription
    fake_stt.generate = fake_generate
    fake_mlx_audio.stt = fake_stt
    monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", fake_stt)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.generate", fake_generate)

    result = run_whisper(wav, srt)
    assert result == srt
    assert srt.exists()


def test_run_whisper_no_segments_no_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"\x00")
    srt = tmp_path / "out.srt"
    jsn = tmp_path / "out.json"

    fake_mlx_audio = types.ModuleType("mlx_audio")
    fake_stt = types.ModuleType("mlx_audio.stt")
    fake_generate = types.ModuleType("mlx_audio.stt.generate")

    class _Seg:
        text = "hello"
        segments = []  # empty -> no json written

    def _fake_generate_transcription(**kwargs):
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
        return _Seg()

    fake_generate.generate_transcription = _fake_generate_transcription
    fake_stt.generate = fake_generate
    fake_mlx_audio.stt = fake_stt
    monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", fake_stt)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.generate", fake_generate)

    run_whisper(wav, srt, output_json=jsn, language="en")
    assert not jsn.exists()


# ---------------- run_hallucination_filter ----------------


def test_run_hallucination_filter(tmp_path: Path) -> None:
    srt = tmp_path / "in.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nthank you for watching\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nreal text\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.srt"
    audio = tmp_path / "a.wav"  # non-existent -> VAD skipped
    result = run_hallucination_filter(srt, audio, out, lang="en", enable_vad=False)
    assert result == out
    assert "real text" in out.read_text(encoding="utf-8")


# ---------------- run_opensubtitles_download ----------------


class _FakeScraper:
    def __init__(self, results: list, download_path: Path | None = None):
        self._results = results
        self._download_path = download_path
        self.searched: list = []
        self.downloaded: list = []
        self.closed = False

    def search(self, query, language):
        self.searched.append((query, language))
        return self._results

    def download(self, sub, output_dir):
        self.downloaded.append((sub, output_dir))
        if self._download_path is not None:
            self._download_path.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nhi\n", encoding="utf-8"
            )
            return self._download_path
        return output_dir / "failed.srt"

    def close(self):
        self.closed = True


def test_run_opensubtitles_download_success(tmp_path: Path) -> None:
    from subtitle_correction.models import SubtitleResult

    dl = tmp_path / "dl.srt"
    scraper = _FakeScraper(
        results=[SubtitleResult(id="1", name="Movie", language="en")],
        download_path=dl,
    )
    out = run_opensubtitles_download("Movie.2020.mp4", "en", tmp_path, scraper=scraper)
    assert out == dl
    assert scraper.closed is False  # external scraper not closed


def test_run_opensubtitles_download_no_results(tmp_path: Path) -> None:
    scraper = _FakeScraper(results=[])
    out = run_opensubtitles_download("Movie.2020.mp4", "en", tmp_path, scraper=scraper)
    assert out is None


def test_run_opensubtitles_download_empty_file_returns_none(tmp_path: Path) -> None:
    from subtitle_correction.models import SubtitleResult

    scraper = _FakeScraper(
        results=[SubtitleResult(id="1", name="Movie", language="en")],
        download_path=None,  # returns failed.srt (empty)
    )
    out = run_opensubtitles_download("Movie.2020.mp4", "en", tmp_path, scraper=scraper)
    assert out is None


def test_run_opensubtitles_download_exception_returns_none(tmp_path: Path) -> None:
    class _Boom:
        def search(self, *a, **k):
            raise RuntimeError("network down")

        def close(self):
            pass

    out = run_opensubtitles_download("Movie.2020.mp4", "en", tmp_path, scraper=_Boom())
    assert out is None


def test_run_opensubtitles_download_creates_own_scraper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # When scraper=None, an OpenSubtitlesScraper is constructed; mock its class
    from subtitle_correction.models import SubtitleResult

    dl = tmp_path / "own.srt"
    fake = _FakeScraper(
        results=[SubtitleResult(id="1", name="Movie", language="en")],
        download_path=dl,
    )
    monkeypatch.setattr("subtitle_correction.pipeline.OpenSubtitlesScraper", lambda *a, **k: fake)
    out = run_opensubtitles_download("Movie.2020.mp4", "en", tmp_path, scraper=None)
    assert out == dl
    assert fake.closed is True  # own scraper closed in finally


def test_run_opensubtitles_download_filters_and_picks_title_match(tmp_path: Path) -> None:
    from subtitle_correction.models import SubtitleResult

    dl = tmp_path / "best.srt"
    results = [
        SubtitleResult(id="1", name="Other Title", language="en"),
        SubtitleResult(id="2", name="Movie Real", language="en"),  # title match
    ]
    scraper = _FakeScraper(results=results, download_path=dl)
    out = run_opensubtitles_download("Movie.2020.mp4", "en", tmp_path, scraper=scraper)
    assert out == dl
    # The best (title-matching) result was downloaded
    assert scraper.downloaded[0][0].id == "2"


# ---------------- process_file ----------------


def test_process_file_skips_complete(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    work = cache / safe_dirname("movie.mp4")
    work.mkdir(parents=True)
    (work / "metadata.json").write_text(
        json.dumps(
            {
                "status": PipelineStep.ALIGNED,
                "source_file": "movie.mp4",
            }
        ),
        encoding="utf-8",
    )
    mp4 = tmp_path / "movie.mp4"
    mp4.write_bytes(b"\x00")
    result = process_file(mp4, cache, skip_existing=True)
    assert result["status"] == "skipped"


def test_process_file_no_subtitle_returns_no_subtitle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    mp4 = tmp_path / "movie.GERMAN.mp4"
    mp4.write_bytes(b"\x00")

    # Mock extract_audio, run_whisper, run_hallucination_filter, run_opensubtitles_download
    import subtitle_correction.pipeline as pl

    def _fake_extract_audio(mp4_path, out_wav):
        out_wav.write_bytes(b"\x00")
        return out_wav

    def _fake_whisper(wav, srt, output_json=None, **kwargs):
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
        if output_json:
            output_json.write_text(json.dumps({"segments": []}), encoding="utf-8")
        return srt

    def _fake_no_download(*a, **k):
        return None

    monkeypatch.setattr(pl, "extract_audio", _fake_extract_audio, raising=False)
    monkeypatch.setattr(pl, "run_whisper", _fake_whisper, raising=False)
    monkeypatch.setattr(pl, "run_hallucination_filter", lambda *a, **k: a[2], raising=False)
    monkeypatch.setattr(pl, "run_opensubtitles_download", _fake_no_download, raising=False)

    result = process_file(mp4, cache, language="en", filter_hallucinations=False)
    assert result["status"] == "no_subtitle"


def test_process_file_full_pipeline_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    mp4 = tmp_path / "movie.english.mp4"
    mp4.write_bytes(b"\x00")

    import subtitle_correction.pipeline as pl

    def _fake_extract_audio(mp4_path, out_wav):
        out_wav.write_bytes(b"\x00")
        return out_wav

    def _fake_whisper(wav, srt, output_json=None, **kwargs):
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nthe cat\n", encoding="utf-8")
        if output_json:
            output_json.write_text(json.dumps({"segments": []}), encoding="utf-8")
        return srt

    def _fake_download(name, lang, out_dir, scraper=None):
        srt = out_dir / f"opensubtitles.{lang}.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nthe bat\n", encoding="utf-8")
        return srt

    def _fake_alignment(whisper, sub, out):
        out.write_text("1\n00:00:00,000 --> 00:00:01,000\nthe bat\n", encoding="utf-8")
        return out

    correction_calls: list = []

    def _fake_correction(input_srt, reference_srt, output_srt, model_path="x", fused=True):
        correction_calls.append(True)
        output_srt.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nthe corrected bat\n", encoding="utf-8"
        )
        return output_srt

    monkeypatch.setattr(pl, "extract_audio", _fake_extract_audio, raising=False)
    monkeypatch.setattr(pl, "run_whisper", _fake_whisper, raising=False)
    monkeypatch.setattr(pl, "run_hallucination_filter", lambda *a, **k: a[2], raising=False)
    monkeypatch.setattr(pl, "run_opensubtitles_download", _fake_download, raising=False)
    monkeypatch.setattr(pl, "run_alignment", _fake_alignment, raising=False)
    monkeypatch.setattr(pl, "compute_alignment_score", lambda *a, **k: 0.9, raising=False)
    monkeypatch.setattr(pl, "run_correction", _fake_correction, raising=False)

    result = process_file(mp4, cache, language="en", filter_hallucinations=False)
    assert result["status"] == PipelineStep.ALIGNED
    assert correction_calls == [True]


def test_process_file_alignment_missing_srt_returns_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    mp4 = tmp_path / "movie.english.mp4"
    mp4.write_bytes(b"\x00")

    import subtitle_correction.pipeline as pl

    def _fake_extract_audio(mp4_path, out_wav):
        out_wav.write_bytes(b"\x00")
        return out_wav

    def _fake_whisper(wav, srt, output_json=None, **kwargs):
        # Don't write the srt -> alignment step sees missing file
        if output_json:
            output_json.write_text(json.dumps({"segments": []}), encoding="utf-8")
        return srt

    def _fake_download(name, lang, out_dir, scraper=None):
        srt = out_dir / f"opensubtitles.{lang}.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nthe bat\n", encoding="utf-8")
        return srt

    monkeypatch.setattr(pl, "extract_audio", _fake_extract_audio, raising=False)
    monkeypatch.setattr(pl, "run_whisper", _fake_whisper, raising=False)
    monkeypatch.setattr(pl, "run_hallucination_filter", lambda *a, **k: a[2], raising=False)
    monkeypatch.setattr(pl, "run_opensubtitles_download", _fake_download, raising=False)

    result = process_file(mp4, cache, language="en", filter_hallucinations=False)
    assert result["status"] == "alignment_failed"


def test_process_file_exception_returns_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    mp4 = tmp_path / "movie.english.mp4"
    mp4.write_bytes(b"\x00")

    import subtitle_correction.pipeline as pl

    def _boom(*a, **k):
        raise RuntimeError("extract failed")

    monkeypatch.setattr(pl, "extract_audio", _boom, raising=False)

    result = process_file(mp4, cache, language="en")
    assert result["status"] == PipelineStep.FAILED
    assert "extract failed" in result["error"]


def test_process_file_skips_correction_when_low_alignment_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    mp4 = tmp_path / "movie.english.mp4"
    mp4.write_bytes(b"\x00")

    import subtitle_correction.pipeline as pl

    def _fake_extract_audio(mp4_path, out_wav):
        out_wav.write_bytes(b"\x00")
        return out_wav

    def _fake_whisper(wav, srt, output_json=None, **kwargs):
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nthe cat\n", encoding="utf-8")
        if output_json:
            output_json.write_text(json.dumps({"segments": []}), encoding="utf-8")
        return srt

    def _fake_download(name, lang, out_dir, scraper=None):
        srt = out_dir / f"opensubtitles.{lang}.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nthe bat\n", encoding="utf-8")
        return srt

    def _fake_alignment(whisper, sub, out):
        out.write_text("1\n00:00:00,000 --> 00:00:01,000\nthe bat\n", encoding="utf-8")
        return out

    correction_calls: list = []

    def _track_correction_call(*a, **k):
        correction_calls.append(True)
        return a[2]

    monkeypatch.setattr(pl, "extract_audio", _fake_extract_audio, raising=False)
    monkeypatch.setattr(pl, "run_whisper", _fake_whisper, raising=False)
    monkeypatch.setattr(pl, "run_hallucination_filter", lambda *a, **k: a[2], raising=False)
    monkeypatch.setattr(pl, "run_opensubtitles_download", _fake_download, raising=False)
    monkeypatch.setattr(pl, "run_alignment", _fake_alignment, raising=False)
    # Low alignment score -> correction gate skips the LLM step
    monkeypatch.setattr(pl, "compute_alignment_score", lambda *a, **k: 0.1, raising=False)
    monkeypatch.setattr(pl, "run_correction", _track_correction_call, raising=False)

    result = process_file(mp4, cache, language="de", filter_hallucinations=False)
    assert result["status"] == PipelineStep.ALIGNED
    assert correction_calls == []
