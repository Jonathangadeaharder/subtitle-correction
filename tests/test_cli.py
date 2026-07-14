from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from subtitle_correction.cli import app


def _write_srt(path: Path, blocks: list[tuple[str, str, str]]) -> None:
    rendered = []
    for idx, timing, text in blocks:
        rendered.append(f"{idx}\n{timing}\n{text}")
    path.write_text("\n\n".join(rendered) + "\n", encoding="utf-8")


# ---------------- align command ----------------


def test_cli_align_missing_whisper_exits(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["align", str(tmp_path / "missing.srt"), str(tmp_path / "sub.srt")],
    )
    assert result.exit_code == 1


def test_cli_align_missing_subtitle_exits(tmp_path: Path) -> None:
    whisper = tmp_path / "w.srt"
    whisper.write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["align", str(whisper), str(tmp_path / "missing.srt")],
    )
    assert result.exit_code == 1


def test_cli_align_alass_failure_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    whisper = tmp_path / "w.srt"
    sub = tmp_path / "s.srt"
    out = tmp_path / "out.srt"
    _write_srt(whisper, [("1", "00:00:01,000 --> 00:00:02,000", "hi")])
    _write_srt(sub, [("1", "00:00:01,000 --> 00:00:02,000", "ho")])

    import subtitle_correction.align as align_mod

    monkeypatch.setattr(align_mod.shutil, "which", lambda _n: None, raising=False)

    result = CliRunner().invoke(
        app,
        ["align", str(whisper), str(sub), "--output", str(out)],
    )
    assert result.exit_code == 1


def test_cli_align_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    whisper = tmp_path / "w.srt"
    sub = tmp_path / "s.srt"
    out = tmp_path / "out.srt"
    _write_srt(whisper, [("1", "00:00:01,000 --> 00:00:02,000", "the cat sat")])
    _write_srt(sub, [("1", "00:00:01,000 --> 00:00:02,000", "the cat ran")])

    import subtitle_correction.align as align_mod

    def _fake_align(ref, bad, output, split_penalty=10):
        output.write_text(sub.read_text(encoding="utf-8"), encoding="utf-8")
        return output

    monkeypatch.setattr(align_mod, "align_with_alass", _fake_align, raising=False)
    monkeypatch.setattr(align_mod, "detect_srt_language", lambda p: "en", raising=False)
    monkeypatch.setattr(align_mod, "extract_language_from_filename", lambda p: "en", raising=False)

    result = CliRunner().invoke(
        app,
        ["align", str(whisper), str(sub), "--output", str(out)],
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists()


# ---------------- scrape command ----------------


def test_cli_scrape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    srt = tmp_path / "downloaded.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n", encoding="utf-8")

    class _FakeScraper:
        def search_and_download(self, parsed, language, output_dir):
            return srt

        def close(self):
            pass

    import subtitle_correction.cli as cli_mod

    monkeypatch.setattr(
        cli_mod, "OpenSubtitlesScraper", lambda *a, **k: _FakeScraper(), raising=False
    )

    result = CliRunner().invoke(
        app,
        ["scrape", "Movie.2020.mp4", "--lang", "en", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout


def test_cli_scrape_no_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeScraper:
        def search_and_download(self, parsed, language, output_dir):
            return None

        def close(self):
            pass

    import subtitle_correction.cli as cli_mod

    monkeypatch.setattr(
        cli_mod, "OpenSubtitlesScraper", lambda *a, **k: _FakeScraper(), raising=False
    )

    result = CliRunner().invoke(
        app,
        ["scrape", "Unknown.2020.mp4", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0


# ---------------- pipeline command ----------------


def test_cli_pipeline_no_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    result = CliRunner().invoke(
        app,
        ["pipeline", "--input-dir", str(input_dir)],
    )
    assert result.exit_code == 0


def test_cli_pipeline_single_file_not_found(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["pipeline", "--input-dir", str(tmp_path), "--file", str(tmp_path / "missing.mp4")],
    )
    assert result.exit_code == 1


def test_cli_pipeline_single_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mp4 = tmp_path / "movie.english.mp4"
    mp4.write_bytes(b"\x00")

    import subtitle_correction.cli as cli_mod

    monkeypatch.setattr(
        cli_mod,
        "process_file",
        lambda *a, **k: {"status": "aligned", "dir": str(tmp_path)},
        raising=False,
    )

    class _FakeScraper:
        def close(self):
            pass

    monkeypatch.setattr(
        cli_mod, "OpenSubtitlesScraper", lambda *a, **k: _FakeScraper(), raising=False
    )

    result = CliRunner().invoke(
        app,
        ["pipeline", "--input-dir", str(tmp_path), "--file", str(mp4), "--all"],
    )
    assert result.exit_code == 0, result.stdout


def test_cli_pipeline_failed_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mp4 = tmp_path / "show.S01E01.english.mp4"
    mp4.write_bytes(b"\x00")
    import subtitle_correction.cli as cli_mod

    monkeypatch.setattr(
        cli_mod,
        "process_file",
        lambda *a, **k: {"status": "failed", "error": "boom", "dir": str(tmp_path)},
        raising=False,
    )

    class _FakeScraper:
        def close(self):
            pass

    monkeypatch.setattr(
        cli_mod, "OpenSubtitlesScraper", lambda *a, **k: _FakeScraper(), raising=False
    )

    result = CliRunner().invoke(
        app,
        ["pipeline", "--input-dir", str(tmp_path)],
    )
    assert result.exit_code == 0


# ---------------- pipeline-status command ----------------


def test_cli_pipeline_status_no_cache(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["pipeline-status", "--input-dir", str(tmp_path)],
    )
    assert result.exit_code == 0


def test_cli_pipeline_status_empty_cache(tmp_path: Path) -> None:
    cache = tmp_path / ".subcache"
    cache.mkdir()
    result = CliRunner().invoke(
        app,
        ["pipeline-status", "--input-dir", str(tmp_path)],
    )
    assert result.exit_code == 0


def test_cli_pipeline_status_with_entries(tmp_path: Path) -> None:
    cache = tmp_path / ".subcache"
    work = cache / "movie"
    work.mkdir(parents=True)
    (work / "metadata.json").write_text(
        json.dumps(
            {
                "source_file": "movie.mp4",
                "status": "aligned",
                "whisper_lang": "en",
                "subtitle_lang": "de",
                "alignment_score": 0.8,
                "error": "",
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        ["pipeline-status", "--input-dir", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "movie" in result.stdout


# ---------------- correct command ----------------


def _install_fake_mlx_cli(monkeypatch: pytest.MonkeyPatch, output: str = "corrected"):
    fake_mlx = types.ModuleType("mlx_lm")
    fake_sample_utils = types.ModuleType("mlx_lm.sample_utils")

    class _FakeTok:
        def apply_chat_template(self, messages, **kwargs):
            return "PROMPT"

    fake_mlx.load = lambda name, adapter_path=None: (f"m:{name}", _FakeTok())
    fake_mlx.generate = lambda *a, **k: output
    fake_sample_utils.make_sampler = lambda temp=0.0: {"temp": temp}
    fake_mlx.make_sampler = lambda temp=0.0: {"temp": temp}
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)


def test_cli_correct_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_mlx_cli(monkeypatch, output="corrected text")
    result = CliRunner().invoke(app, ["correct", "--text", "hello world"])
    assert result.exit_code == 0, result.stdout
    assert "corrected text" in result.stdout


def test_cli_correct_no_args_exits(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["correct"])
    assert result.exit_code == 1


def test_cli_correct_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_mlx_cli(monkeypatch, output="corrected")
    srt = tmp_path / "in.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello world\n", encoding="utf-8")
    out = tmp_path / "out.srt"
    result = CliRunner().invoke(
        app,
        ["correct", "--input-file", str(srt), "--output", str(out), "--lang", "en"],
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists()


# ---------------- srt-from-reference command (already tested) ----------------


def test_cli_srt_from_reference(tmp_path: Path) -> None:
    whisper = tmp_path / "whisper.srt"
    reference = tmp_path / "ref.md"
    out = tmp_path / "out.srt"
    _write_srt(whisper, [("1", "00:00:01,000 --> 00:00:02,000", "Helo world")])
    reference.write_text("Hello world", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "srt-from-reference",
            "--reference",
            str(reference),
            "--whisper-srt",
            str(whisper),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists()


# ---------------- correct-reference-free command ----------------


def test_cli_correct_reference_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_mlx_cli(monkeypatch, output="korrigiert")
    srt = tmp_path / "in.srt"
    out = tmp_path / "out.srt"
    _write_srt(srt, [("1", "00:00:01,000 --> 00:00:02,000", "hallo welt")])

    result = CliRunner().invoke(
        app,
        ["correct-reference-free", "--input", str(srt), "--output", str(out)],
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists()
