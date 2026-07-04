from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from subtitle_correction.cli import app
from subtitle_correction.srt_from_reference import (
    _assign_reference_to_cues,
    align_reference_to_srt,
    strip_markdown_prose,
)


def _write_srt(path: Path, blocks: list[tuple[str, str, str]]) -> None:
    rendered = []
    for idx, timing, text in blocks:
        rendered.append(f"{idx}\n{timing}\n{text}")
    path.write_text("\n\n".join(rendered) + "\n", encoding="utf-8")


def test_strip_markdown_prose_keeps_body_only() -> None:
    md = "# Title\n\nFirst line.\nSecond line."
    assert strip_markdown_prose(md) == "First line.\nSecond line."


def test_assign_reference_to_cues_carries_gaps_forward() -> None:
    assigned = _assign_reference_to_cues(
        cue_token_owner=[0, 0, 1, 1, 2],
        whisper_norm=["a", "bee", "see", "dee", "ee"],
        ref_norm=["a", "b", "see", "dee", "extra", "gap", "ee"],
        n_cues=3,
    )
    assert assigned == [0, 0, 1, 1, 1, 1, 2]


def test_assign_reference_replacements_keep_source_cue_owner() -> None:
    assigned = _assign_reference_to_cues(
        cue_token_owner=[0, 1],
        whisper_norm=["gutes", "maus"],
        ref_norm=["gutes", "haus"],
        n_cues=2,
    )
    assert assigned == [0, 1]


def test_align_reference_to_srt_preserves_timing_and_uses_reference_spans(
    tmp_path: Path,
) -> None:
    whisper = tmp_path / "whisper.srt"
    out = tmp_path / "out.srt"
    _write_srt(
        whisper,
        [
            ("1", "00:00:01,000 --> 00:00:02,000", "a bee"),
            ("2", "00:00:02,000 --> 00:00:03,000", "see dee"),
            ("3", "00:00:03,000 --> 00:00:04,000", "ee"),
        ],
    )
    align_reference_to_srt("a b see dee extra gap ee", whisper, out)
    text = out.read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:02,000" in text
    assert "a b" in text
    assert "extra gap" in text


def test_cli_srt_from_reference_writes_output(tmp_path: Path) -> None:
    whisper = tmp_path / "whisper.srt"
    reference = tmp_path / "ref.md"
    out = tmp_path / "out.srt"
    _write_srt(whisper, [("1", "00:00:01,000 --> 00:00:02,000", "Helo world")])
    reference.write_text("Hello world", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
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
    assert "Hello world" in out.read_text(encoding="utf-8")