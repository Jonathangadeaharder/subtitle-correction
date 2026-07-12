from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from subtitle_correction.evaluate import (
    Levenshtein_distance,
    calculate_wer,
    classify_eval_entry,
    create_dataset_impl,
    evaluate_model_impl,
    normalize_for_compare,
    srt_time_to_seconds,
)


def test_srt_time_to_seconds_comma_format() -> None:
    assert srt_time_to_seconds("01:02:03,500") == pytest.approx(3723.5)


def test_srt_time_to_seconds_dot_format() -> None:
    assert srt_time_to_seconds("00:00:01.250") == pytest.approx(1.25)


def test_srt_time_to_seconds_zero() -> None:
    assert srt_time_to_seconds("00:00:00,000") == 0.0


def test_levenshtein_distance_equal() -> None:
    assert Levenshtein_distance("abc", "abc") == 0


def test_levenshtein_distance_empty_s2() -> None:
    assert Levenshtein_distance("abc", "") == 3


def test_levenshtein_distance_swaps_args_when_s1_shorter() -> None:
    assert Levenshtein_distance("ab", "abc") == 1


def test_levenshtein_distance_substitution() -> None:
    assert Levenshtein_distance("cat", "bat") == 1


def test_calculate_wer_identical() -> None:
    assert calculate_wer("hello world", "hello world") == 0.0


def test_calculate_wer_empty_reference_no_hypothesis() -> None:
    assert calculate_wer("", "") == 0.0


def test_calculate_wer_empty_reference_with_hypothesis() -> None:
    assert calculate_wer("", "hello") == 1.0


def test_calculate_wer_full_substitution() -> None:
    assert calculate_wer("alpha bravo", "charlie delta") == 1.0


def test_calculate_wer_partial() -> None:
    wer = calculate_wer("the cat sat", "the bat sat")
    assert 0.0 < wer < 1.0


def test_calculate_wer_ignores_punctuation_and_case() -> None:
    assert calculate_wer("Hello, World!", "hello world") == 0.0


def test_normalize_for_compare_delegates() -> None:
    assert normalize_for_compare("Hello, World!") == "hello world"
    assert normalize_for_compare("") == ""


def test_classify_eval_entry_delegates() -> None:
    assert classify_eval_entry({"has_error": False}) == "identity"


def test_create_dataset_impl_missing_dir_exits(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(typer.Exit):
        create_dataset_impl(subcache_dir=missing, output_dir=tmp_path / "out")


def test_create_dataset_impl_no_subdirs(tmp_path: Path) -> None:
    subcache = tmp_path / "subcache"
    subcache.mkdir()
    out = tmp_path / "out"
    # Should return None without raising (no folders to scan)
    assert create_dataset_impl(subcache_dir=subcache, output_dir=out) is None


def test_create_dataset_impl_skips_incomplete_folders(tmp_path: Path) -> None:
    subcache = tmp_path / "subcache"
    out = tmp_path / "out"
    # Folder missing required files -> skipped, no pairs -> returns None
    d = subcache / "movie1"
    d.mkdir(parents=True)
    (d / "audio.wav").write_bytes(b"\x00")
    # Missing aligned.srt, whisper.srt, opensubtitles.*.srt
    assert create_dataset_impl(subcache_dir=subcache, output_dir=out) is None


def test_evaluate_model_impl_missing_dataset_exits(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit):
        evaluate_model_impl(dataset_path=tmp_path / "missing.jsonl")


def test_evaluate_model_impl_empty_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text("", encoding="utf-8")
    # Empty dataset -> returns None after printing
    assert evaluate_model_impl(dataset_path=dataset) is None


def test_evaluate_model_impl_runs_with_mock_corrector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    entries = [
        {
            "id": "slice_0001",
            "whisper_text": "the cat sat",
            "ground_truth": "the cat ran",
            "reference": "the cat ran",
            "has_error": True,
            "eval_scope": "in_scope",
        },
        {
            "id": "slice_0002",
            "whisper_text": "hello world",
            "ground_truth": "hello world",
            "reference": "hello world",
            "has_error": False,
        },
        {
            "id": "slice_0003",
            "whisper_text": "the dog ran",
            "ground_truth": "the dog ran",
            "reference": "the dog ran",
            "has_error": True,
            "eval_scope": "no_reference",
        },
        {
            "id": "slice_0004",
            "whisper_text": "the dog ran",
            "ground_truth": "the dog ran",
            "reference": "the dog ran",
            "has_error": True,
            "eval_scope": "misaligned",
        },
        {
            "id": "slice_0005",
            "whisper_text": "♪ song ♪",
            "ground_truth": "♪ song ♪",
            "reference": "♪ song ♪",
            "has_error": True,
            "eval_scope": "lyrics",
        },
    ]
    dataset.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    class _FakeCorrector:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def correct_line(self, line: str, reference: str | None = None, **_: object) -> str:
            # Simulate a "corrected" output: fix to ground truth if reference present
            return reference if reference is not None else line

    # evaluate_model_impl imports SubtitleCorrector at call time via
    # `from .inference import SubtitleCorrector`, so patch on the inference
    # module (not the eval module) to avoid loading real MLX models.
    import subtitle_correction.inference as inf_mod

    monkeypatch.setattr(inf_mod, "SubtitleCorrector", _FakeCorrector, raising=False)

    # Should run through all entries without raising.
    result = evaluate_model_impl(dataset_path=dataset, limit=3)
    assert result is None


def test_evaluate_model_impl_worsened_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force a "worsened" outcome: corrected text differs from both whisper and target,
    # and is farther from target than the original.
    dataset = tmp_path / "dataset.jsonl"
    entries = [
        {
            "id": "s1",
            "whisper_text": "the cat sat",
            "ground_truth": "the cat ran",
            "reference": "the cat ran",
            "has_error": True,
            "eval_scope": "in_scope",
        },
    ]
    dataset.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    class _WorseCorrector:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def correct_line(self, line: str, reference: str | None = None, **_: object) -> str:
            # Return something unrelated -> worsened (farther from target)
            return "zzzzzzzzzz different"

    import subtitle_correction.inference as inf_mod

    monkeypatch.setattr(inf_mod, "SubtitleCorrector", _WorseCorrector, raising=False)
    assert evaluate_model_impl(dataset_path=dataset) is None


def test_create_dataset_impl_full_flow_with_mocked_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subcache = tmp_path / "subcache"
    out = tmp_path / "out"
    d = subcache / "movie1"
    d.mkdir(parents=True)
    (d / "audio.wav").write_bytes(b"\x00" * 100)
    (d / "whisper.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nthe cat sat\n", encoding="utf-8"
    )
    (d / "aligned.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nthe cat ran\n", encoding="utf-8"
    )
    (d / "opensubtitles.en.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nthe cat ran\n", encoding="utf-8"
    )

    # Mock ffmpeg to "succeed" by writing the slice file.
    def _fake_run(cmd: list[str], **_: object):
        # Find the output path (last arg)
        slice_path = Path(cmd[-1])
        slice_path.parent.mkdir(parents=True, exist_ok=True)
        slice_path.write_bytes(b"\x00" * 10)

        class _R:
            returncode = 0
            stderr = ""
            stdout = ""

        return _R()

    import subtitle_correction.evaluate as eval_mod

    monkeypatch.setattr(eval_mod.subprocess, "run", _fake_run, raising=False)
    # Alignment score must be >= 0.5 to proceed
    monkeypatch.setattr(eval_mod, "compute_alignment_score", lambda *a, **k: 0.9, raising=False)

    create_dataset_impl(subcache_dir=subcache, output_dir=out, max_slices=10)
    dataset = out / "dataset.jsonl"
    assert dataset.exists()
    lines = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 1
    assert lines[0]["id"].startswith("slice_")
