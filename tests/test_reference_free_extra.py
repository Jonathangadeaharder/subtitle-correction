from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from subtitle_correction.reference_free import (
    BASE_MODEL,
    SYSTEM_PROMPT,
    ContextCorrector,
    correct_srt,
    main,
    self_check,
    unified_diff,
)


def _install_fake_mlx(monkeypatch: pytest.MonkeyPatch, generate_output: str = "corrected"):
    fake_mlx = types.ModuleType("mlx_lm")
    fake_sample_utils = types.ModuleType("mlx_lm.sample_utils")

    class _FakeTok:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
            return "PROMPT:" + str(messages)

    def _fake_make_sampler(temp: float = 0.0):
        return {"temp": temp}

    def _fake_generate(model, tokenizer, *, prompt, max_tokens, sampler, verbose):
        return generate_output

    fake_mlx.load = lambda name: (f"model:{name}", _FakeTok())
    fake_mlx.generate = _fake_generate
    fake_sample_utils.make_sampler = _fake_make_sampler
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)


def test_base_model_and_prompt() -> None:
    assert "gemma" in BASE_MODEL.lower()
    assert "Untertitel" in SYSTEM_PROMPT


def test_context_corrector_init(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_mlx(monkeypatch)
    c = ContextCorrector(model_name="my-model", temp=0.3)
    assert c.model == "model:my-model"
    assert c.sampler == {"temp": 0.3}


def test_context_corrector_correct_returns_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_mlx(monkeypatch, generate_output="hallo welt")
    c = ContextCorrector()
    out = c.correct(["prev"], "current", ["next"])
    assert out == "hallo welt"


def test_context_corrector_correct_empty_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_mlx(monkeypatch, generate_output="   ")
    c = ContextCorrector()
    out = c.correct(["prev"], "current", ["next"])
    assert out == "current"


def test_context_corrector_correct_too_long_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_mlx(monkeypatch, generate_output="x" * 500)
    c = ContextCorrector()
    out = c.correct(["prev"], "cur", ["next"])
    assert out == "cur"


def test_context_corrector_correct_strips_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_mlx(monkeypatch, generate_output="Korrektur: hallo welt")
    c = ContextCorrector()
    out = c.correct(["prev"], "helo", ["next"])
    assert out == "hallo welt"


def test_correct_srt_changes_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_mlx(monkeypatch, generate_output="hallo welt")
    srt = tmp_path / "in.srt"
    out = tmp_path / "out.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nhelo\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nwelo\n",
        encoding="utf-8",
    )
    changed, total = correct_srt(srt, out)
    assert total == 2
    assert changed == 2
    text = out.read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:02,000" in text
    assert "hallo welt" in text


def test_correct_srt_unchanged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_mlx(monkeypatch, generate_output="same text")
    srt = tmp_path / "in.srt"
    out = tmp_path / "out.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nsame text\n", encoding="utf-8")
    changed, total = correct_srt(srt, out)
    assert changed == 0
    assert total == 1


def test_unified_diff(tmp_path: Path) -> None:
    a = tmp_path / "a.srt"
    b = tmp_path / "b.srt"
    a.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n", encoding="utf-8")
    b.write_text("1\n00:00:01,000 --> 00:00:02,000\nworld\n", encoding="utf-8")
    diff = unified_diff(a, b)
    assert "hello" in diff
    assert "world" in diff
    assert "---" in diff or "-" in diff


def test_unified_diff_accepts_string_paths(tmp_path: Path) -> None:
    a = tmp_path / "a.srt"
    b = tmp_path / "b.srt"
    a.write_text("x\n", encoding="utf-8")
    b.write_text("y\n", encoding="utf-8")
    diff = unified_diff(str(a), str(b))
    assert isinstance(diff, str)


def test_self_check(capsys: pytest.CaptureFixture[str]) -> None:
    self_check()
    captured = capsys.readouterr()
    assert "self-check OK" in captured.out


def test_main_self_check(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["reference_free", "--self-check"])
    main()
    captured = capsys.readouterr()
    assert "self-check OK" in captured.out


def test_main_requires_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["reference_free"])
    with pytest.raises(SystemExit):
        main()
