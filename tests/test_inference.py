from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from subtitle_correction.inference import (
    SubtitleCorrector,
    correct_file_impl,
    has_german_chars,
    has_german_patterns,
    is_english,
    load_mlx_model,
)


def test_has_german_chars_true() -> None:
    assert has_german_chars("Hallo ä ö ü ß") is True


def test_has_german_chars_false() -> None:
    assert has_german_chars("plain ascii") is False


def test_has_german_patterns_true() -> None:
    assert has_german_patterns("das ist eine Ordnung") is True


def test_has_german_patterns_false() -> None:
    assert has_german_patterns("the quick brown fox") is False


def test_is_english_short_text_returns_true() -> None:
    assert is_english("a") is True


def test_is_english_plain_english() -> None:
    assert is_english("the quick brown fox jumps over the lazy dog here") is True


def test_is_english_german_chars_returns_false() -> None:
    assert is_english("Hallo Welt mit ä ö ü") is False


def test_is_english_german_patterns_returns_false() -> None:
    assert is_english("das ist eine Ordnung hier") is False


def test_is_english_two_german_markers_returns_false() -> None:
    assert is_english("der Mann ist hier") is False


def test_is_english_single_german_marker_short_returns_false() -> None:
    # 1 german marker + <= 4 words -> False
    assert is_english("der Mann") is False


def test_is_english_empty_words_returns_true() -> None:
    assert is_english("!!! ???") is True


def test_load_mlx_model_calls_mlx_load(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mlx = types.ModuleType("mlx_lm")

    def _fake_load(name: str):
        return (f"model:{name}", f"tokenizer:{name}")

    fake_mlx.load = _fake_load
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx)
    model, tokenizer = load_mlx_model("some-model")
    assert model == "model:some-model"
    assert tokenizer == "tokenizer:some-model"


class _FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        return "PROMPT:" + str(messages)


def _install_fake_mlx(monkeypatch: pytest.MonkeyPatch, generate_output: str = "corrected text"):
    """Install fake mlx_lm module with load, make_sampler, generate."""
    fake_mlx = types.ModuleType("mlx_lm")
    fake_sample_utils = types.ModuleType("mlx_lm.sample_utils")

    def _fake_make_sampler(temp: float = 0.0):
        return {"temp": temp}

    def _fake_generate(model, tokenizer, *, prompt: str, max_tokens: int, sampler, verbose: bool):
        return generate_output

    def _fake_load(name, adapter_path=None):
        return (f"model:{name}", _FakeTokenizer())

    fake_mlx.load = _fake_load
    fake_mlx.generate = _fake_generate
    fake_mlx.make_sampler = _fake_make_sampler
    fake_sample_utils.make_sampler = _fake_make_sampler
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)


def test_subtitle_corrector_fused_init(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_mlx(monkeypatch)
    c = SubtitleCorrector("my-model", fused=True, temp=0.2)
    assert c.fused is True
    assert c.model == "model:my-model"
    assert isinstance(c.tokenizer, _FakeTokenizer)
    assert c.sampler == {"temp": 0.2}


def test_subtitle_corrector_adapter_init(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_mlx(monkeypatch)
    adapter = Path("/tmp/adapter")
    c = SubtitleCorrector("my-model", adapter_path=adapter, fused=False, temp=0.0)
    assert c.fused is False


def test_subtitle_corrector_correct_line_with_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mlx(monkeypatch, generate_output="corrected text")
    c = SubtitleCorrector("m", fused=True)
    c.tokenizer = _FakeTokenizer()
    out = c.correct_line("hello", reference="hello there")
    assert isinstance(out, str)


def test_subtitle_corrector_correct_line_strips_thinking_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mlx(monkeypatch, generate_output="<channel|>thinking</channel>final answer")
    c = SubtitleCorrector("m", fused=True)
    c.tokenizer = _FakeTokenizer()
    out = c.correct_line("hello")
    assert "thinking" not in out


def test_subtitle_corrector_correct_line_empty_output_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mlx(monkeypatch, generate_output="   ")
    c = SubtitleCorrector("m", fused=True)
    c.tokenizer = _FakeTokenizer()
    out = c.correct_line("hello")
    assert out == "hello"


def test_subtitle_corrector_correct_line_too_long_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mlx(monkeypatch, generate_output="x" * 1000)
    c = SubtitleCorrector("m", fused=True)
    c.tokenizer = _FakeTokenizer()
    out = c.correct_line("hi")
    assert out == "hi"


def test_subtitle_corrector_correct_line_strips_thinking_process_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mlx(monkeypatch, generate_output="Thinking Process:\nanalyze this\nfinal text")
    c = SubtitleCorrector("m", fused=True)
    c.tokenizer = _FakeTokenizer()
    out = c.correct_line("hello", reference="hello there")
    assert "Thinking Process" not in out


def test_subtitle_corrector_correct_line_strips_im_end_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mlx(monkeypatch, generate_output="good text<|im_end|>trailing")
    c = SubtitleCorrector("m", fused=True)
    c.tokenizer = _FakeTokenizer()
    out = c.correct_line("hello")
    assert "trailing" not in out


def test_correct_file_impl_srt_writes_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_mlx(monkeypatch, generate_output="corrected line")
    srt = tmp_path / "in.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nhello world\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nfoo bar\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.srt"
    # Use an English model so is_english passes
    correct_file_impl(srt, out, model="m", fused=True, lang="en")
    text = out.read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:02,000" in text
    assert "corrected line" in text


def test_correct_file_impl_srt_skips_non_english(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_mlx(monkeypatch, generate_output="corrected")
    srt = tmp_path / "in.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nHallo Welt mit ä ö ü\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.srt"
    correct_file_impl(srt, out, model="m", fused=True, lang="en")
    # Non-English skipped -> original block preserved
    text = out.read_text(encoding="utf-8")
    assert "Hallo Welt" in text


def test_correct_file_impl_srt_short_block_passthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_mlx(monkeypatch, generate_output="corrected")
    srt = tmp_path / "in.srt"
    # Block with < 3 lines -> passed through unchanged
    srt.write_text("just one line\n", encoding="utf-8")
    out = tmp_path / "out.srt"
    correct_file_impl(srt, out, model="m", fused=True, lang="en")
    assert out.read_text(encoding="utf-8") == "just one line"


def test_correct_file_impl_no_output_prints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_fake_mlx(monkeypatch, generate_output="corrected")
    srt = tmp_path / "in.srt"
    srt.write_text("plain text content\n", encoding="utf-8")
    correct_file_impl(srt, None, model="m", fused=True, lang="en")
    captured = capsys.readouterr()
    assert "corrected" in captured.out


def test_correct_file_impl_plain_text_with_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_mlx(monkeypatch, generate_output="corrected")
    txt = tmp_path / "in.txt"
    txt.write_text("hello world\n", encoding="utf-8")
    ref = tmp_path / "ref.srt"
    ref.write_text("1\n00:00:01,000 --> 00:00:02,000\nref text\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    correct_file_impl(txt, out, model="m", fused=True, lang="en", reference_file=ref)
    assert "corrected" in out.read_text(encoding="utf-8")


def test_correct_file_impl_plain_text_reference_non_srt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_mlx(monkeypatch, generate_output="corrected")
    txt = tmp_path / "in.txt"
    txt.write_text("hello world\n", encoding="utf-8")
    ref = tmp_path / "ref.txt"
    ref.write_text("reference content\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    correct_file_impl(txt, out, model="m", fused=True, lang="en", reference_file=ref)
    assert "corrected" in out.read_text(encoding="utf-8")
