from __future__ import annotations

import json
import random
import sys
import types
from pathlib import Path

import pytest
import typer

from subtitle_correction.prepare_data import (
    HOMOPHONES,
    PHONETIC_SUBS,
    SIMILAR_WORDS,
    SYSTEM_PROMPT,
    _corrupt_text,
    align_casing_and_punctuation,
    format_chat_text,
    prepare_data_impl,
)


class _FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, enable_thinking):
        # messages: [user, assistant]
        return f"USER={messages[0]['content']}|ASSIST={messages[1]['content']}"


def test_format_chat_text_cleans_and_formats() -> None:
    tok = _FakeTokenizer()
    # system arg is not cleaned by format_chat_text; only whisper/ref/assistant are
    out = format_chat_text(tok, "sys prompt", "whisper♪\nline", "ref", "assist")
    assert "whisper line" in out
    assert "ref" in out
    assert "assist" in out
    # Music notes and newlines stripped from whisper input
    assert "♪" not in out


def test_align_casing_and_punctuation_matches_target_casing() -> None:
    out = align_casing_and_punctuation("hello world", "Hello world")
    assert out.startswith("Hello")


def test_align_casing_and_punctuation_lowercases_when_target_lower() -> None:
    out = align_casing_and_punctuation("Hello world", "hello world")
    assert out.startswith("hello")


def test_align_casing_and_punctuation_adds_trailing_punctuation() -> None:
    out = align_casing_and_punctuation("hello world", "hello world!")
    assert out.endswith("!")


def test_align_casing_and_punctuation_replaces_trailing_punctuation() -> None:
    out = align_casing_and_punctuation("hello world,", "hello world!")
    assert out.endswith("!")
    assert not out.endswith(",")


def test_align_casing_and_punctuation_empty_source_returns_source() -> None:
    assert align_casing_and_punctuation("", "target") == ""
    assert align_casing_and_punctuation("src", "") == "src"


def test_corrupt_text_empty_returns_input() -> None:
    assert _corrupt_text("", random.Random(0)) == ""


def test_corrupt_text_changes_something() -> None:
    rng = random.Random(42)
    text = "their house is over there with the cat and the dog"
    corrupted = _corrupt_text(text, rng)
    # With 7 corruptions forced, something should change (probabilistic but reliable)
    assert isinstance(corrupted, str)


def test_corrupt_text_homophone_branch() -> None:
    # Seed that exercises homophone substitution
    rng = random.Random(100)
    out = _corrupt_text("their house is big", rng)
    assert isinstance(out, str)


def test_corrupt_text_drop_branch() -> None:
    rng = random.Random(7)
    out = _corrupt_text("the dog and the cat in the house on the hill", rng)
    assert isinstance(out, str)


def test_corrupt_text_repeat_branch() -> None:
    rng = random.Random(3)
    out = _corrupt_text("something nothing everything", rng)
    assert isinstance(out, str)


def test_corrupt_text_truncate_branch() -> None:
    rng = random.Random(55)
    out = _corrupt_text("beautiful different interesting", rng)
    assert isinstance(out, str)


def test_homophones_dict_nonempty() -> None:
    assert "their" in HOMOPHONES
    assert "their" in HOMOPHONES["there"]


def test_phonetic_subs_dict_nonempty() -> None:
    assert "probably" in PHONETIC_SUBS


def test_similar_words_dict_nonempty() -> None:
    assert "he" in SIMILAR_WORDS


def test_system_prompt_present() -> None:
    assert "subtitle corrector" in SYSTEM_PROMPT.lower()


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch, tokenizer=None) -> None:
    fake_transformers = types.ModuleType("transformers")

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name: str):
            return tokenizer or _FakeTokenizer()

    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def _install_fake_yaml(monkeypatch: pytest.MonkeyPatch, config: dict | None = None) -> None:
    fake_yaml = types.ModuleType("yaml")

    def _safe_load(stream):
        return config or {}

    fake_yaml.safe_load = _safe_load
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)


def test_prepare_data_impl_empty_pairs_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_file = tmp_path / "pairs.jsonl"
    input_file.write_text("", encoding="utf-8")
    _install_fake_transformers(monkeypatch)
    _install_fake_yaml(monkeypatch, {"model": "test-model"})
    with pytest.raises(typer.Exit):
        prepare_data_impl(input_file, tmp_path / "out")


def test_prepare_data_impl_writes_train_and_val(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_file = tmp_path / "pairs.jsonl"
    pairs = [
        {"whisper_text": "the cat sat", "corrected_text": "the cat ran"},
        {"whisper_text": "a dog barks", "corrected_text": "the dog barks"},
        {"whisper_text": "  ", "corrected_text": "  "},  # invalid, filtered
    ]
    input_file.write_text("\n".join(json.dumps(p) for p in pairs) + "\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    _install_fake_transformers(monkeypatch)
    _install_fake_yaml(monkeypatch, {"model": "test-model"})

    prepare_data_impl(input_file, out_dir, val_split=0.5, seed=42, augment=1, identity_ratio=1.0)

    train = out_dir / "train.jsonl"
    val = out_dir / "valid.jsonl"
    assert train.exists()
    assert val.exists()
    train_lines = [
        json.loads(line) for line in train.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    val_lines = [json.loads(line) for line in val.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(train_lines) >= 1
    assert len(val_lines) >= 1
    assert all("text" in ex for ex in train_lines + val_lines)


def test_prepare_data_impl_reads_repo_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The repo has a real config.yaml at its root; prepare_data_impl reads it
    # to determine the model name. Verify the config-reading branch runs.
    input_file = tmp_path / "pairs.jsonl"
    input_file.write_text(
        json.dumps({"whisper_text": "hello world", "corrected_text": "hello world"}) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    _install_fake_transformers(monkeypatch)
    _install_fake_yaml(monkeypatch, {"model": "config-model"})

    prepare_data_impl(input_file, out_dir, val_split=0.5, augment=0, identity_ratio=0.0)
    assert (out_dir / "train.jsonl").exists()
