from __future__ import annotations

import pytest

from subtitle_correction.prompts import (
    clean_field,
    detect_task_mode,
    format_inference_prompt,
    format_training_text,
    format_user_body,
)


def test_clean_field_strips_music_and_newlines() -> None:
    assert clean_field("♪Hallo\nWelt♪") == "Hallo Welt"


def test_clean_field_collapses_crlf() -> None:
    assert clean_field("a\r\nb") == "a  b"


def test_clean_field_strips_whitespace() -> None:
    assert clean_field("  x  ") == "x"


def test_detect_task_mode_same_lang() -> None:
    # Both clearly English -> same_lang
    assert detect_task_mode("the cat sat on the mat", "the dog ran fast here") == "same_lang"


def test_detect_task_mode_dub_when_reference_differs() -> None:
    # Whisper in English, reference in German -> dub branch
    assert detect_task_mode("the cat sat on the mat", "Hallo Welt das ist Deutsch") == "dub"


def test_detect_task_mode_uses_reference_default_en() -> None:
    # Empty reference falls back to "en" for the reference lang
    assert detect_task_mode("Hallo Welt das ist Deutsch", "") == "dub"


def test_format_user_body_same_lang_mode() -> None:
    body = format_user_body("the cat", "the dog ran fast here")
    assert "ASR transcription: the cat" in body
    assert "Reference subtitles: the dog ran fast here" in body
    assert "Corrected transcription:" in body
    assert "different language" not in body


def test_format_user_body_dub_mode_adds_note() -> None:
    body = format_user_body("the cat sat on the mat", "Hallo Welt das ist Deutsch")
    assert "different language" in body


def test_format_user_body_explicit_mode_override() -> None:
    body = format_user_body(
        "hello world this is english",
        "Hallo Welt das ist Deutsch",
        task_mode="same_lang",
    )
    assert "different language" not in body


def test_format_training_text_shape() -> None:
    text = format_training_text("whisper txt", "ref txt", "target txt")
    assert text.startswith("<bos><start_of_turn>user\n")
    assert "<start_of_turn>model\n" in text
    assert text.endswith("target txt<end_of_turn><eos>")
    assert "ASR transcription: whisper txt" in text
    assert "Reference subtitles: ref txt" in text


def test_format_inference_prompt_ends_with_model_turn() -> None:
    prompt = format_inference_prompt("w", "r")
    assert prompt.startswith("<bos><start_of_turn>user\n")
    assert prompt.endswith("<start_of_turn>model\n")
    assert "ASR transcription: w" in prompt


@pytest.mark.parametrize("mode", ["same_lang", "dub"])
def test_format_user_body_both_branches(mode: str) -> None:
    body = format_user_body("a", "b", task_mode=mode)
    assert "ASR transcription: a" in body
