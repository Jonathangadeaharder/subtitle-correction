from __future__ import annotations

import pytest

from subtitle_correction.lang_detect import detect_lang, is_german_text


def test_detect_lang_empty_returns_default() -> None:
    assert detect_lang("") == "de"
    assert detect_lang("   ") == "de"


def test_detect_lang_german_chars() -> None:
    assert detect_lang("Hallo Welt mit ä ö ü ß") == "de"


def test_detect_lang_german_stopwords() -> None:
    assert detect_lang("ich bin hier und das ist gut") == "de"


def test_detect_lang_english_stopwords() -> None:
    assert detect_lang("you are the one who has been there") == "en"


def test_detect_lang_default_for_unknown() -> None:
    # No recognized stop words and no special chars -> default
    assert detect_lang("xyzzy plugh") == "de"


def test_detect_lang_custom_default() -> None:
    assert detect_lang("xyzzy plugh", default="en") == "en"
    assert detect_lang("", default="en") == "en"


def test_detect_lang_only_punctuation() -> None:
    assert detect_lang("!!! ??? ...") == "de"


def test_is_german_text_true() -> None:
    assert is_german_text("ich bin hier und das ist gut") is True


def test_is_german_text_false() -> None:
    assert is_german_text("you are the one who has been there") is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", "de"),
        ("   ", "de"),
        ("Hallo Welt ä", "de"),
        ("you are the one who has been there", "en"),
    ],
)
def test_detect_lang_parametrized(text: str, expected: str) -> None:
    assert detect_lang(text) == expected
