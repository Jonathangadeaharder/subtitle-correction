from __future__ import annotations

import pytest

from subtitle_correction.parser import (
    parse_filename,
    parse_filename_with_guessit,
    smart_parse,
)


def test_parse_filename_movie_with_year() -> None:
    p = parse_filename("Inception.2010.1080p.BluRay.x264.mkv")
    assert p.title == "Inception"
    assert p.year == 2010
    assert p.is_tv is False
    assert p.season is None
    assert p.episode is None
    assert p.raw == "Inception.2010.1080p.BluRay.x264.mkv"


def test_parse_filename_tv_episode() -> None:
    p = parse_filename("Breaking.Bad.S01E01.Pilot.720p.mkv")
    assert p.is_tv is True
    assert p.season == 1
    assert p.episode == 1
    assert "Breaking" in p.title
    assert p.year is None


def test_parse_filename_strips_tags_and_separators() -> None:
    p = parse_filename("Some_Show_S01E02_WEB-DL.mkv")
    assert p.is_tv is True
    assert p.season == 1
    assert p.episode == 2
    assert p.title == "Some Show"


def test_parse_filename_handles_spaces_in_title() -> None:
    p = parse_filename("The Matrix 1999 1080p.mp4")
    assert p.title == "The Matrix"
    assert p.year == 1999


def test_parse_filename_falls_back_to_stem_when_title_empty() -> None:
    # All tags stripped -> title falls back to the file stem.
    p = parse_filename("720p.mkv")
    assert p.title == "720p"


def test_parse_filename_strips_trailing_dashes() -> None:
    p = parse_filename("Title - 2020.mkv")
    # Year matched, title cleaned of trailing dash
    assert p.year == 2020
    assert p.title == "Title"


def test_parse_filename_with_guessit_movie() -> None:
    p = parse_filename_with_guessit("Inception.2010.1080p.BluRay.x264.mkv")
    assert p.title.lower() == "inception"
    assert p.year == 2010


def test_parse_filename_with_guessit_tv() -> None:
    p = parse_filename_with_guessit("Foo.S02E05.720p.mkv")
    assert p.is_tv is True
    assert p.season == 2
    assert p.episode == 5


def test_smart_parse_uses_regex_when_title_found() -> None:
    p = smart_parse("Inception.2010.1080p.mkv")
    assert p.title == "Inception"
    assert p.year == 2010


def test_smart_parse_falls_back_to_guessit_for_tags_only() -> None:
    # Pure tag filename: regex parse yields title == stem, triggers guessit.
    p = smart_parse("720p.mkv")
    assert p.title  # guessit returns something or falls back to stem


@pytest.mark.parametrize(
    "filename",
    [
        "Movie.2020.mkv",
        "Show.S03E07.x264.mkv",
        "plain title with spaces",
        "Film.(2021).mp4",
    ],
)
def test_parse_filename_does_not_crash_on_various_inputs(filename: str) -> None:
    p = parse_filename(filename)
    assert p.title
    assert p.raw == filename
