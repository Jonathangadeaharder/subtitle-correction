from __future__ import annotations

from pathlib import Path

from subtitle_correction.models import (
    CacheEntry,
    CacheMetadata,
    ParsedFilename,
    SubtitleResult,
)


def test_parsed_filename_defaults() -> None:
    p = ParsedFilename(title="Inception")
    assert p.title == "Inception"
    assert p.season is None
    assert p.episode is None
    assert p.year is None
    assert p.is_tv is False
    assert p.raw == ""


def test_parsed_filename_search_query_with_year() -> None:
    p = ParsedFilename(title="Inception", year=2010)
    assert p.search_query == "Inception 2010"


def test_parsed_filename_search_query_without_year() -> None:
    assert ParsedFilename(title="Inception").search_query == "Inception"


def test_parsed_filename_episode_label() -> None:
    p = ParsedFilename(title="Foo", season=2, episode=3, is_tv=True)
    assert p.episode_label == "S02E03"


def test_parsed_filename_episode_label_none_when_missing() -> None:
    assert ParsedFilename(title="Foo").episode_label is None
    assert ParsedFilename(title="Foo", season=1).episode_label is None
    assert ParsedFilename(title="Foo", episode=1).episode_label is None


def test_subtitle_result_defaults() -> None:
    r = SubtitleResult(id="x", name="n", language="en")
    assert r.download_url == ""
    assert r.file_name == ""
    assert r.rating == 0.0
    assert r.downloads == 0
    assert r.hearing_impaired is False
    assert r.foreign_parts_only is False


def test_cache_entry_defaults() -> None:
    e = CacheEntry(source_file="a.mp4", srt_path=Path("/tmp/a.srt"), language="en")
    assert e.subtitle_id == ""
    assert e.downloaded_at == ""


def test_cache_metadata_default_empty() -> None:
    m = CacheMetadata()
    assert m.entries == []


def test_cache_metadata_with_entries() -> None:
    e = CacheEntry(source_file="a.mp4", srt_path=Path("/tmp/a.srt"), language="en")
    m = CacheMetadata(entries=[e])
    assert m.entries == [e]
