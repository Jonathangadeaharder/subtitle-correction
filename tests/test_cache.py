from __future__ import annotations

from pathlib import Path

from subtitle_correction import cache as cache_mod
from subtitle_correction.cache import (
    add_cache_entry,
    ensure_cache_dir,
    find_cached_subtitle,
    get_cache_metadata,
    save_cache_metadata,
)
from subtitle_correction.models import CacheEntry, CacheMetadata


def test_ensure_cache_dir_creates_dirs() -> None:
    result = ensure_cache_dir()
    assert result == cache_mod.CACHE_DIR
    assert cache_mod.CACHE_DIR.exists()
    assert cache_mod.DOWNLOADS_DIR.exists()


def test_get_cache_metadata_empty_when_missing() -> None:
    assert get_cache_metadata().entries == []


def test_save_and_load_roundtrip() -> None:
    srt = cache_mod.DOWNLOADS_DIR / "foo.srt"
    srt.parent.mkdir(parents=True, exist_ok=True)
    srt.write_text("1\n", encoding="utf-8")
    entry = CacheEntry(
        source_file="movie.mp4",
        srt_path=srt,
        language="en",
        subtitle_id="123",
        downloaded_at="2024-01-01T00:00:00",
    )
    save_cache_metadata(CacheMetadata(entries=[entry]))

    loaded = get_cache_metadata()
    assert len(loaded.entries) == 1
    got = loaded.entries[0]
    assert got.source_file == "movie.mp4"
    assert got.srt_path == srt
    assert got.language == "en"
    assert got.subtitle_id == "123"
    assert got.downloaded_at == "2024-01-01T00:00:00"


def test_save_cache_metadata_fills_downloaded_at() -> None:
    entry = CacheEntry(
        source_file="m.mp4",
        srt_path=Path("/tmp/m.srt"),
        language="de",
    )
    save_cache_metadata(CacheMetadata(entries=[entry]))
    loaded = get_cache_metadata()
    assert loaded.entries[0].downloaded_at != ""


def test_find_cached_subtitle_hit_when_srt_exists() -> None:
    srt = cache_mod.DOWNLOADS_DIR / "hit.srt"
    srt.parent.mkdir(parents=True, exist_ok=True)
    srt.write_text("1\n", encoding="utf-8")
    add_cache_entry("movie.mkv", srt, "en", subtitle_id="9")
    found = find_cached_subtitle("movie.mkv", "en")
    assert found == srt


def test_find_cached_subtitle_miss_when_language_differs() -> None:
    srt = cache_mod.DOWNLOADS_DIR / "lang.srt"
    srt.parent.mkdir(parents=True, exist_ok=True)
    srt.write_text("1\n", encoding="utf-8")
    add_cache_entry("movie.mkv", srt, "en")
    assert find_cached_subtitle("movie.mkv", "de") is None


def test_find_cached_subtitle_miss_when_srt_deleted() -> None:
    # File referenced in metadata no longer exists on disk.
    srt = cache_mod.DOWNLOADS_DIR / "gone.srt"
    srt.parent.mkdir(parents=True, exist_ok=True)
    srt.write_text("1\n", encoding="utf-8")
    add_cache_entry("gone.mkv", srt, "en")
    srt.unlink()
    assert find_cached_subtitle("gone.mkv", "en") is None


def test_find_cached_subtitle_miss_for_unknown_source() -> None:
    add_cache_entry("a.mkv", cache_mod.DOWNLOADS_DIR / "a.srt", "en")
    assert find_cached_subtitle("totally-unknown.mkv", "en") is None


def test_get_cache_metadata_handles_empty_entries_field() -> None:
    # Metadata file present but with no "entries" key.
    cache_mod.METADATA_FILE.write_text("{}", encoding="utf-8")
    assert get_cache_metadata().entries == []


def test_add_cache_entry_appends_without_overwriting() -> None:
    srt1 = cache_mod.DOWNLOADS_DIR / "one.srt"
    srt2 = cache_mod.DOWNLOADS_DIR / "two.srt"
    srt1.parent.mkdir(parents=True, exist_ok=True)
    srt1.write_text("1\n", encoding="utf-8")
    srt2.write_text("1\n", encoding="utf-8")
    add_cache_entry("a.mkv", srt1, "en")
    add_cache_entry("b.mkv", srt2, "de")
    assert len(get_cache_metadata().entries) == 2
