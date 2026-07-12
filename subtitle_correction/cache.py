from pathlib import Path

from .models import CacheEntry, CacheMetadata

CACHE_DIR = Path.home() / ".opensubtitles-cache"
METADATA_FILE = CACHE_DIR / "metadata.json"
DOWNLOADS_DIR = CACHE_DIR / "downloads"


def ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def get_cache_metadata() -> CacheMetadata:
    import json

    if METADATA_FILE.exists():
        data = json.loads(METADATA_FILE.read_text())
        entries = [
            CacheEntry(
                source_file=e["source_file"],
                srt_path=Path(e["srt_path"]),
                language=e["language"],
                subtitle_id=e.get("subtitle_id", ""),
                downloaded_at=e.get("downloaded_at", ""),
            )
            for e in data.get("entries", [])
        ]
        return CacheMetadata(entries=entries)
    return CacheMetadata()


def save_cache_metadata(metadata: CacheMetadata) -> None:
    import json
    from datetime import datetime

    entries = []
    for e in metadata.entries:
        entries.append(
            {
                "source_file": e.source_file,
                "srt_path": str(e.srt_path),
                "language": e.language,
                "subtitle_id": e.subtitle_id,
                "downloaded_at": e.downloaded_at or datetime.now().isoformat(),
            }
        )
    METADATA_FILE.write_text(json.dumps({"entries": entries}, indent=2))


def find_cached_subtitle(source_file: str, language: str) -> Path | None:
    metadata = get_cache_metadata()
    for entry in metadata.entries:
        if entry.source_file == source_file and entry.language == language:
            if entry.srt_path.exists():
                return entry.srt_path
    return None


def add_cache_entry(source_file: str, srt_path: Path, language: str, subtitle_id: str = "") -> None:
    metadata = get_cache_metadata()
    metadata.entries.append(
        CacheEntry(
            source_file=source_file,
            srt_path=srt_path,
            language=language,
            subtitle_id=subtitle_id,
        )
    )
    save_cache_metadata(metadata)
