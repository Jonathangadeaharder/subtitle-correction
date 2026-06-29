from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedFilename:
    title: str
    season: int | None = None
    episode: int | None = None
    year: int | None = None
    is_tv: bool = False
    raw: str = ""

    @property
    def search_query(self) -> str:
        parts = [self.title]
        if self.year:
            parts.append(str(self.year))
        return " ".join(parts)

    @property
    def episode_label(self) -> str | None:
        if self.season is not None and self.episode is not None:
            return f"S{self.season:02d}E{self.episode:02d}"
        return None


@dataclass
class SubtitleResult:
    id: str
    name: str
    language: str
    download_url: str = ""
    file_name: str = ""
    rating: float = 0.0
    downloads: int = 0
    hearing_impaired: bool = False
    foreign_parts_only: bool = False


@dataclass
class CacheEntry:
    source_file: str
    srt_path: Path
    language: str
    subtitle_id: str = ""
    downloaded_at: str = ""


@dataclass
class CacheMetadata:
    entries: list[CacheEntry] = field(default_factory=list)
