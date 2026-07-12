import re
from pathlib import Path

from .models import ParsedFilename

_SEASON_EPISODE_RE = re.compile(r"[Ss](\d{1,2})\s*[Ee](\d{1,2})")

_YEAR_RE = re.compile(r"(?:^|[\s.(])(\d{4})(?:[\s.)]|$)")

_COMMON_TAGS = re.compile(
    r"(?i)(?:_?compressed|_?mp4|_?mkv|720p|1080p|2160p|4k|"
    r"WEB\.?h264|WEB-DL|WEBRip|BDRip|BluRay|x264|x265|HEVC|"
    r"AAC|DD5\.?1|H\.?264|H\.?265|REPACK|PROPER|"
    r"GERMAN|Forced|SAUERKRAUT|DisneyHD|S\.to|"
    r"\.(?:mp4|mkv|avi|srt|ass|ssa))",
    re.IGNORECASE,
)

_SEPARATORS_RE = re.compile(r"[._]+")


def parse_filename(filename: str) -> ParsedFilename:
    name = Path(filename).stem
    name = _COMMON_TAGS.sub("", name)
    name = _SEPARATORS_RE.sub(" ", name).strip()

    se_match = _SEASON_EPISODE_RE.search(name)
    year_match = _YEAR_RE.search(name)

    title = name
    season = None
    episode = None
    year = None
    is_tv = False

    if se_match:
        season = int(se_match.group(1))
        episode = int(se_match.group(2))
        is_tv = True
        title = name[: se_match.start()].strip()

    if year_match and not is_tv:
        year = int(year_match.group(1))
        title = name[: year_match.start()].strip()

    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"[-\s]+$", "", title).strip()
    title = re.sub(r"^[-\s]+", "", title).strip()

    if not title:
        title = Path(filename).stem

    return ParsedFilename(
        title=title,
        season=season,
        episode=episode,
        year=year,
        is_tv=is_tv,
        raw=filename,
    )


def parse_filename_with_guessit(filename: str) -> ParsedFilename:
    try:
        from guessit import guessit as guessit_fn

        result = guessit_fn(filename)
        title = result.get("title", Path(filename).stem)
        season = result.get("season")
        episode = result.get("episode")
        year = result.get("year")
        is_tv = season is not None or episode is not None

        return ParsedFilename(
            title=str(title),
            season=int(season) if season is not None else None,
            episode=int(episode) if episode is not None else None,
            year=int(year) if year is not None else None,
            is_tv=is_tv,
            raw=filename,
        )
    except ImportError:
        return parse_filename(filename)


def smart_parse(filename: str) -> ParsedFilename:
    parsed = parse_filename(filename)
    if not parsed.title or parsed.title == Path(filename).stem:
        return parse_filename_with_guessit(filename)
    return parsed
