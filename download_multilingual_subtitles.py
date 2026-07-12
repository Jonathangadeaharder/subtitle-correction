"""
Downloads Spanish (es), German (de), and French (fr) subtitles for all
episodes present in the subcache directory and aligns them.
"""

import argparse
import os
import random
import time
from pathlib import Path

from subtitle_correction.align import align_with_alass, compute_alignment_score
from subtitle_correction.parser import smart_parse
from subtitle_correction.scraper import OpenSubtitlesScraper


def _default_subcache() -> Path:
    return Path(os.getenv("SUBTITLE_SUBCACHE", Path.home() / "Downloads" / ".subcache"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subcache", type=Path, default=_default_subcache())
    parser.add_argument("--languages", nargs="+", default=["es", "de", "fr"])
    args = parser.parse_args()

    subcache = args.subcache
    scraper = OpenSubtitlesScraper()
    try:
        _download_and_align(subcache, args.languages, scraper)
    finally:
        scraper.close()


def _download_and_align(
    subcache: Path, languages: list[str], scraper: OpenSubtitlesScraper
) -> None:
    for d in sorted(subcache.iterdir()):
        if not d.is_dir():
            continue

        # Parse TV show name and season/episode from directory name
        dir_name = d.name.replace("_S.to", "").replace("_1", "")
        # e.g., Fargo_S02E02_S.to -> Fargo S02E02
        clean_name = dir_name.replace("_", " ")
        parsed = smart_parse(clean_name)

        if not parsed.title:
            print(f"Skipping non-TV folder: {d.name}")
            continue

        print(
            f"\n=== Processing folder: {d.name} ({parsed.title} S{parsed.season or 0:02d}E{parsed.episode or 0:02d}) ==="
        )

        # Audio or Whisper reference to align against
        whisper_srt = d / "whisper.srt"
        if not whisper_srt.exists():
            print(f"No whisper.srt found in {d.name}, skipping alignment.")
            continue

        for lang in languages:
            out_srt = d / f"opensubtitles.{lang}.srt"
            aligned_srt = d / f"aligned.{lang}.srt"

            # Skip if already exists
            if out_srt.exists() and aligned_srt.exists():
                print(f"  [{lang}] Already exists and aligned.")
                continue

            print(f"  [{lang}] Downloading from OpenSubtitles...")
            try:
                # Add delay to avoid rate limiting
                time.sleep(random.uniform(2.0, 5.0))
                res = scraper.search_and_download(parsed, language=lang, output_dir=d)
                if res:
                    # Rename to standard opensubtitles.{lang}.srt
                    Path(res).rename(out_srt)
                    print(f"  [{lang}] Downloaded to {out_srt.name}")
                else:
                    print(f"  [{lang}] No subtitle found.")
                    continue
            except Exception as e:
                print(f"  [{lang}] Download failed: {e}")
                continue

            # Align downloaded subtitle using alass
            if out_srt.exists():
                print(f"  [{lang}] Aligning with alass...")
                try:
                    align_with_alass(whisper_srt, out_srt, aligned_srt, split_penalty=10)
                    score = compute_alignment_score(whisper_srt, aligned_srt)
                    print(f"  [{lang}] Aligned successfully (score: {score:.2%})")
                except Exception as e:
                    print(f"  [{lang}] Alignment failed: {e}")


if __name__ == "__main__":
    main()
