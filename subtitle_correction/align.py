import re
import shutil
import subprocess
from pathlib import Path

import pysrt
from langdetect import detect as detect_lang
from rich.console import Console

console = Console()

WATERMARK_PATTERNS = [
    "untertitelung",
    "zdf",
    "ard",
    "rtl",
    "sat.1",
    "pro7",
    "vox",
    "subtitle",
    "captions provided",
]


def _is_watermark(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in WATERMARK_PATTERNS)


def detect_srt_language(srt_path: Path) -> str:
    subs = pysrt.open(str(srt_path))
    texts = []
    for sub in subs[:50]:
        t = sub.text.strip()
        if t and t != "_" and len(t) > 3:
            texts.append(t)
    text = " ".join(texts)
    if len(text) < 20:
        return "und"
    try:
        return detect_lang(text)
    except Exception:
        return "und"


def extract_language_from_filename(path: Path) -> str | None:
    name = path.stem
    match = re.search(r"[._-]([a-z]{2,3})(?:[._-]|$)", name, re.IGNORECASE)
    if match:
        code = match.group(1).lower()
        common = {
            "en",
            "de",
            "es",
            "fr",
            "it",
            "pt",
            "nl",
            "sv",
            "da",
            "fi",
            "nb",
            "pl",
            "cs",
            "sk",
            "hu",
            "ro",
            "bg",
            "el",
            "tr",
            "ru",
            "ja",
            "ko",
            "zh",
            "ar",
            "he",
            "hi",
            "th",
            "vi",
            "id",
            "ms",
        }
        if code in common:
            return code
    return None


def align_with_alass(
    reference_srt: Path,
    incorrect_srt: Path,
    output_srt: Path,
    split_penalty: int = 10,
) -> Path:
    alass_bin = shutil.which("alass-cli") or shutil.which("alass")
    if not alass_bin:
        raise FileNotFoundError("alass-cli not found. Install with: brew install alass")

    cmd = [
        alass_bin,
        str(reference_srt),
        str(incorrect_srt),
        str(output_srt),
        "--split-penalty",
        str(split_penalty),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"alass failed: {result.stderr}")

    if not output_srt.exists():
        raise FileNotFoundError(f"alass did not produce output: {output_srt}")

    return output_srt


def align_with_ffsubsync(
    video_path: Path,
    input_srt: Path,
    output_srt: Path,
) -> Path:
    cmd = [
        "ffsubsync",
        str(video_path),
        "-i",
        str(input_srt),
        "-o",
        str(output_srt),
        "--skip-sync-on-low-quality",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffsubsync failed: {result.stderr}")

    if not output_srt.exists():
        raise FileNotFoundError(f"ffsubsync did not produce output: {output_srt}")

    return output_srt


def compute_alignment_score(
    whisper_srt: Path, aligned_srt: Path, tolerance_ms: int = 2000
) -> float:
    whisper_subs = pysrt.open(str(whisper_srt))
    aligned_subs = pysrt.open(str(aligned_srt))

    if not whisper_subs or not aligned_subs:
        return 0.0

    whisper_starts = sorted(w.start.ordinal for w in whisper_subs if w.start.ordinal > 0)

    if not whisper_starts:
        return 0.0

    import bisect

    matched = 0
    for a in aligned_subs:
        a_start = a.start.ordinal
        idx = bisect.bisect_left(whisper_starts, a_start)
        best_diff = float("inf")
        for j in (idx - 1, idx):
            if 0 <= j < len(whisper_starts):
                diff = abs(whisper_starts[j] - a_start)
                best_diff = min(best_diff, diff)
        if best_diff <= tolerance_ms:
            matched += 1

    return matched / len(aligned_subs) if aligned_subs else 0.0
