import json
import re
import shutil
import subprocess
from pathlib import Path

import pysrt
from langdetect import detect as detect_lang
from rich.console import Console

console = Console()

WATERMARK_PATTERNS = ["untertitelung", "zdf", "ard", "rtl", "sat.1", "pro7", "vox", "subtitle", "captions provided"]


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
    match = re.search(r'[._-]([a-z]{2,3})(?:[._-]|$)', name, re.IGNORECASE)
    if match:
        code = match.group(1).lower()
        common = {"en", "de", "es", "fr", "it", "pt", "nl", "sv", "da", "fi",
                   "nb", "pl", "cs", "sk", "hu", "ro", "bg", "el", "tr", "ru",
                   "ja", "ko", "zh", "ar", "he", "hi", "th", "vi", "id", "ms"}
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
        "--split-penalty", str(split_penalty),
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
        "-i", str(input_srt),
        "-o", str(output_srt),
        "--skip-sync-on-low-quality",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffsubsync failed: {result.stderr}")

    if not output_srt.exists():
        raise FileNotFoundError(f"ffsubsync did not produce output: {output_srt}")

    return output_srt


def compute_alignment_score(whisper_srt: Path, aligned_srt: Path, tolerance_ms: int = 2000) -> float:
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


def generate_training_pairs(
    whisper_srt: Path,
    aligned_srt: Path,
    source_file: str = "",
    min_score: float = 0.5,
    alignment_score: float = 1.0,
    overlap_threshold: float = 0.05,
) -> list[dict]:
    whisper_subs = pysrt.open(str(whisper_srt))
    aligned_subs = pysrt.open(str(aligned_srt))

    if alignment_score < min_score:
        console.print(f"[yellow]Skipping pairs (alignment score {alignment_score:.2f} < {min_score})[/yellow]")
        return []

    import bisect
    whisper_entries = [(w.start.ordinal, w) for w in whisper_subs]
    whisper_starts = sorted(w.start.ordinal for w in whisper_subs)

    pairs = []
    skipped_unrelated = 0
    skipped_rephrasing = 0

    for a in aligned_subs:
        a_start = a.start.ordinal
        idx = bisect.bisect_left(whisper_starts, a_start)
        best_w = None
        best_diff = float("inf")
        for j in (idx - 1, idx):
            if 0 <= j < len(whisper_entries):
                diff = abs(whisper_entries[j][0] - a_start)
                if diff < best_diff:
                    best_diff = diff
                    best_w = whisper_entries[j][1]

        if not best_w or best_diff > 4000:
            continue

        whisper_text = best_w.text.strip()
        aligned_text = a.text.strip()

        if not whisper_text or not aligned_text:
            continue
        if _is_watermark(whisper_text) or _is_watermark(aligned_text):
            continue
        if whisper_text == aligned_text:
            continue
        if len(whisper_text) < 3 or len(aligned_text) < 3:
            continue

        stop_words = {"the", "a", "an", "to", "of", "and", "or", "but", "in", "on", "at", "for", "with", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their"}

        def _content_words(text: str) -> set[str]:
            return {w.strip(".,!?;:\"'") for w in text.lower().split() if w.strip(".,!?;:\"'") and w.strip(".,!?;:\"'") not in stop_words}

        w_words = _content_words(whisper_text)
        a_words = _content_words(aligned_text)

        shared = w_words & a_words
        overlap_ratio = len(shared) / max(len(w_words), len(a_words)) if w_words and a_words else 0.0

        if overlap_ratio < overlap_threshold:
            skipped_unrelated += 1
            continue

        if overlap_ratio < 0.2:
            w_only = w_words - a_words
            a_only = a_words - w_words
            if w_only and a_only and not _is_phonetic_mishearing(w_only, a_only, whisper_text, aligned_text):
                skipped_rephrasing += 1
                continue

        pairs.append({
            "whisper_text": whisper_text,
            "corrected_text": aligned_text,
            "source_file": source_file,
            "alignment_score": round(alignment_score, 3),
            "whisper_lang": "",
            "subtitle_lang": "",
            "timestamp_start": str(a.start),
            "timestamp_end": str(a.end),
        })

    console.print(f"[dim]Pairs: {len(pairs)} kept, {skipped_unrelated} skipped (unrelated), {skipped_rephrasing} skipped (rephrasing, not mishearing)[/dim]")

    return pairs


def _is_phonetic_mishearing(whisper_only: set, subtitle_only: set, full_whisper: str, full_subtitle: str) -> bool:
    if not whisper_only or not subtitle_only:
        return False

    import re

    def _clean(w: str) -> str:
        return re.sub(r'[^a-zà-ÿ]', '', w.lower())

    def _levenshtein(a: str, b: str) -> int:
        if len(a) < len(b):
            a, b = b, a
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(ca != cb)))
            prev = curr
        return prev[-1]

    def _is_close(w: str, s: str) -> bool:
        wc = _clean(w)
        sc = _clean(s)
        if not wc or not sc:
            return False
        if abs(len(wc) - len(sc)) > 5:
            return False
        dist = _levenshtein(wc, sc)
        return dist / max(len(wc), len(sc)) <= 0.55

    for w in whisper_only:
        for s in subtitle_only:
            if _is_close(w, s):
                return True

    return False


def append_pairs_to_jsonl(pairs: list[dict], output_path: Path) -> int:
    if not pairs:
        return 0

    with open(output_path, "a", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    return len(pairs)
