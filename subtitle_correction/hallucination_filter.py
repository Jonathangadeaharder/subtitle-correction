import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pysrt


@dataclass
class SrtBlock:
    index: int
    start_sec: float
    end_sec: float
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class FilterStats:
    total: int = 0
    dropped_vad: int = 0
    dropped_confidence: int = 0
    dropped_repetition: int = 0
    dropped_hallucination_phrase: int = 0
    kept: int = 0
    reasons: list[str] = field(default_factory=list)


COMMON_HALLUCINATION_PHRASES = {
    "en": [
        "thank you",
        "thanks for watching",
        "please subscribe",
        "thank you for watching",
        "bye",
        "bye-bye",
        "see you next time",
        "i'll see you next time",
        "see you",
        "please like and subscribe",
        "don't forget to subscribe",
        "subscribe for more",
        "thanks for tuning in",
        "thank you for tuning in",
        "that's all for today",
        "thanks for listening",
        "thank you for listening",
        "thanks for having me",
        "thank you for having me",
        "so yeah",
        "yeah so",
        "and so",
        "and yeah",
        "so um",
        "um so",
        "uh huh",
        "mm-hmm",
        "mm hmm",
        "okay so",
        "so okay",
        "right so",
        "alright so",
    ],
    "de": [
        "danke",
        "vielen dank",
        "auf wiedersehen",
        "tschüss",
        "danke schön",
        "danke sehr",
        "vielen dank fürs zuschauen",
        "auf wiederhören",
    ],
}


def parse_srt(srt_path: Path) -> list[SrtBlock]:
    subs = pysrt.open(str(srt_path), encoding="utf-8")
    blocks = []
    for i, sub in enumerate(subs):
        start_sec = sub.start.ordinal / 1000.0
        end_sec = sub.end.ordinal / 1000.0
        blocks.append(
            SrtBlock(
                index=i + 1,
                start_sec=start_sec,
                end_sec=end_sec,
                text=sub.text.strip(),
            )
        )
    return blocks


def parse_srt_with_confidence(srt_path: Path, json_path: Path) -> list[SrtBlock]:
    blocks = parse_srt(srt_path)
    if not json_path.exists():
        return blocks

    data = json.loads(json_path.read_text())
    segments = data.get("segments", [])
    for block, seg in zip(blocks, segments):
        block.no_speech_prob = seg.get("no_speech_prob", 0.0)
        block.avg_logprob = seg.get("avg_logprob", 0.0)
        block.compression_ratio = seg.get("compression_ratio", 1.0)
    return blocks


def write_srt(blocks: list[SrtBlock], output_path: Path) -> None:
    def fmt_time(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

    lines = []
    for i, block in enumerate(blocks, 1):
        lines.append(str(i))
        lines.append(f"{fmt_time(block.start_sec)} --> {fmt_time(block.end_sec)}")
        lines.append(block.text)
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_vad(
    audio_path: Path,
    threshold: float = 0.3,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 500,
    speech_pad_ms: int = 200,
) -> list[dict]:
    from mlx_audio.vad import load as load_vad

    model = load_vad("mlx-community/silero-vad")
    timestamps = model.get_speech_timestamps(
        str(audio_path),
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True,
    )
    return timestamps


def _overlaps_speech(
    block: SrtBlock, speech_segments: list[dict], overlap_threshold: float = 0.15
) -> bool:
    if not speech_segments:
        return True
    block_dur = block.duration
    if block_dur <= 0:
        return True
    for seg in speech_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        overlap_start = max(block.start_sec, seg_start)
        overlap_end = min(block.end_sec, seg_end)
        overlap = overlap_end - overlap_start
        if overlap > 0 and (overlap / block_dur) >= overlap_threshold:
            return True
    return False


def filter_by_vad(
    blocks: list[SrtBlock], speech_segments: list[dict], stats: FilterStats
) -> list[SrtBlock]:
    kept = []
    for block in blocks:
        if _overlaps_speech(block, speech_segments):
            kept.append(block)
        else:
            stats.dropped_vad += 1
            stats.reasons.append(f"  Block {block.index}: VAD silence (text={block.text[:50]!r})")
    return kept


def filter_by_confidence(
    blocks: list[SrtBlock],
    stats: FilterStats,
    no_speech_threshold: float = 0.6,
    logprob_threshold: float = -1.0,
    compression_ratio_threshold: float = 2.4,
) -> list[SrtBlock]:
    kept = []
    for block in blocks:
        reasons = []
        if block.no_speech_prob > no_speech_threshold:
            reasons.append(f"no_speech_prob={block.no_speech_prob:.3f}")
        if block.avg_logprob < logprob_threshold:
            reasons.append(f"avg_logprob={block.avg_logprob:.3f}")
        if block.compression_ratio > compression_ratio_threshold:
            reasons.append(f"compression_ratio={block.compression_ratio:.2f}")

        if reasons and block.no_speech_prob > no_speech_threshold:
            stats.dropped_confidence += 1
            stats.reasons.append(
                f"  Block {block.index}: confidence ({', '.join(reasons)}, text={block.text[:50]!r})"
            )
            continue
        kept.append(block)
    return kept


def _normalize_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower().strip())


def filter_repetitions(
    blocks: list[SrtBlock], stats: FilterStats, repeat_count: int = 3
) -> list[SrtBlock]:
    if len(blocks) < repeat_count:
        return blocks
    kept = []
    i = 0
    while i < len(blocks):
        norm = _normalize_text(blocks[i].text)
        run_len = 1
        j = i + 1
        while j < len(blocks) and _normalize_text(blocks[j].text) == norm and norm:
            run_len += 1
            j += 1
        if run_len >= repeat_count:
            for k in range(i, j):
                stats.dropped_repetition += 1
                stats.reasons.append(
                    f"  Block {blocks[k].index}: repetition x{run_len} (text={blocks[k].text[:50]!r})"
                )
            i = j
        else:
            kept.append(blocks[i])
            i += 1
    return kept


def filter_hallucination_phrases(
    blocks: list[SrtBlock], stats: FilterStats, lang: str = "en"
) -> list[SrtBlock]:
    phrases = COMMON_HALLUCINATION_PHRASES.get(lang, COMMON_HALLUCINATION_PHRASES["en"])
    kept = []
    for block in blocks:
        norm = _normalize_text(block.text)
        if norm in phrases:
            stats.dropped_hallucination_phrase += 1
            stats.reasons.append(
                f"  Block {block.index}: hallucination phrase (text={block.text[:50]!r})"
            )
            continue
        kept.append(block)
    return kept


def filter_hallucinations(
    srt_path: Path,
    audio_path: Path,
    output_path: Path,
    json_path: Path | None = None,
    lang: str = "en",
    vad_threshold: float = 0.5,
    no_speech_threshold: float = 0.6,
    logprob_threshold: float = -1.0,
    compression_ratio_threshold: float = 2.4,
    enable_vad: bool = True,
    enable_confidence: bool = True,
    enable_heuristics: bool = True,
) -> FilterStats:
    stats = FilterStats()

    if json_path and json_path.exists():
        blocks = parse_srt_with_confidence(srt_path, json_path)
    else:
        blocks = parse_srt(srt_path)
    stats.total = len(blocks)

    if not blocks:
        return stats

    if enable_confidence and json_path and json_path.exists():
        blocks = filter_by_confidence(
            blocks,
            stats,
            no_speech_threshold=no_speech_threshold,
            logprob_threshold=logprob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
        )

    if enable_vad and audio_path.exists():
        try:
            speech_segments = run_vad(audio_path, threshold=vad_threshold)
            blocks = filter_by_vad(blocks, speech_segments, stats)
        except Exception:
            pass

    if enable_heuristics:
        blocks = filter_repetitions(blocks, stats)
        blocks = filter_hallucination_phrases(blocks, stats, lang=lang)

    stats.kept = len(blocks)
    write_srt(blocks, output_path)
    return stats
