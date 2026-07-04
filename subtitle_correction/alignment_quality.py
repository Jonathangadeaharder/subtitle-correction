"""Classify evaluation cases and detect aligned mishearings vs out-of-scope cues."""

from __future__ import annotations

import re

from .lang_detect import is_german_text

LYRIC_MARKERS = ("♪", "<i>", "</i>")


def normalize_for_compare(text: str) -> str:
    if not text:
        return ""
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^\w\s]", "", text.replace("\n", " ").lower()),
    ).strip()


def word_overlap(a: str, b: str) -> float:
    wa = set(normalize_for_compare(a).split())
    wb = set(normalize_for_compare(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def reference_is_distinct_from_target(reference: str | None, target: str) -> bool:
    if not reference or not str(reference).strip():
        return False
    return normalize_for_compare(reference) != normalize_for_compare(target)


def is_lyric_cue(*texts: str) -> bool:
    combined = " ".join(t for t in texts if t)
    return any(marker in combined for marker in LYRIC_MARKERS)


def is_aligned_mishearing(
    whisper: str,
    ground_truth: str,
    reference: str | None,
    *,
    gt_ref_min: float = 0.35,
    whisper_content_min: float = 0.12,
) -> bool:
    if not (whisper and ground_truth and reference):
        return False
    if is_lyric_cue(whisper, ground_truth, reference):
        return False

    gr = word_overlap(ground_truth, reference)
    if gr < gt_ref_min and normalize_for_compare(ground_truth) != normalize_for_compare(
        reference
    ):
        return False

    wg = word_overlap(whisper, ground_truth)
    wr = word_overlap(whisper, reference)
    if wg >= whisper_content_min or wr >= whisper_content_min:
        return True

    if is_german_text(whisper) and not is_german_text(reference):
        return word_overlap(ground_truth, reference) >= gt_ref_min
    return False


def classify_eval_entry(entry: dict) -> str:
    if not entry.get("has_error"):
        return "identity"
    whisper = entry.get("whisper_text", "")
    ground_truth = entry.get("ground_truth", "")
    reference = entry.get("reference")
    if not reference or not str(reference).strip():
        return "no_reference"
    if is_lyric_cue(whisper, ground_truth, reference):
        return "lyrics"
    if is_aligned_mishearing(whisper, ground_truth, reference):
        return "in_scope"
    return "misaligned"