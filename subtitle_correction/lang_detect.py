"""Lightweight language detection for subtitle text (no external deps)."""

from __future__ import annotations

import re

GERMAN_STOP = {
    "das", "wie", "sie", "zu", "mal", "soll", "des", "sind", "und", "von", "hier",
    "ja", "mein", "aber", "man", "bin", "dein", "was", "bist", "du", "werde",
    "muss", "kann", "noch", "dem", "der", "als", "nur", "habe", "nein", "die",
    "mir", "wenn", "hab", "eine", "den", "war", "ein", "keine", "dann", "hat",
    "dir", "dort", "doch", "kein", "ich", "mit", "hatte", "er", "wir", "nicht",
    "schon", "auch", "ist", "wird", "dass",
}

ENGLISH_STOP = {
    "said", "your", "you", "did", "this", "what", "get", "their", "there", "put",
    "all", "the", "who", "has", "was", "which", "are", "not", "had", "him", "her",
    "they", "that", "with", "his", "and", "let", "from", "have", "for", "will",
    "out", "she", "when", "were", "how", "been", "about", "but", "each", "too",
    "one", "can",
}


def detect_lang(text: str, default: str = "de") -> str:
    if not text or not str(text).strip():
        return default
    text = str(text)
    if re.search(r"[äöüßÄÖÜ]", text):
        return "de"
    words = re.findall(r"[a-zA-ZäöüßÄÖÜ']+", text.lower())
    if not words:
        return default
    de_ratio = len(set(words) & GERMAN_STOP) / len(words)
    if de_ratio > 0.12:
        return "de"
    en_ratio = len(set(words) & ENGLISH_STOP) / len(words)
    if en_ratio > 0.15:
        return "en"
    return default


def is_german_text(text: str) -> bool:
    return detect_lang(text) == "de"