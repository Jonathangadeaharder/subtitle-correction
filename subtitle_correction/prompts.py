"""Unified training/inference prompt format for the mishearing corrector."""

from __future__ import annotations

from .lang_detect import detect_lang


def clean_field(text: str) -> str:
    return text.replace("♪", "").replace("\n", " ").replace("\r", " ").strip()


def detect_task_mode(whisper: str, reference: str) -> str:
    w_lang = detect_lang(whisper)
    r_lang = detect_lang(reference, default="en")
    return "dub" if w_lang != r_lang else "same_lang"


def format_user_body(whisper: str, reference: str, *, task_mode: str | None = None) -> str:
    mode = task_mode or detect_task_mode(whisper, reference)
    lines = [
        f"ASR transcription: {clean_field(whisper)}",
        f"Reference subtitles: {clean_field(reference)}",
    ]
    if mode == "dub":
        lines.append(
            "(ASR may be in a different language than the reference; "
            "output the reference subtitle line, fixing mishearings.)"
        )
    lines.append("Corrected transcription:")
    return "\n".join(lines)


def format_training_text(whisper: str, reference: str, target: str) -> str:
    user_body = format_user_body(whisper, reference)
    return (
        f"<bos><start_of_turn>user\n{user_body}<end_of_turn>\n"
        f"<start_of_turn>model\n{clean_field(target)}<end_of_turn><eos>"
    )


def format_inference_prompt(whisper: str, reference: str) -> str:
    user_body = format_user_body(whisper, reference)
    return f"<bos><start_of_turn>user\n{user_body}<end_of_turn>\n<start_of_turn>model\n"
